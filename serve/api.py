"""
FastAPI server: streams mesh, telemetry and alerts to the React frontend.

Endpoints:
    GET  /health                 service + model status
    GET  /api/calibration        current rig calibration
    POST /api/scan/start         trigger a dome scan
    GET  /api/scan/latest        most recent mesh payload
    POST /api/hazards/analyse    baseline-vs-current comparison
    POST /api/gas/assess         gas reading -> physics + ML verdict
    WS   /ws/mesh                progressive mesh updates during a scan
    WS   /ws/telemetry           live alert / status feed

Run:
    uvicorn serve.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nexcrawl_ml.gas.physics import classify_atmosphere, time_to_lel
from nexcrawl_ml.gas.windowing import extract_window_features
from nexcrawl_ml.perception.calibration import RigCalibration
from nexcrawl_ml.perception.reconstructor import mesh_to_payload
from nexcrawl_ml.pipeline.alert_fusion import fuse_alerts, gas_alerts, structural_alerts
from nexcrawl_ml.pipeline.dome_scan import DomeScanConfig, DomeScanner
from nexcrawl_ml.pipeline.hazard_pipeline import HazardPipeline

MODEL_DIR = Path("data/models")
CONFIG_DIR = Path("config")

state: dict = {
    "calibration": RigCalibration(),
    "structural_model": None,
    "gas_model": None,
    "knock_detector": None,
    "latest_mesh": None,
    "latest_cloud": None,
    "baseline_cloud": None,
    "scanning": False,
}


class ConnectionManager:
    """Tracks websocket clients and broadcasts, dropping dead connections."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


mesh_clients = ConnectionManager()
telemetry_clients = ConnectionManager()


def _load_models() -> None:
    """Load whatever is available; missing models degrade gracefully rather
    than crashing the server - useful mid-build when only some are trained."""
    calib_path = CONFIG_DIR / "rig_calibration.json"
    if calib_path.exists():
        state["calibration"] = RigCalibration.load(calib_path)

    structural = MODEL_DIR / "structural_model.txt"
    if structural.exists():
        try:
            from nexcrawl_ml.models.structural_classifier import StructuralClassifier

            state["structural_model"] = StructuralClassifier.load(structural)
        except Exception as exc:
            print(f"[api] structural model load failed: {exc}")

    gas = MODEL_DIR / "gas_model.txt"
    if gas.exists():
        try:
            from nexcrawl_ml.models.gas_classifier import GasTrendClassifier

            state["gas_model"] = GasTrendClassifier.load(gas)
        except Exception as exc:
            print(f"[api] gas model load failed: {exc}")

    knock = MODEL_DIR / "knocknet.onnx"
    if knock.exists():
        try:
            from nexcrawl_ml.models.knocknet import ONNXKnockDetector

            state["knock_detector"] = ONNXKnockDetector(knock)
        except Exception as exc:
            print(f"[api] knocknet load failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(title="NexCrawl-M ML API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScanRequest(BaseModel):
    simulate: bool = True
    phi_start_deg: float = -45.0
    phi_end_deg: float = 45.0
    step_deg: float = 2.5
    method: str = "bpa"
    set_as_baseline: bool = False


class GasReading(BaseModel):
    ch4_percent: float
    co_ppm: float
    o2_percent: float
    co2_ppm: float = 400.0
    prior_ch4_percent: float | None = None
    prior_o2_percent: float | None = None
    history: list[float] | None = None
    timestamps: list[float] | None = None


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "models": {
            "structural": state["structural_model"] is not None,
            "gas": state["gas_model"] is not None,
            "knocknet": state["knock_detector"] is not None,
        },
        "calibrated": state["calibration"].residual_rms_m is not None,
        "has_baseline": state["baseline_cloud"] is not None,
        "scanning": state["scanning"],
    }


@app.get("/api/calibration")
async def get_calibration() -> dict:
    c = state["calibration"]
    return {
        "lever_arm_m": list(c.lever_arm_m),
        "phi_zero_offset_deg": float(np.degrees(c.phi_zero_offset_rad)),
        "mast_height_m": c.mast_height_m,
        "residual_rms_mm": (
            c.residual_rms_m * 1000 if c.residual_rms_m is not None else None
        ),
    }


@app.post("/api/scan/start")
async def start_scan(req: ScanRequest) -> dict:
    if state["scanning"]:
        raise HTTPException(409, "a scan is already running")

    state["scanning"] = True
    try:
        if req.simulate:
            from nexcrawl_ml.hw_io.lidar_reader import SimulatedLidar
            from nexcrawl_ml.hw_io.serial_reader import SimulatedServoLink

            lidar, servo = SimulatedLidar(), SimulatedServoLink()
        else:
            from nexcrawl_ml.hw_io.lidar_reader import RPLidarC1Reader
            from nexcrawl_ml.hw_io.serial_reader import ESP32ServoLink

            lidar = RPLidarC1Reader()
            servo = ESP32ServoLink()
            servo.start()

        scanner = DomeScanner(
            lidar,
            servo,
            calibration=state["calibration"],
            config=DomeScanConfig(
                phi_start_deg=req.phi_start_deg,
                phi_end_deg=req.phi_end_deg,
                step_deg=req.step_deg,
                method=req.method,
            ),
        )

        loop = asyncio.get_event_loop()

        def on_progressive(mesh, progress):
            asyncio.run_coroutine_threadsafe(
                mesh_clients.broadcast(
                    {
                        "type": "mesh_partial",
                        "progress": progress,
                        "mesh": mesh_to_payload(mesh),
                    }
                ),
                loop,
            )

        result = await loop.run_in_executor(
            None, lambda: scanner.run(on_progressive_mesh=on_progressive)
        )

        state["latest_cloud"] = result.point_cloud
        if req.set_as_baseline:
            state["baseline_cloud"] = result.point_cloud

        payload = mesh_to_payload(result.mesh) if result.mesh else None
        state["latest_mesh"] = payload

        if payload:
            await mesh_clients.broadcast(
                {"type": "mesh_complete", "progress": 1.0, "mesh": payload}
            )

        return {
            "success": result.mesh is not None,
            "steps_completed": result.steps_completed,
            "steps_attempted": result.steps_attempted,
            "duration_s": result.duration_s,
            "stats": result.stats.as_dict(),
            "is_baseline": req.set_as_baseline,
        }
    finally:
        state["scanning"] = False


@app.get("/api/scan/latest")
async def latest_scan() -> dict:
    if state["latest_mesh"] is None:
        raise HTTPException(404, "no scan has been run yet")
    return {"mesh": state["latest_mesh"]}


@app.post("/api/hazards/analyse")
async def analyse_hazards() -> dict:
    if state["structural_model"] is None:
        raise HTTPException(503, "structural model not loaded")
    if state["baseline_cloud"] is None:
        raise HTTPException(400, "no baseline scan stored")
    if state["latest_cloud"] is None:
        raise HTTPException(400, "no current scan stored")

    pipeline = HazardPipeline(state["structural_model"])
    report = pipeline.run(state["latest_cloud"], state["baseline_cloud"])
    alerts = fuse_alerts(structural_alerts(report))

    payload = {
        "report": report.as_dict(),
        "alerts": [a.as_dict() for a in alerts],
    }
    await telemetry_clients.broadcast({"type": "hazards", **payload})
    return payload


@app.post("/api/gas/assess")
async def assess_gas(reading: GasReading) -> dict:
    atmosphere = classify_atmosphere(
        reading.ch4_percent,
        reading.co_ppm,
        reading.o2_percent,
        co2_ppm=reading.co2_ppm,
        prior_ch4_percent=reading.prior_ch4_percent,
        prior_o2_percent=reading.prior_o2_percent,
    )

    trend_prediction, ttl = None, None
    if reading.history and len(reading.history) >= 4:
        ts = (
            np.array(reading.timestamps)
            if reading.timestamps
            else np.arange(len(reading.history), dtype=float)
        )
        features = extract_window_features(np.array(reading.history), ts)
        ttl = time_to_lel(reading.ch4_percent, features.rate_of_rise)

        if state["gas_model"] is not None:
            try:
                trend_prediction = state["gas_model"].predict_with_severity(
                    features.to_array().reshape(1, -1)
                )[0]
            except Exception as exc:
                print(f"[api] gas prediction failed: {exc}")

    alerts = fuse_alerts(gas_alerts(atmosphere, trend_prediction, ttl))
    payload = {
        "atmosphere": atmosphere,
        "trend": trend_prediction,
        "time_to_lel_s": ttl,
        "alerts": [a.as_dict() for a in alerts],
    }
    await telemetry_clients.broadcast({"type": "gas", **payload})
    return payload


@app.websocket("/ws/mesh")
async def ws_mesh(websocket: WebSocket) -> None:
    await mesh_clients.connect(websocket)
    try:
        if state["latest_mesh"]:
            await websocket.send_json(
                {"type": "mesh_complete", "progress": 1.0,
                 "mesh": state["latest_mesh"]}
            )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        mesh_clients.disconnect(websocket)


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket) -> None:
    await telemetry_clients.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        telemetry_clients.disconnect(websocket)
