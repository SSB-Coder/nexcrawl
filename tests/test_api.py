"""Tests for the FastAPI server: health checks, gas assessment (including
the nitrogen-balance + tamper-detection fixes), and hazard/scan error paths.

Uses TestClient's context manager form so the lifespan handler runs (loading
whatever models happen to be present under data/models/).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="fastapi/httpx not installed"
)


@pytest.fixture
def client():
    from serve.api import app

    with fastapi_testclient.TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "models" in body
        assert set(body["models"].keys()) == {"structural", "gas", "knocknet"}

    def test_calibration_endpoint_returns_defaults_without_file(self, client):
        resp = client.get("/api/calibration")
        assert resp.status_code == 200
        body = resp.json()
        assert "lever_arm_m" in body
        assert len(body["lever_arm_m"]) == 3


class TestGasAssessEndpoint:
    def test_safe_reading_produces_no_alerts(self, client):
        resp = client.post(
            "/api/gas/assess",
            json={"ch4_percent": 0.1, "co_ppm": 5, "o2_percent": 20.9},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["atmosphere"]["overall_safe"]
        assert body["alerts"] == []

    def test_explosive_reading_produces_critical_alert(self, client):
        resp = client.post(
            "/api/gas/assess",
            json={"ch4_percent": 9.5, "co_ppm": 50, "o2_percent": 20.9},
        )
        body = resp.json()
        assert not body["atmosphere"]["overall_safe"]
        assert any(a["priority_name"] == "CRITICAL" for a in body["alerts"])

    def test_co2_ppm_defaults_and_affects_grahams_ratio(self, client):
        """Regression check for the nitrogen-balance fix: co2_ppm must be
        accepted and actually change the result, not just be silently
        ignored (which the old simplified formula would have done)."""
        low_co2 = client.post(
            "/api/gas/assess",
            json={"ch4_percent": 0.2, "co_ppm": 200, "o2_percent": 19.0, "co2_ppm": 400},
        ).json()
        high_co2 = client.post(
            "/api/gas/assess",
            json={"ch4_percent": 0.2, "co_ppm": 200, "o2_percent": 19.0, "co2_ppm": 8000},
        ).json()
        assert low_co2["atmosphere"]["grahams_ratio"] != high_co2["atmosphere"]["grahams_ratio"]

    def test_tamper_detected_when_prior_reading_given(self, client):
        resp = client.post(
            "/api/gas/assess",
            json={
                "ch4_percent": 2.0, "co_ppm": 10, "o2_percent": 20.9,
                "prior_ch4_percent": 0.3, "prior_o2_percent": 20.89,
            },
        )
        body = resp.json()
        assert body["atmosphere"]["sensor_tamper_status"] == "WARNING: CH4_SENSOR_FAULT_OR_TAMPER"
        assert any(a["label"] == "SENSOR_TAMPER_SUSPECTED" for a in body["alerts"])

    def test_tamper_unknown_without_prior_reading(self, client):
        resp = client.post(
            "/api/gas/assess",
            json={"ch4_percent": 0.5, "co_ppm": 10, "o2_percent": 20.9},
        )
        assert resp.json()["atmosphere"]["sensor_tamper_status"] == "UNKNOWN"

    def test_history_enables_trend_extrapolation(self, client):
        resp = client.post(
            "/api/gas/assess",
            json={
                "ch4_percent": 3.0, "co_ppm": 20, "o2_percent": 20.5,
                "history": [1.0, 1.5, 2.0, 2.5, 3.0],
                "timestamps": [0, 10, 20, 30, 40],
            },
        )
        body = resp.json()
        assert body["time_to_lel_s"] is not None
        assert body["time_to_lel_s"] > 0

    def test_missing_required_field_returns_422(self, client):
        resp = client.post("/api/gas/assess", json={"ch4_percent": 0.5})
        assert resp.status_code == 422


class TestHazardsEndpointErrorPaths:
    def test_analyse_without_baseline_returns_400(self, client):
        resp = client.post("/api/hazards/analyse")
        assert resp.status_code in (400, 503)

    def test_latest_scan_without_prior_scan_returns_404(self, client):
        resp = client.get("/api/scan/latest")
        assert resp.status_code == 404


class TestScanEndpointSimulated:
    def test_simulated_scan_completes_and_reports_stats(self, client):
        resp = client.post(
            "/api/scan/start",
            json={
                "simulate": True,
                "phi_start_deg": -20.0,
                "phi_end_deg": 20.0,
                "step_deg": 10.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["steps_attempted"] > 0
        assert "stats" in body

    def test_scan_then_latest_returns_mesh(self, client):
        client.post(
            "/api/scan/start",
            json={"simulate": True, "phi_start_deg": -15.0, "phi_end_deg": 15.0, "step_deg": 10.0},
        )
        resp = client.get("/api/scan/latest")
        assert resp.status_code == 200
        assert "mesh" in resp.json()
