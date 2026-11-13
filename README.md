# NexCrawl-M — ML & Perception Subsystem

**SIH 2026 · Problem Statement 26039** — AI-Powered Underground Mine Safety, Monitoring and Rescue System
Role: Perception & Machine Learning Lead

The complete ML stack for the NexCrawl-M rover: turning 2D LiDAR slices into 3D meshes, detecting structural change, classifying hazards, and monitoring atmosphere and acoustics.

---

## Quick start

```bash
pip install -r requirements-dev.txt        # laptop / simulator only
pytest tests/ -q                           # 68 tests, no hardware needed

# End-to-end demo with the built-in tunnel simulator:
python scripts/generate_synthetic_data.py --out data/raw
python scripts/train_structural.py --data data/raw/structural_synthetic.npz --out data/models
python scripts/train_gas.py        --data data/raw/gas_synthetic.npz        --out data/models
python scripts/run_dome_scan.py --simulate            --out data/processed/baseline.obj
python scripts/run_dome_scan.py --simulate --rockfall --out data/processed/after.obj
python scripts/run_hazard_detection.py \
    --baseline data/processed/baseline.npy \
    --current  data/processed/after.npy \
    --model    data/models/structural_model.txt
```

Everything above runs without a rover. The simulator generates a geometrically correct mine drift with configurable debris, so the full pipeline is testable before hardware exists.

---

## Architecture

```
nexcrawl_ml/
├── perception/          2D slices → calibrated 3D mesh
│   ├── transforms.py        spherical→Cartesian, rigid-body, lever-arm
│   ├── calibration.py       zero-offset + lever-arm least-squares fit
│   ├── time_sync.py         step-and-stare + continuous-nod interpolation
│   ├── reconstructor.py     voxel → SOR → PCA normals → BPA/Poisson
│   └── validation.py        flat-wall, known-dimension, repeatability tests
├── change_detection/    baseline vs current → hazard blobs
│   ├── registration.py      ICP, odometry-seeded
│   ├── differencing.py      two-sided (added AND removed)
│   ├── clustering.py        DBSCAN with density-derived parameters
│   └── features.py          the 5 geometric features
├── models/              trained models
│   ├── structural_classifier.py   LightGBM #1
│   ├── gas_classifier.py          LightGBM #2
│   ├── knocknet.py                4-layer CNN → ONNX
│   └── calibration_wrapper.py     isotonic/sigmoid, prefit
├── audio/               log-mel features, GCC-PHAT bearing
├── gas/                 rolling-window features, Coward triangle, Graham's Ratio
├── fusion/              wheel + IMU EKF (freezable)
├── hw_io/               RPLidar C1, ESP32 servo link, + simulators
└── pipeline/            dome scan, hazard pipeline, alert fusion
```

---

## The 2D → 3D pipeline

A 2D LiDAR reports `(r, θ)` in one flat plane. The servo tilt angle `φ` rotates that plane, which is what supplies the third dimension. Because the RPLidar C1 already spins a full 360°, **one tilt sweep produces a complete dome**, not just a forward wedge.

Per point:

1. `x = r·cos(φ)·cos(θ)`, `y = r·cos(φ)·sin(θ)`, `z = r·sin(φ)`
2. Apply the lever-arm correction (see Calibration below)
3. Rotate/translate into world frame by rover pose, add mast height

Then: accumulate → voxel downsample (2.5 cm) → statistical outlier removal → PCA normals → **Ball Pivoting** (preserves rock texture) or **Poisson** (bridges occlusion gaps) → Laplacian smoothing → `.obj`.

### Scan strategies

| | Step-and-stare (default) | Continuous nod |
|---|---|---|
| How | Servo stops, one full revolution captured, then steps | Servo sweeps while LiDAR spins |
| Tilt angle | One confirmed φ for the entire revolution | Interpolated per point |
| Needs clock sync | No | Yes |
| Use for | Stationary dome scan — the demo path | Driving mode — stretch goal |

Step-and-stare eliminates timestamp interpolation error entirely. Use it for anything demo-critical.

---

## Calibration — do not skip

Three calibrations, run via `scripts/calibrate_rig.py`:

- **A. Servo zero-offset** — raw encoder reading at true horizontal
- **B. Lever-arm / axis offset** — the servo's rotation axis will *not* pass exactly through the LiDAR's optical centre on any DIY bracket. Estimated by scanning a flat wall at several tilt angles and solving for the correction that makes all scans agree on one plane (robust `soft_l1` non-linear least squares).
- **C. Mast height** — measured directly

Skipping **B** produces a range-dependent smear that gets worse the further a point is from the sensor. Verify the solver in simulation first:

```bash
python scripts/calibrate_rig.py --simulate
```

The reported residual RMS is your measured accuracy — quote it to judges.

---

## The ML models

Three trained models, plus deterministic physics that deliberately bypasses ML entirely.

| Model | Input | Output | File |
|---|---|---|---|
| **LightGBM #1** — Structural | 5 geometric features per blob | roof_fall / rib_spall / floor_heave / equipment_moved / artefact | `structural_model.txt` |
| **LightGBM #2** — Gas trend | 8 rolling-window stats | SAFE / SLOW_LEAK / RAPID_ACCUMULATION / SENSOR_FAULT | `gas_model.txt` |
| **KnockNet** — 4-layer CNN | log-mel spectrogram | knock / voice / motor_hum / silence | `knocknet.onnx` |

These are two genuinely separate LightGBM models — different features, different labels, different files. Same library, different job; not two training runs of the same task.

**Why not PointNet / a 3D CNN?** With ~300 staged samples, a deep point-cloud network memorises the training set. Gradient-boosted trees on 5 hand-designed, physically meaningful features match the data budget and give you feature importances you can defend to a judge.

**Why `voice` is its own class:** a shout and a deliberate knock call for different operator responses, and a model with no voice category tends to confuse the two.

**Why `motor_hum` is its own class:** the rover's drivetrain is the single most likely false-positive source for knock detection, so the model gets to learn it directly rather than having it buried in a generic background class.

**Physics outranks ML.** The Coward explosibility triangle and Graham's Ratio are deterministic. If the physics says the atmosphere is explosive, that alert goes top of the list regardless of any model's confidence score. `alert_fusion.py` enforces this ordering.

---

## Verified results (simulator)

```
Dome scan        37 tilt steps → 17,508 pts → 3,595 triangles in 0.79 s
Tunnel width     constant at ~4.07 m across every tilt angle (-40° to +40°) -
                 direct confirmation of the corrected rigid-body geometry
ICP registration fitness 0.987, inlier RMSE 3.3 mm
Rockfall         detected as +204 changed points → 1 cluster → classified
Structural model 85% accuracy; calibration improved Brier 0.0307 → 0.0275
Gas model        94% accuracy
Tests            148 passed, 82% coverage
```

---

## Fix pass — critical geometry correction + hardening

A second pass (against an independently-written fix specification, verified
line-by-line against this project's own master guide and chat history before
any of it was applied — not taken on trust) found one critical math error and
several real, previously-latent bugs:

1. **The 2D-slice-to-3D transform used the wrong physical model.** The
   original formula (`x=r·cosφ·cosθ, y=r·cosφ·sinθ, z=r·sinφ`) treats `φ` as
   a per-beam elevation angle — correct for a fixed-elevation-channel LiDAR
   like a Velodyne, wrong for a 2D disc physically tilted by an external
   servo. The correct model is a rigid-body rotation about the pitch axis:
   `x=r·cosφ·cosθ, y=r·sinθ, z=r·sinφ·cosθ` — under which side beams
   (`θ=±90°`) sit exactly on the rotation axis and are completely unaffected
   by tilt. Verified directly: simulated tunnel width now holds at ~4.07 m
   across every tilt angle from -40° to +40°; under the old formula it would
   have shrunk to ~3.06 m at φ=40° (`cos(40°)≈0.766`). This is not a subtle
   error — it would have visibly warped every dome scan on real hardware.
   Fixed in `transforms.py`, the simulator's ray-casting, and the test that
   had encoded the wrong physics.

2. **`HazardPipeline` zeroed severity for the calibrated model.** If you
   pass the `CalibratedClassifierCV`-wrapped model — the one
   `train_structural.py` produces specifically for shipping — into the
   hazard pipeline instead of the raw wrapper, every hazard silently got
   `severity: 0`, mapping straight to `INFO` priority regardless of actual
   danger. Fixed by looking up real `CLASS_SEVERITY`. Regression-tested in
   `test_calibration.py`.

3. **`torch.onnx.export` silently depended on the installed PyTorch
   version's default exporter.** PyTorch ≥ 2.5 defaults to a newer
   dynamo-based exporter that needs a separate `onnxscript` package and
   ignores the `dynamic_axes` argument this code relies on. Pinned
   `dynamo=False` explicitly so export behaves identically regardless of
   the installed torch version, rather than depending on which exporter
   happens to be default. (Emits a deprecation warning as of PyTorch ≥ 2.9 —
   worth revisiting when the dynamo exporter's `dynamic_shapes` API is
   adopted.)

4. **KnockNet was missing its `voice` class.** The project's own
   specifications consistently call for 4 classes
   (`knock / voice / motor_hum / silence`); the implementation only had 3.
   Without a dedicated voice class, a shout is liable to be confused with a
   deliberate knock — exactly the failure mode a confusion-matrix review
   would need to catch. Fixed across `knocknet.py` and the training script.

5. **Graham's Ratio used a simplified approximation instead of the
   project-specified nitrogen-balance formula.** Verified the correct
   formula directly against the master guide before changing anything:
   `R = CO% / (0.265·N2% − O2%) × 100` with `N2%` computed by difference.
   This matters most in afterdamp conditions, where CH4/CO2 measurably
   shift the nitrogen-by-difference term and the simplified form quietly
   drifts from the correct value. Also added `check_sensor_tamper` — CH4
   rising sharply while O2 stays flat is not physically consistent with a
   real gas event, and is now flagged as a probable masked/clogged intake
   rather than raised as a genuine hazard.

6. **Two dead-code smells from an earlier hasty patch.** Both classifiers'
   `feature_importance()` used a string-literal ternary trick
   (`"structural" in "gas_classifier.py"`) that only avoided crashing by
   accident, via Python's lazy ternary evaluation. Cleaned up in both files.

All fixes were independently verified before being applied — the geometry
correction via direct rederivation from first principles, the gas chemistry
formula against the project's actual master guide, not against the fix
specification's citation of it. The test suite grew from 68 to 148 tests
(82% coverage) specifically to pin these behaviors so they can't silently
regress.

---

## Bugs this codebase already caught

These were found by actually running the pipeline, not by inspection. Each would have been painful to debug live at the hackathon.

1. **`cv='prefit'` no longer exists in scikit-learn ≥ 1.6.** It was replaced by `FrozenEstimator`. The project docs specify `cv='prefit'`, which crashes on any current install. `calibration_wrapper.py` now supports both.

2. **DBSCAN `min_samples=20` silently swallowed a real rockfall.** The blob was correctly detected as 203 changed points, but every point was labelled noise and the hazard vanished — with no error raised anywhere. The blob's 4th-nearest-neighbour spacing was ~9 cm; demanding 20 neighbours within `eps` found nothing. Now derived from the data via the standard `2 × dimensionality` heuristic. Regression test: `test_sparse_blob_is_not_swallowed`.

3. **Save/load round-trip was broken.** `StructuralClassifier.load()` returned a raw `lgb.Booster`, which has `predict()` but no `predict_proba()` or `classes_`. Any deployed model would have failed at inference while training looked perfect.

4. **Synthetic gas data scored a perfect 1.000.** Too separable to mean anything. Class overlap added; now 94%, and both training scripts warn when accuracy ≥ 0.99.

---

## Things to be careful about

**Calibration is not guaranteed to help.** On the gas model it made the Brier score *worse* (0.0047 → 0.0072 isotonic, 0.0074 sigmoid) — isotonic calibration overfits on small held-out sets. Both training scripts measure this and warn. Decide from the number, don't assume.

**Synthetic data is scaffolding, not training data.** It exists so the pipeline can be built before the Week 9 staged dataset. Models trained on it must be retrained on real measurements. `data/raw/SYNTHETIC_README.txt` says the same thing next to the files.

**Pose is pure dead reckoning.** Wheel odometry + IMU EKF, with no LiDAR-based correction anywhere — no scan matching, no loop closure. Over a long traverse it drifts without bound and nothing in this pipeline catches it. Fine for stationary dome scans (where the EKF is frozen and pose is exact); a real limitation for extended driving. Surface `PoseEKF.uncertainty` on the UI so the operator knows when the map is drifting.

**Two microphones cannot resolve front from back.** `DOAResult` reports the mirror bearing explicitly rather than hiding the ambiguity. Resolving it needs a third mic.

**Config values marked `MEASURE` are guesses.** Mast height, wheel base, wheel radius, encoder ticks, mic spacing — all affect accuracy directly. Mic spacing in particular scales every bearing estimate by a constant factor.

---

## Scripts

| Script | Purpose |
|---|---|
| `calibrate_rig.py` | Flat-wall calibration (`--simulate` to verify the solver) |
| `run_dome_scan.py` | Stationary dome scan → `.obj` mesh |
| `run_hazard_detection.py` | Baseline vs current → classified hazards |
| `train_structural.py` | LightGBM #1 + calibration |
| `train_gas.py` | LightGBM #2 + calibration |
| `train_knocknet.py` | CNN training + ONNX export |
| `generate_synthetic_data.py` | Scaffolding datasets |

## API

```bash
uvicorn serve.api:app --host 0.0.0.0 --port 8000
```

`GET /health` · `GET /api/calibration` · `POST /api/scan/start` · `GET /api/scan/latest` · `POST /api/hazards/analyse` · `POST /api/gas/assess` · `WS /ws/mesh` · `WS /ws/telemetry`

The mesh websocket streams *progressively* during a scan — the dome visibly builds up rather than appearing after a loading spinner. That's a meaningfully better demo moment.

## Deployment note

On the Raspberry Pi 5, install `onnxruntime` rather than full PyTorch for KnockNet inference — `ONNXKnockDetector` needs no PyTorch dependency at runtime. `pyserial` and `rplidar-roboticia` are only needed on the rover.
