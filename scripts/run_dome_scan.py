#!/usr/bin/env python3
"""
Run a stationary dome scan and write the mesh.

Usage:
    python run_dome_scan.py --simulate --out ../data/processed/dome.obj
    python run_dome_scan.py --simulate --rockfall --out ../data/processed/after.obj
    python run_dome_scan.py --lidar-port /dev/ttyUSB0 --servo-port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexcrawl_ml.perception.calibration import RigCalibration
from nexcrawl_ml.perception.validation import timing_budget
from nexcrawl_ml.pipeline.dome_scan import DomeScanConfig, DomeScanner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--rockfall", action="store_true",
                        help="inject a debris pile (simulation only)")
    parser.add_argument("--lidar-port", default="/dev/ttyUSB0")
    parser.add_argument("--servo-port", default="/dev/ttyACM0")
    parser.add_argument("--calibration", default="config/rig_calibration.json")
    parser.add_argument("--phi-start", type=float, default=-45.0)
    parser.add_argument("--phi-end", type=float, default=45.0)
    parser.add_argument("--step", type=float, default=2.5)
    parser.add_argument("--method", default="bpa", choices=["bpa", "poisson"])
    parser.add_argument("--out", default="data/processed/dome.obj")
    parser.add_argument("--save-cloud", action="store_true")
    args = parser.parse_args()

    calib_path = Path(args.calibration)
    if calib_path.exists():
        calib = RigCalibration.load(calib_path)
        print(f"loaded calibration: lever_arm={calib.lever_arm_m}, "
              f"phi_zero={np.degrees(calib.phi_zero_offset_rad):.3f} deg")
    else:
        calib = RigCalibration()
        print(
            "WARNING: no calibration file found - using identity calibration.\n"
            "         Run calibrate_rig.py first; uncalibrated meshes will smear."
        )

    config = DomeScanConfig(
        phi_start_deg=args.phi_start,
        phi_end_deg=args.phi_end,
        step_deg=args.step,
        method=args.method,
    )

    if args.simulate:
        from nexcrawl_ml.hw_io.lidar_reader import SimulatedLidar
        from nexcrawl_ml.hw_io.serial_reader import SimulatedServoLink

        lidar = SimulatedLidar()
        if args.rockfall:
            lidar.add_rockfall(center=(3.0, 0.5, 0.4), radius=0.6)
            print("injected simulated rockfall at (3.0, 0.5, 0.4) r=0.6m")
        servo = SimulatedServoLink()
    else:
        from nexcrawl_ml.hw_io.lidar_reader import RPLidarC1Reader
        from nexcrawl_ml.hw_io.serial_reader import ESP32ServoLink

        lidar = RPLidarC1Reader(port=args.lidar_port)
        servo = ESP32ServoLink(port=args.servo_port)
        servo.start()
        servo.sync_clock()

    scanner = DomeScanner(lidar, servo, calibration=calib, config=config)

    def on_slice(i, n, added):
        bar = "#" * int(30 * (i + 1) / n)
        print(f"\r  [{bar:<30}] step {i+1}/{n}  +{added} pts", end="", flush=True)

    print(f"\nrunning dome scan: {config.n_steps} tilt steps "
          f"({args.phi_start} to {args.phi_end} deg)")
    result = scanner.run(output_path=args.out, on_slice=on_slice)
    print()

    s = result.stats
    print(f"\ncompleted {result.steps_completed}/{result.steps_attempted} steps "
          f"in {result.duration_s:.2f}s")
    print(f"  raw points      : {s.raw_points:,}")
    print(f"  after voxel     : {s.after_voxel:,}")
    print(f"  after outlier   : {s.after_outlier:,}")
    print(f"  mesh            : {s.vertices:,} vertices / {s.triangles:,} triangles")
    print(f"  method          : {s.method}")

    budget = timing_budget(config.n_steps, config.settle_time_s, 0.15, s.seconds)
    print(f"\ntiming: sweep {budget['sweep_seconds']:.1f}s + "
          f"reconstruct {budget['reconstruction_seconds']:.1f}s "
          f"= {budget['total_seconds']:.1f}s "
          f"({'fits' if budget['fits_7min_demo'] else 'DOES NOT FIT'} demo window)")

    if args.save_cloud:
        cloud_path = Path(args.out).with_suffix(".pcd")
        scanner.reconstructor.save_cloud(cloud_path)
        print(f"  saved cloud -> {cloud_path}")

    np.save(Path(args.out).with_suffix(".npy"), result.point_cloud)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
