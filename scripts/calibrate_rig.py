#!/usr/bin/env python3
"""
Rig calibration driver (Section 4 of the implementation plan).

Procedure:
  1. Park the rover facing a flat wall, 2-4 m away.
  2. This script sweeps a set of tilt angles, capturing one revolution at each.
  3. A least-squares fit finds the lever arm + zero-offset that makes every
     scan agree on a single plane.
  4. Calibration is written to JSON for the runtime pipeline to load.

Quote the reported residual RMS to judges - it is your measured accuracy.

Usage:
    python calibrate_rig.py --simulate --out ../config/rig_calibration.json
    python calibrate_rig.py --lidar-port /dev/ttyUSB0 --servo-port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexcrawl_ml.perception.calibration import calibrate_lever_arm
from nexcrawl_ml.perception.validation import flat_wall_test
from nexcrawl_ml.perception.transforms import spherical_to_local, apply_lever_arm


def collect_simulated(angles_deg, true_lever=(0.012, -0.007, 0.023)):
    """Simulate a wall at 3 m with a known injected lever-arm error.

    The point of simulating is to verify the solver recovers a lever arm
    close to the injected one before trusting it on real hardware.
    """
    rng = np.random.default_rng(7)
    scans = []
    for phi_deg in angles_deg:
        phi = np.deg2rad(phi_deg)
        theta = np.linspace(-np.pi / 4, np.pi / 4, 220)
        # Wall at x = 3.0; ray direction determined by (phi, theta).
        denom = np.cos(phi) * np.cos(theta)
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        r = 3.0 / denom
        r += rng.normal(0, 0.008, len(r))
        scans.append((r, theta, phi))
    return scans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--lidar-port", default="/dev/ttyUSB0")
    parser.add_argument("--servo-port", default="/dev/ttyACM0")
    parser.add_argument("--angles", default="-30,-15,0,15,30")
    parser.add_argument("--mast-height", type=float, default=0.25)
    parser.add_argument("--out", default="config/rig_calibration.json")
    args = parser.parse_args()

    angles_deg = [float(a) for a in args.angles.split(",")]
    print(f"calibrating over tilt angles: {angles_deg}")

    if args.simulate:
        print("SIMULATION MODE - verifying the solver, not the hardware")
        scans = collect_simulated(angles_deg)
    else:
        from nexcrawl_ml.hw_io.lidar_reader import RPLidarC1Reader
        from nexcrawl_ml.hw_io.serial_reader import ESP32ServoLink

        lidar = RPLidarC1Reader(port=args.lidar_port)
        servo = ESP32ServoLink(port=args.servo_port)
        servo.start()
        servo.sync_clock()

        scans = []
        for phi_deg in angles_deg:
            phi = np.deg2rad(phi_deg)
            servo.set_tilt(phi)
            if not servo.wait_until_settled(phi):
                print(f"  servo failed to settle at {phi_deg} deg - skipping")
                continue
            rev = next(lidar.revolutions())
            status = servo.latest
            mask = (rev.ranges > 0.3) & (rev.ranges < 8.0)
            scans.append(
                (
                    rev.ranges[mask],
                    rev.beam_angles[mask],
                    status.angle_rad if status else phi,
                )
            )
            print(f"  captured {mask.sum()} points at {phi_deg} deg")

        servo.stop()
        lidar.stop()

    print("\nrunning least-squares lever-arm fit...")
    calib = calibrate_lever_arm(scans)
    calib.mast_height_m = args.mast_height

    print(f"  lever arm      : {tuple(round(v, 5) for v in calib.lever_arm_m)} m")
    print(f"  phi zero offset: {np.degrees(calib.phi_zero_offset_rad):.4f} deg")
    print(f"  residual RMS   : {calib.residual_rms_m*1000:.2f} mm "
          f"over {calib.n_calibration_points} points")

    # Independent verification on the middle scan.
    r, theta, phi = scans[len(scans) // 2]
    pts = apply_lever_arm(
        spherical_to_local(r, theta, phi - calib.phi_zero_offset_rad),
        calib.lever_arm_m,
    )
    planarity = flat_wall_test(pts)
    print(f"\n  post-calibration planarity RMS: {planarity.rms_error_m*1000:.2f} mm "
          f"({'PASS' if planarity.passed else 'FAIL'})")

    if calib.residual_rms_m > 0.03:
        print(
            "\n  WARNING: residual above 30 mm. Check that the wall really is flat,\n"
            "  that the LiDAR saw only the wall (crop out floor/ceiling returns),\n"
            "  and that the servo bracket is not flexing under load."
        )

    out = Path(args.out)
    calib.save(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
