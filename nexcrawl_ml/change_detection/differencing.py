"""
Cloud differencing: what is here today that was not here before (and vice versa).

Two-sided by design:
  - ADDED points   -> material has appeared  (roof fall, rib spall, debris)
  - REMOVED points -> material has gone      (equipment moved, void opened)

A one-sided diff would silently miss half the hazards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.spatial import cKDTree

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


@dataclass
class DifferenceResult:
    added_points: np.ndarray       # in current, absent from baseline
    removed_points: np.ndarray     # in baseline, absent from current
    added_count: int
    removed_count: int
    threshold_m: float

    @property
    def total_changed(self) -> int:
        return self.added_count + self.removed_count


def cloud_difference(
    current: np.ndarray,
    baseline: np.ndarray,
    threshold_m: float = 0.08,
    detect_removed: bool = True,
) -> DifferenceResult:
    """Nearest-neighbour distance thresholding.

    threshold_m should sit comfortably above your reconstruction noise floor
    (from the repeatability test) but below the smallest hazard you care
    about. 8 cm is a reasonable starting point for a rig with ~2 cm RMS
    noise; re-tune after validation.
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required: pip install scipy")

    cur = np.asarray(current, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)

    if len(base) == 0:
        return DifferenceResult(cur, np.empty((0, 3)), len(cur), 0, threshold_m)
    if len(cur) == 0:
        return DifferenceResult(np.empty((0, 3)), base, 0, len(base), threshold_m)

    base_tree = cKDTree(base)
    dist_to_base, _ = base_tree.query(cur, k=1)
    added = cur[dist_to_base > threshold_m]

    if detect_removed:
        cur_tree = cKDTree(cur)
        dist_to_cur, _ = cur_tree.query(base, k=1)
        removed = base[dist_to_cur > threshold_m]
    else:
        removed = np.empty((0, 3))

    return DifferenceResult(
        added_points=added,
        removed_points=removed,
        added_count=len(added),
        removed_count=len(removed),
        threshold_m=threshold_m,
    )
