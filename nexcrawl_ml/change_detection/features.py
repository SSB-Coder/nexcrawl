"""
Geometric feature extraction: turns a hazard blob into the 5-element vector
consumed by the LightGBM structural classifier.

Feature design rationale (worth knowing for judge Q&A - each feature exists
to separate a specific pair of classes):

  1. volume_m3        - separates a major roof fall from minor spall
  2. height_above_floor_m - roof falls sit high, floor heave sits low
  3. aspect_ratio     - equipment is boxy/regular, rockfall is irregular
  4. surface_roughness - broken rock is rough, manufactured equipment smooth
  5. planarity        - artefacts/registration ghosts are often thin sheets
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

FEATURE_NAMES: list[str] = [
    "volume_m3",
    "height_above_floor_m",
    "aspect_ratio",
    "surface_roughness",
    "planarity",
]


@dataclass
class BlobFeatures:
    volume_m3: float
    height_above_floor_m: float
    aspect_ratio: float
    surface_roughness: float
    planarity: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.volume_m3,
                self.height_above_floor_m,
                self.aspect_ratio,
                self.surface_roughness,
                self.planarity,
            ],
            dtype=np.float64,
        )

    def as_dict(self) -> dict:
        return asdict(self)


def _pca_eigenvalues(points: np.ndarray) -> np.ndarray:
    centred = points - points.mean(axis=0)
    cov = np.cov(centred.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    return np.sort(eigenvalues)[::-1]  # descending


def extract_features(
    points: np.ndarray,
    floor_z: float = 0.0,
) -> BlobFeatures:
    """Compute the 5 geometric features for one blob.

    floor_z is the estimated floor height in world coordinates; pass the
    value from estimate_floor_z() on the full cloud rather than assuming 0.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 4:
        return BlobFeatures(0.0, 0.0, 1.0, 0.0, 0.0)

    extent = pts.max(axis=0) - pts.min(axis=0)
    volume = float(np.prod(np.maximum(extent, 1e-6)))

    height = float(pts[:, 2].mean() - floor_z)

    sorted_extent = np.sort(extent)[::-1]
    aspect = float(sorted_extent[0] / max(sorted_extent[-1], 1e-6))

    eigenvalues = _pca_eigenvalues(pts)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    total = eigenvalues.sum()

    # Roughness: how much variance lies off the dominant plane.
    roughness = float(eigenvalues[2] / total)
    # Planarity: classic Weinmann-style dimensionality descriptor.
    planarity = float((eigenvalues[1] - eigenvalues[2]) / eigenvalues[0])

    return BlobFeatures(
        volume_m3=volume,
        height_above_floor_m=height,
        aspect_ratio=aspect,
        surface_roughness=roughness,
        planarity=planarity,
    )


def estimate_floor_z(points: np.ndarray, percentile: float = 2.0) -> float:
    """Robust floor-height estimate.

    Uses a low percentile rather than the minimum, because the single lowest
    point in a cloud is very often a stray outlier that survived SOR.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return 0.0
    return float(np.percentile(pts[:, 2], percentile))


def features_to_matrix(feature_list: list[BlobFeatures]) -> np.ndarray:
    """Stack features into the (N, 5) matrix LightGBM expects."""
    if not feature_list:
        return np.empty((0, len(FEATURE_NAMES)))
    return np.vstack([f.to_array() for f in feature_list])
