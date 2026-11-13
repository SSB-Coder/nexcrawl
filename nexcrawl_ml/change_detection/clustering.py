"""
DBSCAN clustering of residual difference points into discrete hazard blobs.

Parameter choice matters more here than almost anywhere else in the project.
Generic defaults do not work: eps too small shatters one rockfall into a
dozen phantom hazards, eps too large merges a rockfall and a nearby piece of
equipment into a single unclassifiable blob.

eps should be tied to your actual post-voxel point spacing, not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from sklearn.cluster import DBSCAN

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


@dataclass
class Blob:
    """One candidate hazard: a spatially coherent group of changed points."""

    points: np.ndarray
    label: int
    centroid: np.ndarray = field(init=False)

    def __post_init__(self):
        self.centroid = self.points.mean(axis=0)

    @property
    def n_points(self) -> int:
        return len(self.points)


@dataclass
class ClusterResult:
    blobs: list[Blob]
    n_noise_points: int
    eps_used: float
    min_samples_used: int

    @property
    def n_blobs(self) -> int:
        return len(self.blobs)


def suggest_eps(points: np.ndarray, voxel_size: float = 0.025, k: int = 4) -> float:
    """Data-driven eps suggestion instead of a guessed constant.

    Uses the median k-th nearest-neighbour distance, floored at ~2 voxels.
    This is the practical version of the classic k-distance elbow method and
    keeps eps tied to the density your pipeline actually produces.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) <= k:
        return max(voxel_size * 3, 0.05)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=k + 1)
        median_kth = float(np.median(dists[:, -1]))
    except ImportError:  # pragma: no cover
        median_kth = voxel_size * 3

    return max(median_kth * 1.5, voxel_size * 2)


def suggest_min_samples(n_points: int, dim: int = 3) -> int:
    """Density-aware min_samples.

    The standard heuristic (Sander et al.) is min_samples ~= 2 * dim, i.e.
    6 for 3D data. A hand-picked constant like 20 silently fails on sparse
    blobs: a real rockfall producing 200 changed points spread over a
    1 m surface has a 4th-NN spacing around 9 cm, and demanding 20
    neighbours inside eps finds nothing at all - every point is labelled
    noise and the hazard vanishes.

    Scales up mildly for large clouds where 6 would be over-sensitive.
    """
    base = 2 * dim
    if n_points > 5000:
        return base * 3
    if n_points > 1000:
        return base * 2
    return base


def cluster_changes(
    points: np.ndarray,
    eps: float | None = None,
    min_samples: int | None = None,
    min_blob_points: int = 25,
    voxel_size: float = 0.025,
) -> ClusterResult:
    """Group changed points into hazard blobs.

    eps and min_samples both default to None = derive from the data. Passing
    fixed constants here is the single easiest way to make this pipeline
    silently miss hazards - see suggest_min_samples for what goes wrong.

    min_blob_points filters out clusters too small to be a real hazard; those
    are usually reconstruction artefacts at scan boundaries. Keep it above
    the noise scale you measured in the repeatability test.
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return ClusterResult([], 0, eps or 0.0, min_samples or 0)

    if eps is None:
        eps = suggest_eps(pts, voxel_size=voxel_size)
    if min_samples is None:
        min_samples = suggest_min_samples(len(pts))

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)

    blobs = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        member_pts = pts[labels == label]
        if len(member_pts) >= min_blob_points:
            blobs.append(Blob(points=member_pts, label=int(label)))

    blobs.sort(key=lambda b: b.n_points, reverse=True)

    return ClusterResult(
        blobs=blobs,
        n_noise_points=int(np.sum(labels == -1)),
        eps_used=float(eps),
        min_samples_used=min_samples,
    )
