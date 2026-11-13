"""
ICP registration: align today's scan to the stored baseline.

Blind global ICP is unnecessary here - the rover knows roughly where it is
from odometry, so that pose is used as the initial guess. Point-to-plane ICP
converges faster and more reliably than point-to-point on the kind of large
planar-ish rock faces found in a mine drift, so it is the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:  # pragma: no cover
    o3d = None
    _HAS_OPEN3D = False


@dataclass
class RegistrationResult:
    transformation: np.ndarray
    fitness: float          # fraction of inlier correspondences (higher better)
    inlier_rmse: float      # metres
    converged: bool

    def summary(self) -> str:
        return (
            f"fitness={self.fitness:.3f} rmse={self.inlier_rmse*1000:.1f}mm "
            f"converged={self.converged}"
        )


def _to_pcd(points):
    if hasattr(points, "points"):
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def initial_guess_from_odometry(
    dx: float = 0.0, dy: float = 0.0, dyaw: float = 0.0, dz: float = 0.0
) -> np.ndarray:
    """Build a 4x4 seed transform from the rover's own pose estimate."""
    c, s = np.cos(dyaw), np.sin(dyaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = [dx, dy, dz]
    return T


def register_to_baseline(
    current_points,
    baseline_points,
    initial_transform: np.ndarray | None = None,
    max_correspondence_distance: float = 0.20,
    voxel_size: float = 0.03,
    method: str = "point_to_plane",
    max_iterations: int = 60,
) -> RegistrationResult:
    """Align the current scan onto the baseline scan.

    A low fitness score is itself informative: it can mean the rover is not
    where odometry thinks it is, OR that the geometry genuinely changed a lot
    (i.e. a large collapse). Always check fitness before trusting the
    downstream difference map.
    """
    if not _HAS_OPEN3D:
        raise ImportError("open3d is required: pip install open3d")

    src = _to_pcd(current_points).voxel_down_sample(voxel_size)
    dst = _to_pcd(baseline_points).voxel_down_sample(voxel_size)

    if initial_transform is None:
        initial_transform = np.eye(4)

    if method == "point_to_plane":
        for cloud in (src, dst):
            if not cloud.has_normals():
                cloud.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=voxel_size * 4, max_nn=30
                    )
                )
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    result = o3d.pipelines.registration.registration_icp(
        src,
        dst,
        max_correspondence_distance,
        initial_transform,
        estimator,
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        ),
    )

    return RegistrationResult(
        transformation=np.asarray(result.transformation),
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        converged=float(result.fitness) > 0.3,
    )


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an (N,3) array."""
    pts = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    return (homogeneous @ np.asarray(transform).T)[:, :3]
