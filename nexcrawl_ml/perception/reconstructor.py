"""
Point-cloud accumulation and surface reconstruction.

Pipeline (all pure algorithms, no training anywhere in this file):
    accumulate -> voxel downsample -> SOR -> PCA normals -> BPA/Poisson
    -> Laplacian smoothing -> cleanup -> export
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration import RigCalibration
from .transforms import (
    apply_lever_arm,
    local_to_world,
    range_gate,
    spherical_to_local,
    spherical_to_local_varying_phi,
)

try:  # Open3D is heavy; keep the transform half of this module importable without it
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:  # pragma: no cover
    o3d = None
    _HAS_OPEN3D = False


@dataclass
class ReconstructionStats:
    """Everything worth logging or showing on the demo HUD."""

    raw_points: int = 0
    after_voxel: int = 0
    after_outlier: int = 0
    vertices: int = 0
    triangles: int = 0
    n_slices: int = 0
    seconds: float = 0.0
    method: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class TiltedDomeLidarReconstructor:
    """Turns tilt-tagged 2D LiDAR revolutions into a 3D triangle mesh.

    Usage (step-and-stare dome scan):
        rec = TiltedDomeLidarReconstructor(calib)
        for ranges, angles, phi in sweep:
            rec.process_slice(ranges, angles, phi)
        mesh = rec.reconstruct(method="bpa")
    """

    def __init__(
        self,
        calibration: RigCalibration | None = None,
        voxel_size: float = 0.025,
        min_range: float = 0.15,
        max_range: float = 10.0,
        sor_neighbors: int = 25,
        sor_std_ratio: float = 2.0,
        normal_radius: float = 0.12,
        normal_max_nn: int = 30,
        bpa_radii: tuple[float, ...] = (0.04, 0.08, 0.16),
        poisson_depth: int = 9,
        smoothing_iterations: int = 2,
    ):
        self.calib = calibration or RigCalibration()
        self.voxel_size = voxel_size
        self.min_range = min_range
        self.max_range = max_range
        self.sor_neighbors = sor_neighbors
        self.sor_std_ratio = sor_std_ratio
        self.normal_radius = normal_radius
        self.normal_max_nn = normal_max_nn
        self.bpa_radii = bpa_radii
        self.poisson_depth = poisson_depth
        self.smoothing_iterations = smoothing_iterations

        self.accumulated_points: list[np.ndarray] = []
        self.viewpoints: list[np.ndarray] = []
        self.stats = ReconstructionStats()

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def process_slice(
        self,
        ranges: np.ndarray,
        beam_angles: np.ndarray,
        raw_tilt_angle_rad: float,
        rover_x: float = 0.0,
        rover_y: float = 0.0,
        rover_yaw: float = 0.0,
        quality: np.ndarray | None = None,
    ) -> int:
        """Step-and-stare path: one revolution, one confirmed tilt angle."""
        r, theta, mask = range_gate(
            ranges, beam_angles, self.min_range, self.max_range, quality
        )
        if len(r) == 0:
            return 0

        phi = raw_tilt_angle_rad - self.calib.phi_zero_offset_rad
        pts = spherical_to_local(r, theta, phi)
        pts = apply_lever_arm(pts, self.calib.lever_arm_m)
        pts = local_to_world(
            pts, rover_x, rover_y, rover_yaw, self.calib.mast_height_m
        )

        self.accumulated_points.append(pts)
        self.viewpoints.append(
            np.array([rover_x, rover_y, self.calib.mast_height_m])
        )
        self.stats.n_slices += 1
        return len(pts)

    def process_slice_continuous(
        self,
        ranges: np.ndarray,
        beam_angles: np.ndarray,
        raw_tilt_angles_rad: np.ndarray,
        rover_x: float = 0.0,
        rover_y: float = 0.0,
        rover_yaw: float = 0.0,
        quality: np.ndarray | None = None,
    ) -> int:
        """Continuous-nod path: every point carries its own interpolated phi."""
        r = np.asarray(ranges, dtype=np.float64)
        theta = np.asarray(beam_angles, dtype=np.float64)
        phis = np.asarray(raw_tilt_angles_rad, dtype=np.float64)

        mask = (r > self.min_range) & (r < self.max_range) & np.isfinite(r)
        if quality is not None:
            mask &= np.asarray(quality) > 0
        if not mask.any():
            return 0

        phi = phis[mask] - self.calib.phi_zero_offset_rad
        pts = spherical_to_local_varying_phi(r[mask], theta[mask], phi)
        pts = apply_lever_arm(pts, self.calib.lever_arm_m)
        pts = local_to_world(
            pts, rover_x, rover_y, rover_yaw, self.calib.mast_height_m
        )

        self.accumulated_points.append(pts)
        self.viewpoints.append(
            np.array([rover_x, rover_y, self.calib.mast_height_m])
        )
        self.stats.n_slices += 1
        return len(pts)

    def clear(self) -> None:
        self.accumulated_points.clear()
        self.viewpoints.clear()
        self.stats = ReconstructionStats()

    @property
    def point_count(self) -> int:
        return sum(len(p) for p in self.accumulated_points)

    def merged_points(self) -> np.ndarray:
        if not self.accumulated_points:
            return np.empty((0, 3))
        return np.vstack(self.accumulated_points)

    # ------------------------------------------------------------------ #
    # Reconstruction
    # ------------------------------------------------------------------ #

    def _require_open3d(self) -> None:
        if not _HAS_OPEN3D:
            raise ImportError(
                "open3d is required for reconstruction: pip install open3d"
            )

    def build_point_cloud(self, orient_per_scan: bool = True):
        """Downsample, denoise and estimate normals.

        orient_per_scan matters once you merge dome scans taken from
        *different* rover positions. Open3D's single global
        orient_normals_towards_camera_location assumes one viewpoint; with
        several it flips normals inconsistently and BPA/Poisson then produce
        holes or inside-out patches. Orienting each scan against its own
        viewpoint before merging avoids that.
        """
        self._require_open3d()
        if not self.accumulated_points:
            raise ValueError("point buffer empty - run scan cycles first")

        t0 = time.perf_counter()
        self.stats.raw_points = self.point_count

        if orient_per_scan and len(self.accumulated_points) > 1:
            clouds = []
            for pts, vp in zip(self.accumulated_points, self.viewpoints):
                c = o3d.geometry.PointCloud()
                c.points = o3d.utility.Vector3dVector(pts)
                c = c.voxel_down_sample(self.voxel_size)
                c.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.normal_radius, max_nn=self.normal_max_nn
                    )
                )
                c.orient_normals_towards_camera_location(camera_location=vp)
                clouds.append(c)
            pcd = clouds[0]
            for c in clouds[1:]:
                pcd += c
            pcd = pcd.voxel_down_sample(self.voxel_size)
            self.stats.after_voxel = len(pcd.points)
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.sor_neighbors, std_ratio=self.sor_std_ratio
            )
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.merged_points())
            pcd = pcd.voxel_down_sample(self.voxel_size)
            self.stats.after_voxel = len(pcd.points)
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.sor_neighbors, std_ratio=self.sor_std_ratio
            )
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.normal_radius, max_nn=self.normal_max_nn
                )
            )
            vp = (
                self.viewpoints[0]
                if self.viewpoints
                else np.array([0.0, 0.0, self.calib.mast_height_m])
            )
            pcd.orient_normals_towards_camera_location(camera_location=vp)

        self.stats.after_outlier = len(pcd.points)
        self.stats.seconds += time.perf_counter() - t0
        return pcd

    def reconstruct(
        self,
        method: str = "bpa",
        output_path: str | Path | None = None,
        orient_per_scan: bool = True,
    ):
        """Full mesh reconstruction. method is 'bpa' or 'poisson'."""
        self._require_open3d()
        t0 = time.perf_counter()

        pcd = self.build_point_cloud(orient_per_scan=orient_per_scan)
        if len(pcd.points) < 100:
            raise ValueError(
                f"only {len(pcd.points)} points survived filtering - "
                "check range gating and calibration"
            )

        method = method.lower()
        if method == "bpa":
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(list(self.bpa_radii))
            )
        elif method == "poisson":
            mesh, densities = (
                o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=self.poisson_depth
                )
            )
            densities = np.asarray(densities)
            mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))
        else:
            raise ValueError(f"unknown method {method!r}; use 'bpa' or 'poisson'")

        if self.smoothing_iterations > 0:
            mesh = mesh.filter_smooth_laplacian(
                number_of_iterations=self.smoothing_iterations
            )
        mesh.compute_vertex_normals()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()

        self.stats.vertices = len(mesh.vertices)
        self.stats.triangles = len(mesh.triangles)
        self.stats.method = method
        self.stats.seconds = time.perf_counter() - t0

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(str(output_path), mesh)

        return mesh

    def decimate(self, mesh, target_triangles: int = 60_000):
        """Quadric edge-collapse decimation for smoother WebSocket streaming."""
        self._require_open3d()
        if len(mesh.triangles) <= target_triangles:
            return mesh
        return mesh.simplify_quadric_decimation(target_triangles)

    def save_cloud(self, path: str | Path) -> None:
        """Dump the raw accumulated cloud - useful for offline debugging."""
        self._require_open3d()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.merged_points())
        o3d.io.write_point_cloud(str(path), pcd)


def mesh_to_payload(mesh, include_normals: bool = True) -> dict:
    """Serialise an Open3D mesh into a JSON-friendly dict for the WebSocket."""
    payload = {
        "vertices": np.asarray(mesh.vertices, dtype=np.float32).ravel().tolist(),
        "triangles": np.asarray(mesh.triangles, dtype=np.int32).ravel().tolist(),
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.triangles),
    }
    if include_normals and mesh.has_vertex_normals():
        payload["normals"] = (
            np.asarray(mesh.vertex_normals, dtype=np.float32).ravel().tolist()
        )
    return payload
