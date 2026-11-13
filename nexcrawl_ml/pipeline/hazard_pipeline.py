"""
Hazard assessment pipeline: baseline vs current -> classified, calibrated alerts.

Flow:
    current cloud + baseline cloud
      -> ICP registration
      -> two-sided differencing
      -> DBSCAN clustering
      -> 5 geometric features per blob
      -> calibrated LightGBM structural classifier
      -> prioritised alerts

Gas and audio are handled in parallel branches and merged at the alert layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..change_detection.clustering import cluster_changes
from ..change_detection.differencing import cloud_difference
from ..change_detection.features import (
    estimate_floor_z,
    extract_features,
    features_to_matrix,
)
from ..change_detection.registration import (
    apply_transform,
    initial_guess_from_odometry,
    register_to_baseline,
)
from ..models.structural_classifier import CLASS_SEVERITY


@dataclass
class Hazard:
    label: str
    confidence: float
    severity: int
    centroid: tuple[float, float, float]
    n_points: int
    features: dict
    distribution: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "severity": self.severity,
            "centroid": list(self.centroid),
            "n_points": self.n_points,
            "features": self.features,
            "distribution": self.distribution,
        }


@dataclass
class HazardReport:
    hazards: list[Hazard]
    registration_fitness: float
    registration_rmse: float
    n_added_points: int
    n_removed_points: int
    n_blobs: int
    eps_used: float
    duration_s: float
    warnings: list[str] = field(default_factory=list)

    @property
    def max_severity(self) -> int:
        return max((h.severity for h in self.hazards), default=0)

    def as_dict(self) -> dict:
        return {
            "hazards": [h.as_dict() for h in self.hazards],
            "registration": {
                "fitness": self.registration_fitness,
                "rmse_m": self.registration_rmse,
            },
            "n_added_points": self.n_added_points,
            "n_removed_points": self.n_removed_points,
            "n_blobs": self.n_blobs,
            "eps_used": self.eps_used,
            "max_severity": self.max_severity,
            "duration_s": self.duration_s,
            "warnings": self.warnings,
        }


class HazardPipeline:
    """Runs change detection and structural classification end to end."""

    def __init__(
        self,
        classifier,
        diff_threshold_m: float = 0.08,
        min_blob_points: int = 25,
        voxel_size: float = 0.025,
        min_registration_fitness: float = 0.3,
    ):
        self.classifier = classifier
        self.diff_threshold_m = diff_threshold_m
        self.min_blob_points = min_blob_points
        self.voxel_size = voxel_size
        self.min_registration_fitness = min_registration_fitness

    def run(
        self,
        current_points: np.ndarray,
        baseline_points: np.ndarray,
        odometry_guess: tuple[float, float, float] | None = None,
    ) -> HazardReport:
        t0 = time.perf_counter()
        warnings: list[str] = []

        # 1. Register current onto baseline, seeded with odometry.
        guess = (
            initial_guess_from_odometry(*odometry_guess)
            if odometry_guess
            else None
        )
        reg = register_to_baseline(
            current_points, baseline_points, initial_transform=guess,
            voxel_size=self.voxel_size,
        )
        if reg.fitness < self.min_registration_fitness:
            warnings.append(
                f"low ICP fitness ({reg.fitness:.2f}) - either the rover is not "
                "where odometry thinks, or the geometry changed substantially. "
                "Treat the change map below with caution."
            )
        aligned = apply_transform(current_points, reg.transformation)

        # 2. Two-sided difference.
        diff = cloud_difference(
            aligned, baseline_points, threshold_m=self.diff_threshold_m
        )

        # 3. Cluster the added material into candidate hazards.
        clusters = cluster_changes(
            diff.added_points,
            min_blob_points=self.min_blob_points,
            voxel_size=self.voxel_size,
        )

        # 4. Features + classification.
        hazards: list[Hazard] = []
        if clusters.blobs:
            floor_z = estimate_floor_z(baseline_points)
            feature_objs = [
                extract_features(b.points, floor_z=floor_z) for b in clusters.blobs
            ]
            X = features_to_matrix(feature_objs)

            try:
                if hasattr(self.classifier, "predict_with_severity"):
                    predictions = self.classifier.predict_with_severity(X)
                else:  # bare CalibratedClassifierCV wrapper (no severity helper
                    # of its own) - this is exactly the object train_structural.py
                    # saves as structural_model_calibrated.joblib, so it must be
                    # a fully-supported input here, not a degraded fallback.
                    proba = self.classifier.predict_proba(X)
                    classes = [str(c) for c in self.classifier.classes_]
                    predictions = [
                        {
                            "label": classes[int(np.argmax(row))],
                            "confidence": float(np.max(row)),
                            "severity": CLASS_SEVERITY.get(
                                classes[int(np.argmax(row))], 0
                            ),
                            "distribution": {
                                c: float(p) for c, p in zip(classes, row)
                            },
                        }
                        for row in proba
                    ]
            except Exception as exc:
                warnings.append(f"classification failed: {exc}")
                predictions = []

            for blob, feats, pred in zip(clusters.blobs, feature_objs, predictions):
                hazards.append(
                    Hazard(
                        label=pred["label"],
                        confidence=pred["confidence"],
                        severity=pred.get("severity", 0),
                        centroid=tuple(float(v) for v in blob.centroid),
                        n_points=blob.n_points,
                        features=feats.as_dict(),
                        distribution=pred.get("distribution", {}),
                    )
                )

        hazards.sort(key=lambda h: (h.severity, h.confidence), reverse=True)

        return HazardReport(
            hazards=hazards,
            registration_fitness=reg.fitness,
            registration_rmse=reg.inlier_rmse,
            n_added_points=diff.added_count,
            n_removed_points=diff.removed_count,
            n_blobs=clusters.n_blobs,
            eps_used=clusters.eps_used,
            duration_s=time.perf_counter() - t0,
            warnings=warnings,
        )
