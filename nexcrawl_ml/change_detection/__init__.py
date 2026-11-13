"""Change detection: baseline vs current scan -> classified hazard blobs."""

from .clustering import (
    Blob,
    ClusterResult,
    cluster_changes,
    suggest_eps,
    suggest_min_samples,
)
from .differencing import DifferenceResult, cloud_difference
from .features import (
    FEATURE_NAMES,
    BlobFeatures,
    estimate_floor_z,
    extract_features,
    features_to_matrix,
)
from .registration import (
    RegistrationResult,
    apply_transform,
    initial_guess_from_odometry,
    register_to_baseline,
)

__all__ = [
    "Blob",
    "ClusterResult",
    "cluster_changes",
    "suggest_eps",
    "suggest_min_samples",
    "DifferenceResult",
    "cloud_difference",
    "FEATURE_NAMES",
    "BlobFeatures",
    "extract_features",
    "estimate_floor_z",
    "features_to_matrix",
    "RegistrationResult",
    "register_to_baseline",
    "initial_guess_from_odometry",
    "apply_transform",
]
