"""Tests for geometric feature extraction."""

import numpy as np

from nexcrawl_ml.change_detection.features import (
    FEATURE_NAMES,
    estimate_floor_z,
    extract_features,
    features_to_matrix,
)


def _box(nx=10, ny=10, nz=10, sx=1.0, sy=1.0, sz=1.0, z0=0.0):
    g = np.meshgrid(
        np.linspace(0, sx, nx), np.linspace(0, sy, ny), np.linspace(z0, z0 + sz, nz)
    )
    return np.column_stack([a.ravel() for a in g])


def test_volume_matches_bounding_box():
    pts = _box(sx=2.0, sy=1.0, sz=0.5)
    f = extract_features(pts)
    assert np.isclose(f.volume_m3, 1.0, rtol=0.01)


def test_height_relative_to_floor():
    pts = _box(z0=2.0, sz=0.2)
    f = extract_features(pts, floor_z=0.0)
    assert 2.0 < f.height_above_floor_m < 2.3


def test_aspect_ratio_of_elongated_blob():
    pts = _box(sx=4.0, sy=1.0, sz=0.5)
    f = extract_features(pts)
    assert f.aspect_ratio > 5.0


def test_planar_blob_has_high_planarity():
    """A flat sheet - the classic registration artefact signature."""
    rng = np.random.default_rng(0)
    pts = np.column_stack(
        [rng.uniform(0, 1, 500), rng.uniform(0, 1, 500), rng.normal(0, 0.001, 500)]
    )
    f = extract_features(pts)
    assert f.planarity > 0.8
    assert f.surface_roughness < 0.05


def test_rough_blob_has_high_roughness():
    rng = np.random.default_rng(1)
    pts = rng.normal(0, 0.3, (500, 3))  # isotropic -> not planar, rough
    f = extract_features(pts)
    assert f.surface_roughness > 0.2


def test_degenerate_blob_returns_safe_defaults():
    f = extract_features(np.array([[0.0, 0.0, 0.0]]))
    assert f.volume_m3 == 0.0
    assert f.aspect_ratio == 1.0


def test_floor_estimate_ignores_single_outlier():
    """The lowest single point is usually a stray that survived SOR."""
    pts = np.column_stack(
        [np.zeros(100), np.zeros(100), np.concatenate([[-5.0], np.full(99, 1.0)])]
    )
    assert estimate_floor_z(pts) > 0.0


def test_feature_matrix_shape():
    feats = [extract_features(_box()) for _ in range(3)]
    X = features_to_matrix(feats)
    assert X.shape == (3, len(FEATURE_NAMES))


def test_empty_feature_matrix():
    assert features_to_matrix([]).shape == (0, len(FEATURE_NAMES))
