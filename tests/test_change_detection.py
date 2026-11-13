"""Tests for clustering and differencing - the stages where a silent
parameter failure makes hazards disappear without any error."""

import numpy as np
import pytest

from nexcrawl_ml.change_detection.clustering import (
    cluster_changes,
    suggest_eps,
    suggest_min_samples,
)
from nexcrawl_ml.change_detection.differencing import cloud_difference


def _blob(center, n=200, spread=0.15, seed=0):
    rng = np.random.default_rng(seed)
    return np.array(center) + rng.normal(0, spread, (n, 3))


def test_min_samples_follows_2dim_heuristic():
    assert suggest_min_samples(200) == 6
    assert suggest_min_samples(2000) == 12
    assert suggest_min_samples(10000) == 18


def test_sparse_blob_is_not_swallowed():
    """Regression test for the real bug: a 200-point rockfall spread over
    ~1m was labelled entirely as noise by a fixed min_samples=20, so the
    hazard silently vanished with no error raised anywhere."""
    pts = _blob([2.5, 0.4, 0.4], n=200, spread=0.2)
    result = cluster_changes(pts)
    assert result.n_blobs >= 1, "sparse hazard blob was swallowed as noise"
    assert result.blobs[0].n_points > 50


def test_two_separated_blobs_stay_separate():
    a = _blob([0.0, 0.0, 0.0], seed=1)
    b = _blob([5.0, 5.0, 0.0], seed=2)
    result = cluster_changes(np.vstack([a, b]))
    assert result.n_blobs == 2


def test_blobs_sorted_by_size():
    a = _blob([0.0, 0.0, 0.0], n=300, seed=1)
    b = _blob([5.0, 0.0, 0.0], n=80, seed=2)
    result = cluster_changes(np.vstack([a, b]))
    sizes = [blob.n_points for blob in result.blobs]
    assert sizes == sorted(sizes, reverse=True)


def test_empty_input_is_safe():
    result = cluster_changes(np.empty((0, 3)))
    assert result.n_blobs == 0


def test_suggest_eps_scales_with_density():
    dense = _blob([0, 0, 0], n=500, spread=0.05, seed=3)
    sparse = _blob([0, 0, 0], n=500, spread=0.5, seed=3)
    assert suggest_eps(sparse) > suggest_eps(dense)


def test_identical_clouds_show_no_change():
    pts = _blob([0, 0, 0], n=500)
    diff = cloud_difference(pts, pts, threshold_m=0.05)
    assert diff.added_count == 0
    assert diff.removed_count == 0


def test_differencing_is_two_sided():
    """A one-sided diff misses material that has been REMOVED - an opened
    void is as important as a new debris pile."""
    base = _blob([0, 0, 0], n=300, seed=1)
    extra = _blob([4.0, 0, 0], n=150, seed=2)

    added = cloud_difference(np.vstack([base, extra]), base)
    assert added.added_count > 100 and added.removed_count == 0

    removed = cloud_difference(base, np.vstack([base, extra]))
    assert removed.removed_count > 100 and removed.added_count == 0


def test_empty_baseline_treats_everything_as_added():
    pts = _blob([0, 0, 0], n=100)
    diff = cloud_difference(pts, np.empty((0, 3)))
    assert diff.added_count == 100
