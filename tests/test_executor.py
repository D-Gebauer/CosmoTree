import numpy as np
import warnings
import pytest
from CosmoTree.tree import build_tree
from CosmoTree.traversal import traverse
from CosmoTree.executor import execute_tree_correlation

try:
    import cupy
    cupy_available = True
except ImportError:
    cupy_available = False

def _build_inputs():
    ra = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    dec = np.array([0.0, 0.0, 0.01], dtype=np.float64)
    w = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    g1 = np.array([0.1, -0.1, 0.2], dtype=np.float64)
    g2 = np.array([0.0, 0.05, -0.02], dtype=np.float64)
    maps_legacy = np.stack([g1, g2], axis=1)

    tree = build_tree(ra, dec, w)
    inter, inter_bins, leaves, leaf_bins = traverse(
        tree, None, min_sep=1e-6, max_sep=1.0, nbins=8, slop=1.0
    )
    assert len(inter_bins) == len(inter)
    assert len(leaf_bins) == len(leaves)

    px = np.cos(dec) * np.cos(ra)
    py = np.cos(dec) * np.sin(ra)
    pz = np.sin(dec)
    return tree, maps_legacy, w, inter, leaves, leaf_bins, (px, py, pz)


def test_executor_tomographic_output_shape():
    tree, maps_legacy, w, inter, leaves, leaf_bins, coords = _build_inputs()
    n_bins = 8

    maps_tomo = np.stack([maps_legacy.T, (0.5 * maps_legacy).T], axis=0)
    w_tomo = np.stack([w, 0.5 * w], axis=0)

    if not cupy_available:
        with warnings.catch_warnings(record=True) as w_log:
            warnings.simplefilter("always")
            corr = execute_tree_correlation(
                maps_tomo,
                w_tomo,
                tree,
                inter,
                leaves,
                n_bins=n_bins,
                leaf_bins=leaf_bins,
                particle_coords=coords,
            )
            assert corr.shape == (2, 2, n_bins)
            assert np.all(corr == 0.0)
            assert len(w_log) >= 1
            assert "CuPy not found" in str(w_log[0].message)
    else:
        corr = execute_tree_correlation(
            maps_tomo,
            w_tomo,
            tree,
            inter,
            leaves,
            n_bins=n_bins,
            leaf_bins=leaf_bins,
            particle_coords=coords,
        )
        assert corr.shape == (2, 2, n_bins)
        assert np.issubdtype(corr.dtype, np.floating)
        assert np.all(corr[np.tril_indices(2, -1)] == 0.0)


def test_executor_single_bin_shorthand_shape():
    tree, maps_legacy, w, inter, leaves, leaf_bins, coords = _build_inputs()
    n_bins = 8
    maps_2_by_n = maps_legacy.T

    corr = execute_tree_correlation(
        maps_2_by_n,
        w,
        tree,
        inter,
        leaves,
        n_bins=n_bins,
        leaf_bins=leaf_bins,
        particle_coords=coords,
    )
    assert corr.shape == (1, 1, n_bins)


def test_executor_legacy_n_by_2_shape():
    tree, maps_legacy, w, inter, leaves, leaf_bins, coords = _build_inputs()
    n_bins = 8

    corr = execute_tree_correlation(
        maps_legacy,
        w,
        tree,
        inter,
        leaves,
        n_bins=n_bins,
        leaf_bins=leaf_bins,
        particle_coords=coords,
    )
    assert corr.shape == (1, 1, n_bins)


def test_executor_validation_errors():
    tree, maps_legacy, w, inter, leaves, leaf_bins, _ = _build_inputs()

    with pytest.raises(ValueError, match="n_bins"):
        execute_tree_correlation(maps_legacy, w, tree, inter, leaves, n_bins=0, leaf_bins=leaf_bins)

    with pytest.raises(ValueError, match="leaf_bins is required"):
        execute_tree_correlation(maps_legacy, w, tree, inter, leaves, n_bins=8)

    bad_inter = inter.copy()
    if bad_inter.shape[0] == 0:
        bad_inter = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    bad_inter[0, 4] = 999
    with pytest.raises(ValueError, match="interaction_list bin indices"):
        execute_tree_correlation(
            maps_legacy,
            w,
            tree,
            bad_inter,
            np.empty((0, 2), dtype=np.int64),
            n_bins=8,
            leaf_bins=np.empty(0, dtype=np.int32),
        )

    bad_leaf_bins = leaf_bins.copy()
    if bad_leaf_bins.shape[0] == 0:
        leaves = np.array([[0, 1]], dtype=np.int64)
        bad_leaf_bins = np.array([999], dtype=np.int32)
    else:
        bad_leaf_bins[0] = 999
    with pytest.raises(ValueError, match="leaf_bins indices"):
        execute_tree_correlation(
            maps_legacy,
            w,
            tree,
            inter,
            leaves,
            n_bins=8,
            leaf_bins=bad_leaf_bins,
            particle_coords=(np.ones(3), np.ones(3), np.ones(3)),
        )

    with pytest.raises(ValueError, match="w_map with ndim=2"):
        execute_tree_correlation(
            maps_legacy,
            np.ones((2, maps_legacy.shape[0] + 1)),
            tree,
            inter,
            np.empty((0, 2), dtype=np.int64),
            n_bins=8,
            leaf_bins=np.empty(0, dtype=np.int32),
        )

if __name__ == "__main__":
    test_executor_tomographic_output_shape()
    test_executor_single_bin_shorthand_shape()
    test_executor_legacy_n_by_2_shape()
    test_executor_validation_errors()
    print("Test passed")
