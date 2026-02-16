import numpy as np
import pytest

pytest.importorskip("h5py")

from CosmoTree.io import load_geometry, save_geometry
from CosmoTree.tree import build_tree
from CosmoTree.traversal import traverse


_TREE_DTYPES = {
    "parents": np.int32,
    "child_left": np.int32,
    "child_right": np.int32,
    "x": np.float64,
    "y": np.float64,
    "z": np.float64,
    "radius": np.float64,
    "w": np.float64,
    "node_start": np.int64,
    "node_end": np.int64,
    "idx_array": np.int64,
}


def _build_geometry():
    ra = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    dec = np.array([0.0, 0.0, 0.01, -0.01], dtype=np.float64)
    w = np.ones_like(ra)

    tree = build_tree(ra, dec, w, min_size=0.0)
    inter, inter_bins, leaf_pairs, leaf_bins = traverse(
        tree, None, min_sep=1e-6, max_sep=1.0, nbins=8, bin_slop=0.0
    )
    return tree, inter, inter_bins, leaf_pairs, leaf_bins, ra, dec


def test_geometry_roundtrip_preserves_arrays_and_dtypes(tmp_path):
    tree, inter, inter_bins, leaf_pairs, leaf_bins, ra, dec = _build_geometry()

    filename = tmp_path / "geometry.h5"
    config = {
        "nbins": 8,
        "min_sep": 1e-6,
        "max_sep": 1.0,
        "slop": 1.0,
        "leaf_bins": leaf_bins,
        "ra": ra,
        "dec": dec,
    }
    save_geometry(filename, tree, inter, leaf_pairs, config)

    tree2, inter2, inter_bins2, leaf_pairs2, leaf_bins2, config2 = load_geometry(filename)

    for key, dtype in _TREE_DTYPES.items():
        assert tree2[key].dtype == np.dtype(dtype)
        np.testing.assert_array_equal(tree2[key], tree[key].astype(dtype))

    assert inter2.dtype == np.float64
    np.testing.assert_array_equal(inter2, inter.astype(np.float64))
    assert inter_bins2.dtype == np.int32
    np.testing.assert_array_equal(inter_bins2, inter_bins.astype(np.int32))

    assert leaf_pairs2.dtype == np.int64
    np.testing.assert_array_equal(leaf_pairs2, leaf_pairs.astype(np.int64))
    assert leaf_bins2.dtype == np.int32
    np.testing.assert_array_equal(leaf_bins2, leaf_bins.astype(np.int32))

    assert config2["nbins"] == 8
    assert np.isclose(config2["min_sep"], 1e-6)
    assert np.isclose(config2["max_sep"], 1.0)
    assert np.isclose(config2["slop"], 1.0)
    assert config2["leaf_bins"].dtype == np.int32
    np.testing.assert_array_equal(config2["ra"], ra)
    np.testing.assert_array_equal(config2["dec"], dec)


def test_save_geometry_requires_leaf_bins_when_leaf_pairs_present(tmp_path):
    tree, inter, _, leaf_pairs, _, _, _ = _build_geometry()
    filename = tmp_path / "geometry_missing_leaf_bins.h5"

    with pytest.raises(ValueError, match="leaf_bins"):
        save_geometry(
            filename,
            tree,
            inter,
            leaf_pairs,
            {"nbins": 8, "min_sep": 1e-6, "max_sep": 1.0},
        )


def test_save_geometry_normalizes_tree_dtypes(tmp_path):
    tree, inter, inter_bins, leaf_pairs, leaf_bins, _, _ = _build_geometry()
    filename = tmp_path / "geometry_casts.h5"

    tree_mixed = {
        "parents": tree["parents"].astype(np.int64),
        "child_left": tree["child_left"].astype(np.int64),
        "child_right": tree["child_right"].astype(np.int64),
        "x": tree["x"].astype(np.float32),
        "y": tree["y"].astype(np.float32),
        "z": tree["z"].astype(np.float32),
        "radius": tree["radius"].astype(np.float32),
        "w": tree["w"].astype(np.float32),
        "node_start": tree["node_start"].astype(np.int32),
        "node_end": tree["node_end"].astype(np.int32),
        "idx_array": tree["idx_array"].astype(np.int32),
    }

    save_geometry(
        filename,
        tree_mixed,
        inter,
        leaf_pairs,
        {"nbins": 8, "min_sep": 1e-6, "max_sep": 1.0, "leaf_bins": leaf_bins},
    )
    tree2, inter2, inter_bins2, _, leaf_bins2, _ = load_geometry(filename)

    for key, dtype in _TREE_DTYPES.items():
        assert tree2[key].dtype == np.dtype(dtype)
    assert inter2.dtype == np.float64
    assert inter_bins2.dtype == np.int32
    assert leaf_bins2.dtype == np.int32
    np.testing.assert_array_equal(inter_bins2, inter_bins.astype(np.int32))
