import numpy as np
import pytest

pytest.importorskip("h5py")

from CosmoTree.interface import CosmoTree


def _sample_data():
    ra = np.array([0.0, 0.1, 0.2, 0.25], dtype=np.float64)
    dec = np.array([0.0, 0.0, 0.01, -0.01], dtype=np.float64)
    w = np.ones_like(ra)

    g1 = np.array([0.1, -0.1, 0.2, -0.05], dtype=np.float64)
    g2 = np.array([0.0, 0.05, -0.02, 0.03], dtype=np.float64)
    maps = np.stack([g1, g2], axis=1)
    return ra, dec, w, maps


def test_cosmotree_process_requires_preprocess():
    _, _, _, maps = _sample_data()
    ct = CosmoTree(min_sep=1e-6, max_sep=1.0, nbins=8)
    with pytest.raises(RuntimeError, match="preprocess"):
        ct.process(maps)


def test_cosmotree_preprocess_and_process():
    ra, dec, w, maps = _sample_data()
    ct = CosmoTree(min_sep=1e-6, max_sep=1.0, nbins=8, bin_slop=0.0, dtype=np.float32)
    ct.preprocess({"ra": ra, "dec": dec, "w": w})
    assert ct.is_preprocessed

    corr = ct.process(maps, w_map=w)
    assert corr.shape == (1, 1, 8)
    assert corr.dtype == np.float32


def test_cosmotree_save_and_load_roundtrip(tmp_path):
    ra, dec, w, maps = _sample_data()
    ct = CosmoTree(min_sep=1e-6, max_sep=1.0, nbins=8, bin_slop=0.0, dtype=np.float32)
    ct.preprocess((ra, dec, w))

    filename = tmp_path / "ct_geometry.h5"
    ct.save(filename)

    ct_loaded = CosmoTree(min_sep=0.1, max_sep=0.2, nbins=2, dtype=np.float64)
    ct_loaded.load(filename)

    assert ct_loaded.nbins == 8
    assert np.isclose(ct_loaded.min_sep, 1e-6)
    assert np.isclose(ct_loaded.max_sep, 1.0)
    assert ct_loaded.is_preprocessed

    # Loaded geometry does not include ra/dec by default, so provide them here.
    corr = ct_loaded.process(maps, w_map=w, ra=ra, dec=dec, dtype=np.float64)
    assert corr.shape == (1, 1, 8)
    assert corr.dtype == np.float64

