import numpy as np
import warnings
from CosmoTree.tree import build_tree
from CosmoTree.traversal import traverse
from CosmoTree.executor import execute_tree_correlation

try:
    import cupy
    cupy_available = True
except ImportError:
    cupy_available = False

def test_executor_cpu_fallback():
    # Test that it doesn't crash without CuPy and returns (0j, 0.0)
    # or if CuPy is present, it runs (we can't verify result without GPU in this env if no GPU).
    
    # Simple setup
    ra = np.array([0.0, 0.1])
    dec = np.array([0.0, 0.0])
    w = np.array([1.0, 1.0])
    g1 = np.array([0.1, -0.1])
    g2 = np.array([0.0, 0.0])
    shear_map = np.stack([g1, g2], axis=1)
    
    tree = build_tree(ra, dec, w)
    inter, leaves = traverse(tree, None, min_sep=0.0, max_sep=1.0, slop=1.0)
    
    # Mock particle coords
    x, y, z = tree['x'][tree['idx_array']], tree['y'][tree['idx_array']], tree['z'][tree['idx_array']]
    # Wait, tree['x'] are node centers. 
    # We need to re-compute particle coords from RA/Dec?
    # Yes.
    px = np.cos(dec) * np.cos(ra)
    py = np.cos(dec) * np.sin(ra)
    pz = np.sin(dec)
    
    # Run
    if not cupy_available:
        with warnings.catch_warnings(record=True) as w_log:
            warnings.simplefilter("always")
            xi, weight = execute_tree_correlation(shear_map, w, tree, inter, leaves, (px, py, pz))
            assert xi == 0j
            assert weight == 0.0
            assert len(w_log) >= 1
            assert "CuPy not found" in str(w_log[0].message)
    else:
        # If CuPy is available (e.g. CI with GPU), it should run.
        xi, weight = execute_tree_correlation(shear_map, w, tree, inter, leaves, (px, py, pz))
        # Basic check
        assert isinstance(xi, (complex, np.complex128))
        assert isinstance(weight, (float, np.float64))

if __name__ == "__main__":
    test_executor_cpu_fallback()
    print("Test passed")
