import numpy as np
import pytest
from CosmoTree.tree import build_tree
from CosmoTree.traversal import traverse

def test_traverse_two_clusters():
    # Create two small clusters separated by some distance
    # Cluster A around (0, 0)
    ra_a = np.array([0.0, 0.01])
    dec_a = np.array([0.0, 0.01])
    w_a = np.ones(2)
    
    # Cluster B around (0.5, 0) - approx 0.5 rad separation
    ra_b = np.array([0.5, 0.51])
    dec_b = np.array([0.0, 0.01])
    w_b = np.ones(2)
    
    tree_a = build_tree(ra_a, dec_a, w_a, min_size=0.0)
    tree_b = build_tree(ra_b, dec_b, w_b, min_size=0.0)
    
    # Separation is approx 0.5.
    # Radii of clusters approx 0.01 or less.
    # If we set min_sep=0.1, max_sep=1.0, they should interact.
    
    # 1. Approximation allowed
    # TreeCorr-style criterion: d > (rA + rB) / slopeff, with slopeff=min(bin_slop, angle_slop).
    # d ~ 0.5. rA ~ 0.01, rB ~ 0.01.
    # With bin_slop=0.1 and default angle_slop=1.0, slopeff=0.1, so threshold is ~0.2.
    # 0.5 > 0.2, so approximation is allowed.
    # Should get interaction pair (RootA, RootB).
    inter, inter_bins, leaves, leaf_bins = traverse(
        tree_a, tree_b, min_sep=0.1, max_sep=1.0, nbins=8, bin_slop=0.1
    )
    
    assert len(inter) > 0
    assert len(inter_bins) == len(inter)
    assert len(leaves) == 0
    assert len(leaf_bins) == 0
    # Should be root-root interaction if approximation allowed
    # Roots are index 0.
    assert inter[0, 0] == 0
    assert inter[0, 1] == 0
    assert inter.shape[1] == 5
    assert np.isfinite(inter[0, 2])
    assert np.isfinite(inter[0, 3])
    assert np.issubdtype(inter_bins.dtype, np.integer)
    assert np.all(inter_bins >= 0)
    assert np.all(inter_bins < 8)
    assert int(inter[0, 4]) == int(inter_bins[0])
    
    # 2. Approximation disallowed
    # Force descent by setting bin_slop=0.0 => slopeff=0 => never approximate.
    # Should descend to leaves.
    inter, inter_bins, leaves, leaf_bins = traverse(
        tree_a, tree_b, min_sep=0.1, max_sep=1.0, nbins=8, bin_slop=0.0
    )
    
    assert len(inter) == 0
    assert len(inter_bins) == 0
    assert len(leaves) == 4 # 2x2 pairs
    assert len(leaf_bins) == len(leaves)
    assert np.issubdtype(leaf_bins.dtype, np.integer)
    assert np.all(leaf_bins >= 0)
    assert np.all(leaf_bins < 8)
    
    # Check leaf pairs
    # In traversal, leaves contains tree-order indices.
    # Map them back to original indices using idx_array.
    
    # Cluster A indices: 0, 1. Cluster B indices: 0, 1 (in their respective arrays).
    # Since we passed separate arrays, indices are 0-based for each tree.
    # Pairs should be (0,0), (0,1), (1,0), (1,1).
    expected = {(0,0), (0,1), (1,0), (1,1)}
    found = set()
    for i in range(len(leaves)):
        idx_a = tree_a['idx_array'][leaves[i, 0]]
        idx_b = tree_b['idx_array'][leaves[i, 1]]
        found.add((idx_a, idx_b))
    assert found == expected

def test_interaction_list_format():
    ra_a = np.array([0.0, 0.01])
    dec_a = np.array([0.0, 0.01])
    w_a = np.ones(2)

    ra_b = np.array([0.5, 0.51])
    dec_b = np.array([0.0, 0.01])
    w_b = np.ones(2)

    tree_a = build_tree(ra_a, dec_a, w_a, min_size=0.0)
    tree_b = build_tree(ra_b, dec_b, w_b, min_size=0.0)

    inter, inter_bins, leaves, leaf_bins = traverse(
        tree_a, tree_b, min_sep=0.1, max_sep=1.0, nbins=8, bin_slop=0.1
    )

    assert inter.ndim == 2
    assert inter.shape[1] == 5
    assert len(inter_bins) == len(inter)
    assert len(leaves) == 0
    assert len(leaf_bins) == 0

def test_auto_correlation_symmetry():
    # 4 points close to each other
    ra = np.array([0.0, 0.01, 0.02, 0.03])
    dec = np.array([0.0, 0.0, 0.0, 0.0])
    w = np.ones(4)
    
    tree = build_tree(ra, dec, w, min_size=0.0)
    
    # Auto correlation
    # min_sep very small.
    # bin_slop=0 to force leaves.
    inter, inter_bins, leaves, leaf_bins = traverse(
        tree, None, min_sep=1e-6, max_sep=1.0, nbins=8, bin_slop=0.0
    )
    
    # Should find all pairs (i, j) with i < j (since i==j avoided in code for auto)
    # Wait, code: `if is_auto and (i == j): start_kb = ka + 1`.
    # So we get strictly upper triangular pairs for same-leaf.
    # What about cross-leaf?
    # If we split root->(L, R). We process (L, R).
    # L indices are disjoint from R indices.
    # We get all pairs in LxR.
    # Since L and R are disjoint, we don't have i==j.
    # And we effectively get upper triangular of the full matrix if sorted by tree index?
    # Basically we get unique pairs.
    
    # N=4. Pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3). Total 6.
    # Plus maybe (0,0)? No, `start_kb = ka + 1`. Self-pairs excluded.
    
    assert len(leaves) == 6
    assert len(inter) == 0
    assert len(inter_bins) == 0
    assert len(leaf_bins) == len(leaves)
    
    # Check indices (map tree-order back to original indices)
    found = set()
    for i in range(len(leaves)):
        idx_a = tree['idx_array'][leaves[i, 0]]
        idx_b = tree['idx_array'][leaves[i, 1]]
        pair = tuple(sorted((idx_a, idx_b)))
        found.add(pair)
        
    expected = {(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)}
    assert found == expected

def test_range_limit():
    # Two points far apart
    ra = np.array([0.0, 1.0]) # 1 rad separation
    dec = np.array([0.0, 0.0])
    w = np.ones(2)
    tree = build_tree(ra, dec, w, min_size=0.0)
    
    # Max sep 0.5
    inter, inter_bins, leaves, leaf_bins = traverse(
        tree, None, min_sep=1e-6, max_sep=0.5, nbins=8, bin_slop=0.0
    )
    assert len(leaves) == 0
    assert len(inter) == 0
    assert len(inter_bins) == 0
    assert len(leaf_bins) == 0
    
    # Min sep 0.5, Max 1.5
    inter, inter_bins, leaves, leaf_bins = traverse(
        tree, None, min_sep=0.5, max_sep=1.5, nbins=8, bin_slop=0.0
    )
    assert len(leaves) == 1 # (0, 1)
    assert len(inter) == 0
    assert len(inter_bins) == 0
    assert len(leaf_bins) == 1


def test_arc_metric_range_selection_differs_from_euclidean():
    # Two points separated by ~1 radian on the sphere.
    ra = np.array([0.0, 1.0], dtype=np.float64)
    dec = np.array([0.0, 0.0], dtype=np.float64)
    w = np.ones(2, dtype=np.float64)
    tree = build_tree(ra, dec, w, min_size=0.0)

    # In arc metric this window includes the pair (true arc distance ~=1.0).
    # In euclidean/chord metric it excludes it (chord distance ~=0.9589).
    inter_e, inter_bins_e, leaves_e, leaf_bins_e = traverse(
        tree, None, min_sep=0.99, max_sep=1.01, nbins=8, metric="Euclidean", bin_slop=0.0
    )
    assert len(inter_e) == 0
    assert len(inter_bins_e) == 0
    assert len(leaves_e) == 0
    assert len(leaf_bins_e) == 0

    inter_a, inter_bins_a, leaves_a, leaf_bins_a = traverse(
        tree, None, min_sep=0.99, max_sep=1.01, nbins=8, metric="Arc", bin_slop=0.0
    )
    assert len(inter_a) == 0
    assert len(inter_bins_a) == 0
    assert len(leaves_a) == 1
    assert len(leaf_bins_a) == 1


def test_arc_metric_rejects_max_sep_above_pi():
    ra = np.array([0.0, 0.2], dtype=np.float64)
    dec = np.array([0.0, 0.0], dtype=np.float64)
    tree = build_tree(ra, dec)
    with pytest.raises(ValueError, match="<= pi"):
        traverse(tree, None, min_sep=0.1, max_sep=np.pi + 1e-3, nbins=8, metric="Arc")

if __name__ == "__main__":
    test_traverse_two_clusters()
    test_auto_correlation_symmetry()
    test_range_limit()
    test_arc_metric_range_selection_differs_from_euclidean()
    test_arc_metric_rejects_max_sep_above_pi()
    print("All tests passed")
