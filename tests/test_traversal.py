import numpy as np
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
    # slop = 0.0. d > rA + rB + slop?
    # d ~ 0.5. rA ~ 0.01, rB ~ 0.01.
    # 0.5 > 0.02 + 0.
    # Should get interaction pair (RootA, RootB).
    inter, leaves = traverse(tree_a, tree_b, min_sep=0.1, max_sep=1.0, slop=0.0)
    
    assert len(inter) > 0
    assert len(leaves) == 0
    # Should be root-root interaction if approximation allowed
    # Roots are index 0.
    assert inter[0, 0] == 0
    assert inter[0, 1] == 0
    assert inter.shape[1] == 4
    assert np.isfinite(inter[0, 2])
    assert np.isfinite(inter[0, 3])
    
    # 2. Approximation disallowed (large slop or small theta implied)
    # Force descent by setting slop large, e.g., 1.0.
    # d (0.5) < rA + rB + 1.0.
    # Should descend to leaves.
    inter, leaves = traverse(tree_a, tree_b, min_sep=0.1, max_sep=1.0, slop=1.0)
    
    assert len(inter) == 0
    assert len(leaves) == 4 # 2x2 pairs
    
    # Check leaf pairs
    # Indices in leaves refer to idx_array indices? No, original indices?
    # In build_tree, idx_array maps tree_order -> original_index.
    # In traversal, we output aidx[k] which is original index.
    # So leaves contains pairs of original indices.
    
    # Cluster A indices: 0, 1. Cluster B indices: 0, 1 (in their respective arrays).
    # Since we passed separate arrays, indices are 0-based for each tree.
    # Pairs should be (0,0), (0,1), (1,0), (1,1).
    expected = {(0,0), (0,1), (1,0), (1,1)}
    found = set()
    for i in range(len(leaves)):
        found.add((leaves[i, 0], leaves[i, 1]))
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

    inter, leaves = traverse(tree_a, tree_b, min_sep=0.1, max_sep=1.0, slop=0.0)

    assert inter.ndim == 2
    assert inter.shape[1] == 4
    assert len(leaves) == 0

def test_auto_correlation_symmetry():
    # 4 points close to each other
    ra = np.array([0.0, 0.01, 0.02, 0.03])
    dec = np.array([0.0, 0.0, 0.0, 0.0])
    w = np.ones(4)
    
    tree = build_tree(ra, dec, w, min_size=0.0)
    
    # Auto correlation
    # min_sep very small.
    # slop large to force leaves.
    inter, leaves = traverse(tree, None, min_sep=0.0, max_sep=1.0, slop=1.0)
    
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
    
    # Check indices
    found = set()
    for i in range(len(leaves)):
        pair = tuple(sorted((leaves[i, 0], leaves[i, 1])))
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
    inter, leaves = traverse(tree, None, min_sep=0.0, max_sep=0.5, slop=1.0)
    assert len(leaves) == 0
    assert len(inter) == 0
    
    # Min sep 0.5, Max 1.5
    inter, leaves = traverse(tree, None, min_sep=0.5, max_sep=1.5, slop=1.0)
    assert len(leaves) == 1 # (0, 1)

if __name__ == "__main__":
    test_traverse_two_clusters()
    test_auto_correlation_symmetry()
    test_range_limit()
    print("All tests passed")
