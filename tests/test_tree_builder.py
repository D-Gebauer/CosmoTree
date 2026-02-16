import numpy as np
import pytest
from CosmoTree.tree import build_tree

def test_build_tree_basics():
    # Simple case with 4 points forming a square on the equator
    # RA: 0, 0.1, 0, 0.1
    # Dec: 0, 0, 0.1, 0.1
    ra = np.array([0.0, 0.1, 0.0, 0.1])
    dec = np.array([0.0, 0.0, 0.1, 0.1])
    w = np.ones(4)
    
    # Build tree with very small min_size to force full depth
    tree = build_tree(ra, dec, w, min_size=0.0)
    
    # Check that we have 4 leaves (since N=4 and min_size=0)
    # Actually, leaves might have 1 point each.
    # Total nodes for N=4 binary tree is 7 (1 root, 2 children, 4 grandchildren)
    # But it depends on the structure.
    
    # Check basic structure
    assert 'parents' in tree
    assert 'child_left' in tree
    assert 'child_right' in tree
    assert 'x' in tree
    assert 'idx_array' in tree
    
    n_nodes = len(tree['x'])
    assert n_nodes >= 4
    
    # Check parent of root is -1
    assert tree['parents'][0] == -1
    
    # Check leaf count
    is_leaf = (tree['child_left'] == -1) & (tree['child_right'] == -1)
    n_leaves = np.sum(is_leaf)
    assert n_leaves == 4
    
    # Check that all points are covered
    # Each leaf covers a range in idx_array
    # Total range length sum should be 4
    total_points = 0
    for i in range(n_nodes):
        if is_leaf[i]:
            start = tree['node_start'][i]
            end = tree['node_end'][i]
            total_points += (end - start)
            
    assert total_points == 4

def test_build_tree_large_random():
    np.random.seed(42)
    N = 1000
    ra = np.random.uniform(0, 2*np.pi, N)
    dec = np.random.uniform(-np.pi/2, np.pi/2, N)
    w = np.random.uniform(0.5, 1.5, N)
    
    # Use larger min_size to test pruning
    min_size = 0.1 # Radians (approx 6 degrees)
    tree = build_tree(ra, dec, w, min_size=min_size)
    
    n_nodes = len(tree['x'])
    is_leaf = (tree['child_left'] == -1)
    
    # Check consistency
    for i in range(n_nodes):
        if is_leaf[i]:
            # If leaf, size should be <= min_size (unless it contains only 1 point, then size is 0)
            # OR we stopped due to depth limit (default -1)
            # OR end - start <= 1
            # Note: build_tree stops if end-start <= 1 OR size <= min_size
            pass
        else:
            # Internal node must have children
            l = tree['child_left'][i]
            r = tree['child_right'][i]
            assert l != -1
            assert r != -1
            assert tree['parents'][l] == i
            assert tree['parents'][r] == i
            
            # Check radius containment logic (approx)
            # Not strictly enforced to be minimal bounding sphere, but should be reasonable.
            
    # Check that reordered indices are a permutation of 0..N-1
    idx = tree['idx_array']
    assert len(idx) == N
    assert np.all(np.sort(idx) == np.arange(N))

def test_max_depth():
    N = 100
    ra = np.random.uniform(0, 0.1, N)
    dec = np.random.uniform(0, 0.1, N)
    
    # Force max depth 2
    tree = build_tree(ra, dec, max_depth=2)
    
    # Traverse to find max depth
    depths = np.zeros(len(tree['x']), dtype=int)
    max_d = 0
    
    # We can compute depths because parents are known.
    # Root is depth 0.
    # We need to traverse from root down or just check parent depths (requires topological order or BFS)
    # Since parents are always created before children (in our implementation, wait),
    # actually parent indices are smaller than child indices because of `next_node_idx`.
    # So we can iterate i from 1 to end.
    
    for i in range(1, len(tree['x'])):
        p = tree['parents'][i]
        depths[i] = depths[p] + 1
        if depths[i] > max_d:
            max_d = depths[i]
            
    assert max_d <= 2


def test_build_tree_arc_metric_and_validation():
    ra = np.array([0.0, 0.3, 0.6], dtype=np.float64)
    dec = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    tree = build_tree(ra, dec, min_size=0.1, metric="Arc")
    assert "radius" in tree
    assert tree["radius"].dtype == np.float64

    with pytest.raises(ValueError, match="<= pi"):
        build_tree(ra, dec, min_size=np.pi + 1e-3, metric="Arc")

if __name__ == "__main__":
    test_build_tree_basics()
    test_build_tree_large_random()
    test_max_depth()
    test_build_tree_arc_metric_and_validation()
    print("All tests passed")
