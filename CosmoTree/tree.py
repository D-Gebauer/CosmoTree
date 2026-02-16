import numpy as np
from numba import njit, float64, int32, int64


def _normalize_metric(metric):
    if isinstance(metric, str):
        m = metric.strip().lower()
        if m in ("euclidean", "chord"):
            return "euclidean"
        if m in ("arc", "great_circle", "great-circle", "greatcircle"):
            return "arc"
    raise ValueError("metric must be one of: 'Euclidean', 'Arc'")


@njit(fastmath=True)
def _angular_to_cartesian(ra, dec):
    """
    Convert RA/Dec to Cartesian coordinates on the unit sphere.
    RA, Dec in radians.
    """
    cos_dec = np.cos(dec)
    x = cos_dec * np.cos(ra)
    y = cos_dec * np.sin(ra)
    z = np.sin(dec)
    return x, y, z

@njit(fastmath=True)
def _compute_node_properties(idx_start, idx_end, idx_array, x, y, z, w):
    """
    Compute centroid and radius (squared) of a node.
    """
    sum_w = 0.0
    sum_wx = 0.0
    sum_wy = 0.0
    sum_wz = 0.0
    
    for k in range(idx_start, idx_end):
        i = idx_array[k]
        wi = w[i]
        sum_w += wi
        sum_wx += wi * x[i]
        sum_wy += wi * y[i]
        sum_wz += wi * z[i]
        
    if sum_w == 0.0:
        # Should not happen with positive weights
        return 0.0, 0.0, 0.0, 0.0, 0.0

    cen_x = sum_wx / sum_w
    cen_y = sum_wy / sum_w
    cen_z = sum_wz / sum_w
    
    # Normalize for Sphere
    norm = np.sqrt(cen_x*cen_x + cen_y*cen_y + cen_z*cen_z)
    if norm > 0:
        cen_x /= norm
        cen_y /= norm
        cen_z /= norm
        
    # Calculate radius squared (max dist sq from centroid)
    max_dsq = 0.0
    for k in range(idx_start, idx_end):
        i = idx_array[k]
        dx = x[i] - cen_x
        dy = y[i] - cen_y
        dz = z[i] - cen_z
        dsq = dx*dx + dy*dy + dz*dz
        if dsq > max_dsq:
            max_dsq = dsq
            
    return cen_x, cen_y, cen_z, max_dsq, sum_w

@njit(fastmath=True)
def _partition(idx_array, x, y, z, start, end, split_dim, split_val):
    """
    Partition idx_array[start:end] such that elements with pos[split_dim] < split_val come first.
    Returns the split index.
    """
    i = start
    j = end - 1
    
    # Using Hoare partition scheme or similar
    # We want elements < split_val on left, >= on right
    
    while i <= j:
        # Find element on left that belongs on right
        val_i = 0.0
        idx_i = idx_array[i]
        if split_dim == 0: val_i = x[idx_i]
        elif split_dim == 1: val_i = y[idx_i]
        else: val_i = z[idx_i]
            
        while i <= j and val_i < split_val:
            i += 1
            if i <= j:
                idx_i = idx_array[i]
                if split_dim == 0: val_i = x[idx_i]
                elif split_dim == 1: val_i = y[idx_i]
                else: val_i = z[idx_i]
        
        # Find element on right that belongs on left
        val_j = 0.0
        idx_j = idx_array[j]
        if split_dim == 0: val_j = x[idx_j]
        elif split_dim == 1: val_j = y[idx_j]
        else: val_j = z[idx_j]

        while i <= j and val_j >= split_val:
            j -= 1
            if i <= j:
                idx_j = idx_array[j]
                if split_dim == 0: val_j = x[idx_j]
                elif split_dim == 1: val_j = y[idx_j]
                else: val_j = z[idx_j]
        
        if i < j:
            # Swap
            temp = idx_array[i]
            idx_array[i] = idx_array[j]
            idx_array[j] = temp
            i += 1
            j -= 1
            
    return i

@njit(fastmath=True)
def _partition_median(idx_array, x, y, z, start, end, split_dim):
    """
    Partition idx_array[start:end] based on median value along split_dim.
    Simple approximation: just find the median position using quickselect or similar?
    Or just split by count (middle index). 
    TreeCorr 'Median' split method splits by count (equal number of points).
    Actually, to match TreeCorr's fallback, we should just split at (start+end)//2?
    Wait, TreeCorr Median split actually partitions by the coordinate value of the median element.
    But effectively it ensures equal split.
    For simplicity and speed, we can just use the (start+end)//2 split if Mean fails,
    BUT we need to ensure spatial locality. 
    So we should use `std::nth_element` equivalent.
    """
    mid = (start + end) // 2
    
    # Selection algorithm (Quickselect) to put the median element at 'mid'
    # and partition around it.
    
    left = start
    right = end - 1
    k = mid
    
    while left < right:
        # Pivot selection (median of 3 or random)
        pivot_idx = (left + right) // 2
        pivot_val = 0.0
        pidx = idx_array[pivot_idx]
        if split_dim == 0: pivot_val = x[pidx]
        elif split_dim == 1: pivot_val = y[pidx]
        else: pivot_val = z[pidx]
        
        # Partition
        i = left
        j = right
        while i <= j:
            val_i = 0.0
            idx_i = idx_array[i]
            if split_dim == 0: val_i = x[idx_i]
            elif split_dim == 1: val_i = y[idx_i]
            else: val_i = z[idx_i]
            while val_i < pivot_val:
                i += 1
                idx_i = idx_array[i]
                if split_dim == 0: val_i = x[idx_i]
                elif split_dim == 1: val_i = y[idx_i]
                else: val_i = z[idx_i]

            val_j = 0.0
            idx_j = idx_array[j]
            if split_dim == 0: val_j = x[idx_j]
            elif split_dim == 1: val_j = y[idx_j]
            else: val_j = z[idx_j]
            while val_j > pivot_val:
                j -= 1
                idx_j = idx_array[j]
                if split_dim == 0: val_j = x[idx_j]
                elif split_dim == 1: val_j = y[idx_j]
                else: val_j = z[idx_j]
                
            if i <= j:
                temp = idx_array[i]
                idx_array[i] = idx_array[j]
                idx_array[j] = temp
                i += 1
                j -= 1
        
        if k <= j:
            right = j
        elif k >= i:
            left = i
        else:
            break # k is in the gap or pivot was exactly median
            
    return mid

@njit(fastmath=True)
def _build_tree_numba(x, y, z, w, min_size_sq, max_depth):
    N = len(x)
    idx_array = np.arange(N, dtype=np.int64)
    
    # Allocate arrays for nodes. 
    # Max nodes in a binary tree with N leaves is 2*N - 1.
    max_nodes = 2 * N
    
    node_start = np.zeros(max_nodes, dtype=np.int64)
    node_end = np.zeros(max_nodes, dtype=np.int64)
    node_child_left = np.full(max_nodes, -1, dtype=np.int32)
    node_child_right = np.full(max_nodes, -1, dtype=np.int32)
    node_x = np.zeros(max_nodes, dtype=np.float64)
    node_y = np.zeros(max_nodes, dtype=np.float64)
    node_z = np.zeros(max_nodes, dtype=np.float64)
    node_radius = np.zeros(max_nodes, dtype=np.float64)
    node_w = np.zeros(max_nodes, dtype=np.float64) # Sum of weights
    node_parent = np.full(max_nodes, -1, dtype=np.int32)

    # Stack for traversal: (node_idx, start_idx, end_idx, depth)
    stack_node = np.zeros(max_nodes, dtype=np.int32)
    stack_start = np.zeros(max_nodes, dtype=np.int64)
    stack_end = np.zeros(max_nodes, dtype=np.int64)
    stack_depth = np.zeros(max_nodes, dtype=np.int32)
    
    stack_ptr = 0
    
    # Root node
    next_node_idx = 1 # 0 is root
    stack_node[0] = 0
    stack_start[0] = 0
    stack_end[0] = N
    stack_depth[0] = 0
    stack_ptr = 1
    
    while stack_ptr > 0:
        stack_ptr -= 1
        curr_node = stack_node[stack_ptr]
        start = stack_start[stack_ptr]
        end = stack_end[stack_ptr]
        depth = stack_depth[stack_ptr]
        
        # Store start/end
        node_start[curr_node] = start
        node_end[curr_node] = end
        
        # Compute properties
        cen_x, cen_y, cen_z, max_dsq, sum_w = _compute_node_properties(
            start, end, idx_array, x, y, z, w
        )
        
        node_x[curr_node] = cen_x
        node_y[curr_node] = cen_y
        node_z[curr_node] = cen_z
        node_radius[curr_node] = np.sqrt(max_dsq)
        node_w[curr_node] = sum_w
        
        # Check leaf conditions
        is_leaf = False
        if (end - start) <= 1:
            is_leaf = True
        elif max_dsq <= min_size_sq:
            is_leaf = True
        elif (max_depth > 0) and (depth >= max_depth):
            is_leaf = True
            
        if is_leaf:
            continue
            
        # Split logic
        # 1. Compute bounds to find split dimension
        min_x, max_x = x[idx_array[start]], x[idx_array[start]]
        min_y, max_y = y[idx_array[start]], y[idx_array[start]]
        min_z, max_z = z[idx_array[start]], z[idx_array[start]]
        
        for k in range(start + 1, end):
            i = idx_array[k]
            if x[i] < min_x: min_x = x[i]
            if x[i] > max_x: max_x = x[i]
            if y[i] < min_y: min_y = y[i]
            if y[i] > max_y: max_y = y[i]
            if z[i] < min_z: min_z = z[i]
            if z[i] > max_z: max_z = z[i]
            
        dx = max_x - min_x
        dy = max_y - min_y
        dz = max_z - min_z
        
        split_dim = 0
        if dy > dx:
            split_dim = 1
            if dz > dy:
                split_dim = 2
        elif dz > dx:
            split_dim = 2
            
        # 2. Mean split
        split_val = 0.0
        if split_dim == 0: split_val = cen_x
        elif split_dim == 1: split_val = cen_y
        else: split_val = cen_z
            
        mid = _partition(idx_array, x, y, z, start, end, split_dim, split_val)
        
        # Fallback if split failed (all on one side)
        if mid == start or mid == end:
            # Use Median split (partition by count/median)
            mid = _partition_median(idx_array, x, y, z, start, end, split_dim)
            
        if mid == start or mid == end:
            # Should basically be impossible with Median split unless all points identical
            # Force split in half
            mid = (start + end) // 2
            
        # Create children
        left_node = next_node_idx
        right_node = next_node_idx + 1
        next_node_idx += 2
        
        node_child_left[curr_node] = left_node
        node_child_right[curr_node] = right_node
        
        node_parent[left_node] = curr_node
        node_parent[right_node] = curr_node
        
        # Push to stack
        stack_node[stack_ptr] = right_node
        stack_start[stack_ptr] = mid
        stack_end[stack_ptr] = end
        stack_depth[stack_ptr] = depth + 1
        stack_ptr += 1
        
        stack_node[stack_ptr] = left_node
        stack_start[stack_ptr] = start
        stack_end[stack_ptr] = mid
        stack_depth[stack_ptr] = depth + 1
        stack_ptr += 1
        
    # Trim arrays
    final_count = next_node_idx
    return (
        node_parent[:final_count],
        node_child_left[:final_count],
        node_child_right[:final_count],
        node_x[:final_count],
        node_y[:final_count],
        node_z[:final_count],
        node_radius[:final_count],
        node_w[:final_count],
        node_start[:final_count],
        node_end[:final_count],
        idx_array
    )

def build_tree(ra, dec, w=None, min_size=0.0, max_depth=-1, metric="Euclidean"):
    """
    Build a Ball Tree from RA/Dec/Weight.
    
    Args:
        ra, dec: arrays of coordinates in radians.
        w: array of weights (optional, defaults to 1).
        min_size: minimum node size to split (interpreted in selected metric).
        max_depth: maximum depth of the tree.
        metric: 'Euclidean' (chord) or 'Arc' (great-circle).
        
    Returns:
        Dictionary containing flat arrays:
        - parents
        - child_left
        - child_right
        - x, y, z (centers)
        - radius
        - w (sum weights)
        - node_start, node_end (indices into idx_array)
        - idx_array (reordered particle indices)
    """
    metric_name = _normalize_metric(metric)

    if min_size < 0.0:
        raise ValueError("min_size must be >= 0")
    if metric_name == "arc":
        if min_size > np.pi:
            raise ValueError("for Arc metric, min_size must be <= pi")
        # Tree geometry is built in Cartesian space; convert user arc threshold to chord.
        min_size_internal = 2.0 * np.sin(0.5 * min_size)
    else:
        min_size_internal = float(min_size)

    if w is None:
        w = np.ones_like(ra)
        
    x, y, z = _angular_to_cartesian(ra, dec)
    min_size_sq = min_size_internal * min_size_internal
    
    (parents, left, right, nx, ny, nz, nr, nw, nstart, nend, idx) = _build_tree_numba(
        x, y, z, w, min_size_sq, max_depth
    )
    
    return {
        'parents': parents,
        'child_left': left,
        'child_right': right,
        'x': nx,
        'y': ny,
        'z': nz,
        'radius': nr,
        'w': nw,
        'node_start': nstart,
        'node_end': nend,
        'idx_array': idx
    }
