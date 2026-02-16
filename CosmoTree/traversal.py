import numpy as np
from numba import njit, float64, int32, int64

@njit(fastmath=True)
def _push_stack(stack, ptr, node_a, node_b):
    if ptr >= stack.shape[0]:
        return -1 
    stack[ptr, 0] = node_a
    stack[ptr, 1] = node_b
    return ptr + 1

@njit(fastmath=True)
def _traverse_numba(
    ax, ay, az, arad, achild_left, achild_right, astart, aend, aidx,
    bx, by, bz, brad, bchild_left, bchild_right, bstart, bend, bidx,
    min_sep, max_sep, slop, is_auto,
    max_stack, max_inter, max_leaf
):
    stack = np.empty((max_stack, 2), dtype=np.int32)
    stack_ptr = 0
    
    interaction_list = np.empty((max_inter, 2), dtype=np.int32)
    inter_ptr = 0
    
    leaf_pairs = np.empty((max_leaf, 2), dtype=np.int64)
    leaf_ptr = 0
    
    # Push root
    stack[0, 0] = 0
    stack[0, 1] = 0
    stack_ptr = 1
    
    min_sep_sq = min_sep * min_sep
    max_sep_sq = max_sep * max_sep
    
    while stack_ptr > 0:
        stack_ptr -= 1
        i = stack[stack_ptr, 0]
        j = stack[stack_ptr, 1]
        
        ra = arad[i]
        rb = brad[j]
        
        dx = ax[i] - bx[j]
        dy = ay[i] - by[j]
        dz = az[i] - bz[j]
        dsq = dx*dx + dy*dy + dz*dz
        d = np.sqrt(dsq)
        
        # Check 1: Range overlap
        # If [d - ra - rb, d + ra + rb] intersects [min_sep, max_sep]
        if (d - ra - rb > max_sep) or (d + ra + rb < min_sep):
            continue
            
        # Check 2: Approximation Criterion
        can_approximate = False
        if d > (ra + rb + slop):
            if d >= min_sep and d <= max_sep:
                can_approximate = True
                
        if can_approximate:
            if inter_ptr < max_inter:
                interaction_list[inter_ptr, 0] = i
                interaction_list[inter_ptr, 1] = j
                inter_ptr += 1
            else:
                return interaction_list, leaf_pairs, -1
            continue
            
        # Check 3: Descend
        leaf_a = (achild_left[i] == -1)
        leaf_b = (bchild_left[j] == -1)
        
        if leaf_a and leaf_b:
            # Add all constituent pairs
            start_a = astart[i]
            end_a = aend[i]
            start_b = bstart[j]
            end_b = bend[j]
            
            for ka in range(start_a, end_a):
                idx_a = aidx[ka]
                
                start_kb = start_b
                # If same leaf in auto-corr, avoid self-pairs and double counting
                if is_auto and (i == j):
                    start_kb = ka + 1
                
                for kb in range(start_kb, end_b):
                    idx_b = bidx[kb]
                    
                    if leaf_ptr < max_leaf:
                        leaf_pairs[leaf_ptr, 0] = idx_a
                        leaf_pairs[leaf_ptr, 1] = idx_b
                        leaf_ptr += 1
                    else:
                        return interaction_list, leaf_pairs, -2
            continue
            
        # Split logic
        split_a = False
        if is_auto and (i == j):
            # Always split A (symmetric)
            split_a = True
        elif leaf_a:
            split_a = False # Must split B
        elif leaf_b:
            split_a = True # Must split A
        else:
            split_a = (ra >= rb) # Split larger
            
        if split_a:
            left = achild_left[i]
            right = achild_right[i]
            
            if is_auto and (i == j):
                # Symmetric split of A vs A
                # Push (L, L), (R, R), (L, R)
                
                # Check stack overflow
                if stack_ptr + 3 > max_stack:
                    return interaction_list, leaf_pairs, -3
                     
                stack[stack_ptr, 0] = left
                stack[stack_ptr, 1] = right
                stack_ptr += 1
                
                stack[stack_ptr, 0] = right
                stack[stack_ptr, 1] = right
                stack_ptr += 1
                
                stack[stack_ptr, 0] = left
                stack[stack_ptr, 1] = left
                stack_ptr += 1
            else:
                if stack_ptr + 2 > max_stack:
                    return interaction_list, leaf_pairs, -3

                stack[stack_ptr, 0] = right
                stack[stack_ptr, 1] = j
                stack_ptr += 1
                
                stack[stack_ptr, 0] = left
                stack[stack_ptr, 1] = j
                stack_ptr += 1
        else:
            # Split B
            left = bchild_left[j]
            right = bchild_right[j]

            if stack_ptr + 2 > max_stack:
                return interaction_list, leaf_pairs, -3

            stack[stack_ptr, 0] = i
            stack[stack_ptr, 1] = right
            stack_ptr += 1
            
            stack[stack_ptr, 0] = i
            stack[stack_ptr, 1] = left
            stack_ptr += 1
            
    return interaction_list[:inter_ptr], leaf_pairs[:leaf_ptr], 0

def traverse(
    tree_a,
    tree_b=None,
    min_sep=0.0,
    max_sep=1.0,
    slop=0.0,
    max_stack=1000000,
    max_inter=10000000,
    max_leaf=10000000,
    growth_factor=2.0,
    max_retries=5
):
    """
    Perform dual-tree traversal to find interaction pairs.
    """
    is_auto = False
    if tree_b is None:
        tree_b = tree_a
        is_auto = True
        
    # Unpack arrays
    # A
    ax = tree_a['x']
    ay = tree_a['y']
    az = tree_a['z']
    arad = tree_a['radius']
    achild_left = tree_a['child_left']
    achild_right = tree_a['child_right']
    astart = tree_a['node_start']
    aend = tree_a['node_end']
    aidx = tree_a['idx_array']
    
    # B
    bx = tree_b['x']
    by = tree_b['y']
    bz = tree_b['z']
    brad = tree_b['radius']
    bchild_left = tree_b['child_left']
    bchild_right = tree_b['child_right']
    bstart = tree_b['node_start']
    bend = tree_b['node_end']
    bidx = tree_b['idx_array']
    
    cur_stack = int(max_stack)
    cur_inter = int(max_inter)
    cur_leaf = int(max_leaf)
    retries = 0

    while True:
        inter, leaves, status = _traverse_numba(
            ax, ay, az, arad, achild_left, achild_right, astart, aend, aidx,
            bx, by, bz, brad, bchild_left, bchild_right, bstart, bend, bidx,
            min_sep, max_sep, float(slop), is_auto,
            cur_stack, cur_inter, cur_leaf
        )

        if status == 0:
            return inter, leaves

        if status == -1:
            print("Warning: Interaction list overflow, retrying with larger buffer")
            cur_inter = int(cur_inter * growth_factor)
        elif status == -2:
            print("Warning: Leaf pairs list overflow, retrying with larger buffer")
            cur_leaf = int(cur_leaf * growth_factor)
        elif status == -3:
            print("Warning: Stack overflow, retrying with larger buffer")
            cur_stack = int(cur_stack * growth_factor)
        else:
            raise RuntimeError("Traversal failed with unknown status")

        retries += 1
        if retries > max_retries:
            raise RuntimeError("Traversal exceeded max_retries; increase buffer sizes or max_retries")
