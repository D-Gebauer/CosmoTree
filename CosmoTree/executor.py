import numpy as np
import warnings

try:
    import cupy as cp
    
    _leaf_sum_kernel = cp.ElementwiseKernel(
        'int64 start, int64 end, raw int64 idx_array, raw complex128 g_pix, raw float64 w_pix',
        'complex128 g_node, float64 w_node',
        '''
        thrust::complex<double> g_sum = 0;
        double w_sum = 0;
        for (long long i = start; i < end; ++i) {
            long long idx = idx_array[i];
            double w = w_pix[idx];
            thrust::complex<double> val = g_pix[idx];
            g_sum += val * w;
            w_sum += w;
        }
        g_node = g_sum;
        w_node = w_sum;
        ''',
        'leaf_sum_kernel',
        preamble='#include <thrust/complex.h>'
    )
    
    _node_agg_kernel = cp.ElementwiseKernel(
        'int32 left, int32 right, raw complex128 node_shear, raw float64 node_weight',
        'complex128 g_parent, float64 w_parent',
        '''
        thrust::complex<double> gl = node_shear[left];
        thrust::complex<double> gr = node_shear[right];
        g_parent = gl + gr;
        w_parent = node_weight[left] + node_weight[right];
        ''',
        'node_agg_kernel',
        preamble='#include <thrust/complex.h>'
    )

    _tree_corr_kernel = cp.ElementwiseKernel(
        'int32 i, int32 j, raw complex128 node_shear, raw float64 node_weight, raw float64 nx, raw float64 ny, raw float64 nz',
        'complex128 corr, float64 w_prod',
        '''
        thrust::complex<double> gi = node_shear[i];
        thrust::complex<double> gj = node_shear[j];
        
        // Compute phi using projection onto tangent plane of i
        // i is center (ax, ay, az). j is target (bx, by, bz).
        double ax = nx[i];
        double ay = ny[i];
        double az = nz[i];
        
        double bx = nx[j];
        double by = ny[j];
        double bz = nz[j];
        
        // Tangent basis vectors u (East), v (North) at i
        // u = (-sin_ra, cos_ra, 0)
        // v = (-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec)
        // cos_dec = sqrt(ax^2 + ay^2)
        // sin_dec = az
        // cos_ra = ax / cos_dec
        // sin_ra = ay / cos_dec
        
        double r2 = ax*ax + ay*ay;
        double r = sqrt(r2);
        double ux, uy, uz;
        double vx, vy, vz;
        
        if (r > 1e-9) {
            double cos_ra = ax / r;
            double sin_ra = ay / r;
            double cos_dec = r;
            double sin_dec = az;
            
            ux = -sin_ra;
            uy = cos_ra;
            uz = 0.0;
            
            vx = -sin_dec * cos_ra;
            vy = -sin_dec * sin_ra;
            vz = cos_dec;
        } else {
            // Pole (assume North Pole for simplicity, or handle South)
            // At North Pole (0,0,1), RA is undefined. 
            // Usually we define a local frame.
            // u = (0, 1, 0) (East along y?)
            // v = (-1, 0, 0) (South along -x?)
            // Just use placeholder.
            ux = 1.0; uy = 0.0; uz = 0.0;
            vx = 0.0; vy = 1.0; vz = 0.0;
        }
        
        // Separation vector
        double dx = bx - ax;
        double dy = by - ay;
        double dz = bz - az;
        
        // Project onto tangent plane
        double xp = dx * ux + dy * uy + dz * uz;
        double yp = dx * vx + dy * vy + dz * vz;
        
        // Angle relative to East (u)
        double phi = atan2(yp, xp);
        
        thrust::complex<double> rot = thrust::exp(thrust::complex<double>(0, -2 * phi));
        
        corr = gi * thrust::conj(gj) * rot;
        w_prod = node_weight[i] * node_weight[j];
        ''',
        'tree_corr_kernel',
        preamble='#include <thrust/complex.h>'
    )

    _leaf_pair_kernel = cp.ElementwiseKernel(
        'int64 idx_a, int64 idx_b, raw complex128 g_pix, raw float64 w_pix, raw float64 px, raw float64 py, raw float64 pz',
        'complex128 corr, float64 w_prod',
        '''
        thrust::complex<double> ga = g_pix[idx_a];
        thrust::complex<double> gb = g_pix[idx_b];
        double wa = w_pix[idx_a];
        double wb = w_pix[idx_b];
        
        double ax = px[idx_a];
        double ay = py[idx_a];
        double az = pz[idx_a];
        
        double bx = px[idx_b];
        double by = py[idx_b];
        double bz = pz[idx_b];
        
        double r2 = ax*ax + ay*ay;
        double r = sqrt(r2);
        double ux, uy, uz;
        double vx, vy, vz;
        
        if (r > 1e-9) {
            double cos_ra = ax / r;
            double sin_ra = ay / r;
            double cos_dec = r;
            double sin_dec = az;
            
            ux = -sin_ra;
            uy = cos_ra;
            uz = 0.0;
            
            vx = -sin_dec * cos_ra;
            vy = -sin_dec * sin_ra;
            vz = cos_dec;
        } else {
            ux = 1.0; uy = 0.0; uz = 0.0;
            vx = 0.0; vy = 1.0; vz = 0.0;
        }
        
        double dx = bx - ax;
        double dy = by - ay;
        double dz = bz - az;
        
        double xp = dx * ux + dy * uy + dz * uz;
        double yp = dx * vx + dy * vy + dz * vz;
        
        double phi = atan2(yp, xp);
        thrust::complex<double> rot = thrust::exp(thrust::complex<double>(0, -2 * phi));
        
        corr = (wa * ga) * thrust::conj(wb * gb) * rot;
        w_prod = wa * wb;
        ''',
        'leaf_pair_kernel',
        preamble='#include <thrust/complex.h>'
    )

except ImportError:
    cp = None
    _leaf_sum_kernel = None
    _node_agg_kernel = None
    _tree_corr_kernel = None
    _leaf_pair_kernel = None

def _get_levels(parents):
    n_nodes = len(parents)
    depths = np.zeros(n_nodes, dtype=np.int32)
    depths[0] = 0
    for i in range(1, n_nodes):
        p = parents[i]
        depths[i] = depths[p] + 1
    max_depth = np.max(depths)
    levels = []
    for d in range(max_depth, -1, -1):
        levels.append(np.where(depths == d)[0])
    return levels

def _fill_tree_gpu(shear_map, w_map, tree, levels):
    if cp is None:
        raise RuntimeError("CuPy not installed.")
        
    n_nodes = len(tree['x'])
    node_shear = cp.zeros(n_nodes, dtype=cp.complex128)
    node_weight = cp.zeros(n_nodes, dtype=cp.float64)
    
    t_child_left = cp.asarray(tree['child_left'])
    t_child_right = cp.asarray(tree['child_right'])
    t_node_start = cp.asarray(tree['node_start'])
    t_node_end = cp.asarray(tree['node_end'])
    t_idx_array = cp.asarray(tree['idx_array'])
    
    d_shear = cp.asarray(shear_map)
    d_w = cp.asarray(w_map)
    g_pix = d_shear[:, 0] + 1j * d_shear[:, 1]
    
    # Leaves
    is_leaf = (t_child_left == -1)
    leaf_indices = cp.where(is_leaf)[0]
    
    if leaf_indices.size > 0:
        _leaf_sum_kernel(
            t_node_start[leaf_indices],
            t_node_end[leaf_indices],
            t_idx_array,
            g_pix,
            d_w,
            node_shear[leaf_indices],
            node_weight[leaf_indices]
        )
    
    # Upward Pass
    for level_nodes in levels:
        d_indices = cp.asarray(level_nodes)
        node_children = t_child_left[d_indices]
        is_internal = (node_children != -1)
        internal_indices = d_indices[is_internal]
        
        if internal_indices.size > 0:
            _node_agg_kernel(
                t_child_left[internal_indices],
                t_child_right[internal_indices],
                node_shear,
                node_weight,
                node_shear[internal_indices],
                node_weight[internal_indices]
            )
            
    return node_shear, node_weight

def execute_tree_correlation(
    shear_map,
    w_map,
    tree,
    interaction_list,
    leaf_pairs,
    particle_coords=None,
    ra=None,
    dec=None
):
    if cp is None:
        warnings.warn("CuPy not found. Returning 0.")
        return 0j, 0.0

    n_pix = shear_map.shape[0]
    if w_map.shape[0] != n_pix:
        raise ValueError("w_map length must match shear_map length")

    if leaf_pairs is not None and len(leaf_pairs) > 0:
        leaf_pairs_np = np.asarray(leaf_pairs)
        if leaf_pairs_np.ndim != 2 or leaf_pairs_np.shape[1] != 2:
            raise ValueError("leaf_pairs must be a 2D array with shape (N, 2)")
        if leaf_pairs_np.size > 0:
            min_idx = int(leaf_pairs_np.min())
            max_idx = int(leaf_pairs_np.max())
            if min_idx < 0 or max_idx >= n_pix:
                raise ValueError("leaf_pairs indices are out of bounds for shear_map")

    parents = tree['parents']
    levels = _get_levels(parents)
    
    node_shear, node_weight = _fill_tree_gpu(shear_map, w_map, tree, levels)
    
    d_inter = cp.asarray(interaction_list)
    d_leaves = cp.asarray(leaf_pairs)
    
    xi_inter = cp.zeros(d_inter.shape[0], dtype=cp.complex128)
    w_inter = cp.zeros(d_inter.shape[0], dtype=cp.float64)
    xi_leaf = cp.zeros(d_leaves.shape[0], dtype=cp.complex128)
    w_leaf = cp.zeros(d_leaves.shape[0], dtype=cp.float64)
    
    t_x = cp.asarray(tree['x'])
    t_y = cp.asarray(tree['y'])
    t_z = cp.asarray(tree['z'])
    
    if d_inter.shape[0] > 0:
        _tree_corr_kernel(
            d_inter[:, 0],
            d_inter[:, 1],
            node_shear,
            node_weight,
            t_x, t_y, t_z,
            xi_inter,
            w_inter
        )
        
    if d_leaves.shape[0] > 0:
        d_shear = cp.asarray(shear_map)
        d_w = cp.asarray(w_map)
        g_pix = d_shear[:, 0] + 1j * d_shear[:, 1]

        px, py, pz = None, None, None
        if particle_coords is not None:
            if len(particle_coords) != 3:
                raise ValueError("particle_coords must be a 3-tuple of arrays")
            n_coords = particle_coords[0].shape[0]
            if (particle_coords[1].shape[0] != n_coords) or (particle_coords[2].shape[0] != n_coords):
                raise ValueError("particle_coords arrays must have matching lengths")
            if n_coords != n_pix:
                raise ValueError("particle_coords length must match shear_map length")
            px = cp.asarray(particle_coords[0])
            py = cp.asarray(particle_coords[1])
            pz = cp.asarray(particle_coords[2])
        elif ra is not None and dec is not None:
            if len(ra) != len(dec):
                raise ValueError("ra and dec must have matching lengths")
            if len(ra) != n_pix:
                raise ValueError("ra/dec length must match shear_map length")
            x = np.cos(dec) * np.cos(ra)
            y = np.cos(dec) * np.sin(ra)
            z = np.sin(dec)
            px = cp.asarray(x)
            py = cp.asarray(y)
            pz = cp.asarray(z)
        else:
            raise ValueError("Must provide particle coordinates (particle_coords or ra/dec) for leaf calculations")

        _leaf_pair_kernel(
            d_leaves[:, 0],
            d_leaves[:, 1],
            g_pix,
            d_w,
            px, py, pz,
            xi_leaf,
            w_leaf
        )
        
    total_xi = cp.sum(xi_inter) + cp.sum(xi_leaf)
    total_w = cp.sum(w_inter) + cp.sum(w_leaf)
    
    return total_xi.get(), total_w.get()
