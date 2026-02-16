import numpy as np
import warnings

try:
    import cupy as cp

    _leaf_sum_kernel = cp.ElementwiseKernel(
        "int64 start, int64 end, raw int64 idx_array, raw complex128 g_pix, raw float64 w_pix",
        "complex128 g_node, float64 w_node",
        """
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
        """,
        "leaf_sum_kernel",
        preamble="#include <thrust/complex.h>",
    )

    _node_agg_kernel = cp.ElementwiseKernel(
        "int32 left, int32 right, raw complex128 node_shear, raw float64 node_weight",
        "complex128 g_parent, float64 w_parent",
        """
        thrust::complex<double> gl = node_shear[left];
        thrust::complex<double> gr = node_shear[right];
        g_parent = gl + gr;
        w_parent = node_weight[left] + node_weight[right];
        """,
        "node_agg_kernel",
        preamble="#include <thrust/complex.h>",
    )

    _tomo_interaction_kernel = cp.RawKernel(
        r"""
        #include <cuComplex.h>
        extern "C" __global__
        void tomo_interaction_kernel(
            const int* i_idx,
            const int* j_idx,
            const double* rot_re,
            const double* rot_im,
            const int* bins,
            const cuDoubleComplex* node_shear,
            const int n_tomo,
            const int n_nodes,
            const int n_bins,
            const long long n_inter,
            double* out
        ) {
            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= n_inter) return;

            const int i = i_idx[tid];
            const int j = j_idx[tid];
            const int bin = bins[tid];
            if (i < 0 || j < 0 || i >= n_nodes || j >= n_nodes || bin < 0 || bin >= n_bins) return;

            const double rr = rot_re[tid];
            const double ri = rot_im[tid];
            for (int a = 0; a < n_tomo; ++a) {
                const cuDoubleComplex gi = node_shear[(long long)a * n_nodes + i];
                for (int b = a; b < n_tomo; ++b) {
                    const cuDoubleComplex gj = node_shear[(long long)b * n_nodes + j];
                    const double prod_re = gi.x * gj.x + gi.y * gj.y;
                    const double prod_im = gi.y * gj.x - gi.x * gj.y;
                    const double val = prod_re * rr - prod_im * ri;
                    const long long out_idx = ((long long)a * n_tomo + b) * n_bins + bin;
                    atomicAdd(out + out_idx, val);
                }
            }
        }
        """,
        "tomo_interaction_kernel",
    )

    _tomo_leaf_kernel = cp.RawKernel(
        r"""
        #include <cuComplex.h>
        #include <math.h>
        extern "C" __global__
        void tomo_leaf_kernel(
            const long long* idx_a,
            const long long* idx_b,
            const int* bins,
            const cuDoubleComplex* g_pix,
            const double* w_pix,
            const double* px,
            const double* py,
            const double* pz,
            const int n_tomo,
            const int n_pix,
            const int n_bins,
            const long long n_pairs,
            double* out
        ) {
            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= n_pairs) return;

            const long long ia = idx_a[tid];
            const long long ib = idx_b[tid];
            const int bin = bins[tid];
            if (ia < 0 || ib < 0 || ia >= n_pix || ib >= n_pix || bin < 0 || bin >= n_bins) return;

            const double ax = px[ia];
            const double ay = py[ia];
            const double az = pz[ia];
            const double bx = px[ib];
            const double by = py[ib];
            const double bz = pz[ib];

            const double r2 = ax * ax + ay * ay;
            const double r = sqrt(r2);
            double ux, uy, uz;
            double vx, vy, vz;
            if (r > 1e-9) {
                const double cos_ra = ax / r;
                const double sin_ra = ay / r;
                const double cos_dec = r;
                const double sin_dec = az;
                ux = -sin_ra;
                uy = cos_ra;
                uz = 0.0;
                vx = -sin_dec * cos_ra;
                vy = -sin_dec * sin_ra;
                vz = cos_dec;
            } else {
                ux = 1.0;
                uy = 0.0;
                uz = 0.0;
                vx = 0.0;
                vy = 1.0;
                vz = 0.0;
            }

            const double dx = bx - ax;
            const double dy = by - ay;
            const double dz = bz - az;
            const double xp = dx * ux + dy * uy + dz * uz;
            const double yp = dx * vx + dy * vy + dz * vz;
            const double phi = atan2(yp, xp);
            const double angle = -2.0 * phi;
            const double rr = cos(angle);
            const double ri = sin(angle);

            for (int a = 0; a < n_tomo; ++a) {
                const long long base_a = (long long)a * n_pix;
                const cuDoubleComplex ga = g_pix[base_a + ia];
                const double wa = w_pix[base_a + ia];
                for (int b = a; b < n_tomo; ++b) {
                    const long long base_b = (long long)b * n_pix;
                    const cuDoubleComplex gb = g_pix[base_b + ib];
                    const double wb = w_pix[base_b + ib];

                    const double gax = wa * ga.x;
                    const double gay = wa * ga.y;
                    const double gbx = wb * gb.x;
                    const double gby = wb * gb.y;
                    const double prod_re = gax * gbx + gay * gby;
                    const double prod_im = gay * gbx - gax * gby;
                    const double val = prod_re * rr - prod_im * ri;
                    const long long out_idx = ((long long)a * n_tomo + b) * n_bins + bin;
                    atomicAdd(out + out_idx, val);
                }
            }
        }
        """,
        "tomo_leaf_kernel",
    )

except ImportError:
    cp = None
    _leaf_sum_kernel = None
    _node_agg_kernel = None
    _tomo_interaction_kernel = None
    _tomo_leaf_kernel = None


def _get_levels(parents):
    n_nodes = len(parents)
    depths = np.zeros(n_nodes, dtype=np.int32)
    for i in range(1, n_nodes):
        p = parents[i]
        depths[i] = depths[p] + 1
    max_depth = int(np.max(depths))
    levels = []
    for d in range(max_depth, -1, -1):
        levels.append(np.where(depths == d)[0])
    return levels


def _normalize_maps(maps):
    maps_np = np.asarray(maps, dtype=np.float64)

    if maps_np.ndim == 3:
        if maps_np.shape[1] != 2:
            raise ValueError("maps with ndim=3 must have shape (n_tomo_bins, 2, n_pixels)")
        norm = maps_np
    elif maps_np.ndim == 2:
        if maps_np.shape[0] == 2:
            norm = maps_np[np.newaxis, :, :]
        elif maps_np.shape[1] == 2:
            norm = maps_np.T[np.newaxis, :, :]
        else:
            raise ValueError("maps with ndim=2 must have shape (2, n_pixels) or (n_pixels, 2)")
    else:
        raise ValueError("maps must have shape (n_tomo_bins, 2, n_pixels), (2, n_pixels), or (n_pixels, 2)")

    if norm.shape[2] <= 0:
        raise ValueError("maps must contain at least one pixel")
    return np.ascontiguousarray(norm, dtype=np.float64)


def _normalize_weights(w_map, n_tomo, n_pix):
    w_np = np.asarray(w_map, dtype=np.float64)

    if w_np.ndim == 1:
        if w_np.shape[0] != n_pix:
            raise ValueError("w_map with ndim=1 must have length n_pixels")
        out = np.broadcast_to(w_np[np.newaxis, :], (n_tomo, n_pix))
    elif w_np.ndim == 2:
        if w_np.shape == (n_tomo, n_pix):
            out = w_np
        elif w_np.shape == (1, n_pix):
            out = np.broadcast_to(w_np, (n_tomo, n_pix))
        else:
            raise ValueError("w_map with ndim=2 must have shape (n_tomo_bins, n_pixels)")
    else:
        raise ValueError("w_map must have shape (n_pixels,) or (n_tomo_bins, n_pixels)")

    return np.ascontiguousarray(out, dtype=np.float64)


def _validate_n_bins(n_bins):
    if not isinstance(n_bins, (int, np.integer)):
        raise ValueError("n_bins must be a positive integer")
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    return n_bins


def _compute_rotation_from_tree_nodes(tree, idx_i, idx_j):
    x = np.asarray(tree["x"], dtype=np.float64)
    y = np.asarray(tree["y"], dtype=np.float64)
    z = np.asarray(tree["z"], dtype=np.float64)

    ax = x[idx_i]
    ay = y[idx_i]
    az = z[idx_i]
    bx = x[idx_j]
    by = y[idx_j]
    bz = z[idx_j]

    r = np.sqrt(ax * ax + ay * ay)
    ux = np.ones_like(r)
    uy = np.zeros_like(r)
    uz = np.zeros_like(r)
    vx = np.zeros_like(r)
    vy = np.ones_like(r)
    vz = np.zeros_like(r)

    mask = r > 1e-9
    if np.any(mask):
        cos_ra = ax[mask] / r[mask]
        sin_ra = ay[mask] / r[mask]
        sin_dec = az[mask]
        ux[mask] = -sin_ra
        uy[mask] = cos_ra
        vx[mask] = -sin_dec * cos_ra
        vy[mask] = -sin_dec * sin_ra
        vz[mask] = r[mask]

    dx = bx - ax
    dy = by - ay
    dz = bz - az
    xp = dx * ux + dy * uy + dz * uz
    yp = dx * vx + dy * vy + dz * vz

    angle = -2.0 * np.arctan2(yp, xp)
    return np.cos(angle), np.sin(angle)


def _normalize_interactions(interaction_list, tree, n_bins):
    if interaction_list is None:
        n0_int32 = np.empty(0, dtype=np.int32)
        n0_float64 = np.empty(0, dtype=np.float64)
        return n0_int32, n0_int32, n0_float64, n0_float64, n0_int32

    inter_np = np.asarray(interaction_list)
    if inter_np.size == 0:
        n0_int32 = np.empty(0, dtype=np.int32)
        n0_float64 = np.empty(0, dtype=np.float64)
        return n0_int32, n0_int32, n0_float64, n0_float64, n0_int32

    if inter_np.ndim != 2 or inter_np.shape[1] not in (2, 4, 5):
        raise ValueError("interaction_list must have shape (N, 2), (N, 4), or (N, 5)")

    i_idx = np.ascontiguousarray(inter_np[:, 0].astype(np.int32))
    j_idx = np.ascontiguousarray(inter_np[:, 1].astype(np.int32))

    n_nodes = len(tree["x"])
    if i_idx.size > 0:
        if int(i_idx.min()) < 0 or int(i_idx.max()) >= n_nodes:
            raise ValueError("interaction_list i indices are out of bounds for tree nodes")
        if int(j_idx.min()) < 0 or int(j_idx.max()) >= n_nodes:
            raise ValueError("interaction_list j indices are out of bounds for tree nodes")

    if inter_np.shape[1] == 5:
        rot_re = np.ascontiguousarray(inter_np[:, 2].astype(np.float64))
        rot_im = np.ascontiguousarray(inter_np[:, 3].astype(np.float64))
        bins = np.ascontiguousarray(inter_np[:, 4].astype(np.int32))
    elif inter_np.shape[1] == 4:
        warnings.warn(
            "interaction_list has shape (N, 4): assigning all interactions to angular bin 0",
            stacklevel=2,
        )
        rot_re = np.ascontiguousarray(inter_np[:, 2].astype(np.float64))
        rot_im = np.ascontiguousarray(inter_np[:, 3].astype(np.float64))
        bins = np.zeros(inter_np.shape[0], dtype=np.int32)
    else:
        warnings.warn(
            "interaction_list has shape (N, 2): computing rotations from tree nodes and assigning angular bin 0",
            stacklevel=2,
        )
        rot_re, rot_im = _compute_rotation_from_tree_nodes(tree, i_idx, j_idx)
        rot_re = np.ascontiguousarray(rot_re, dtype=np.float64)
        rot_im = np.ascontiguousarray(rot_im, dtype=np.float64)
        bins = np.zeros(inter_np.shape[0], dtype=np.int32)

    if bins.size > 0 and (int(bins.min()) < 0 or int(bins.max()) >= n_bins):
        raise ValueError("interaction_list bin indices are out of bounds for n_bins")

    return i_idx, j_idx, rot_re, rot_im, bins


def _normalize_leaf_inputs(leaf_pairs, leaf_bins, n_bins, n_pix):
    if leaf_pairs is None:
        pairs = np.empty((0, 2), dtype=np.int64)
    else:
        pairs = np.asarray(leaf_pairs)
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=np.int64)
        elif pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("leaf_pairs must be a 2D array with shape (N, 2)")
        else:
            pairs = np.ascontiguousarray(pairs.astype(np.int64))

    if pairs.shape[0] > 0 and leaf_bins is None:
        raise ValueError("leaf_bins is required when leaf_pairs is non-empty")

    if leaf_bins is None:
        bins = np.empty(0, dtype=np.int32)
    else:
        bins = np.asarray(leaf_bins)
        if bins.size == 0:
            bins = np.empty(0, dtype=np.int32)
        elif bins.ndim != 1:
            raise ValueError("leaf_bins must be a 1D array")
        bins = np.ascontiguousarray(bins.astype(np.int32))

    if bins.shape[0] != pairs.shape[0]:
        raise ValueError("leaf_bins length must match the number of leaf_pairs")

    if pairs.shape[0] > 0:
        if int(pairs.min()) < 0 or int(pairs.max()) >= n_pix:
            raise ValueError("leaf_pairs indices are out of bounds for maps")
        if int(bins.min()) < 0 or int(bins.max()) >= n_bins:
            raise ValueError("leaf_bins indices are out of bounds for n_bins")

    return pairs, bins


def _prepare_particle_coords(particle_coords, ra, dec, n_pix, order):
    if particle_coords is not None:
        if len(particle_coords) != 3:
            raise ValueError("particle_coords must be a 3-tuple of arrays")
        px = np.asarray(particle_coords[0], dtype=np.float64)
        py = np.asarray(particle_coords[1], dtype=np.float64)
        pz = np.asarray(particle_coords[2], dtype=np.float64)
        if px.shape[0] != n_pix or py.shape[0] != n_pix or pz.shape[0] != n_pix:
            raise ValueError("particle_coords arrays must each have length n_pixels")
    elif ra is not None and dec is not None:
        ra = np.asarray(ra, dtype=np.float64)
        dec = np.asarray(dec, dtype=np.float64)
        if ra.shape[0] != n_pix or dec.shape[0] != n_pix:
            raise ValueError("ra/dec arrays must each have length n_pixels")
        px = np.cos(dec) * np.cos(ra)
        py = np.cos(dec) * np.sin(ra)
        pz = np.sin(dec)
    else:
        raise ValueError("Must provide particle_coords or ra/dec when leaf_pairs is non-empty")

    return (
        np.ascontiguousarray(px[order], dtype=np.float64),
        np.ascontiguousarray(py[order], dtype=np.float64),
        np.ascontiguousarray(pz[order], dtype=np.float64),
    )


def _fill_tree_gpu(shear_map, w_map, tree, levels):
    if cp is None:
        raise RuntimeError("CuPy not installed.")
    if shear_map.shape[0] != 2:
        raise ValueError("shear_map must have shape (2, n_pixels)")

    n_nodes = len(tree["x"])
    node_shear = cp.zeros(n_nodes, dtype=cp.complex128)
    node_weight = cp.zeros(n_nodes, dtype=cp.float64)

    t_child_left = cp.asarray(tree["child_left"])
    t_child_right = cp.asarray(tree["child_right"])
    t_node_start = cp.asarray(tree["node_start"])
    t_node_end = cp.asarray(tree["node_end"])
    t_idx_array = cp.asarray(tree["idx_array"])

    d_shear = cp.asarray(shear_map)
    d_w = cp.asarray(w_map)
    g_pix = d_shear[0] + 1j * d_shear[1]

    is_leaf = t_child_left == -1
    leaf_indices = cp.where(is_leaf)[0]
    if leaf_indices.size > 0:
        _leaf_sum_kernel(
            t_node_start[leaf_indices],
            t_node_end[leaf_indices],
            t_idx_array,
            g_pix,
            d_w,
            node_shear[leaf_indices],
            node_weight[leaf_indices],
        )

    for level_nodes in levels:
        d_indices = cp.asarray(level_nodes)
        node_children = t_child_left[d_indices]
        internal_indices = d_indices[node_children != -1]
        if internal_indices.size > 0:
            _node_agg_kernel(
                t_child_left[internal_indices],
                t_child_right[internal_indices],
                node_shear,
                node_weight,
                node_shear[internal_indices],
                node_weight[internal_indices],
            )

    return node_shear, node_weight


def execute_tree_correlation(
    maps,
    w_map,
    tree,
    interaction_list,
    leaf_pairs,
    n_bins,
    leaf_bins=None,
    particle_coords=None,
    ra=None,
    dec=None,
):
    maps_np = _normalize_maps(maps)
    n_tomo, _, n_pix = maps_np.shape
    weights_np = _normalize_weights(w_map, n_tomo, n_pix)
    n_bins = _validate_n_bins(n_bins)

    inter_i, inter_j, inter_rot_re, inter_rot_im, inter_bins = _normalize_interactions(
        interaction_list, tree, n_bins
    )
    leaf_pairs_np, leaf_bins_np = _normalize_leaf_inputs(leaf_pairs, leaf_bins, n_bins, n_pix)

    order = np.asarray(tree["idx_array"], dtype=np.int64)
    if order.shape[0] != n_pix:
        raise ValueError("tree['idx_array'] length must match n_pixels")

    ordered_coords = None
    if leaf_pairs_np.shape[0] > 0:
        ordered_coords = _prepare_particle_coords(particle_coords, ra, dec, n_pix, order)

    if cp is None:
        warnings.warn("CuPy not found. Returning zeros.", stacklevel=2)
        return np.zeros((n_tomo, n_tomo, n_bins), dtype=np.float64)

    parents = tree["parents"]
    levels = _get_levels(parents)
    n_nodes = len(tree["x"])
    node_shear_all = cp.empty((n_tomo, n_nodes), dtype=cp.complex128)

    for t in range(n_tomo):
        node_shear_t, _ = _fill_tree_gpu(maps_np[t], weights_np[t], tree, levels)
        node_shear_all[t] = node_shear_t

    out = cp.zeros((n_tomo, n_tomo, n_bins), dtype=cp.float64)

    if inter_i.size > 0:
        d_i = cp.asarray(inter_i)
        d_j = cp.asarray(inter_j)
        d_rot_re = cp.asarray(inter_rot_re)
        d_rot_im = cp.asarray(inter_rot_im)
        d_bins = cp.asarray(inter_bins)
        threads = 256
        blocks = (inter_i.size + threads - 1) // threads
        _tomo_interaction_kernel(
            (blocks,),
            (threads,),
            (
                d_i,
                d_j,
                d_rot_re,
                d_rot_im,
                d_bins,
                node_shear_all,
                np.int32(n_tomo),
                np.int32(n_nodes),
                np.int32(n_bins),
                np.int64(inter_i.size),
                out,
            ),
        )

    if leaf_pairs_np.shape[0] > 0:
        ordered_maps = np.ascontiguousarray(maps_np[:, :, order], dtype=np.float64)
        ordered_weights = np.ascontiguousarray(weights_np[:, order], dtype=np.float64)

        d_ordered_maps = cp.asarray(ordered_maps)
        d_g_pix = d_ordered_maps[:, 0, :] + 1j * d_ordered_maps[:, 1, :]
        d_w_pix = cp.asarray(ordered_weights)

        leaf_i = np.ascontiguousarray(leaf_pairs_np[:, 0], dtype=np.int64)
        leaf_j = np.ascontiguousarray(leaf_pairs_np[:, 1], dtype=np.int64)
        d_leaf_i = cp.asarray(leaf_i)
        d_leaf_j = cp.asarray(leaf_j)
        d_leaf_bins = cp.asarray(leaf_bins_np)

        px, py, pz = ordered_coords
        d_px = cp.asarray(px)
        d_py = cp.asarray(py)
        d_pz = cp.asarray(pz)

        threads = 256
        blocks = (leaf_i.size + threads - 1) // threads
        _tomo_leaf_kernel(
            (blocks,),
            (threads,),
            (
                d_leaf_i,
                d_leaf_j,
                d_leaf_bins,
                d_g_pix,
                d_w_pix,
                d_px,
                d_py,
                d_pz,
                np.int32(n_tomo),
                np.int32(n_pix),
                np.int32(n_bins),
                np.int64(leaf_i.size),
                out,
            ),
        )

    return out.get()
