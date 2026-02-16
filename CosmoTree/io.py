import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - exercised via runtime checks
    h5py = None


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

_META_KEYS = ("nbins", "min_sep", "max_sep")
_NON_ATTR_CONFIG_KEYS = {"leaf_bins"}


def _require_h5py():
    if h5py is None:
        raise ImportError("h5py is required for geometry I/O. Install h5py to use save_geometry/load_geometry.")


def _to_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_config(config):
    if config is None:
        raise ValueError("config must be provided and contain nbins, min_sep, and max_sep")
    for key in _META_KEYS:
        if key not in config:
            raise ValueError(f"config must include '{key}'")

    nbins = int(config["nbins"])
    min_sep = float(config["min_sep"])
    max_sep = float(config["max_sep"])
    if nbins <= 0:
        raise ValueError("config['nbins'] must be a positive integer")
    if min_sep <= 0.0:
        raise ValueError("config['min_sep'] must be > 0")
    if max_sep <= min_sep:
        raise ValueError("config['max_sep'] must be larger than config['min_sep']")
    return nbins, min_sep, max_sep


def _normalize_tree(tree):
    if not isinstance(tree, dict):
        raise ValueError("tree must be a dictionary")

    missing = [k for k in _TREE_DTYPES if k not in tree]
    if missing:
        raise ValueError(f"tree is missing required keys: {missing}")

    out = {}
    n_nodes = None
    for key, dtype in _TREE_DTYPES.items():
        arr = np.asarray(tree[key], dtype=dtype)
        if arr.ndim != 1:
            raise ValueError(f"tree['{key}'] must be a 1D array")
        arr = np.ascontiguousarray(arr)
        out[key] = arr
        if key != "idx_array":
            if n_nodes is None:
                n_nodes = arr.shape[0]
            elif arr.shape[0] != n_nodes:
                raise ValueError("all node-level tree arrays must have the same length")
    return out


def _normalize_interactions(interaction_list, nbins):
    inter = np.asarray(interaction_list, dtype=np.float64)
    if inter.ndim != 2 or inter.shape[1] != 5:
        raise ValueError("interaction_list must have shape (N, 5) including angular bin in column 4")
    inter = np.ascontiguousarray(inter)

    inter_bins = np.ascontiguousarray(inter[:, 4].astype(np.int32))
    if inter_bins.size > 0:
        if int(inter_bins.min()) < 0 or int(inter_bins.max()) >= nbins:
            raise ValueError("interaction_list bin indices are out of bounds for config['nbins']")
    inter[:, 4] = inter_bins.astype(np.float64)
    return inter, inter_bins


def _normalize_leaf_pairs(leaf_pairs):
    pairs = np.asarray(leaf_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("leaf_pairs must have shape (N, 2)")
    return np.ascontiguousarray(pairs)


def _normalize_leaf_bins(config, n_leaf, nbins):
    leaf_bins_val = config.get("leaf_bins")
    if n_leaf > 0 and leaf_bins_val is None:
        raise ValueError("config must include 'leaf_bins' when leaf_pairs is non-empty")
    if leaf_bins_val is None:
        return np.empty(0, dtype=np.int32)

    leaf_bins = np.asarray(leaf_bins_val, dtype=np.int32)
    if leaf_bins.ndim != 1:
        raise ValueError("config['leaf_bins'] must be a 1D array")
    if leaf_bins.shape[0] != n_leaf:
        raise ValueError("config['leaf_bins'] length must match number of leaf_pairs")
    if leaf_bins.size > 0:
        if int(leaf_bins.min()) < 0 or int(leaf_bins.max()) >= nbins:
            raise ValueError("config['leaf_bins'] values are out of bounds for config['nbins']")
    return np.ascontiguousarray(leaf_bins)


def save_geometry(filename, tree, interaction_list, leaf_pairs, config):
    """
    Save tree traversal geometry to an HDF5 file.

    The file stores:
    - `tree/*` datasets for all tree arrays produced by `build_tree`.
    - `traversal/interaction_list` and `traversal/interaction_bins`.
    - `traversal/leaf_pairs` and `traversal/leaf_bins` (from `config['leaf_bins']`, if provided).
    - metadata attrs: `nbins`, `min_sep`, `max_sep`.
    """
    _require_h5py()

    nbins, min_sep, max_sep = _validate_config(config)
    tree_norm = _normalize_tree(tree)
    inter, inter_bins = _normalize_interactions(interaction_list, nbins)
    leaf_pairs_norm = _normalize_leaf_pairs(leaf_pairs)
    leaf_bins = _normalize_leaf_bins(config, leaf_pairs_norm.shape[0], nbins)

    with h5py.File(filename, "w") as f:
        f.attrs["format"] = "cosmotree_geometry"
        f.attrs["version"] = np.int32(1)
        f.attrs["nbins"] = np.int64(nbins)
        f.attrs["min_sep"] = np.float64(min_sep)
        f.attrs["max_sep"] = np.float64(max_sep)

        for key, value in config.items():
            if key in _META_KEYS or key in _NON_ATTR_CONFIG_KEYS:
                continue
            scalar = _to_scalar(value)
            if np.isscalar(scalar) or isinstance(scalar, (str, bytes)):
                f.attrs[key] = scalar

        grp_tree = f.create_group("tree")
        for key, arr in tree_norm.items():
            grp_tree.create_dataset(key, data=arr, dtype=_TREE_DTYPES[key])

        grp_trav = f.create_group("traversal")
        grp_trav.create_dataset("interaction_list", data=inter, dtype=np.float64)
        grp_trav.create_dataset("interaction_bins", data=inter_bins, dtype=np.int32)
        grp_trav.create_dataset("leaf_pairs", data=leaf_pairs_norm, dtype=np.int64)
        grp_trav.create_dataset("leaf_bins", data=leaf_bins, dtype=np.int32)


def load_geometry(filename):
    """
    Load geometry from an HDF5 file written by `save_geometry`.

    Returns:
        tree, interaction_list, interaction_bins, leaf_pairs, leaf_bins, config
    """
    _require_h5py()

    with h5py.File(filename, "r") as f:
        for key in _META_KEYS:
            if key not in f.attrs:
                raise ValueError(f"missing required metadata attribute '{key}'")

        config = {
            "nbins": int(f.attrs["nbins"]),
            "min_sep": float(f.attrs["min_sep"]),
            "max_sep": float(f.attrs["max_sep"]),
        }
        for key, value in f.attrs.items():
            if key in {"format", "version", *_META_KEYS}:
                continue
            config[key] = _to_scalar(value)

        if "tree" not in f:
            raise ValueError("missing 'tree' group in geometry file")
        grp_tree = f["tree"]
        tree = {}
        for key, dtype in _TREE_DTYPES.items():
            if key not in grp_tree:
                raise ValueError(f"missing tree dataset '{key}'")
            arr = np.asarray(grp_tree[key][()], dtype=dtype)
            if arr.ndim != 1:
                raise ValueError(f"tree dataset '{key}' must be 1D")
            tree[key] = np.ascontiguousarray(arr)

        if "traversal" not in f:
            raise ValueError("missing 'traversal' group in geometry file")
        grp_trav = f["traversal"]

        if "interaction_list" not in grp_trav:
            raise ValueError("missing traversal dataset 'interaction_list'")
        interaction_list = np.asarray(grp_trav["interaction_list"][()], dtype=np.float64)
        if interaction_list.ndim != 2 or interaction_list.shape[1] != 5:
            raise ValueError("interaction_list dataset must have shape (N, 5)")
        interaction_list = np.ascontiguousarray(interaction_list)

        if "interaction_bins" in grp_trav:
            interaction_bins = np.asarray(grp_trav["interaction_bins"][()], dtype=np.int32)
        else:
            interaction_bins = interaction_list[:, 4].astype(np.int32)
        if interaction_bins.ndim != 1 or interaction_bins.shape[0] != interaction_list.shape[0]:
            raise ValueError("interaction_bins must be 1D and aligned with interaction_list")
        interaction_bins = np.ascontiguousarray(interaction_bins)
        interaction_list[:, 4] = interaction_bins.astype(np.float64)

        if "leaf_pairs" not in grp_trav:
            raise ValueError("missing traversal dataset 'leaf_pairs'")
        leaf_pairs = np.asarray(grp_trav["leaf_pairs"][()], dtype=np.int64)
        if leaf_pairs.ndim != 2 or leaf_pairs.shape[1] != 2:
            raise ValueError("leaf_pairs dataset must have shape (N, 2)")
        leaf_pairs = np.ascontiguousarray(leaf_pairs)

        if "leaf_bins" in grp_trav:
            leaf_bins = np.asarray(grp_trav["leaf_bins"][()], dtype=np.int32)
        else:
            leaf_bins = np.empty(0, dtype=np.int32)
        if leaf_bins.ndim != 1 or leaf_bins.shape[0] != leaf_pairs.shape[0]:
            raise ValueError("leaf_bins must be 1D and aligned with leaf_pairs")
        leaf_bins = np.ascontiguousarray(leaf_bins)
        config["leaf_bins"] = leaf_bins

    return tree, interaction_list, interaction_bins, leaf_pairs, leaf_bins, config
