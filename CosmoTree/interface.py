import numpy as np

from .executor import execute_tree_correlation
from .io import load_geometry, save_geometry
from .tree import build_tree
from .traversal import traverse


class CosmoTree:
    def __init__(
        self,
        min_sep,
        max_sep,
        nbins,
        bin_slop=0.1,
        angle_slop=1.0,
        metric="Euclidean",
        min_size=0.0,
        max_depth=-1,
        max_stack=1000000,
        max_inter=10000000,
        max_leaf=10000000,
        growth_factor=2.0,
        max_retries=5,
        dtype=np.float32,
    ):
        self.min_sep = float(min_sep)
        self.max_sep = float(max_sep)
        self.nbins = int(nbins)
        self.bin_slop = float(bin_slop)
        self.angle_slop = float(angle_slop)
        self.metric = metric
        self.min_size = float(min_size)
        self.max_depth = int(max_depth)
        self.max_stack = int(max_stack)
        self.max_inter = int(max_inter)
        self.max_leaf = int(max_leaf)
        self.growth_factor = float(growth_factor)
        self.max_retries = int(max_retries)
        self.dtype = np.dtype(dtype)

        self._geometry = None
        self._ra = None
        self._dec = None
        self._particle_coords = None

    @property
    def is_preprocessed(self):
        return self._geometry is not None

    def _set_geometry(self, tree, interaction_list, interaction_bins, leaf_pairs, leaf_bins, ra=None, dec=None):
        self._geometry = {
            "tree": tree,
            "interaction_list": interaction_list,
            "interaction_bins": interaction_bins,
            "leaf_pairs": leaf_pairs,
            "leaf_bins": leaf_bins,
            "ra": None if ra is None else np.ascontiguousarray(np.asarray(ra, dtype=np.float64)),
            "dec": None if dec is None else np.ascontiguousarray(np.asarray(dec, dtype=np.float64)),
        }

    def _set_coords_cache(self, ra=None, dec=None, particle_coords=None):
        self._ra = None if ra is None else np.ascontiguousarray(np.asarray(ra, dtype=np.float64))
        self._dec = None if dec is None else np.ascontiguousarray(np.asarray(dec, dtype=np.float64))
        self._particle_coords = particle_coords

    def _parse_geometry_input(self, geometry_data):
        if isinstance(geometry_data, dict):
            if all(k in geometry_data for k in ("tree", "interaction_list", "leaf_pairs")):
                tree = geometry_data["tree"]
                interaction_list = np.asarray(geometry_data["interaction_list"])
                interaction_bins = geometry_data.get("interaction_bins")
                if interaction_bins is None:
                    interaction_bins = interaction_list[:, 4].astype(np.int32) if interaction_list.size else np.empty(0, dtype=np.int32)
                leaf_pairs = np.asarray(geometry_data["leaf_pairs"])
                leaf_bins = geometry_data.get("leaf_bins")
                if leaf_bins is None:
                    if leaf_pairs.shape[0] > 0:
                        raise ValueError("geometry_data must include 'leaf_bins' when 'leaf_pairs' is non-empty")
                    leaf_bins = np.empty(0, dtype=np.int32)
                return tree, interaction_list, np.asarray(interaction_bins), leaf_pairs, np.asarray(leaf_bins), geometry_data.get("ra"), geometry_data.get("dec"), geometry_data.get("particle_coords")

            if "ra" in geometry_data and "dec" in geometry_data:
                return geometry_data["ra"], geometry_data["dec"], geometry_data.get("w")

            raise ValueError(
                "geometry_data dict must contain either {'ra','dec'} or "
                "{'tree','interaction_list','leaf_pairs','leaf_bins'}"
            )

        if isinstance(geometry_data, (tuple, list)):
            if len(geometry_data) == 2:
                return geometry_data[0], geometry_data[1], None
            if len(geometry_data) == 3:
                return geometry_data[0], geometry_data[1], geometry_data[2]

        raise ValueError(
            "geometry_data must be a dict with coordinates/geometry, or a tuple/list (ra, dec[, w])"
        )

    def preprocess(self, geometry_data):
        parsed = self._parse_geometry_input(geometry_data)

        # Case 1: precomputed geometry dictionary
        if len(parsed) == 8:
            tree, interaction_list, interaction_bins, leaf_pairs, leaf_bins, ra, dec, particle_coords = parsed
            self._set_geometry(
                tree=tree,
                interaction_list=np.ascontiguousarray(interaction_list, dtype=np.float64),
                interaction_bins=np.ascontiguousarray(interaction_bins, dtype=np.int32),
                leaf_pairs=np.ascontiguousarray(leaf_pairs, dtype=np.int64),
                leaf_bins=np.ascontiguousarray(leaf_bins, dtype=np.int32),
                ra=ra,
                dec=dec,
            )
            self._set_coords_cache(ra=ra, dec=dec, particle_coords=particle_coords)
            return self

        # Case 2: raw coordinates -> build tree + traversal
        ra, dec, w = parsed
        ra = np.asarray(ra, dtype=np.float64)
        dec = np.asarray(dec, dtype=np.float64)
        if ra.shape != dec.shape:
            raise ValueError("ra and dec must have matching shapes")
        if ra.ndim != 1:
            raise ValueError("ra and dec must be 1D arrays")

        if w is not None:
            w = np.asarray(w, dtype=np.float64)
            if w.shape != ra.shape:
                raise ValueError("w must match ra/dec length")

        tree = build_tree(
            ra,
            dec,
            w=w,
            min_size=self.min_size,
            max_depth=self.max_depth,
            metric=self.metric,
        )
        interaction_list, interaction_bins, leaf_pairs, leaf_bins = traverse(
            tree,
            None,
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.nbins,
            metric=self.metric,
            bin_slop=self.bin_slop,
            angle_slop=self.angle_slop,
            max_stack=self.max_stack,
            max_inter=self.max_inter,
            max_leaf=self.max_leaf,
            growth_factor=self.growth_factor,
            max_retries=self.max_retries,
        )

        self._set_geometry(
            tree=tree,
            interaction_list=np.ascontiguousarray(interaction_list, dtype=np.float64),
            interaction_bins=np.ascontiguousarray(interaction_bins, dtype=np.int32),
            leaf_pairs=np.ascontiguousarray(leaf_pairs, dtype=np.int64),
            leaf_bins=np.ascontiguousarray(leaf_bins, dtype=np.int32),
            ra=ra,
            dec=dec,
        )
        self._set_coords_cache(ra=ra, dec=dec, particle_coords=None)
        return self

    def _infer_npix(self, shear_maps):
        maps = np.asarray(shear_maps)
        if maps.ndim == 3 and maps.shape[1] == 2:
            return maps.shape[2]
        if maps.ndim == 2:
            if maps.shape[0] == 2:
                return maps.shape[1]
            if maps.shape[1] == 2:
                return maps.shape[0]
        raise ValueError("shear_maps must have shape (n_tomo_bins, 2, n_pixels), (2, n_pixels), or (n_pixels, 2)")

    def process(
        self,
        shear_maps,
        w_map=None,
        particle_coords=None,
        ra=None,
        dec=None,
        dtype=None,
        device="auto",
    ):
        if self._geometry is None:
            raise RuntimeError("preprocess must be run before process")

        run_dtype = self.dtype if dtype is None else np.dtype(dtype)
        n_pix = self._infer_npix(shear_maps)

        if w_map is None:
            w_map = np.ones(n_pix, dtype=run_dtype)

        coords_arg = particle_coords if particle_coords is not None else self._particle_coords
        ra_arg = ra if ra is not None else self._ra
        dec_arg = dec if dec is not None else self._dec

        return execute_tree_correlation(
            maps=shear_maps,
            w_map=w_map,
            tree=self._geometry["tree"],
            interaction_list=self._geometry["interaction_list"],
            leaf_pairs=self._geometry["leaf_pairs"],
            n_bins=self.nbins,
            leaf_bins=self._geometry["leaf_bins"],
            particle_coords=coords_arg,
            ra=ra_arg,
            dec=dec_arg,
            dtype=run_dtype,
            device=device,
        )

    def save(self, filename):
        if self._geometry is None:
            raise RuntimeError("Nothing to save: run preprocess first")

        config = {
            "nbins": self.nbins,
            "min_sep": self.min_sep,
            "max_sep": self.max_sep,
            "metric": str(self.metric),
            "bin_slop": self.bin_slop,
            "angle_slop": self.angle_slop,
            "min_size": self.min_size,
            "max_depth": self.max_depth,
            "leaf_bins": self._geometry["leaf_bins"],
        }
        if self._ra is not None and self._dec is not None:
            config["ra"] = self._ra
            config["dec"] = self._dec
        save_geometry(
            filename=filename,
            tree=self._geometry["tree"],
            interaction_list=self._geometry["interaction_list"],
            leaf_pairs=self._geometry["leaf_pairs"],
            config=config,
        )
        return self

    def load(self, filename):
        tree, interaction_list, interaction_bins, leaf_pairs, leaf_bins, config = load_geometry(filename)
        self._set_geometry(
            tree=tree,
            interaction_list=interaction_list,
            interaction_bins=interaction_bins,
            leaf_pairs=leaf_pairs,
            leaf_bins=leaf_bins,
            ra=config.get("ra"),
            dec=config.get("dec"),
        )
        self._set_coords_cache(ra=config.get("ra"), dec=config.get("dec"), particle_coords=None)

        self.nbins = int(config["nbins"])
        self.min_sep = float(config["min_sep"])
        self.max_sep = float(config["max_sep"])
        if "metric" in config:
            self.metric = config["metric"]
        if "bin_slop" in config:
            self.bin_slop = float(config["bin_slop"])
        if "angle_slop" in config:
            self.angle_slop = float(config["angle_slop"])
        if "min_size" in config:
            self.min_size = float(config["min_size"])
        if "max_depth" in config:
            self.max_depth = int(config["max_depth"])
        return self
