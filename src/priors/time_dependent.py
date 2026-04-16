"""
time_dependent.py — Real-time spatial prior updaters for bEPIC.

Provides a base class and a concrete ETAS implementation for priors that
must be recomputed as new seismic events arrive.

Architecture
------------
Time-dependent priors have two distinct cost profiles:

  Inversion  (slow, minutes) — calibrates model parameters from a catalog.
                                Done once offline.  Results stored in JSON.

  Evaluation (fast, seconds) — evaluates lambda(x,y,t | H_t) on the
                                pre-built grid.  Called on every update().

EtasPriorUpdater holds the inversion output and a rolling catalog in memory
so that update() only runs the fast evaluation step.  The returned object
is always a SeismicPrior, the common currency consumed by bEPIC.

Usage
-----
    updater = EtasPriorUpdater.from_inversion_json(
        json_path='etas_2/output_data/parameters_0.json',
        bounds=(-129, -112, 30, 45),
        grid_spacing=0.1,
    )

    # In the event loop:
    prior = updater.update(pd.Timestamp.utcnow())
    # prime bEPIC
    params.prior = prior

    # When a new event arrives:
    updater.append_events(new_events_df)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from .prior_model import SeismicPrior


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TimeDependentPriorUpdater(ABC):
    """
    Interface for priors that need periodic recomputation.

    Subclasses must implement update() and append_events().  Both should be
    cheap relative to the one-time model calibration that happens at
    construction time.
    """

    @abstractmethod
    def update(self, forecast_time: pd.Timestamp, **kwargs) -> SeismicPrior:
        """
        Compute and return an updated SeismicPrior for forecast_time.

        Parameters
        ----------
        forecast_time : pd.Timestamp
        **kwargs : forwarded to the underlying model (e.g. cache_path).

        Returns
        -------
        SeismicPrior
        """

    @abstractmethod
    def append_events(self, new_events: pd.DataFrame) -> None:
        """
        Add new events to the rolling catalog used by update().

        Parameters
        ----------
        new_events : pd.DataFrame
            Must have columns: time (datetime64), latitude, longitude,
            magnitude.  Duplicates (matched on time+lat+lon) are dropped.
        """


# ---------------------------------------------------------------------------
# ETAS implementation
# ---------------------------------------------------------------------------

class EtasPriorUpdater(TimeDependentPriorUpdater):
    """
    Real-time SeismicPrior updater backed by a pre-inverted ETAS model.

    Holds theta (ETAS parameters), mc, the masked evaluation grid, and a
    rolling catalog in memory.  Each call to update() evaluates the ETAS
    conditional intensity — just NumPy math, no I/O unless cache_path is given.

    Parameters
    ----------
    theta : dict
        Inverted ETAS parameters from 'final_parameters' in parameters_0.json.
    mc : float
        Reference magnitude of completeness (m_ref).
    grid_lats_masked : array-like, shape (n_grid,)
        Latitudes of the pre-masked grid points (inside the ETAS polygon).
    grid_lons_masked : array-like, shape (n_grid,)
        Longitudes of the pre-masked grid points.
    catalog : pd.DataFrame
        Initial event catalog with columns: time, latitude, longitude,
        magnitude.
    bounds : tuple or None
        (lon_min, lon_max, lat_min, lat_max) used to expand the prior to
        cover the full bEPIC search region.  If None the prior covers only
        the ETAS polygon extent.
    out_of_bounds_fill : float
        Fill value (event rate) assigned to cells outside the ETAS polygon.
        A small positive value (e.g. 1e-4) keeps the posterior well-behaved
        for events near the polygon edge.
    metadata_base : dict or None
        Fixed metadata fields written to every prior's sidecar JSON
        (e.g. inversion file path, catalog path).
    """

    def __init__(
        self,
        theta: dict,
        mc: float,
        grid_lats_masked: np.ndarray,
        grid_lons_masked: np.ndarray,
        catalog: pd.DataFrame,
        bounds: tuple | None = None,
        out_of_bounds_fill: float = 1e-4,
        metadata_base: dict | None = None,
    ):
        self.theta              = theta
        self.mc                 = mc
        self.grid_lats_masked   = np.asarray(grid_lats_masked, dtype=float)
        self.grid_lons_masked   = np.asarray(grid_lons_masked, dtype=float)
        self.catalog            = catalog.copy()
        self.bounds             = bounds
        self.out_of_bounds_fill = float(out_of_bounds_fill)
        self.metadata_base      = metadata_base or {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_inversion_json(
        cls,
        json_path: str,
        catalog_df: pd.DataFrame | None = None,
        bounds: tuple | None = None,
        grid_spacing: float = 0.1,
        out_of_bounds_fill: float = 1e-4,
    ) -> EtasPriorUpdater:
        """
        Build an updater from an etas_2 inversion output JSON file.

        Reads theta, mc, and the polygon from the JSON; constructs the
        masked evaluation grid at the requested spacing; and optionally
        loads the catalog referenced in the JSON.

        Parameters
        ----------
        json_path : str
            Path to parameters_0.json produced by etas_2 inversion.
        catalog_df : pd.DataFrame or None
            Event catalog to use.  If None, the file referenced by
            'fn_catalog' inside the JSON is loaded automatically.
        bounds : tuple or None
            (lon_min, lon_max, lat_min, lat_max) for the output prior.
            Pass the same bounds you use for the other SeismicPriors
            (e.g. (-129, -112, 30, 45) for California).
        grid_spacing : float
            Grid resolution in degrees for intensity evaluation.
            0.1° matches the native resolution of GEAR1/NSHM/Helmstetter.
        out_of_bounds_fill : float
            Rate assigned outside the polygon.

        Returns
        -------
        EtasPriorUpdater
        """
        from shapely.geometry import Point, Polygon

        json_path = str(json_path)
        with open(json_path) as fh:
            config = json.load(fh)

        theta = config['final_parameters']
        mc    = float(config['m_ref'])

        # shape_coords is stored as a string repr of a numpy array
        # in (lat, lon) order — matching the convention in conditional_intensity.py
        shape_coords = np.array(eval(config['shape_coords']))
        polygon      = Polygon(shape_coords)

        if catalog_df is None:
            fn = config.get('fn_catalog', '')
            if not fn:
                raise ValueError(
                    "catalog_df is None and 'fn_catalog' is missing from the JSON. "
                    "Pass catalog_df explicitly."
                )
            catalog_df = pd.read_csv(
                fn,
                index_col=0,
                parse_dates=['time'],
                dtype={'url': str, 'alert': str},
            )

        # Build a rectangular grid over the polygon bounding box, then mask
        # to points inside the polygon.  The (lat, lon) Point convention
        # matches how the polygon was constructed by the inversion pipeline.
        min_lat, min_lon, max_lat, max_lon = polygon.bounds
        lats = np.arange(min_lat, max_lat + grid_spacing * 0.5, grid_spacing)
        lons = np.arange(min_lon, max_lon + grid_spacing * 0.5, grid_spacing)
        grid_lons_2d, grid_lats_2d = np.meshgrid(lons, lats)
        lats_flat = grid_lats_2d.ravel()
        lons_flat = grid_lons_2d.ravel()

        mask = np.array([
            polygon.contains(Point(la, lo))
            for la, lo in zip(lats_flat, lons_flat)
        ])

        metadata_base = {
            'inversion_json':   json_path,
            'catalog':          config.get('fn_catalog', ''),
            'timewindow_start': config.get('timewindow_start', ''),
            'timewindow_end':   config.get('timewindow_end', ''),
            'grid_spacing_deg': grid_spacing,
            'n_grid_points':    int(mask.sum()),
        }

        return cls(
            theta             = theta,
            mc                = mc,
            grid_lats_masked  = lats_flat[mask],
            grid_lons_masked  = lons_flat[mask],
            catalog           = catalog_df,
            bounds            = bounds,
            out_of_bounds_fill = out_of_bounds_fill,
            metadata_base     = metadata_base,
        )

    # ------------------------------------------------------------------
    # Real-time interface
    # ------------------------------------------------------------------

    def append_events(self, new_events: pd.DataFrame) -> None:
        """
        Append new events to the rolling catalog.

        Duplicates are dropped by matching on (time, latitude, longitude)
        after concatenation, so it is safe to call with overlapping batches.
        """
        self.catalog = (
            pd.concat([self.catalog, new_events], ignore_index=True)
            .drop_duplicates(subset=['time', 'latitude', 'longitude'])
            .sort_values('time')
            .reset_index(drop=True)
        )

    def update(
        self,
        forecast_time: pd.Timestamp,
        cache_path: str | None = None,
    ) -> SeismicPrior:
        """
        Evaluate the ETAS intensity and return a fresh SeismicPrior.

        This is the fast path — only the intensity evaluation runs.  No
        inversion, no file I/O unless cache_path is provided.

        Parameters
        ----------
        forecast_time : pd.Timestamp
            Time at which to evaluate lambda(x,y,t | H_t).
        cache_path : str or None
            If given, write the prior to this .tt3 path.  Useful for
            logging, replay, or handing off to the seismic_benchmark runner.

        Returns
        -------
        SeismicPrior
            Ready to assign to params.prior in bEPIC.
        """
        from etas.intensity import conditional_intensity_grid

        lambda_vals = conditional_intensity_grid(
            forecast_time,
            self.grid_lats_masked,
            self.grid_lons_masked,
            self.catalog,
            self.theta,
            self.mc,
        )

        prior = SeismicPrior.from_etas(
            lats             = self.grid_lats_masked,
            lons             = self.grid_lons_masked,
            lambda_grid      = lambda_vals,
            forecast_time    = forecast_time,
            metadata         = dict(self.metadata_base),
            bounds           = self.bounds,
            out_of_bounds_fill = self.out_of_bounds_fill,
        )

        if cache_path is not None:
            prior.to_tt3(cache_path)

        return prior

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_catalog_events(self) -> int:
        return len(self.catalog)

    @property
    def n_grid_points(self) -> int:
        return len(self.grid_lats_masked)

    def __repr__(self) -> str:
        return (
            f"EtasPriorUpdater("
            f"n_grid={self.n_grid_points}, "
            f"n_catalog={self.n_catalog_events}, "
            f"bounds={self.bounds})"
        )
