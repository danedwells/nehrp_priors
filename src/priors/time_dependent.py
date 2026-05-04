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
    cheap (ideally) relative to the one-time model calibration that happens at
    construction time.

    For a computationally expensive update, some alteration to this framework may be needed.
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
        max_lookback_days: float | None = None,
    ):
        self.theta              = theta
        self.mc                 = mc
        self.grid_lats_masked   = np.asarray(grid_lats_masked, dtype=float)
        self.grid_lons_masked   = np.asarray(grid_lons_masked, dtype=float)
        self.catalog            = catalog.copy()
        self.bounds             = bounds
        self.out_of_bounds_fill = float(out_of_bounds_fill)
        self.metadata_base      = metadata_base or {}
        self.max_lookback_days  = max_lookback_days

        # Precompute time-independent spatial weights for the historical catalog.
        # _w_hist[j, i] = aftershock_number(m_i) * space_decay(r_sq[j,i], m_i)
        # Shape: (n_grid, n_hist), float32.  Haversine runs once here instead
        # of once per update() call.
        self._hist_times = self.catalog['time'].copy().reset_index(drop=True)
        self._w_hist     = self._precompute_spatial_weights(self.catalog)

        # Appended events (via append_events) are tracked separately so their
        # spatial weights can be built incrementally — one haversine call per
        # new event, not a full rebuild each time.
        self._appended      = pd.DataFrame(
            columns=['time', 'latitude', 'longitude', 'magnitude'])
        self._w_appended    = np.empty((len(self.grid_lats_masked), 0), dtype=np.float32)
        self._n_app_built   = 0   # events in _appended already reflected in _w_appended

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
        max_lookback_days: float | None = None,
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
        from numpy import array  # needed for eval() of shape_coords repr
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
            theta              = theta,
            mc                 = mc,
            grid_lats_masked   = lats_flat[mask],
            grid_lons_masked   = lons_flat[mask],
            catalog            = catalog_df,
            bounds             = bounds,
            out_of_bounds_fill = out_of_bounds_fill,
            metadata_base      = metadata_base,
            max_lookback_days  = max_lookback_days,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precompute_spatial_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Return (n_grid, n_events) float32 spatial weight matrix for df."""
        from etas.intensity import _compute_spatial_weights
        if df.empty:
            return np.empty((len(self.grid_lats_masked), 0), dtype=np.float32)
        # Explicit float64 cast guards against object-dtype columns that arise
        # when concatenating into the initially-empty _appended DataFrame.
        return _compute_spatial_weights(
            self.grid_lats_masked, self.grid_lons_masked,
            df['latitude'].values.astype(np.float64),
            df['longitude'].values.astype(np.float64),
            df['magnitude'].values.astype(np.float64),
            self.theta, self.mc,
        )

    # ------------------------------------------------------------------
    # Real-time interface
    # ------------------------------------------------------------------

    def append_events(self, new_events: pd.DataFrame) -> None:
        """
        Append new events to the rolling catalog.

        Duplicates are dropped by matching on (time, latitude, longitude)
        after concatenation, so it is safe to call with overlapping batches.
        Events that are genuinely new (not already in the historical catalog)
        are also stored in _appended so their spatial weights can be built
        incrementally — avoiding a full haversine recompute over _w_hist.
        """
        hist_idx = self.catalog.set_index(['time', 'latitude', 'longitude']).index
        new_idx  = new_events.set_index(['time', 'latitude', 'longitude']).index
        truly_new = new_events[~new_idx.isin(hist_idx)].copy()

        self.catalog = (
            pd.concat([self.catalog, new_events], ignore_index=True)
            .drop_duplicates(subset=['time', 'latitude', 'longitude'])
            .sort_values('time')
            .reset_index(drop=True)
        )

        if not truly_new.empty:
            self._appended = (
                pd.concat([self._appended, truly_new], ignore_index=True)
                .drop_duplicates(subset=['time', 'latitude', 'longitude'])
            )
            # _n_app_built is intentionally not updated here; update() will
            # build weights only for the new tail on the next call.

    def update(
        self,
        forecast_time: pd.Timestamp,
        cache_path: str | None = None,
    ) -> SeismicPrior:
        """
        Evaluate the ETAS intensity and return a fresh SeismicPrior.

        Uses precomputed spatial weights (_w_hist) so haversine is not
        recomputed on each call on the entire catalag (only appended events).  
        Only the time-decay vector is evaluated
        per update, reducing cost from O(n_grid × n_catalog) haversine
        operations to O(n_catalog) scalar operations plus a matrix–vector
        product.

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
        import datetime
        from etas.intensity import _compute_time_decay

        mu = 10 ** self.theta['log10_mu']

        n_app = len(self._appended)
        if n_app > self._n_app_built:
            new_chunk = self._precompute_spatial_weights(
                self._appended.iloc[self._n_app_built:])
            self._w_appended  = np.hstack([self._w_appended, new_chunk])
            self._n_app_built = n_app

        # --- Historical contribution (spatial weights precomputed at init) ---
        # Mask events: only before forecast time
        hist_mask = self._hist_times < forecast_time
        if self.max_lookback_days is not None:
            cutoff    = forecast_time - datetime.timedelta(days=self.max_lookback_days)
            # Add to the mask - must be after the time cutoff (2 years?)
            hist_mask = hist_mask & (self._hist_times >= cutoff)

        # IF any events in historical catalog within mask time
        if hist_mask.any():
            td          = _compute_time_decay(
                              self._hist_times[hist_mask], forecast_time, self.theta)
            lambda_vals = (mu + (self._w_hist[:, hist_mask.values] * td).sum(axis=1))
        # Else background seismicity rate (mu)
        else:
            lambda_vals = np.full(len(self.grid_lats_masked), mu)

        # --- Appended-events contribution (weights built once per append) ---
        # This repeats the same logic as above, but for the appended events.
        if self._w_appended is not None and not self._appended.empty:
            app_mask = self._appended['time'] < forecast_time
            if self.max_lookback_days is not None:
                cutoff   = forecast_time - datetime.timedelta(days=self.max_lookback_days)
                app_mask = app_mask & (self._appended['time'] >= cutoff)
            if app_mask.any():
                td_app      = _compute_time_decay(
                                  self._appended.loc[app_mask, 'time'],
                                  forecast_time, self.theta)
                lambda_vals = lambda_vals + (
                    self._w_appended[:, app_mask.values] * td_app).sum(axis=1)

        prior = SeismicPrior.from_etas(
            lats               = self.grid_lats_masked,
            lons               = self.grid_lons_masked,
            lambda_grid        = lambda_vals.astype(float),
            forecast_time      = forecast_time,
            metadata           = dict(self.metadata_base),
            bounds             = self.bounds,
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
