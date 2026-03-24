#!/usr/bin/env python3
"""
prior_model.py — SeismicPrior class for bEPIC.

Provides a unified interface for building, saving, and loading spatial seismic
prior distributions.  Five source models are supported:

  GEAR1              — global seismic hazard table (Bird & Kreemer 2015)
  NSHM               — USGS National Seismic Hazard Model gridded moment rates
  Helmstetter        — Helmstetter (2007) aftershock forecast via pyCSEP
  Smooth seismicity  — pre-built US/Canada seismicity grid (prior_seis_grid_US_Canada.tt3)
  ETAS               — ETAS conditional intensity lambda(x,y,t) evaluated on a grid

Priors can be serialized to / deserialized from the .tt3 binary grid format
that bEPIC uses.  Once loaded, a SeismicPrior can be passed to
bEPIC's prior_file.compute_prior_from_model() instead of the default
ANSS-KDE prior.

When a SeismicPrior is built from ETAS output, a sidecar JSON file is written
alongside the .tt3 (same stem, .json extension) recording the forecast time,
generation time, ETAS parameters, and any other metadata supplied by the
caller.  The sidecar is loaded automatically by from_tt3().

.tt3 file format
----------------
Line 1 : mx     (number of longitude grid points)
Line 2 : my     (number of latitude grid points)
Line 3 : xlower (minimum longitude, degrees)
Line 4 : ylower (minimum latitude, degrees)
Line 5 : dx     (longitude spacing, degrees)
Line 6 : dy     (latitude spacing, degrees)
Remaining lines : my × mx grid of probability values written with np.flipud
                  (bEPIC's PriorFile reader flips the grid back on load).
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from shapely.geometry import Point


class SeismicPrior:
    """
    A spatial seismic prior distribution on a regular lon/lat grid.

    Attributes
    ----------
    data_dir : str
        Class-level path to the priors directory.  Override to point at a
        different location, e.g. SeismicPrior.data_dir = '/my/priors/'.
        Used as the default directory by from_smooth_seismicity().
    name : str
        Label identifying the prior type (e.g. 'gear1', 'nshm', 'etas').
    lons : np.ndarray
        1-D array of grid longitudes, sorted ascending.
    lats : np.ndarray
        1-D array of grid latitudes, sorted ascending.
    grid : np.ndarray
        2-D probability array, shape (len(lats), len(lons)).
        grid[row, col] → probability at (lats[row], lons[col]).
        Values are non-negative and normalized to sum to 1.
    metadata : dict
        Arbitrary key-value pairs describing how the prior was generated.
        Populated automatically by from_etas(); empty for other sources.
        Written to a sidecar JSON by to_tt3() and loaded back by from_tt3().
    """

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    def __init__(self, name, lons, lats, grid, metadata=None):
        self.name     = str(name)
        self.lons     = np.asarray(lons, dtype=float)
        self.lats     = np.asarray(lats, dtype=float)
        self.grid     = np.asarray(grid, dtype=float)
        self.metadata = metadata if metadata is not None else {}

    # ------------------------------------------------------------------
    # Construction from raw data sources
    # ------------------------------------------------------------------

    @classmethod
    def from_gear1(cls, data_path, polygon=None, bounds=(-127, -113, 30, 45)):
        """
        Build a prior from the GEAR1 global seismic hazard table.

        Parameters
        ----------
        data_path : str
            Path to GL_HAZTBLT_M5_B2_2013.TMP.
        polygon : shapely.geometry.Polygon, optional
            Region mask in standard (lon, lat) coordinates.  Points outside
            the polygon are excluded.  If None, only the bounds filter is applied.
        bounds : tuple
            (lon_min, lon_max, lat_min, lat_max) used to pre-filter the global
            table before the (slower) polygon masking step.

        Returns
        -------
        SeismicPrior
        """
        df = pd.read_csv(
            data_path,
            sep=r'\s+',
            header=None,
            skiprows=17,
            index_col=False,
            names=[
                'longitude', 'latitude', 'probability>5.8',
                'fm_taxis_pl', 'fm_taxis_az', 'fm_paxis_pl', 'fm_paxis_az',
                'rotationdeg', 'probability>5.8_s', 'probability_ratio',
            ],
        )

        lon_min, lon_max, lat_min, lat_max = bounds
        df = df[
            (df['latitude']  > lat_min) & (df['latitude']  < lat_max) &
            (df['longitude'] > lon_min) & (df['longitude'] < lon_max)
        ].copy()

        x     = df['longitude'].values
        y     = df['latitude'].values
        rates = df['probability>5.8'].values

        if polygon is not None:
            mask  = np.array([polygon.contains(Point(xi, yi)) for xi, yi in zip(x, y)])
            x, y, rates = x[mask], y[mask], rates[mask]

        lons = np.sort(np.unique(x))
        lats = np.sort(np.unique(y))

        grid = (
            pd.DataFrame({'lon': x, 'lat': y, 'rate': rates})
            .pivot(index='lat', columns='lon', values='rate')
            .values
        )
        grid = grid / grid.sum()

        return cls(name='gear1', lons=lons, lats=lats, grid=grid)

    @classmethod
    def from_nshm(cls, data_path, polygon=None, bounds=(-127, -113, 30, 45)):
        """
        Build a prior from the USGS NSHM gridded moment-rate file.

        Parameters
        ----------
        data_path : str
            Path to gridded_moment_rates.xyz (space-separated lon, lat, moment_rate).
        polygon : shapely.geometry.Polygon, optional
            Region mask in standard (lon, lat) coordinates.
        bounds : tuple
            (lon_min, lon_max, lat_min, lat_max).

        Returns
        -------
        SeismicPrior
        """
        df = pd.read_csv(
            data_path,
            sep=r'\s+',
            header=None,
            index_col=False,
            names=['longitude', 'latitude', 'moment_rate'],
        )
        df = df.dropna(subset=['moment_rate'])

        lon_min, lon_max, lat_min, lat_max = bounds
        df = df[
            (df['latitude']  > lat_min) & (df['latitude']  < lat_max) &
            (df['longitude'] > lon_min) & (df['longitude'] < lon_max)
        ].copy()

        if polygon is not None:
            mask = np.array([
                polygon.contains(Point(row.longitude, row.latitude))
                for row in df.itertuples(index=False)
            ])
            df = df[mask].copy()

        lons = np.sort(df['longitude'].unique())
        lats = np.sort(df['latitude'].unique())

        grid = (
            df.pivot(index='latitude', columns='longitude', values='moment_rate')
            .values
        )
        grid = grid / grid.sum()

        return cls(name='nshm', lons=lons, lats=lats, grid=grid)

    @classmethod
    def from_helmstetter(cls, polygon=None, bounds=(-127, -113, 30, 45),
                          mag_cutoff=6.0):
        """
        Build a prior from the Helmstetter (2007) aftershock forecast via pyCSEP.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon, optional
            Region mask in standard (lon, lat) coordinates.
        bounds : tuple
            (lon_min, lon_max, lat_min, lat_max).
        mag_cutoff : float
            Use rates from the first magnitude bin >= this value.

        Returns
        -------
        SeismicPrior
        """
        import csep
        from csep.utils import datasets, time_utils

        start_date = time_utils.strptime_to_utc_datetime('2006-11-12 00:00:00.0')
        end_date   = time_utils.strptime_to_utc_datetime('2011-11-12 00:00:00.0')
        forecast   = csep.load_gridded_forecast(
            datasets.helmstetter_aftershock_fname,
            start_date=start_date,
            end_date=end_date,
            name='helmstetter_aftershock',
        )

        magnitudes = forecast.magnitudes
        xy         = forecast.region.midpoints()
        x          = xy[:, 0]   # longitude
        y          = xy[:, 1]   # latitude

        mag_idx = int(np.searchsorted(magnitudes, mag_cutoff))
        rates   = forecast.data[:, mag_idx].astype(float)

        lon_min, lon_max, lat_min, lat_max = bounds
        bounds_mask = (
            (x > lon_min) & (x < lon_max) &
            (y > lat_min) & (y < lat_max)
        )
        x, y, rates = x[bounds_mask], y[bounds_mask], rates[bounds_mask]

        if polygon is not None:
            poly_mask = np.array([
                polygon.contains(Point(xi, yi)) for xi, yi in zip(x, y)
            ])
            x, y, rates = x[poly_mask], y[poly_mask], rates[poly_mask]

        lons = np.sort(np.unique(x))
        lats = np.sort(np.unique(y))

        grid = (
            pd.DataFrame({'lon': x, 'lat': y, 'rate': rates})
            .pivot(index='lat', columns='lon', values='rate')
            .fillna(0.0)
            .values
        )
        grid = grid / grid.sum()

        return cls(name='helmstetter', lons=lons, lats=lats, grid=grid)

    @classmethod
    def from_smooth_seismicity(cls):
        """
        Load the pre-built US/Canada smooth seismicity prior.

        Reads prior_seis_grid_US_Canada.tt3 from the priors data_dir.

        Returns
        -------
        SeismicPrior
        """
        filepath = os.path.join(cls.data_dir, 'prior_seis_grid_US_Canada.tt3')
        return cls.from_tt3(filepath, name='smooth_seismicity')

    @classmethod
    def from_etas(cls, lats, lons, lambda_grid, forecast_time, metadata=None):
        """
        Build a prior from ETAS conditional intensity output.

        Takes the masked-point output of conditional_intensity_grid() directly.
        Points outside the polygon (not present in lats/lons) are filled with
        zero in the rectangular grid.

        Parameters
        ----------
        lats : array-like
            1-D array of grid point latitudes (masked subset).
        lons : array-like
            1-D array of grid point longitudes (masked subset).
        lambda_grid : array-like
            1-D array of conditional intensity values, same length as lats/lons.
        forecast_time : pd.Timestamp or datetime
            The time at which the conditional intensity was evaluated.  Stored
            in metadata and used to auto-generate filenames via suggested_filename().
        metadata : dict, optional
            Additional key-value pairs to store in the sidecar JSON, e.g.
            {'inversion_config': path, 'mc': 2.95, 'theta': {...}}.

        Returns
        -------
        SeismicPrior
        """
        lats        = np.asarray(lats, dtype=float)
        lons        = np.asarray(lons, dtype=float)
        lambda_grid = np.asarray(lambda_grid, dtype=float)

        grid = (
            pd.DataFrame({'lon': lons, 'lat': lats, 'rate': lambda_grid})
            .pivot(index='lat', columns='lon', values='rate')
            .fillna(0.0)
        )
        grid_lons = np.asarray(grid.columns, dtype=float)
        grid_lats = np.asarray(grid.index,   dtype=float)
        grid_vals = grid.values
        grid_vals = grid_vals / grid_vals.sum()

        # Build metadata, always including forecast_time and generated_at
        ts = forecast_time
        if hasattr(ts, 'isoformat'):
            forecast_time_str = ts.isoformat()
        else:
            forecast_time_str = str(ts)

        full_metadata = {
            'forecast_time': forecast_time_str,
            'generated_at':  datetime.now(timezone.utc).isoformat(),
            'n_source_points': int(len(lats)),
        }
        if metadata:
            full_metadata.update(metadata)

        return cls(name='etas', lons=grid_lons, lats=grid_lats,
                   grid=grid_vals, metadata=full_metadata)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def suggested_filename(self):
        """
        Return a suggested .tt3 filename based on name and forecast_time metadata.

        For ETAS priors this produces e.g. 'etas_prior_20080101_000000.tt3'.
        For other priors it produces e.g. 'gear1_prior.tt3'.

        Returns
        -------
        str
        """
        if self.name == 'etas' and 'forecast_time' in self.metadata:
            # Parse ISO string back to datetime for formatting
            ft = self.metadata['forecast_time'].replace(':', '').replace('-', '')
            ft = ft[:15].replace('T', '_')   # YYYYMMDD_HHMMSS
            return f'etas_prior_{ft}.tt3'
        return f'{self.name}_prior.tt3'

    def to_tt3(self, filepath):
        """
        Write the prior to a .tt3 file readable by bEPIC.

        The grid is stored flipped (np.flipud) because bEPIC's PriorFile
        reader flips it back on load.

        If self.metadata is non-empty, a sidecar JSON is written alongside
        the .tt3 (same path, .json extension) for bookkeeping.

        Parameters
        ----------
        filepath : str
            Output path (e.g. 'etas_prior_20080101_000000.tt3').
        """
        mx     = len(self.lons)
        my     = len(self.lats)
        xlower = float(self.lons.min())
        ylower = float(self.lats.min())
        dx     = float(np.diff(self.lons).mean())
        dy     = float(np.diff(self.lats).mean())

        with open(filepath, 'w') as f:
            f.write(f"{mx}\n{my}\n{xlower}\n{ylower}\n{dx:.6f}\n{dy:.6f}\n")
            np.savetxt(f, np.flipud(self.grid), fmt='%.6e')

        if self.metadata:
            json_path = os.path.splitext(filepath)[0] + '.json'
            with open(json_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)

    @classmethod
    def from_tt3(cls, filepath, name=None):
        """
        Load a prior from a .tt3 file.

        The grid is stored flipped; this method unflips it so that
        self.grid[row, col] corresponds to (self.lats[row], self.lons[col]).

        If a sidecar JSON exists (same stem, .json extension), it is loaded
        into self.metadata automatically.

        Parameters
        ----------
        filepath : str
            Path to a .tt3 file.
        name : str, optional
            Label for the prior; defaults to the filename stem.

        Returns
        -------
        SeismicPrior
        """
        with open(filepath, 'r') as f:
            mx     = int(f.readline())
            my     = int(f.readline())
            xlower = float(f.readline())
            ylower = float(f.readline())
            dx     = float(f.readline())
            dy     = float(f.readline())
            grid_flipped = np.loadtxt(f)

        grid = np.flipud(grid_flipped)
        lons = xlower + np.arange(mx) * dx
        lats = ylower + np.arange(my) * dy

        if name is None:
            name = os.path.splitext(os.path.basename(filepath))[0]

        metadata  = {}
        json_path = os.path.splitext(filepath)[0] + '.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metadata = json.load(f)

        return cls(name=name, lons=lons, lats=lats, grid=grid, metadata=metadata)
