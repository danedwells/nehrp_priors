#!/usr/bin/env python3
"""
compare_priors.py

5x5 matrix comparing all seismic prior models.
  Diagonal     : raw normalized prior (log10 scale, viridis)
  Off-diagonal : prior_i - prior_j (diverging, RdBu_r)
  Colorbar     : one shared per row covering the off-diagonal range

All priors are interpolated onto a common grid (intersection of extents,
finest resolution) and renormalized before comparison.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from prior_model import SeismicPrior


# ---------------------------------------------------------------------------
# Load priors
# ---------------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

candidates = [
    ('GEAR1',        'GEAR1_prior.tt3'),
    ('NSHM',         'USGS_NSHM_prior.tt3'),
    ('Helmstetter',  'helmstetter_prior.tt3'),
    ('Smooth Seis.', 'prior_seis_grid_US_Canada.tt3'),
    ('ETAS',         'etas_prior_20080101_000000.tt3'),
]

priors = {}
for label, fname in candidates:
    path = os.path.join(data_dir, fname)
    if os.path.exists(path):
        priors[label] = SeismicPrior.from_tt3(path)
    else:
        print(f'Warning: {fname} not found — skipping {label}.')

names = list(priors.keys())
n     = len(names)
if n < 2:
    raise RuntimeError('Need at least 2 priors to compare.')


# ---------------------------------------------------------------------------
# Common grid: intersection of all extents at finest resolution
# ---------------------------------------------------------------------------
lon_min = max(p.lons[0]  for p in priors.values())
lon_max = min(p.lons[-1] for p in priors.values())
lat_min = max(p.lats[0]  for p in priors.values())
lat_max = min(p.lats[-1] for p in priors.values())
dx      = min(float(np.diff(p.lons).mean()) for p in priors.values())
dy      = min(float(np.diff(p.lats).mean()) for p in priors.values())

lons_c = np.arange(lon_min, lon_max, dx)
lats_c = np.arange(lat_min, lat_max, dy)

# Query points on common grid: (lat, lon) pairs, lat on axis 0
LAT_C, LON_C = np.meshgrid(lats_c, lons_c, indexing='ij')
pts = np.stack([LAT_C.ravel(), LON_C.ravel()], axis=1)


# ---------------------------------------------------------------------------
# Interpolate and renormalize each prior onto common grid
# ---------------------------------------------------------------------------
grids = {}
for name, p in priors.items():
    interp = RegularGridInterpolator(
        (p.lats, p.lons), p.grid,
        method='linear', bounds_error=False, fill_value=0.0,
    )
    g = interp(pts).reshape(LAT_C.shape)
    g = np.clip(g, 0.0, None)           # remove sub-zero interpolation artifacts
    total = g.sum()
    if total > 0:
        g /= total
    grids[name] = g


# ---------------------------------------------------------------------------
# Figure: n×(n+1) GridSpec — last column holds row colorbars
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(3 * n + 1, 3 * n))
gs  = GridSpec(n, n + 1, figure=fig,
               width_ratios=[*([1] * n), 0.05],
               hspace=0.06, wspace=0.08)

axes      = np.array([[fig.add_subplot(gs[r, c]) for c in range(n)] for r in range(n)])
cbar_axes = [fig.add_subplot(gs[r, n]) for r in range(n)]

extent = [lons_c[0], lons_c[-1], lats_c[0], lats_c[-1]]

for row, name_i in enumerate(names):

    # Symmetric vmax from all off-diagonal differences in this row
    row_vmax = max(
        np.abs(grids[name_i] - grids[name_j]).max()
        for col, name_j in enumerate(names) if col != row
    )
    diff_im = None

    for col, name_j in enumerate(names):
        ax = axes[row, col]
        ax.set_xticks([])
        ax.set_yticks([])

        if row == col:
            g_log = np.where(grids[name_i] > 0, np.log10(grids[name_i]), np.nan)
            ax.imshow(g_log, origin='lower', extent=extent,
                      aspect='auto', cmap='viridis')
            ax.text(0.97, 0.03, 'log₁₀(p)', transform=ax.transAxes,
                    fontsize=6, color='white', ha='right', va='bottom')
        else:
            diff    = grids[name_i] - grids[name_j]
            diff_im = ax.imshow(diff, origin='lower', extent=extent,
                                aspect='auto', cmap='RdBu_r',
                                vmin=-row_vmax, vmax=row_vmax)

        if col == 0:
            ax.set_ylabel(name_i, fontsize=9)
        if row == 0:
            ax.set_title(name_j, fontsize=9)

    # Shared colorbar for off-diagonal panels in this row
    if diff_im is not None:
        cbar = fig.colorbar(diff_im, cax=cbar_axes[row])
        cbar.ax.tick_params(labelsize=7)
        if row == n // 2:
            cbar.set_label('Δ probability', fontsize=8, labelpad=8)


out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prior_comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved to {out_path}')
plt.show()
