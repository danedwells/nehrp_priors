#%%
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
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Point, Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
    ('Smooth_Seis.', 'prior_seis_grid_US_Canada_filled.tt3'),
    ('ETAS',         'etas_prior_20080101_000000_filled.tt3'),
]

# Check units
"""
Gear1: rate of shallow earthquakes magn >5.8, eq *day^-1 * km^-2, 0.1*0.1 degree cells 
NSHM: perhaps N*m/yr per cell? Website says events/yr, unclear if log10 already applied.
Helmstetter: rate counts per cell, expected number of eqs per 0.1*0.1 degree cell 
    above m 4.95 per year
Smooth_Seis: probability density of eq occurance per km^2 (log10 already applied)?
ETAS - conditional intensity, in units of events/(day*km^2)

Existing Priors Links:

    • Gear1 () - https://pubs.geoscienceworld.org/ssa/bssa/article/105/5/2538/332070/GEAR1-A-Global-Earthquake-Activity-Rate-Model
    • NSHM () - https://data.opensha.org/nshm23/reports/branch_averaged_gridded/resources/gridded_moment_rates.xyz
    or https://data.opensha.org/nshm23/reports/branch_averaged_gridded/#regional-nucleation-rates
    • Helmstetter (2007) – from PyCSEP package https://github.com/cseptesting/pycsep/blob/main/csep/artifacts/ExampleForecasts/GriddedForecasts/helmstetter_et_al.hkj-fromXML.dat
        AND from paper itself https://hal.science/hal-00195399/document
    • Smoothed Seismicity (Williamson) – Google Drive https://drive.google.com/drive/folders/1qjJDD1CV43Afhp0xP7-Z9g4VJY8RIvYZ and/or email from Amy Williamson @ Amy.Williamson@berkeley.edu
    • ETAS_2 output – Running on example catalog, see ETAS_2 github (forked from ETAS) https://github.com/danedwells/etas_2


"""
#%%
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

# etas_raw = priors['ETAS'].grid
# len(etas_raw[etas_raw==0])
# etas_raw[etas_raw == 0] = 0.00005
# plt.imshow(np.log10(etas_raw))

#%%

print("PRE - interpolation")
for name, p in priors.items():
    print(name, len(p.grid[p.grid==0]))
    print(np.nanmin(p.grid))
    fig,ax = plt.subplots(figsize=(8,6))
    ax.imshow(p.grid)
    ax.set_title(name)
    plt.show()



#%%





# ---------------------------------------------------------------------------
# Common grid: union of all extents at finest resolution
# Priors that don't cover a cell will receive a uniform fill value.
# ---------------------------------------------------------------------------
lon_min = -129 #min(p.lons[0]  for p in priors.values())
lon_max = -112 #max(p.lons[-1] for p in priors.values())
lat_min = 30 #min(p.lats[0]  for p in priors.values())
lat_max = 45 #max(p.lats[-1] for p in priors.values())
dx      = min(float(np.diff(p.lons).mean()) for p in priors.values())
dy      = min(float(np.diff(p.lats).mean()) for p in priors.values())

lons_c = np.arange(lon_min, lon_max, dx)
lats_c = np.arange(lat_min, lat_max, dy)

# Query points on common grid: (lat, lon) pairs, lat on axis 0
LAT_C, LON_C = np.meshgrid(lats_c, lons_c, indexing='ij')
pts = np.stack([LAT_C.ravel(), LON_C.ravel()], axis=1)

# # ---------------------------------------------------------------------------
# # Load polygon from ETAS inversion config
# # ---------------------------------------------------------------------------
# etas_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                 '..', 'etas_2', 'output_data', 'parameters_0.json')
# with open(etas_config_path, 'r') as f:
#     inversion_config = json.load(f)
# from numpy import array  # needed for eval of shape_coords
# polygon = Polygon(np.array(eval(inversion_config['shape_coords'])))
# # ---------------------------------------------------------------------------
# # Polygon mask on common grid
# # ---------------------------------------------------------------------------
# poly_mask = np.array([polygon.contains(Point(lat, lon))
#                       for lat, lon in zip(LAT_C.ravel(), LON_C.ravel())],
#                      dtype=bool).reshape(LAT_C.shape)


# ---------------------------------------------------------------------------
# Interpolate, mask, and renormalize each prior onto common grid
# ---------------------------------------------------------------------------
grids = {}
for name, p in priors.items():
    #if name == "Smooth_Seis.":
        #p.grid = np.grid
    #if name == "Helmstetter":
    #    p.grid += 0.00001
    #if name == "ETAS":
    #    p.grid += 0.00001
    interp = RegularGridInterpolator(
        (p.lats, p.lons), p.grid,
        method='linear', bounds_error=False, fill_value=np.nan,
    )
    g = interp(pts).reshape(LAT_C.shape)

    # Fill cells outside this prior's native extent with a uniform value.
    # Use the mean of in-bounds values so offshore cells carry equal, low weight.
    in_bounds = (
        (LAT_C >= p.lats[0]) & (LAT_C <= p.lats[-1]) &
        (LON_C >= p.lons[0]) & (LON_C <= p.lons[-1])
    )
    uniform_fill = float(np.nanmean(g[in_bounds]))
    g[~in_bounds] = uniform_fill
    fig,ax = plt.subplots(figsize=(8,6))
    ax.imshow(g)
    plt.show()
    #print(name, (g==0).sum()/g.size)
    #g = np.clip(g, 0.0, None)           # remove sub-zero interpolation artifacts
    #g[~poly_mask] = np.nan
    total = np.nansum(g)
    print(total)
    if total > 0:
        g /= total
    total = np.nansum(g)
    print("Total new: ",total)
    grids[name] = g

#%%
print("POST- interpolation")
for name, p in grids.items():
    print(name, "Number of values == 0: ", len(p[p==0]))
    print("Min value: ",np.nanmin(p))

#%%
# ---------------------------------------------------------------------------
# Figure: n×(n+1) GridSpec — last column holds row colorbars
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(3 * n + 1, 3 * n))
gs  = GridSpec(n, n + 1, figure=fig,
               width_ratios=[*([1] * n), 0.05],
               hspace=0.06, wspace=0.08)

proj      = ccrs.PlateCarree()
axes      = np.array([[fig.add_subplot(gs[r, c], projection=proj) for c in range(n)] for r in range(n)])
cbar_axes = [fig.add_subplot(gs[r, n]) for r in range(n)]

extent = [lons_c[0], lons_c[-1], lats_c[0], lats_c[-1]]

def signed_log(a, eps):
    return np.sign(a) * (np.log10(np.abs(a) + eps) - np.log10(eps))

for row, name_i in enumerate(names):

    eps = grids[name_i][grids[name_i] > 0].min() * 1e-2

    # Consistent color scale for off-diagonal
    row_vmax = max(
        #np.nanmax(np.abs(signed_log(grids[name_i] - grids[name_j], eps)))
        np.nanmax(np.log10(grids[name_i] - np.log10(grids[name_j])))
        for col, name_j in enumerate(names) if col != row
    )
    diff_im = None

    for col, name_j in enumerate(names):
        ax = axes[row, col]
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='black')
        ax.set_xticks([])
        ax.set_yticks([])
        if name_j == 'helmstetter':
            helm = grids[name_j]
        if row == col:
            g_log = np.where(grids[name_i] > 0, np.log10(grids[name_i]), np.nan)
            ax.imshow(g_log, origin='lower', extent=extent, transform=proj,
                      aspect='auto', cmap='viridis')
            ax.text(0.97, 0.03, 'log₁₀(p)', transform=ax.transAxes,
                    fontsize=6, color='white', ha='right', va='bottom')
        else:
            #diff = signed_log(grids[name_i] - grids[name_j], eps)
            diff = np.log10(grids[name_i]) - np.log10(grids[name_j])
            diff_im = ax.imshow(diff, origin='lower', extent=extent, transform=proj,
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
            cbar.set_label('log₁₀(|Δ|) of priors', fontsize=8, labelpad=8)


out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prior_comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved to {out_path}')
plt.show()

#%%

# ---------------------------------------------------------------------------
# Figure 2: 1×n strip — diagonal panels only (log10 of each prior)
# ---------------------------------------------------------------------------
g_logs = {name: np.where(grids[name] > 0, np.log10(grids[name]), np.nan)
          for name in names}

fig2, axes2 = plt.subplots(
    1, n,
    figsize=(3.6 * n, 4.0),
    subplot_kw={'projection': ccrs.PlateCarree()},
)
if n == 1:
    axes2 = [axes2]

for ax, name in zip(axes2, names):
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='black')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(name, fontsize=9)
    im = ax.imshow(
        g_logs[name], origin='lower', extent=extent, transform=proj,
        aspect='auto', cmap='viridis',
    )
    cbar2 = fig2.colorbar(im, ax=ax, orientation='horizontal',
                          fraction=0.05, pad=0.06)
    cbar2.set_label('log₁₀(p)', fontsize=7)
    cbar2.ax.tick_params(labelsize=6)

fig2.suptitle('Prior models — log₁₀ scale', fontsize=11, y=1.01)

out_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prior_diagonal_strip.png')
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f'Saved to {out_path2}')
plt.show()

# %%



#%%