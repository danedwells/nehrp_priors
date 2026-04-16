#%%
"""
test_etas_updater.py — Interactive test of EtasPriorUpdater.

Loads the pre-inverted ETAS model, then builds three priors:
  (0) Baseline   — historical catalog only, at FORECAST_TIME
  (1) Mainshock  — after appending MAINSHOCK to the catalog
  (2) Sequence   — after appending MAINSHOCK + AFTERSHOCKS

Edit the "CONFIGURE" sections below to change the forecast time, mainshock
location/magnitude, or aftershock sequence, then run all cells.

Requirements: priors, etas_2 (both pip install -e'd), cartopy, matplotlib.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from priors import EtasPriorUpdater

# ── Paths ────────────────────────────────────────────────────────────────────

ETAS_JSON    = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'etas_2', 'output_data', 'parameters_0.json',
)
CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'etas_2', 'input_data', 'example_catalog.csv',
)

# ── CONFIGURE: forecast time and prior grid ───────────────────────────────────
# Baseline prior is evaluated at this time using only the historical catalog.
# Must be after the inversion window end (2007-01-01).

FORECAST_TIME = pd.Timestamp('2008-01-01 00:00:00')

# Bounding box for the output prior (passed to EtasPriorUpdater).
# Should match the bounds used for your other SeismicPriors.
BOUNDS = (-129, -112, 30, 45)   # (lon_min, lon_max, lat_min, lat_max)

# Grid resolution for intensity evaluation (degrees).
# 0.1 matches GEAR1/NSHM/Helmstetter; 0.05 gives finer spatial detail.
GRID_SPACING = 0.1

# ── CONFIGURE: example events ─────────────────────────────────────────────────
# Edit these to explore how the prior responds to different scenarios.
# All times must be > FORECAST_TIME for the update to take effect.
# Magnitude floor for this model: mc = 3.6.

# Mainshock
MAINSHOCK = pd.DataFrame([{
    'time':      pd.Timestamp('2008-01-15 10:32:00'),
    'latitude':  35.80,     # Parkfield / central CA
    'longitude': -120.37,
    'magnitude': 6.5,
}])

# Aftershock sequence (edit times, locations, magnitudes freely)
AFTERSHOCKS = pd.DataFrame([
    {'time': pd.Timestamp('2008-01-15 11:00:00'), 'latitude': 35.82, 'longitude': -120.40, 'magnitude': 4.8},
    {'time': pd.Timestamp('2008-01-15 13:15:00'), 'latitude': 35.77, 'longitude': -120.33, 'magnitude': 4.2},
    {'time': pd.Timestamp('2008-01-16 02:10:00'), 'latitude': 35.84, 'longitude': -120.38, 'magnitude': 3.9},
    {'time': pd.Timestamp('2008-01-16 09:45:00'), 'latitude': 35.79, 'longitude': -120.42, 'magnitude': 3.7},
    {'time': pd.Timestamp('2008-01-17 18:30:00'), 'latitude': 35.75, 'longitude': -120.35, 'magnitude': 4.1},
    {'time': pd.Timestamp('2008-01-18 06:00:00'), 'latitude': 35.81, 'longitude': -120.44, 'magnitude': 3.6},
])

#%%
# ── Build updater and compute baseline prior ──────────────────────────────────

print("Loading catalog and building EtasPriorUpdater...")
catalog = pd.read_csv(
    CATALOG_PATH,
    index_col=0,
    parse_dates=['time'],
    dtype={'url': str, 'alert': str},
)

updater = EtasPriorUpdater.from_inversion_json(
    json_path    = ETAS_JSON,
    catalog_df   = catalog,
    bounds       = BOUNDS,
    grid_spacing = GRID_SPACING,
)
print(updater)

# Baseline — historical catalog only
print(f"\nEvaluating baseline prior at {FORECAST_TIME}...")
prior_baseline = updater.update(FORECAST_TIME)
print(f"  grid shape: {prior_baseline.grid.shape},  max: {prior_baseline.grid.max():.3e}")

#%%
# ── Append mainshock, evaluate at the time just after it ─────────────────────

T_AFTER_MAINSHOCK = MAINSHOCK['time'].iloc[0] + pd.Timedelta(minutes=30)

updater.append_events(MAINSHOCK)
print(f"\nAppended mainshock (M{MAINSHOCK['magnitude'].iloc[0]:.1f} "
      f"at {MAINSHOCK['latitude'].iloc[0]:.2f}°N, "
      f"{MAINSHOCK['longitude'].iloc[0]:.2f}°E).")
print(f"Evaluating prior at {T_AFTER_MAINSHOCK}...")
prior_mainshock = updater.update(T_AFTER_MAINSHOCK)
print(f"  grid shape: {prior_mainshock.grid.shape},  max: {prior_mainshock.grid.max():.3e}")

#%%
# ── Append aftershock sequence, evaluate at the time after the last one ───────

T_AFTER_SEQUENCE = AFTERSHOCKS['time'].max() + pd.Timedelta(hours=2)

updater.append_events(AFTERSHOCKS)
print(f"\nAppended {len(AFTERSHOCKS)} aftershocks "
      f"(M{AFTERSHOCKS['magnitude'].min():.1f}–{AFTERSHOCKS['magnitude'].max():.1f}).")
print(f"Evaluating prior at {T_AFTER_SEQUENCE}...")
prior_sequence = updater.update(T_AFTER_SEQUENCE)
print(f"  grid shape: {prior_sequence.grid.shape},  max: {prior_sequence.grid.max():.3e}")
print(f"\nTotal catalog events in updater: {updater.n_catalog_events}")

#%%
# ── Plot ─────────────────────────────────────────────────────────────────────

PLOT_BOUNDS = [-128, -113, 31, 44]   # display extent (lon_min, lon_max, lat_min, lat_max)

priors  = [prior_baseline, prior_mainshock, prior_sequence]
titles  = [
    f'Baseline\n{FORECAST_TIME.strftime("%Y-%m-%d")}',
    f'After mainshock\nM{MAINSHOCK["magnitude"].iloc[0]:.1f} '
    f'@ {MAINSHOCK["latitude"].iloc[0]:.1f}°N',
    f'After sequence\n({len(AFTERSHOCKS)} aftershocks added)',
]

# Shared colour scale across all three panels
vmax = max(p.grid.max() for p in priors)
norm = mcolors.LogNorm(vmin=vmax * 1e-4, vmax=vmax)

proj = ccrs.PlateCarree()
fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 6),
    subplot_kw={'projection': proj},
    constrained_layout=True,
)

for ax, prior, title in zip(axes, priors, titles):
    ax.set_extent(PLOT_BOUNDS, crs=proj)
    ax.add_feature(cfeature.STATES,     linewidth=0.5, edgecolor='0.4')
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.7)
    ax.add_feature(cfeature.OCEAN,      color='#d0e8f5', zorder=0)
    ax.add_feature(cfeature.LAND,       color='#f5f0e8', zorder=0)

    LON, LAT = np.meshgrid(prior.lons, prior.lats)
    im = ax.pcolormesh(
        LON, LAT, prior.grid,
        norm=norm, cmap='hot_r',
        transform=proj, shading='auto',
    )
    ax.set_title(title, fontsize=11)

# Mark event locations on the mainshock and sequence panels
for ax in axes[1:]:
    ax.scatter(
        MAINSHOCK['longitude'], MAINSHOCK['latitude'],
        s=120, marker='*', color='cyan', edgecolors='k', linewidths=0.5,
        transform=proj, zorder=5, label='Mainshock',
    )
    ax.scatter(
        AFTERSHOCKS['longitude'], AFTERSHOCKS['latitude'],
        s=30, marker='o', color='white', edgecolors='k', linewidths=0.5,
        transform=proj, zorder=5, label='Aftershocks',
    )
    ax.legend(loc='lower left', fontsize=8)

cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.85, pad=0.02)
cbar.set_label('Normalized prior probability', fontsize=10)

fig.suptitle('EtasPriorUpdater — prior evolution after synthetic sequence', fontsize=13)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'figures', 'etas_updater_test.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to {out_path}")
plt.show()

#%%
# ── Spot-check: intensity at mainshock epicentre ──────────────────────────────
# Shows how the peak near the mainshock grows after each update.

def grid_value_at(prior, lat, lon):
    """Nearest-cell value from a SeismicPrior at (lat, lon)."""
    i = int(np.argmin(np.abs(prior.lats - lat)))
    j = int(np.argmin(np.abs(prior.lons - lon)))
    return prior.grid[i, j]

ms_lat = float(MAINSHOCK['latitude'].iloc[0])
ms_lon = float(MAINSHOCK['longitude'].iloc[0])

print(f"\nPrior value at mainshock epicentre ({ms_lat}°N, {ms_lon}°E):")
print(f"  Baseline:         {grid_value_at(prior_baseline,  ms_lat, ms_lon):.4e}")
print(f"  After mainshock:  {grid_value_at(prior_mainshock, ms_lat, ms_lon):.4e}")
print(f"  After sequence:   {grid_value_at(prior_sequence,  ms_lat, ms_lon):.4e}")
