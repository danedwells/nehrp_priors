# priors

Spatial seismic prior distributions for bEPIC. The core module (`src/prior_model.py`) provides a unified interface for building, serializing, and loading spatial priors from five source models.

---

## Installation

Via SSH:

```bash
mkdir priors/
cd priors/
git clone git@github.com:danedwells/nehrp_priors.git .
pip install -e .
```

**Dependencies** (installed automatically via `pyproject.toml`):

| Package | Use |
|---------|-----|
| `numpy` | Grid arithmetic |
| `pandas` | Data ingestion and pivoting |
| `scipy` | (Available for extensions) |
| `shapely` | Polygon masking |
| `pycsep` | Required only for `from_helmstetter()` |

`pycsep` is not declared in `pyproject.toml` and must be installed separately if you use the Helmstetter source:

```bash
pip install git+https://github.com/SCECcode/pycsep.git
```

---

## Directory structure

```
priors/
├── src/
│   └── prior_model.py       # SeismicPrior class (main module)
├── legacy/                  # Superseded scripts (logic absorbed into prior_model.py)
│   ├── Gear1.py
│   ├── Helmstetter.py
│   ├── USGS_NSHM.py
│   ├── functions.py
│   └── __init__.py
├── data/
│   ├── prior_seis_grid_US_Canada.tt3   # Pre-built US/Canada smooth seismicity prior
│   ├── GEAR1_prior.tt3                 # Saved GEAR1 prior
│   ├── helmstetter_prior.tt3           # Saved Helmstetter prior
│   ├── USGS_NSHM_prior.tt3             # Saved NSHM prior
│   ├── GEAR1_data/
│   │   └── GL_HAZTBLT_M5_B2_2013.TMP  # GEAR1 global hazard table (Bird & Kreemer 2015)
│   └── USGS_NSHM_data/
│       └── gridded_moment_rates.xyz    # USGS NSHM gridded moment rates
└── pyproject.toml
```

---

## `src/prior_model.py`

### `SeismicPrior`

A spatial seismic prior on a regular lon/lat grid. Values are non-negative and normalized to sum to 1.

**Class attribute**

| Attribute | Description |
|-----------|-------------|
| `data_dir` | Path to `data/`. Override to point at a different location before calling `from_smooth_seismicity()`. |

**Instance attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Label for the prior source (e.g. `'gear1'`, `'etas'`). |
| `lons` | `np.ndarray` | 1-D array of grid longitudes, sorted ascending. |
| `lats` | `np.ndarray` | 1-D array of grid latitudes, sorted ascending. |
| `grid` | `np.ndarray` | 2-D probability array, shape `(len(lats), len(lons))`. `grid[row, col]` is the probability at `(lats[row], lons[col])`. |
| `metadata` | `dict` | Provenance key-value pairs. Populated automatically by `from_etas()`; empty for all other sources. Written to a sidecar `.json` by `to_tt3()` and reloaded by `from_tt3()`. |

---

### Constructors

#### `SeismicPrior.from_gear1(data_path, polygon=None, bounds=(-127, -113, 30, 45))`

Build a prior from the GEAR1 global seismic hazard table.

| Parameter | Description |
|-----------|-------------|
| `data_path` | Path to `GL_HAZTBLT_M5_B2_2013.TMP`. |
| `polygon` | `shapely.Polygon` region mask (lon, lat). Points outside are excluded. |
| `bounds` | `(lon_min, lon_max, lat_min, lat_max)` bounding box used to pre-filter before polygon masking. |

---

#### `SeismicPrior.from_nshm(data_path, polygon=None, bounds=(-127, -113, 30, 45))`

Build a prior from the USGS NSHM gridded moment-rate file.

| Parameter | Description |
|-----------|-------------|
| `data_path` | Path to `gridded_moment_rates.xyz` (space-separated lon, lat, moment_rate). |
| `polygon` | `shapely.Polygon` region mask. |
| `bounds` | `(lon_min, lon_max, lat_min, lat_max)`. |

---

#### `SeismicPrior.from_helmstetter(polygon=None, bounds=(-127, -113, 30, 45), mag_cutoff=6.0)`

Build a prior from the Helmstetter (2007) aftershock forecast loaded via pyCSEP. Requires `pycsep`.

| Parameter | Description |
|-----------|-------------|
| `polygon` | `shapely.Polygon` region mask. |
| `bounds` | `(lon_min, lon_max, lat_min, lat_max)`. |
| `mag_cutoff` | Rates are taken from the first magnitude bin ≥ this value. Default `6.0`. |

---

#### `SeismicPrior.from_smooth_seismicity()`

Load the pre-built US/Canada smooth seismicity prior directly from `data/prior_seis_grid_US_Canada.tt3`. No arguments.

---

#### `SeismicPrior.from_etas(lats, lons, lambda_grid, forecast_time, metadata=None)`

Build a prior from ETAS conditional intensity output. Takes the masked-point output of `conditional_intensity_grid()` directly; points outside the polygon are filled with zero in the rectangular grid.

| Parameter | Description |
|-----------|-------------|
| `lats` | 1-D array of grid point latitudes (masked subset). |
| `lons` | 1-D array of grid point longitudes (masked subset). |
| `lambda_grid` | 1-D array of conditional intensity values, same length as `lats`/`lons`. |
| `forecast_time` | `pd.Timestamp` or `datetime` at which the intensity was evaluated. Stored in `metadata`. |
| `metadata` | Optional dict of additional provenance fields (e.g. `mc`, `theta`, `inversion_config`). |

The resulting `metadata` dict always includes `forecast_time`, `generated_at`, and `n_source_points`.

---

#### `SeismicPrior.from_tt3(filepath, name=None)`

Load a prior from a `.tt3` file. Automatically loads a sidecar `.json` into `metadata` if one exists.

| Parameter | Description |
|-----------|-------------|
| `filepath` | Path to a `.tt3` file. |
| `name` | Label for the prior; defaults to the filename stem. |

---

### Serialization

#### `prior.to_tt3(filepath)`

Write the prior to a `.tt3` file readable by bEPIC. If `metadata` is non-empty, a sidecar `.json` is written alongside it (same stem, `.json` extension).

#### `prior.suggested_filename()`

Return a suggested `.tt3` filename. For ETAS priors, includes the forecast time (e.g. `etas_prior_20080101_000000.tt3`). For all others, returns `{name}_prior.tt3`.

---

### `.tt3` file format

```
Line 1:  mx      — number of longitude grid points
Line 2:  my      — number of latitude grid points
Line 3:  xlower  — minimum longitude (degrees)
Line 4:  ylower  — minimum latitude (degrees)
Line 5:  dx      — longitude spacing (degrees)
Line 6:  dy      — latitude spacing (degrees)
Lines 7+: my × mx grid of probability values (np.flipud applied on write; unflipped on read)
```

The flip convention matches bEPIC's `PriorFile` reader.

---

### Example usage

```python
from shapely.geometry import Polygon
from src.prior_model import SeismicPrior

polygon = Polygon([(-127, 30), (-113, 30), (-113, 45), (-127, 45)])

# Build from GEAR1
prior = SeismicPrior.from_gear1('data/GEAR1_data/GL_HAZTBLT_M5_B2_2013.TMP', polygon=polygon)
prior.to_tt3('data/GEAR1_prior.tt3')

# Build from ETAS output
prior = SeismicPrior.from_etas(lats, lons, lambda_grid, forecast_time=t,
                                metadata={'mc': 2.95, 'theta': theta})
prior.to_tt3(prior.suggested_filename())

# Load back
prior = SeismicPrior.from_tt3('etas_prior_20080101_000000.tt3')

# Use pre-built smooth seismicity prior
prior = SeismicPrior.from_smooth_seismicity()
```

---

## `legacy/`

Standalone scripts from before the logic was consolidated into `prior_model.py`. Kept for reference; not intended for active use.

| File | Description |
|------|-------------|
| `Gear1.py` | Script that reads `GL_HAZTBLT_M5_B2_2013.TMP`, masks to a polygon, and writes `GEAR1_prior.tt3`. |
| `USGS_NSHM.py` | Script that reads `gridded_moment_rates.xyz` and writes `USGS_NSHM_prior.tt3`. |
| `Helmstetter.py` | Script that loads the Helmstetter (2007) forecast via pyCSEP and plots rates within a polygon. |
| `functions.py` | Shared plotting helpers (Cartopy map plots) used by the scripts above. |
| `__init__.py` | Package init for the old module layout. |
