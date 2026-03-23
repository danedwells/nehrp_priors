#%%
import matplotlib.pyplot as plt
import numpy as np

from etas.inversion import triggering_kernel, parameter_dict2array, to_days, haversine
import numpy as np
from shapely.geometry import Point, Polygon
import json
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm

from functions import *

import pandas as pd

df = pd.read_csv('USGS_NSHM_data/gridded_moment_rates.xyz',
                 sep='\s+', 
                 header=None,
                 index_col=False,
                 names=['longitude',
                        'latitude',
                        'moment_rate'])

with open("/home/daned/2024_NEHRP/etas_2/output_data/parameters_0.json", 'r') as f:
    inversion_config = json.load(f)
forecast_time = pd.Timestamp(inversion_config["timewindow_end"])
from numpy import array
polygon = Polygon(np.array(eval(inversion_config["shape_coords"])))
theta = inversion_config["final_parameters"]
mc = inversion_config["m_ref"]

#%%
bounds = [-127,-113,30,45]
df=df[(df['latitude']>bounds[2]) & (df['latitude']<bounds[3])]
df=df[(df['longitude']<bounds[1]) & (df['longitude']>bounds[0])]

magnitudes = df['moment_rate'].values
x = df['longitude'].values
y = df['latitude'].values
x,y = np.meshgrid(x,y)
x = x.ravel()
y = y.ravel()

mask = np.array([polygon.contains(Point(lat, lon)) 
                 for lat, lon in zip(y,x)])


plot_cartopy(x['mask'],y['mask'],magnitudes['mask'],'USGS NSHM Moment Rate',bounds=bounds)
# %%
"""
Make a bEPIC prior
"""
# Assumes df has columns: longitude, latitude, moment_rate
# and is on a regular grid

lons = np.sort(df['longitude'].unique())
lats = np.sort(df['latitude'].unique())

mx = len(lons)
my = len(lats)
dx = np.diff(lons).mean()
dy = np.diff(lats).mean()
xlower = lons.min()
ylower = lats.min()

# Pivot to 2D grid (rows=lat, cols=lon), normalize to sum=1
grid = df.pivot(index='latitude', columns='longitude', values='moment_rate').values
grid = grid / grid.sum()

# flipud before writing (PriorFile flips on read, so we pre-flip)
grid_out = np.flipud(grid)

output_path = 'USGS_NSHM_prior.tt3'
with open(output_path, 'w') as f:
    f.write(f"{mx}\n{my}\n{xlower}\n{ylower}\n{dx:.6f}\n{dy:.6f}\n")
    np.savetxt(f, grid_out, fmt='%.6e')
# %%
