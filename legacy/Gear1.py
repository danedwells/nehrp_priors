#%%

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
df = pd.read_csv('../data/GEAR1_data/GL_HAZTBLT_M5_B2_2013.TMP',
                 sep='\s+', 
                 header=None,
                 skiprows=17,
                 index_col=False,
                 names=['longitude',
                        'latitude',
                        'probability>5.8',
                        'fm_taxis_pl',
                        'fm_taxis_az',
                        'fm_paxis_pl',
                        'fm_paxis_az',
                        'rotationdeg',
                        'probability>5.8_s',
                        'probability_ratio'])
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

magnitudes = df['probability>5.8'].values
x = df['longitude'].values
y = df['latitude'].values

bounds = [-127,-113,30,45]
mask = np.array([polygon.contains(Point(lat, lon)) 
                 for lat, lon in zip(y,x)])


plot_cartopy(x[mask],y[mask],magnitudes[mask],'USGS GEAR1',bounds=bounds)


# %%
# Save as bEPIC prior

x_masked = x[mask]
y_masked = y[mask]
rates_masked = magnitudes[mask]
lons = np.sort(np.unique(x_masked))
lats = np.sort(np.unique(y_masked))

mx = len(lons)
my = len(lats)
dx = np.diff(lons).mean()
dy = np.diff(lats).mean()
xlower = lons.min()
ylower = lats.min()

grid = pd.DataFrame({'lon': x_masked, 'lat': y_masked, 'rate': rates_masked}) \
         .pivot(index='lat', columns='lon', values='rate').values
grid = grid / grid.sum()

with open('../data/GEAR1_prior.tt3', 'w') as f:
    f.write(f"{mx}\n{my}\n{xlower}\n{ylower}\n{dx:.6f}\n{dy:.6f}\n")
    np.savetxt(f, np.flipud(grid), fmt='%.6e')