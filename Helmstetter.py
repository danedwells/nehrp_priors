#%%
from csep.core import poisson_evaluations as poisson
from csep.utils import datasets, time_utils
from csep import plots
import csep

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


start_date = time_utils.strptime_to_utc_datetime('2006-11-12 00:00:00.0')
end_date = time_utils.strptime_to_utc_datetime('2011-11-12 00:00:00.0')

forecast = csep.load_gridded_forecast(datasets.helmstetter_aftershock_fname,
                                      start_date=start_date,
                                      end_date=end_date,
                                      name='helmstetter_aftershock')
with open("/home/daned/2024_NEHRP/etas_2/output_data/parameters_0.json", 'r') as f:
    inversion_config = json.load(f)
forecast_time = pd.Timestamp(inversion_config["timewindow_end"])
from numpy import array
polygon = Polygon(np.array(eval(inversion_config["shape_coords"])))
theta = inversion_config["final_parameters"]
mc = inversion_config["m_ref"]

#%%
data = forecast.data
magnitudes = forecast.magnitudes
xy = forecast.region.midpoints()
x = xy[:,0]
y = xy[:,1]

mag_cutoff = 6


# Find first magnitude bin >= cutoff
mag_idx = np.searchsorted(magnitudes, mag_cutoff)

# Extract rates for that bin and plot
rates = data[:, mag_idx]

bounds = [-127,-113,30,45]
mask = np.array([polygon.contains(Point(lat, lon)) 
                 for lat, lon in zip(y,x)])

plot_cartopy(x[mask],y[mask],rates[mask],'Helmstetter (2007)',bounds=bounds)
# %%
