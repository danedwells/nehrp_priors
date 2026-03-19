#%%
import matplotlib.pyplot as plt
import numpy as np
import csep
from csep.core import poisson_evaluations as poisson
from csep.utils import datasets, time_utils
from csep import plots

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from functions import *


start_date = time_utils.strptime_to_utc_datetime('2006-11-12 00:00:00.0')
end_date = time_utils.strptime_to_utc_datetime('2011-11-12 00:00:00.0')

forecast = csep.load_gridded_forecast(datasets.helmstetter_aftershock_fname,
                                      start_date=start_date,
                                      end_date=end_date,
                                      name='helmstetter_aftershock')


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

plot_cartopy(x,y,rates,'Helmstetter (2007)',bounds=bounds)
# %%
