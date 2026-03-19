#%%
import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
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

#%%
bounds = [-127,-113,30,45]
df=df[(df['latitude']>bounds[2]) & (df['latitude']<bounds[3])]
df=df[(df['longitude']<bounds[1]) & (df['longitude']>bounds[0])]

magnitudes = df['moment_rate'].values
x = df['longitude'].values
y = df['latitude'].values

plot_cartopy(x,y,magnitudes,'USGS NSHM Moment Rate',bounds=bounds)
# %%
