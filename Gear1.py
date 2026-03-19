#%%

import matplotlib.pyplot as plt
import numpy as np

from functions import *

import pandas as pd
df = pd.read_csv('GEAR1_data/GL_HAZTBLT_M5_B2_2013.TMP',
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

#%%   

bounds = [-127,-113,30,45]
df=df[(df['latitude']>bounds[2]) & (df['latitude']<bounds[3])]
df=df[(df['longitude']<bounds[1]) & (df['longitude']>bounds[0])]

magnitudes = df['probability>5.8'].values
x = df['longitude'].values
y = df['latitude'].values


plot_cartopy(x,y,magnitudes,'USGS GEAR1',bounds=bounds)


# %%
