import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_cartopy(lon,
                 lat,
                 vals,
                 valname, 
                 bounds = [-127,-113,30,45],
                 transform=ccrs.PlateCarree()):
    fig, ax = plt.subplots(subplot_kw={'projection': transform})

    ax.set_extent(bounds)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)

    sc = ax.scatter(lon, lat, 
                    c=vals, s=4, marker='s',
                    cmap='hot_r', norm=LogNorm(),
                    transform=transform)

    plt.colorbar(sc, label=f'{valname}')
    plt.title(f'{valname}')
    plt.show()