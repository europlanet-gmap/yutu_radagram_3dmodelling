# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %pylab inline
import rasterio
from smalllib import *

# %% [markdown]
# We now want to cross this information with the available DEM to estimate the elevation along the horizons, which are needed to correct the absolute vertical positioning of the horizons, which are measured from the surface. Notice this is not clear from the papers, we will assume so, givig the oppurtinuty to see how to perform this correction.
# This step it might not be striclty necessary for a small area like the onw investigated by Chang'e III Yutu rover

# %% [markdown]
# Load the data we need:

# %%
horizons_f = of("horizons.npz")
hs = np.load(horizons_f, allow_pickle=True)["arr_0"].tolist()
dem_f  = "./rasters/NAC_DTM_CHANGE3.TIF"
dem = rasterio.open(dem_f)

# %% [markdown]
# # crop the raster to a meaningful region
# We first determine the boundaries for the region we are dealing with

# %%
# estimate the bounding box
mi = np.min(hs[0][:,:2],axis=0)
ma = np.max(hs[0][:,:2],axis=0)


# %%
# mi = np.array([minx, miny]) # lower left corner
# ma = np.array([maxx, maxy]) # upper right corner

enlarge = 3 # enlarge the region of this size in m to have also the sourrandings

mi -= np.ones(2) * enlarge
ma += np.ones(2) * enlarge

# we also save this data to file for any future use
bb = np.array([*mi, *ma])
np.savetxt("region.txt", bb)


# %%
out_img, new_meta = crop_raster_rectangle(dem, mi, ma, "cropped_dem.tif")
imshow(out_img[0])

# %% [markdown]
# # Upscale the raster
# This is also not striclty necessary, but is useful to smooth out elevation values on such low-res raster, it would be better to higher resolution data, but it is not freely available

# %%
dem = rasterio.open("cropped_dem.tif")

# %%
upscale_factor=5 # split each pixel in 25

# %%
up, newmeta = upsample_raster(dem, "cropped_dem_up.tif")
imshow(up[0])

# %% [markdown]
# # Sample elevation values

# %% [markdown]
# we can now use the upscaled dem for estimating elevations along the path

# %%
dem = rasterio.open("cropped_dem_up.tif") # reopen the dem

# %%
from smalllib import sampleRasterAtPoints # some methods are in this small module, have a look at that for doc

# %%
import geopandas as gpd
from shapely.geometry import Point

tables = [] # we will create a geopandas table for each horizon

for hor in hs: # we are repeating for each horizon, they have different nodes
    tab = gpd.GeoDataFrame()
    
    elevations = sampleRasterAtPoints(dem, hor[:,:2]) # sample elevations
    
    points = [Point(x,y,z+el) for (x,y,z), el in zip(hor, elevations)] # create the points, notice we are adding the computed elevations to z
    tab.geometry = points # and use them as geometry
    tab["surface_elevation"] = elevations
    tab["original_z"] = hor[:,2]
    tables.append(tab)
    


# %%
# see the last table we generated, just for reference
tab.plot()

# %%
tab[0:5] # see an example

# %%
# save each horizon in the same geopackage, one per layer, so we can also open them in qgis
for i, t in enumerate(tables):
    t.to_file("horizons.gpkg", driver="GPKG", layer=f"{str(i)}")

