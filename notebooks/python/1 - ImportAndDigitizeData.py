# -*- coding: utf-8 -*-
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
# load the path image and the path data
from PIL import Image
import geopandas as gpd

# %%
from smalllib import of, ff

# %%
# the input files
path_image = "../images/path.png"
gpkg_path = "../images/path.gpkg"

# %%
# load everything
im = Image.open(path_image)
ref_frame = gpd.read_file(gpkg_path, layer="reference_frame")
ref_frame.crs = None # force to none, just for clarity, we might have digitized in a valid crs but we are not interested

# %%
ref_frame # this is the table of digitized points

# %% [markdown]
# The table above is used to reconstruct the transform from pixel-coordinates to metric coordinates, which are needed for digitizing the dataset of the path

# %%
px_coords = np.array([np.array(p.coords)[0] for p in ref_frame.geometry]) # extract the pixel coordinates
px_coords

# %%
m_coords = np.array(ref_frame[["x", "y"]]) # extract the coordinates in meters for the same points
m_coords

# %% [markdown]
# we determine the transform between the two, we know the two set of points do correspont to each other and there must be a transform between the two
# we use an affine transform for this... via affine6p (or nudged library if you prefer)

# %%
#pip install affine6p --user if not installed already

# %%
import affine6p
# using a complete affine transform is a little overkill in our case, but in the general situation migh be desirable, especially when workin on scanned imagery

# %%
T = affine6p.estimate(px_coords.tolist(), m_coords.tolist())
err = affine6p.estimate_error(T, px_coords.tolist(), m_coords.tolist()) # mean squared distance
np.sqrt(err) # around 20 cm average error due to digitalization

# %% [markdown]
# we got a very low error, good!

# %% [markdown]
# we have our transform, now we work with the picture
#

# %%
figure(figsize=(10,10)) # fast inspection
imshow(im)

# %% [markdown]
# Now we want to save the raster as a geotiff file using rasterio
#
# We define the absolute position of the landing site in an absolute reference frame. These values are obtanied from QGIS and picking the landing site onto the orthoimage (see companion dataset)
#
# Note that we do not actually know the crs of the map we imported, but we know is a metric framework. We will try to use the same crs with eqc projection as the orthoimages and the DEM from LROC NAC. For such small distances using the wrong projection will not be noticeable

# %%
coords_start = np.array([3500707.20,1337897.55]) # landing site coords in the ref frame below
crs = "+proj=eqc +lat_ts=44 +lat_0=0 +lon_0=180 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs +type=crs" # the crs of the coords above

np.savetxt(of("landing_coordinates.txt"), coords_start) # we also save them for future use


# T provide the transform from pixel to meter coordinates in a reference frame with the origin in the top left corner with y-axis upward
# we need to alter the transform to a different reference with y downward: for this reason we chenge the sign of mt[1,1]
mt = np.array(T.get_matrix())[:2,:]
mt[1,1] = - mt[1,1]
mt[:,2] += coords_start # add the global shift

import rasterio
at = rasterio.transform.Affine(*mt.ravel()) # fnally recreate a rasterio-compatible affine transform

# %%
im_asarr = np.array(im) # to a numpy array
im_asarr = np.moveaxis(im_asarr, 2, 0) # move the 3d axis (the one with colors) to the first position, rasterio is expecting the bands (r,g,b) to be the first-axis dimension

# %%

# %%
# save it, finally
import os
with rasterio.open(
    of('path_georef.tif'),
    'w',
    photometric="RGB",
    driver='GTiff',
    height=im_asarr.shape[1],
    width=im_asarr.shape[2],
    count=4,
    dtype=im_asarr.dtype,
    crs=crs,
    transform=at # here we plug the affine transform computed above
)as dst:
    dst.write(im_asarr, [1,2,3,4])



# %% [markdown]
# Now the raster is georeferenced, you can open it up in qgis and verify it.

# %% [markdown]
# # Now the path-data

# %%
path = gpd.read_file(gpkg_path, layer="path") # load the data for the path
path_px_coords = np.array([np.array(p.coords[0]) for p in path.geometry]) # as a pure numpy array

# %%
path_m_coords = np.array(T.transform(path_px_coords.tolist())) # apply transform
scatter(*path_m_coords.T) # and plot
axis("equal")
grid()



# %% [markdown]
# We correctly digitized the points of the path in meters

# %%
from shapely.geometry import Point, LineString
asgeom_m = [Point(p) for p in path_m_coords]


# when digitizing we did not digitize the landing point site (station 0), we jsut add i at Point(0,0)
numbers = list(path.progressive_number)
numbers.insert(0, np.nan)
asgeom_m.insert(0, Point(0,0))

# recreate the dataframe
path = gpd.GeoDataFrame(geometry=asgeom_m)
path["progressive_number"] = numbers


# %% [markdown]
# we can now plot the data and save it

# %%
path.plot()
grid()

# %%
newgeom = path.translate(*coords_start) # we add the coordinates of the landing site to match the crs

path_t = path.copy() # copy
path_t.geometry = newgeom # set the new geometry
path.crs = crs # and the crs

# %%
path_t.to_file(of("path.gpkg"), driver="GPKG")

# %%
