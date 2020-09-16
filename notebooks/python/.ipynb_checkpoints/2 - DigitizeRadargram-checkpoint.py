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
from PIL import Image
import geopandas as gpd

# %%
rad_im_f = "../images/radargram.png"
rad_gpkg_f = "../images/radargram.gpkg"

import fiona # we use fiona directly just to list the available layer
fiona.listlayers(rad_gpkg_f) # list the available layers

# %%
rad = Image.open(rad_im_f)
rad

# %% [markdown]
# We determine the scale and translation of the plt using the depth_grid data in the geopackage

# %%
depths = gpd.read_file(rad_gpkg_f, layer='depth_grid')
depths.crs = None # force to none, irrelevant if was set


# %% [markdown]
# to estimate the reference frame (the transform from pixels to meters, we use a different approach here)
# We create the system to be solved directly. This approach is not perfect but still ok for our purposes
#

# %%
# create the matrix and the b vector of the system Ax=b, we will find x
A = []
b = []
for id, (xm, ym, pt) in depths.iterrows():
    xp, yp = np.array(pt.coords)[0]
    A.append([xp,0,1,0])    
    b.append(xm)
    A.append([0,yp,0,1])
    b.append(ym)
    
A = np.array(A)        
b = np.array(b)

# %%
not_nans = ~np.isnan(b) # we remove the nans, where a measure was not available, in x or y dimension (meters)
A = A[not_nans]
b = b[not_nans]

# %%
x, sqnorm, rank, s  = numpy.linalg.lstsq(A, b) # solution
a,b,x_0, y_0 = x

# %%
ratio = a/b
print(f"ratio of scales {ratio}") # the original plot has vertical exagerration of 6x

# %%
# we use homogeneus coordinats and set up the transform, scale and translation are considered
# we assembly the matrix 
T = np.array([[a, 0, x_0], 
              [0, b, y_0],
              [0, 0,  1 ]])

np.savetxt(o("radargram_transform.txt"), T) # and save for future uses

# %%
# extract and prepare the points to be transformed
pt_px = np.array([np.array(p.coords)[0] for p in depths.geometry])
pt_px = np.column_stack([pt_px, np.ones(len(pt_px))])



# %%
# apply the transform on te input points just to check how far we are
pt_mt_est = np.array([T@pt for pt in pt_px])[:,:2]
pt_mt_obs = np.column_stack([depths.x, depths.y])
pt_mt_obs - pt_mt_est 

# %% [markdown]
# just to see if we are near enough, around 0.1 meters is the error in progressive positioning due to digitalization
#
# lower than 0.01 m is the error in depth positioning for the depth positioning due to digitalization
#
# note this is not measurement error! it is just related to our ability to pass from px to m coordinates, given the two axis of the original image have different scales, we are better in determining depth rather than positioning

# %% [markdown]
# # Now we read the stations
# and we estimate the progressive positioning of them in meters

# %%
stations = gpd.read_file(rad_gpkg_f, layer='stations')

# %%
stations_x_px = np.array([np.array(p.coords)[0][0] for p in stations.geometry])
stations_x_px # the progressive positioning in px

# %%
stations_x_m = a*stations_x_px+x_0 # trasnsform the values in px to m
stations_x_m # progressive positions in meters of the corresponding station point
stations["progressive_m"] = stations_x_m # add it to the dataframe

# %%
stations

# %% [markdown]
# # The horizons now
# similarly as above, we estimate the positions in meters for each node of the polylines defining the horizons

# %%
horizons = gpd.read_file(rad_gpkg_f, layer='horizons') # read the horizons as 3 linestrings
horizons # we have just 3 linestrins we need to transform

# %%
# apply th transform to the linestrings, this is a nice way of doing that:
gtr = horizons.geometry.affine_transform([a, 0, 0, b,x_0, y_0])

horizons_m = horizons.copy() # duplicate the dataframe for the meters versio
horizons_m["geometry"] = gtr



# %% [markdown]
# # We now have everything to reproduce the plot 

# %% [markdown]
# We reproduce the plot, to check that the digitalizaion went fine

# %%
figure()
title("Horizons and stations")
horizons_m.plot(ax=gca())
axis("auto")
ylim(-12, 0)
xlim(0, 95)
grid(1)

vlines(stations.progressive_m, -12, 0, zorder=-10, alpha=0.5, color="red")

for i, (sid, m) in stations[["sid", "progressive_m"]].iterrows():
    text( m, -1, str(sid))

# %%
# we reconstruct the per pixel positioning of the image, we could also use imshow() with extend argument
xx = np.arange(rad.size[0])+0.5 
yy = -(np.arange(rad.size[1])+0.5)
X, Y= np.meshgrid(xx,yy)

# %%
px = np.column_stack([X.ravel(), Y.ravel()])
px = np.column_stack([px, np.ones(len(px))]) # add the ones as last column, we use an augmented matrix for the affine transform

# %%
tr = px.dot(T.T) # apply the transform
XX = tr[:,0].reshape(X.shape)
YY = tr[:,1].reshape(Y.shape)


# %%
# now we can plot the original radargram together with the digitized traces
figure(figsize=(8,6))
pcolormesh(XX,YY,np.array(rad)[:,:,0], cmap="gray", alpha=0.1)
horizons_m.plot(ax=gca(), color="blue")
grid()
axis("auto");
vlines(stations.progressive_m, -14, 0, zorder=10, alpha=0.5, color="orange")

for i, (sid, m) in stations[["sid", "progressive_m"]].iterrows():
    text( m, 0.5, str(sid), color="orange",horizontalalignment='center')
    
xlabel("Distance")
ylabel("Depth")
savefig("recreated.png", dpi=300)


# %%
# text?

# %%
horizons_m.to_pickle("horizons.pickle") #  we save them as pickle
stations.to_pickle("stations.pickle")
