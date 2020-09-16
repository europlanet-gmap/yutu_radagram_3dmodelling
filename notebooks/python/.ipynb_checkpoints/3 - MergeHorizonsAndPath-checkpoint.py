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
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

# %%
# load input data
horizons = pd.read_pickle (of("horizons.pickle"))
stations = pd.read_pickle (of("./stations.pickle"))
path = gpd.read_file("path.gpkg", driver="GPKG")

# %%
asline = LineString(path.geometry) #  transform the set of points to a linestring
asline

# %%
# compute progressive and cumulative distance along the path
path['dist_prev'] = 0
for i in path.index[:-1]:
      path.loc[i+1, 'dist_prev'] = path.loc[i, 'geometry'].distance(path.loc[i+1, 'geometry'])
path["cum_dist"] = path["dist_prev"].cumsum()

# %%

# %%
figure(figsize=(8,8))
path.plot(ax=gca())
lll = gpd.GeoDataFrame (geometry=[LineString(path.geometry)])
lll.plot(ax=gca())

for id, (geom, d, n) in path[["geometry", "cum_dist", "progressive_number"]].iterrows():
    pp = np.array(geom.coords)[0]
    text(*pp, s="{:.1f}".format(d))
    
grid()

savefig("prog_distance.png")

# %%

# %%
path[0:5] # we display just some of the data to give an idea

# %% [markdown]
# Basically the progressive meters in the radargrams starts from stations n 3, and end at 16, the radargram does not cover the full path that was done by the rover. Our "horizons" linestring are expressed in the radargram reference frame, so to correctly translate them to the progressive distance along the path we actually need to find a coordinate transform, for this we can use the stations we know should match up. There might be several ways of doing so, but we prefer to distribute the errors along the path via an inerpolation

# %%
out = []

for id, (sid, geometry, progressive_m) in stations.iterrows():
    try :
        sid = np.int(sid) # this will exlude that "8/9" station that was reported in the paper in the same place
    except:
        continue
    
    toget = path.progressive_number == sid
    val = path.cum_dist[toget]
    out.append([sid, progressive_m, np.double(val)])
out   =  np.array(out) 
out
# station id, position in the radargram, position in the overall path
# notice e.g. station 16 is out of several meters, this might be a problem of digitalization of the path. But it is no such a big problem
# because we know that the radargram starts at station 3 and must end at station 16

# %%

# %%

# %%
# we just use a 1-degree polynomial to fit the data
pars = np.polyfit(out[:,1], out[:,2], deg=1) 
predicted =  np.polyval(pars, out[:,1])
err = predicted - out[:,2]

print(pars) # the parameters of the approximating polynomial (first parameter is expected to be around 1)
print(err) # the error at each station

# %%
# we create a nice plot to show the idea here
figure(figsize=(10,3))
xlabel("Distance along path [m]")

out = []
tout = []
for id, (sid, d) in path[["progressive_number", "cum_dist"]].iterrows():  
    x = d
    y = 0
    out.append([x,y])
    if ~np.isnan(sid):
        tout.append([sid, [x,y]])
    
    
    
out=np.array(out)

plot(*out.T, label="Distance along path")
dy = 0.025

for id, (x,y) in tout:
    text(x,y+dy, f"{np.int(id)}", horizontalalignment="center")
    scatter(x,y, color="b")

line = []
for id, (sid, d) in stations[["sid", "progressive_m"]].iterrows():
    x = d*pars[0] + pars[1]
    y = 0.1
    scatter(x,y, color="black")
    text(x,y+dy, f"{sid}", horizontalalignment="center")
    line.append([x,y])
    
plot(*np.array(line).T,  label="Distance along radargram")
lgd =legend()



gca().get_yaxis().set_visible(False)
grid()
xticks(np.arange(0, 120, 10.0))
title("Mapping between radargram and path")
# tight_layout()

ylim(-0.1, 0.35)
tight_layout()

savefig("mapping.png", dpi=300, transparent=0,)


# %%

# %%
# we can now process each horizon and append the result to out
out = []
for id, (hid, geometry) in horizons.iterrows():
    pts = np.array(geometry.coords) # original coord in the radargram
    newpts = []
    for pd,elev in pts:
        # point cumulative distance position on the full radargram
        onpath_prog_d = np.polyval(pars, pd) # apply the trasform from radargram progressive distance to
                                             # full path progressive distance
                                            
        # estimate the point on the "asline" path
        xy_pt = asline.interpolate(onpath_prog_d)
        x,y = np.array(xy_pt.coords)[0] # extract x,y value
        newpts.append([x,y,elev]) # append a point of the horizon
    out.append(np.array(newpts)) # append the points of the horizon to out

# %% [markdown]
# out contains the two dimensional coordinate (map coordinates) of the nodes of the horizons, third column is the elevation in m from the surface

# %%

# %%
# a first very simple 3d view of the data, we will see soon how to actually visualize all the data in 3D, 
# but this is enough for a visual check

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
for i in np.arange(3):
    ax.scatter(*out[i].T)
    ax.plot(*out[i].T)
    
tight_layout()
savefig("3d_simple.png", dpi=300)

# %%
np.savez("horizons", out) # save, using numpy.savez here

# %%
