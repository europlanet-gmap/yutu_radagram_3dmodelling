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
import rasterio
from smalllib import crop_raster_rectangle, raster_to_pyvista_mesh, of, ff, save_mesh_and_texture_as_obj
import vtk, pyvista

# %%
center_coords = np.loadtxt(of("center_coords.txt"))

# %%
# crop_raster_rectangle?

# %%
newbounds = gpd.read_file("../images/crater_bounds.gpkg")
newbounds = newbounds.geometry[0]

base ="../rasters/NAC_DTM_CHANGE3_M1144922100_160CM.TIF"
basemap = rasterio.open(base)
cropped2, bmeta = crop_raster_rectangle(basemap, newbounds.bounds[:2], newbounds.bounds[2:], outfile=of("crater_cropped_dem.tif"))


dem = "../rasters/NAC_DTM_CHANGE3.TIF"
dem = rasterio.open(dem)
dem_c, dmeta = crop_raster_rectangle(dem, newbounds.bounds[:2], newbounds.bounds[2:],  outfile=of("crater_cropped_ortho.tif"))

# %%
from smalllib import generate_pixel_coordinates, transform_pixel_coorindates, show_raster


show_raster(dem_c[0], dmeta)

# %%
show_raster(cropped2[0], bmeta, cmap="gray")

# %%
T = dmeta["transform"]

T = np.array(T).reshape(3, 3)

asmesh = raster_to_pyvista_mesh(dem_c[0], T)

asmesh.points -= center_coords # recenter the mesh
asmesh.points /=1000


asmesh = asmesh.compute_normals(flip_normals=True)

# %%
mapper = vtk.vtkTextureMapToPlane()
mapper.SetInputData(asmesh)
mapper.Update()



# %%
save_mesh_and_texture_as_obj(of("crater_cropped_dem_textured.obj"), mapper.GetOutput(), cropped2[0])

# %%
