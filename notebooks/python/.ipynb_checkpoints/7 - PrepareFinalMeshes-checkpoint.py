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
from smalllib import crop_raster_rectangle, raster_to_pyvista_mesh, of, ff

# %%
center_coords = np.loadtxt(of("center_coords.txt"))

# %%
newbounds = gpd.read_file("../images/crater_bounds.gpkg")
newbounds = newbounds.geometry[0]

base ="rasters/NAC_DTM_CHANGE3_M1144922100_160CM.TIF"
basemap = rasterio.open(base)
cropped2, bmeta = crop_raster_rectangle(basemap, newbounds.bounds[:2], newbounds.bounds[2:])


dem = "rasters/NAC_DTM_CHANGE3.TIF"
dem = rasterio.open(dem)
dem_c, dmeta = crop_raster_rectangle(dem, newbounds.bounds[:2], newbounds.bounds[2:])

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

# add the mesh to the 3D view
# bp.add_mesh(asmesh)
asmesh.save("dtm2.vtp")


asmesh = asmesh.compute_normals(flip_normals=True)

# %%

# %%

# %%
import vtk, pyvista


# %%
mapper = vtk.vtkTextureMapToPlane()
mapper.SetInputData(asmesh)
mapper.Update()

# %%

# %%
o = mapper.GetOutput()

tx = pyvista.Texture(cropped2[0])
asim = tx.GetImageDataInput(0)


# %%
import pyvista
pyvista.PolyData(o).save("dtm3.vtp")

# %%
w = vtk.vtkOBJWriter()

w.SetInputData(pyvista.PolyData(o))
w.AddInputDataObject (1,asim)
w.SetFileName("dem.obj")
w.Update()

# %%

# %%
save_mesh_and_texture_as_obj("mydem.obj", asmesh, cropped2[0])

# %%
