import shapefile, rasterio
import numpy as np
import os
def of(filename):
    fulldir = f"../output/"
    os.makedirs(fulldir, exist_ok=True)
    return os.path.abspath(os.path.join(fulldir, filename))

def ff(filename):
    fulldir = f"../figures/"
    os.makedirs(fulldir, exist_ok=True)
    return os.path.abspath(os.path.join(fulldir, filename))

def loadCoordsOfPoints(shp):
    r = shapefile.Reader(shp)
    shapes = r.shapes()

    coords = []

    for shape in shapes:
        coords.append(shape.points[0])
    coords = np.array(coords)
    
    return coords

def sampleRasterAtPoints(rasterio_dataset, points):
    values = np.array(list(rasterio_dataset.sample(points))).T[0]
    return values

def getBoundingBox(pts):
    return [np.min(pts, axis=0), np.max(pts, axis=0)]

# mostly from https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
from rasterio import mask
from shapely.geometry import Polygon

def crop_raster_rectangle(raster:rasterio.io.DatasetReader, mins, maxs, outfile="raster.tif"):
    region = Polygon.from_bounds(*mins, *maxs) # the region as a polygon
    out_img, out_transform = mask.mask(dataset=raster, shapes=[region], crop=True, filled=False, pad=True) # do the crop
    
    # update the metadata and save it
    out_meta = raster.meta.copy()
    mycrs = raster.crs
    out_meta.update({"driver": "GTiff",
                         "height": out_img.shape[1],
                         "width": out_img.shape[2],
                         "transform": out_transform,
                         "crs": mycrs}
                                 )

    with rasterio.open(outfile, "w", **out_meta) as dest:
        dest.write(out_img)
        
    return out_img, out_meta

from rasterio.enums import Resampling

def upsample_raster(raster, out_file=None, upscale_factor=5):
    out_meta = raster.meta.copy()
    # resample data to target shape
    data = raster.read(
            out_shape=(
                raster.count,
                int(raster.height * upscale_factor),
                int(raster.width * upscale_factor)
            ),
            resampling=Resampling.cubic_spline
        )

        # scale image transform
    transform = raster.transform * raster.transform.scale(
            (raster.width / data.shape[-1]),
            (raster.height / data.shape[-2])
        )
    
    # update metadata
    out_meta.update({"height": data.shape[1], "width": data.shape[2], "transform": transform})
    
    # finally write the cropped, upscaled dem, if needed
    if out_file is not None:
        with rasterio.open(out_file, "w", **out_meta) as dest:
            dest.write(data)
        
    return data, out_meta


# this method will generate the faces for a triangular mesh for the DEM
def generate_faces(Nr,Nc):
    # directly from https://stackoverflow.com/questions/44934631/making-grid-triangular-mesh-quickly-with-numpy
    out = np.empty((Nr-1,Nc-1,2,3),dtype=int)

    r = np.arange(Nr*Nc).reshape(Nr,Nc)

    out[:,:, 0,0] = r[:-1,:-1]
    out[:,:, 1,0] = r[:-1,1:]
    out[:,:, 0,1] = r[:-1,1:]

    out[:,:, 1,1] = r[1:,1:]
    out[:,:, :,2] = r[1:,:-1,None]

    out.shape =(-1,3)
    return out

def resample_linestring(linestring, sample_step=0.1): # simple method for upsampling a linestring
    points = []
    for d in np.r_[0: linestring.length: sample_step]:
        points.append(linestring.interpolate(d))
    return points

import pyvista
# transform points to a vtkPolyData with lines
def polyline_from_points(points):
    "from https://docs.pyvista.org/examples/00-load/create-spline.html#sphx-glr-examples-00-load-create-spline-py"
    poly = pyvista.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly

from copy import deepcopy
import pyvista as pvs

def generate_plane_quads(pts, depth=1):
    """
    generates a quad-based mesh to represent the radargram in the 3D scene
    """
    pts2 = deepcopy(pts)
    pts2[:,2] -= depth
    verts = np.ascontiguousarray(np.row_stack([pts, pts2])).astype(np.float)
    n = len(pts)
    cells = []
    for i in np.arange(n-1):
        cell = [4, i,  i+n, i+n+1, i+1 ]
        cells.append(cell)
    
    cells = np.array(cells)
    cells = np.ascontiguousarray(cells.astype(np.int16).ravel())
    dd = pvs.PolyData(verts, cells)
    return dd, verts, cells

from shapely.geometry import Point
def generate_uv(fullline, sline):
    """this will generate the UV coordinates for a linestring fulline
       for the points given in sline. it is meant to work together with generate_plane_quads()
       UV coordinaes are normalized coordinate 0->1 on both axis
       the origin of the reference for UV coordinates are lower left corner of the image
    """
    u = []
    for p in sline:
        a = fullline.project(Point(p), normalized=1) # normalized distance along the fulline path
        u.append(a)

    u = np.array(u)

    v= np.ones(len(u))

    uv = np.column_stack([u,v])
    uv2 = np.column_stack([u, np.zeros(len(v))])
    uv = np.row_stack([uv, uv2])

    return uv

def raster_to_pyvista_mesh(arr, affine_2d=None):
    """affine_2d is a 3x3 matrix of the augmented affine 2d transform to apply before producing the output"""
    x= np.arange(arr.shape[1]) + 0.5 # pixel on the the x-axis, shape[1] is the number of columns
    y= np.arange(arr.shape[0]) + 0.5 # pixels on the y-axis, shape[0] is the number of rows
    X,Y = np.meshgrid(x,y) # compose the grid
    
    pts = np.column_stack([X.ravel(), Y.ravel(), np.ones(X.size)]) #  compose the list of augmented pixel coordinates: [x,y,1] for each pixel
    if affine_2d is not None:
        pts = pts.dot(affine_2d.T) # a fancy way of transforming a list of points at once
        
    pts[:,2] = arr.ravel() # we set the 3d coordiate to correspond to z-elevations as stored  
    
    faces  =  generate_faces(*arr.shape) # compute the faces
    faces = np.column_stack([np.ones(len(faces)) * 3, faces ]).ravel() # basically add the multiplicity of the face (3 in this case) in front of each face row.
    
    asmesh = pyvista.PolyData(pts,  faces) # set up a polydata from the points and the faces
    asmesh.point_arrays["elevation"] =pts[:,2] # add the elevation scalar field
    return asmesh

import networkx as nx
def extract_lines(vtp_data):
    edges = vtp_data.lines.reshape(-1,3)
    g = nx.Graph() # create the empty graph

    for e in edges: # fill the edges
        g.add_edge(*e[1:])
    
    
    ccs = list(nx.connected_components(g)) # separate multiple lines
    
    lines = []
    for cc in ccs:
        subg = g.subgraph(cc) # get the pertinent subgraph
        leaves = [] 
        for id, n in enumerate(subg.nodes): # find the leaves
            a = len(list(g.neighbors(n)))
            if a ==1:
                leaves.append(id)

        if len(leaves) != 2:
            print("subgraph with more than 2 leaves")

        path = nx.shortest_path(subg, leaves[0], leaves[1]) # find path between leaves
        lines.append(path) # append the found line
    return lines

def generate_pixel_coordinates(image:np.ndarray):
    x = np.arange(image.shape[1]) + 0.5
    y = np.arange(image.shape[0]) + 0.5
    X,Y = np.meshgrid(x,y)
    return X, Y
    
def transform_pixel_coorindates(X:np.ndarray,Y:np.ndarray, AffineT, p_0:np.ndarray = None):
    pts = np.column_stack([X.ravel(), Y.ravel()])
    pts = np.array([np.array(AffineT*p)   for p in pts])
    if p_0 is not None:
        pts -= p_0[:2]

    newX, newY = pts[:,0].reshape(X.shape),pts[:,1].reshape(X.shape)
    return newX, newY
import vtk
def save_mesh_and_texture_as_obj(file, mesh:vtk.vtkPolyData, texture_arr:np.ndarray):
    
    tx = pyvista.Texture(texture_arr)
    asim = tx.GetImageDataInput(0)
    
    w = vtk.vtkOBJWriter()

    w.SetInputData(pyvista.PolyData(mesh))
    w.AddInputDataObject (1,asim)
    w.SetFileName(file)
    w.Update()
    
def save_as_obj(file, vtp):
    w = vtk.vtkOBJWriter()
    w.SetInputData(vtp)
    w.SetFileName(file)
    w.Update()

from matplotlib.pyplot import pcolormesh
def show_raster(array, meta, *args, **kwargs):
    X,Y = generate_pixel_coordinates(array)
    Xt, Yt = transform_pixel_coorindates(X,Y, meta["transform"])
    pcolormesh(Xt, Yt, array,*args, **kwargs)

