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

import sys
if sys.version_info[0] >2:
    unicode = str

import numpy as np
import scipy.spatial
from scipy.special import xlogy
import scipy.linalg


class RadialBasisInterpolation:
    """
    Augmented Radial Basis function interpolation with (optional) polynomial 
    augmentation. The construction follows [1] with code borrowed from [2] 
    for the kernel (which they call the "function")
    
    Inputs
    ------
    X (array like)
        (N,ndim) array or (N,) vector of build points
        
    f (array like)
        (N,) vector of function values
    
    Options: 
    --------
    degree [0]
        Use an augmented RBF. Specify the *total* degree fit if given as an 
        integer. If given as a matrix, must be (P+1,ndim) shape. This is the 
        degree of the augmenting polynomial.
    
    kernel : ['gaussian'] 
        The radial basis kernel, based on the radius, r
        If string:
            'multiquadric': sqrt((r/epsilon)**2 + 1)
            'inverse': 1.0/sqrt((r/epsilon)**2 + 1)
            'gaussian': exp(-(r/epsilon)**2)
            'linear': r
            'cubic': r**3
            'quintic': r**5
            'thin_plate': r**2 * log(r)
        If callable
            must take the arguments (r,epsilon) and ignore epsilon if 
            not needed. Tip: If epsilon is not needed, set it to
            0 to avoid trying to optimize it
        If integer:
            Spline of the form: r**kernel if kernel is odd else r**kernel*log(r)

    epsilon [None]
        Adjustable parameter for some kernels.
        
    smooth : float, optional
        Values greater than zero increase the smoothness of the
        approximation.  0 is for interpolation (default). Actually set to
        max(smooth,1e-10 * ||f||) for numerical precision

    Theory:
    ------
    The RBF is based on a radial distance 
    
        r_j = ||x - x_j||                                               (1)
    
    with the kernel phi(r) and an optional "Agumenting" polynomial of set 
    degree. The interpolant is of the form: [1]
    
        s(x) = sum_{j=1}^N c_j * phi(r_j) + sum_{k \in P} g_k * p_k     (2)
    
    where `P` is a polynomial index and `p_k` is the corresponding
    polynomial (e.g. k = {1,2,0} then p_k = x*y^2*z^0).
    
    In order to solve for c_j and g_k we enforce 
        (a) s(x_j) = f_j for all j (i.e. exact interpolation)
        (b) sum_{j=1}^N c_j*p_k(x_j) = 0 for all k=1,...,N 
    
    That results in a matrix system:
        ____________________________________________________   ___     ___
       |                            |                       | |   |   |   |
       |   phi(||x_i - x_j||) (N,N) |   p_k(x_j) (N,len(K)) | | c |   | f |
       |                            |                       | |   | = |   | (3)
       |----------------------------|-----------------------| |---|   |---|
       |       p_k(x_j).T           |           0           | | g |   | 0 |
        ----------------------------------------------------   ---     ---
      
    which is then solved. Given c and g, (2) can be used to solve for
    any given x
    
    References:
    ----------  
    [1] G. B. Wright. Radial Basis Function Interpolation: Numerical and 
        Analytical Developments. PhD thesis, University of Colorado, 2003.

    [2] SciPy version 1.1.0
        https://github.com/scipy/scipy/blob/v1.1.0/scipy/interpolate/rbf.py
    
    [3] S. Rippa. An algorithm for selecting a good value for the parameter 
        c in radial basis function interpolation. Advances in Computational 
        Mathematics, 11(2-3):193–210, 1999.
        
    [4] J. D. Martin and T. W. Simpson. Use of Kriging Models to Approximate 
        Deterministic Computer Models. AIAA journal, 43(4):853–863, 2005.

    """
    def __init__(self,X,f,degree=0,
                 epsilon=1,smooth=0,
                 kernel='gaussian',_solve=True):
        self.X = X = np.atleast_2d(X)
        self.N,self.ndim = X.shape
        self.f = f = np.ravel(f)
        self.degree = degree
        self.kernel = kernel
        self.epsilon = epsilon
        
        
        r = scipy.spatial.distance.cdist(X,X)

        self.smooth = max(float(smooth),1e-10*scipy.linalg.norm(f)/np.sqrt(len(f))) # Scale by ||f||
        
        K = self._kernel(r) + np.eye(self.N)*self.smooth
        P = RadialBasisInterpolation.vandermond(X,degree=self.degree)
        
        # Build the matrix
        Z = np.zeros([P.shape[1]]*2)
        z = np.zeros(P.shape[1])
        KP = np.block([[K   , P],
                       [P.T , Z]])
        b = np.hstack([f,z])
        coef = scipy.linalg.solve(KP,b)
        
        self.A = KP
        self.b = b
        self.rbf_coef = coef[:self.N]
        self.poly_coef = coef[self.N:]
        

    def __call__(self,X):
        X = np.atleast_2d(X)
        K = self._kernel( scipy.spatial.distance.cdist(X,self.X) )
        P = RadialBasisInterpolation.vandermond(X,degree=self.degree)
        return K.dot(self.rbf_coef) + P.dot(self.poly_coef)

    def _kernel(self,r):
        r = np.asarray(r)
        if callable(self.kernel):
            return self.kernel(r,self.epsilon)
        elif isinstance(self.kernel,(int,np.integer)):
            if np.mod(self.kernel,2) == 0:
                return xlogy(r**self.kernel,r)
            else:
                return r**self.kernel
        elif not isinstance(self.kernel,(str,unicode)):
            raise ValueError('Kernel must be a callable with signature (r,epsilon), an integer (spline) or a valid string')
        
        kernel = self.kernel.lower().replace(' ','_').replace('-','_')
        
        if kernel in ['multiquadric']:
            return np.sqrt((r/self.epsilon)**2 + 1)
        elif kernel in ['inverse_multiquadric','inverse']:
            return 1.0/np.sqrt((1.0/self.epsilon*r)**2 + 1)
        elif kernel in ['gaussian','squared_exponential']:
            return np.exp(-(1.0/self.epsilon*r)**2)
        elif kernel in ['linear']:
            return r
        elif kernel in ['cubic']:
            return r**3
        elif kernel in ['quintic']:
            return r**5
        elif kernel in ['thin_plate']:
            return xlogy(r**2, r)
        else:
            raise ValueError('Not valid kernel name')
    
    @staticmethod
    def vandermond(X,degree):
        """
        Return a Vandermond matrix of X up to the
        *total* order degree
        """
        X = np.atleast_2d(X)
        ndim = X.shape[1]
        index = np.asarray(RadialBasisInterpolation.total_index(degree,ndim),dtype=int)
        V = np.ones([X.shape[0],len(index)])
        for d in range(ndim):
            v = np.fliplr(np.vander(X[:,d],N=degree+1))
            V *= v[:,index[:,d]]
        return V

    
    @staticmethod
    def total_index(P,ndim,sort=True):
        P = tuple([P]*ndim)
        curr = [0]*ndim
        ind = []
        ind.append(tuple(curr))
        while True:
            for d in range(ndim):
                if sum(curr) < P[0]:
                    curr[d] += 1
                    break
                else:
                    curr[d] = 0
            else:
                break
            ind.append(tuple(curr))
        if sort:
            ind.sort(key=lambda a:(sum(a),(np.array(a)**2).sum(),a[::-1]))
        return ind