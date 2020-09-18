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
from smalllib import ff, of
import shutil

# %%
import attr

@attr.attrs
class ArchivableFile:
    filename = attr.ib()
    extend_to_aux = attr.ib(default=False)
    group = attr.ib(default=None)
    rename_to= attr.ib(default=None)
    is_aux = attr.ib(default=False)
    
    

# %%
files = [ArchivableFile(of("crater_cropped_dem_textured.obj"), True , "model/meshes"),
        ArchivableFile(of("radargram.obj"), True , "model/meshes")]

# %%
for i in [0,1,2]:
    files.append(ArchivableFile(of(f"tube_{i}.obj"), False, "model/meshes", f"plane_topography_intersection_horizon_{i}.obj"))
    
for i in [0,1,2]:
    files.append(ArchivableFile(of(f"large_plane_{i}.obj"), False, "model/meshes", f"modelled_horizon_{i}.obj"))

# %%
vectors = [["prediction.gpkg", "predicted_intersections.gpkg"],
          ["horizons.gpkg","horizons_2.5D.gpkg"],
          ["path.gpkg","yutu_path_gereferenced.gpkg"]]

rasters = [["path_georef.tif", "xiao_georeferenced.tif"]]

# %%
for inf, out in vectors:
    files.append(ArchivableFile(of(inf), False, "vector", out))
    
for inf, out in rasters:
    files.append(ArchivableFile(of(inf), False, "raster", out))
    
    
files.append(ArchivableFile("../package_README.md", False, "", f"README.md"))

# %%
out_folder = "../PM-MOO-D-YutuGPR"
import os
os.makedirs(out_folder, exist_ok=True)


# %%
import glob

# %%
auxes = []

for f in files:
#     print(f.filename)
    
    if f.extend_to_aux:
        basename = os.path.basename(f.filename)
        print(basename)
        path = os.path.dirname(f.filename)
        print(path)
        noextname = os.path.splitext(basename)[0]
        aux = glob.glob(f"{path}/{noextname}.*")
        
        aux.remove(f.filename)
        for au in aux:
            from copy import deepcopy
            newf = deepcopy(f)
            newf.filename =au
            newf.extend_to_aux = False
            newf.is_aux = True
            auxes.append(newf)
            
        print(aux)

        
files += auxes

# %%
for f in files:
    flag = f"AUX: [{f.is_aux}]"
    print(f"Processing {f.filename} {flag}")
    
    if not os.path.exists(f.filename):
        print("File do not exist. aboorting")
        break
        
    outf = out_folder + f"/{f.group}/"
    print(f"oufolder will be {outf}" )
    if f.rename_to is not None:
        outfile = f.rename_to
    else:
        outfile = os.path.basename(f.filename)
        
    print(f"outfile {outfile}")
    os.makedirs(outf, exist_ok=True)
    fullout = os.path.join(outf, outfile)
    
    shutil.copy2(f.filename, fullout)
    print(f"expected output {fullout}\n")
        
    

# %%
bounds = "../images/crater_bounds.gpkg"
import geopandas as gpd
ext = gpd.read_file(bounds)

# %%
aslatlon = ext.to_crs("ESRI:104903")
print(aslatlon.bounds)
asarray = np.array(aslatlon.bounds)[0]
minlon, minlat, maxlon, maxlat = asarray

# %%
if minlon < 0:
    minlon = 360 +minlon
    
if maxlon <0:
    maxlon = 360+maxlon

# %%
print(minlon, minlat, maxlon, maxlat)


# %%
def decdeg2dms(dd):
    mnt,sec = divmod(dd*3600,60)
    deg,mnt = divmod(mnt,60)
    return deg,mnt,sec

def format_degs(d,m,s):
    return f"{int(d)}d{int(m)}" + "'" +"{:.2f}\"".format(s)


# %%
exp = [format_degs(*decdeg2dms(val)) for val in asarray]
[print(ex) for ex in exp];

# %%
cs = ext.crs

# %%
cs.to_proj4()

# %%
import os
import zipfile

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        print(root)
        for file in files:
            fullpath = os.path.join(root, file)
            ziph.write(fullpath, arcname=fullpath[len(out_folder) :])

zipf = zipfile.ZipFile(f"{out_folder}.zip", 'w', zipfile.ZIP_DEFLATED)
zipdir(out_folder, zipf)
zipf.close()
