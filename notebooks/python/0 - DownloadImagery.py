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
import requests, os

# %% [markdown]
# This will download the needed rasters into the raster folder of the repo root.

# %%
out_folder = "../rasters/"

urls = ["http://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/CHANGE3/NAC_DTM_CHANGE3.TIF",
        "http://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/BROWSE/NAC_DTM/CHANGE3/NAC_DTM_CHANGE3_M1144922100_160CM.TIF",
        "http://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/BROWSE/NAC_DTM/CHANGE3/NAC_DTM_CHANGE3_M1144936321_160CM.TIF"]


# %%
def downloadFile(url, outfolder, skip_if_exists=True):
    fname = os.path.basename(url)
    outname  = os.path.join(outfolder, fname)
    
    if skip_if_exists:
        if os.path.exists(outname) :
            return fname
        
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
            
    r = requests.get(url)
        
    with open(outname, 'wb') as f:
        f.write(r.content)
        
    return fname
    
    

# %%
# This may take a while
for url in urls:
    downloadFile(url, outfolder=out_folder,skip_if_exists= True)

# %%
os.makedirs("../output", exist_ok=True)
os.makedirs("../figures", exist_ok=True)

# %%
