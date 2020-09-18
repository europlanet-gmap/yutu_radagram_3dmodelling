import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import numpy as np

nbfolder = "./notebooks/"

import glob

found = glob.glob(nbfolder + "*.ipynb")
print(found)

import re

numbers = []
matched = []
for f in found:
    m = re.search(".*([0-9])\s.*.ipynb", f)
    if m is not None:
        n = m.group(1)
        numbers.append(n)
        matched.append(f)

print(numbers)
order = np.argsort(numbers)
found = np.array(matched)[order].tolist()
print(found)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

for nbf in found:
    print(f"Executing {nbf}")
    nb = nbformat.read(open(nbf), as_version=4)

    try:
        out = ep.preprocess(nb, {'metadata': {'path': nbfolder}})
    except CellExecutionError:
        msg = 'Error executing the notebook "%s".\n\n' % nbf
        # msg += 'See notebook "%s" for the traceback.' %
        print(msg)
        raise
        break