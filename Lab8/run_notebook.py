import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(r'C:\Users\Михайло\Data_Science_Labs\Lab8')
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTHONIOENCODING'] = 'utf-8'
path = root / 'Lab8_Vakhniuk.ipynb'
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(root)}})
try:
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                print(f'Running cell {i+1}/{len(nb.cells)}', flush=True)
                client.execute_cell(cell, i)
                nbformat.write(nb, path)
finally:
    nbformat.write(nb, path)
print('NOTEBOOK COMPLETED', flush=True)
