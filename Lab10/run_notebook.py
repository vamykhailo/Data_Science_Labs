import os
os.environ['MPLBACKEND']='Agg'
os.environ['PYTHONIOENCODING']='utf-8'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parent
path=root/'Lab10_Vakhniuk.ipynb'
nb=nbformat.read(path,as_version=4)
client=NotebookClient(nb,timeout=7200,kernel_name='python3',resources={'metadata':{'path':str(root)}})
try:
    with client.setup_kernel():
        for i,c in enumerate(nb.cells):
            if c.cell_type=='code':
                print(f'Running cell {i+1}/{len(nb.cells)}',flush=True)
                client.execute_cell(c,i)
                nbformat.write(nb,path)
finally:
    nbformat.write(nb,path)
print('NOTEBOOK COMPLETED',flush=True)
