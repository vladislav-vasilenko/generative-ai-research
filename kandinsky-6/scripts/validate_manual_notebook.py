"""Validate notebook cells; --execute runs only its default lightweight route."""
import argparse
import ast
from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--execute', action='store_true')
args = parser.parse_args()
path = ROOT / 'notebooks/00_manual_experiment_comfy_mlx_film.ipynb'
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == 'code':
        ast.parse(cell.source, filename=f'{path.name}:cell-{index}')
print('Schema and Python syntax passed:', len(notebook.cells), 'cells', flush=True)
if args.execute:
    from jupyter_client import KernelManager
    from nbclient import NotebookClient
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=120,
                            resources={'metadata': {'path': str(path.parent)}},
                            on_cell_execute=lambda cell, cell_index: print('Cell', cell_index, flush=True))
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    out = ROOT / 'reports/executed'
    out.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, out / path.name)
    print('Executed successfully. No model downloads, pretrained generation or voice training.')
    print('Readiness and educational measurements:',
          ROOT / 'artifacts/manual_experiment/manual-readiness-and-educational-results.json')
