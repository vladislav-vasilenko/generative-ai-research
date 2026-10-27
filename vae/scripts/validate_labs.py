"""Check all Hunyuan VAE labs; --execute runs fresh CPU kernels, no weights/downloads."""
import argparse,ast,hashlib,json,re,sys,time
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import unquote
import nbformat
C=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(C))
from labkit.vae_lab import source_hash
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--only');a=p.parse_args()
rows=[];errors=[]
for path in sorted((C/'notebooks').glob('*.ipynb')):
    if a.only and not path.stem.startswith(a.only):continue
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    row={'notebook':path.name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'cells':len(nb.cells),'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed':False}
    for i,cell in enumerate(nb.cells):
        if cell.cell_type=='code':ast.parse(cell.source,filename=f'{path.name}:{i}')
        if cell.cell_type=='markdown':
            if cell.source.count('$$')%2:errors.append(f'{path.name}:{i}: unclosed display math')
            if r'\operatorname' in cell.source:errors.append(f'{path.name}:{i}: unsupported operatorname')
            for link in re.findall(r'\]\(([^\s)]+)\)',cell.source):
                if link.startswith(('https:','http:','#','mailto:')):continue
                if not (path.parent/unquote(link.split('#')[0])).exists():errors.append(f'{path.name}:{i}: broken link: {link}')
    if a.execute:
        from jupyter_client import KernelManager
        from nbclient import NotebookClient
        km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        start=time.perf_counter();print('EXECUTE',path.name,flush=True)
        try:
            NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(path.parent)}}).execute()
            row.update(executed=True,elapsed_seconds=time.perf_counter()-start,
                       measurements=json.loads((C/'artifacts'/path.stem/'metrics.json').read_text()))
            out=C/'artifacts/executed';out.mkdir(parents=True,exist_ok=True);nbformat.write(nb,out/path.name)
            print('PASS',path.name,round(row['elapsed_seconds'],2),'s',flush=True)
        except Exception as exc:
            errors.append(f'{path.name}: {type(exc).__name__}: {exc}');print('FAIL',str(exc)[-3500:],flush=True)
        finally:
            if km.has_kernel:km.shutdown_kernel(now=True)
    rows.append(row)
report={'created_utc':datetime.now(timezone.utc).isoformat(),'mode':'execute' if a.execute else 'static',
        'python':sys.version,'pinned_source_sha256':source_hash(),
        'helper_sha256':hashlib.sha256((C/'labkit/vae_lab.py').read_bytes()).hexdigest(),
        'notebooks':rows,'passed':not errors,'errors':errors,'pretrained_weights_loaded':False,
        'GPU_peak_measured':False,'comfyui_jobs_submitted':0}
name='execution' if a.execute else 'static'
if a.only:name+='-'+a.only
(C/'reports'/f'{name}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print('PASS' if not errors else 'FAIL',len(rows),'notebooks',len(errors),'errors')
if errors:raise SystemExit(1)
