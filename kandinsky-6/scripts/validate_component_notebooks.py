"""Validate K6 teaching labs; --execute uses fresh CPU kernels, no downloads."""
from __future__ import annotations
import argparse,ast,hashlib,json,re,sys,time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import unquote

import nbformat

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--execute',action='store_true')
parser.add_argument('--only',help='Optional notebook stem prefix for a repair run')
args=parser.parse_args()
catalog=json.loads((ROOT/'reports/component-catalog.json').read_text())
rows=[];errors=[]
manifest=json.loads((ROOT/'reference/source-manifest.json').read_text())
for file in manifest['files']:
    path=ROOT/'reference'/file['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=file['sha256']:
        errors.append('Modified pinned source: '+file['path'])
for item in catalog['notebooks']:
    stem=item['stem']
    if args.only and not stem.startswith(args.only):continue
    path=ROOT/'notebooks'/f'{stem}.ipynb'
    nb=nbformat.read(path,as_version=4)
    nbformat.validate(nb)
    row={'notebook':path.name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'cells':len(nb.cells),'code_cells':sum(c.cell_type=='code' for c in nb.cells),
         'executed':False}
    for index,cell in enumerate(nb.cells):
        if cell.cell_type=='code':ast.parse(cell.source,filename=f'{path.name}:{index}')
        if cell.cell_type=='markdown':
            for link in re.findall(r'\]\(([^\s)]+)\)',cell.source):
                if link.startswith(('http:','https:','#','mailto:')):continue
                target=path.parent/unquote(link.split('#')[0])
                if not target.exists():errors.append(f'{path.name}:{index}: missing {link}')
            if cell.source.count('$$')%2:errors.append(f'{path.name}:{index}: unclosed display math')
            if r'\operatorname' in cell.source:errors.append(f'{path.name}:{index}: unsupported math macro')
    if args.execute:
        from jupyter_client import KernelManager
        from nbclient import NotebookClient
        print('EXECUTE',path.name,flush=True)
        manager=KernelManager(kernel_name='python3')
        manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        client=NotebookClient(nb,km=manager,timeout=180,
              resources={'metadata':{'path':str(path.parent)}})
        start=time.perf_counter()
        try:
            client.execute()
            row['executed']=True
            row['elapsed_seconds']=time.perf_counter()-start
            result=ROOT/'artifacts/component_labs'/stem/'metrics.json'
            row['measurements']=json.loads(result.read_text())['metrics']
            out=ROOT/'reports/executed';out.mkdir(parents=True,exist_ok=True)
            nbformat.write(nb,out/path.name)
            print('PASS',path.name,round(row['elapsed_seconds'],2),'s',flush=True)
        except Exception as exc:
            errors.append(f'{path.name}: {type(exc).__name__}: {exc}')
            print('FAIL',path.name,str(exc)[-2500:],flush=True)
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
    rows.append(row)
report={'created_utc':datetime.now(timezone.utc).isoformat(),
        'mode':'execute' if args.execute else 'static','python':sys.version,
        'no_pretrained_weights':True,'comfyui_jobs_submitted':0,
        'pinned_source_hashes_checked':len(manifest['files']),
        'colab_bootstrap_executed':False,
        'helper_sha256':hashlib.sha256((ROOT/'labkit/components.py').read_bytes()).hexdigest(),
        'notebooks':rows,'passed':not errors,'errors':errors}
name='component-validation' if args.execute else 'component-static-validation'
if args.only:name+='-'+args.only
out=ROOT/'reports'/f'{name}.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print('REPORT',out,'PASS' if not errors else 'FAIL',flush=True)
if errors:raise SystemExit(1)
