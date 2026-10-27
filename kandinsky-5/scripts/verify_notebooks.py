from pathlib import Path
import json,sys,time,ast
ROOT=Path(__file__).resolve().parents[1]
import nbformat
from nbclient import NotebookClient
files=sorted((ROOT/'notebooks').glob('*.ipynb'))
if len(sys.argv)>1: files=[p for p in files if p.name in sys.argv[1:]]
previous=json.loads((ROOT/'reports/notebook-validation.json').read_text()) if (ROOT/'reports/notebook-validation.json').exists() else []
results=[]
for path in files:
    nb=nbformat.read(path,as_version=4); nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code': ast.parse(cell.source)
    start=time.perf_counter()
    try:
        NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
        out=ROOT/'reports/executed'/path.name; out.parent.mkdir(exist_ok=True,parents=True)
        nbformat.write(nb,out)
        results.append({'notebook':path.name,'status':'passed','seconds':round(time.perf_counter()-start,2)})
        print('PASS',path.name,flush=True)
    except Exception as error:
        results.append({'notebook':path.name,'status':'failed','error':str(error)})
        print('FAIL',path.name,str(error)[-2000:],flush=True)
        break
if len(sys.argv)>1:
    names={r['notebook'] for r in results}
    results=sorted([r for r in previous if r['notebook'] not in names]+results,key=lambda r:r['notebook'])
(ROOT/'reports/notebook-validation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
sys.exit(1 if any(r['status']=='failed' for r in results) else 0)
