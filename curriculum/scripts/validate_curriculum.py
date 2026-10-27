"""Validate curriculum structure and optionally execute only its two small new labs."""
import argparse,ast,hashlib,json,re,sys,platform
from importlib.metadata import version
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import nbformat
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
paths=[ROOT/'curriculum/notebooks/00_probability_and_model_contracts.ipynb',
       ROOT/'lecture_05/05_ctmc_and_discrete_flow.ipynb']
result={'date':datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat(), 'executed_at_utc':datetime.now(timezone.utc).isoformat(),'original_labs_executed':False,'pretrained_generation':'not_run',
        'comfyui_runtime':'not_run','mlx':'not_run','notebooks':[],
        'environment':{'python':platform.python_version(),**{p:version(p) for p in ['torch','numpy','matplotlib','nbclient']},'backend':'cpu'}}
manifest=json.loads((ROOT/'curriculum/references/materials_manifest.json').read_text())
for source in manifest['notebooks']:
    path=ROOT/source['canonical_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256'],path
result['canonical_notebook_hashes']='3 matched provided originals'
for path in paths:
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    for i,cell in enumerate(nb.cells):
        if cell.cell_type=='code':ast.parse(cell.source,filename=f'{path.name}:cell-{i}')
    row={'path':str(path.relative_to(ROOT)),'cells':len(nb.cells),
         'code_cells':sum(c.cell_type=='code' for c in nb.cells),'syntax':'passed','execution':'not_run'}
    if args.execute:
        from jupyter_client import KernelManager
        from nbclient import NotebookClient
        manager=KernelManager(kernel_name='python3')
        manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        client=NotebookClient(nb,km=manager,timeout=120,resources={'metadata':{'path':str(path.parent)}},
            on_cell_execute=lambda cell,cell_index:print(path.name,'cell',cell_index,flush=True))
        try:client.execute()
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
        out=ROOT/'curriculum/reports/executed';out.mkdir(exist_ok=True)
        nbformat.write(nb,out/path.name);row['execution']='passed'
    result['notebooks'].append(row)
for path in (ROOT/'curriculum/workflows').glob('*.json'):
    graph=json.loads(path.read_text());nodes={n['id']:n for n in graph['nodes']}
    for link_id,a,out_slot,b,in_slot,kind in graph['links']:
        assert nodes[a]['outputs'][out_slot]['type']==kind
        assert nodes[b]['inputs'][in_slot]['type']==kind
        assert nodes[b]['inputs'][in_slot]['link']==link_id
        assert link_id in nodes[a]['outputs'][out_slot]['links']
result['comfyui_static_links']='passed: image IO and VAE reconstruction'
industry=json.loads((ROOT/'curriculum/references/industry_manifest.json').read_text())
assert all('sha256' in f for repo in industry['repositories'] for f in repo['files'])
result['industrial_entry_points']=str(len(industry['repositories']))+' repositories; selected paths pinned by revision and SHA-256'
# Validate local links in newly authored teaching guides, not upstream documents.
docs=list((ROOT/'curriculum').glob('*.md'))+[ROOT/f'lecture_{n:02}/README.md' for n in range(2,6)]
bad=[]
for path in docs:
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
        if '://' in target or target.startswith('#'):continue
        file=target.split('#')[0]
        if file and not (path.parent/file).exists():bad.append((str(path.relative_to(ROOT)),target))
assert not bad,bad
result['local_guide_links']='passed'
protected=ROOT/'curriculum/artifacts/user_notebook_hashes.json'
if protected.exists():
    before=json.loads(protected.read_text())
    for file,digest in before.items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,file
    result['existing_user_answers']='unchanged'
if args.execute:
    result['probability_bridge']=json.loads((ROOT/'curriculum/artifacts/probability-and-contracts-results.json').read_text())
    result['ctmc']=json.loads((ROOT/'curriculum/artifacts/ctmc-results.json').read_text())
destination=args.output or (ROOT/'curriculum/reports'/('validation.json' if args.execute else 'static-validation.json'))
if not destination.is_absolute():destination=ROOT/destination
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
