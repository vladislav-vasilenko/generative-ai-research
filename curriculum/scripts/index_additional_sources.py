"""Pin supplementary reading/code metadata, never install or execute downloaded code."""
import ast, concurrent.futures, hashlib, html, json, re, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'curriculum/artifacts/additional_sources';OUT.mkdir(parents=True,exist_ok=True)
REPOS={
 'lucidrains/rectified-flow-pytorch':['rectified_flow_pytorch/rectified_flow.py','rectified_flow_pytorch/reflow.py','pyproject.toml'],
 'facebookresearch/flow_matching':['flow_matching/path/affine.py','flow_matching/path/scheduler/scheduler.py','flow_matching/solver/ode_solver.py','flow_matching/path/mixture.py'],
 'gnobitab/RectifiedFlow':['ImageGeneration/sde_lib.py','ImageGeneration/sampling.py']}
BLOGS=['https://lilianweng.github.io/posts/2021-07-11-diffusion-models/',
       'https://lilianweng.github.io/posts/2024-04-12-diffusion-video/']
def get(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'generative-ai-research-reading-index'}),timeout=60) as response:return response.read()
def repo_record(item):
 repo,wanted=item
 rev=json.loads(get(f'https://api.github.com/repos/{repo}/commits/main'))['sha']
 tree=json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1'))['tree']
 paths={x['path'] for x in tree if x['type']=='blob'}
 if not wanted:
  wanted=sorted(p for p in paths if p.endswith('.py') and ('rectified_flow' in p.lower() or p.endswith('/sde_lib.py')))[:2]
 selected=['README.md',*wanted];records=[]
 for path in selected:
  if path not in paths:
   records.append({'path':path,'status':'not_found'});continue
  raw=get(f'https://raw.githubusercontent.com/{repo}/{rev}/{path}')
  target=OUT/repo.replace('/','--')/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  row={'path':path,'sha256':hashlib.sha256(raw).hexdigest(),'url':f'https://github.com/{repo}/blob/{rev}/{path}'}
  if path.endswith('.py'):row['symbols']=[n.name for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
  records.append(row)
 return {'repository':repo,'revision':rev,'files':records}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:repositories=list(pool.map(repo_record,REPOS.items()))
blogs=[]
for url in BLOGS:
 raw=get(url);text=raw.decode();target=OUT/(url.rstrip('/').rsplit('/',1)[-1]+'.html');target.write_bytes(raw)
 headings=[]
 for level,attrs,body in re.findall(r'<h([1-4])([^>]*)>(.*?)</h\1>',text,re.S):
  label=html.unescape(re.sub('<[^>]+>','',body)).strip().rstrip('#').strip()
  anchor=re.search(r'id=["\x27]([^"\x27]+)',attrs)
  headings.append({'heading':label,**({'anchor':anchor.group(1)} if anchor else {})})
 blogs.append({'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'role':'secondary explanatory reading; technical claims checked against primary papers/code','headings':headings})
manifest={'checked_date':datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat(),'checked_at_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'text and metadata only; no package install, model download or upstream code execution',
 'repositories':repositories,'articles':blogs,
 'primary_papers':['https://arxiv.org/abs/2006.11239','https://arxiv.org/abs/2202.00512','https://arxiv.org/abs/2204.03458','https://arxiv.org/abs/2209.03003','https://arxiv.org/abs/2210.02747','https://arxiv.org/abs/2412.06264']}
p=ROOT/'curriculum/references/additional_sources.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for row in repositories:
 print(row['repository'],row['revision'])
 for f in row['files']:print(' ',f['path'],f.get('status','OK'),f.get('symbols',[])[:6])
for row in blogs:print(row['url'],len(row['headings']),'headings')
