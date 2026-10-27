"""Pin public architecture entry points. Downloads text/metadata only, never weights."""
import ast, concurrent.futures, hashlib, json, urllib.request
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[2]
REPOS = {
'Tencent-Hunyuan/HunyuanVideo': ['hyvideo/modules/models.py','hyvideo/vae/__init__.py','hyvideo/diffusion/schedulers/scheduling_flow_match_discrete.py'],
'Wan-Video/Wan2.1': ['wan/modules/model.py','wan/modules/vae.py','wan/modules/t5.py','wan/text2video.py','wan/utils/fm_solvers.py'],
'Wan-Video/Wan2.2': ['wan/modules/model.py','wan/modules/vae2_2.py','wan/text2video.py','wan/configs/wan_ti2v_5B.py','wan/configs/wan_t2v_A14B.py'],
'kandinskylab/kandinsky-5': ['kandinsky/models/dit.py','kandinsky/models/vae.py','kandinsky/models/text_embedders.py','kandinsky/generation_utils.py'],
'kandinskylab/kandinsky-6': ['kandinsky/core/components/dit.py','kandinsky/core/components/piflow_dit.py','kandinsky/core/algo/piflow_sampler.py','kandinsky/core/algo/flow_matching.py'],
'Lightricks/LTX-Video': ['ltx_video/models/autoencoders/causal_video_autoencoder.py','ltx_video/models/transformers/transformer3d.py','ltx_video/schedulers/rf.py'],
'genmoai/mochi': ['src/genmo/mochi_preview/dit/joint_model/asymm_models_joint.py','src/genmo/mochi_preview/vae/models.py','src/genmo/mochi_preview/pipelines.py'],
'hpcaitech/Open-Sora': ['opensora/models/mmdit/model.py','opensora/models/hunyuan_vae/autoencoder_kl_causal_3d.py','opensora/utils/sampling.py']}
def get(url):
    request = urllib.request.Request(url, headers={'User-Agent':'generative-ai-research-curriculum'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()
def inspect(item):
    repo, wanted = item
    revision = json.loads(get(f'https://api.github.com/repos/{repo}/commits/main'))['sha']
    tree = json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1'))
    available = {row['path'] for row in tree['tree'] if row['type']=='blob'}
    out = ROOT / 'curriculum/artifacts/source_analysis/industry' / repo.replace('/','--')
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for path in ['README.md',*wanted]:
        if path not in available:
            files.append({'path':path,'status':'not_found_in_revision'})
            continue
        raw = get(f'https://raw.githubusercontent.com/{repo}/{revision}/{path}')
        target = out / path; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
        row = {'path':path,'sha256':hashlib.sha256(raw).hexdigest(),
               'url':f'https://github.com/{repo}/blob/{revision}/{path}'}
        if path.endswith('.py'):
            row['symbols'] = [node.name for node in ast.parse(raw).body if isinstance(node,(ast.ClassDef,ast.FunctionDef))]
        files.append(row)
    return {'repository':repo,'revision':revision,'url':'https://github.com/'+repo,'files':files}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    rows = list(pool.map(inspect,REPOS.items()))
manifest = {'checked_date':datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat(), 'checked_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'public code/docs only; no model download or execution','repositories':rows}
output = ROOT / 'curriculum/references/industry_manifest.json'
output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for row in rows:
    print(row['repository'],row['revision'])
    for f in row['files']:
        print(' ',f['path'],f.get('status','OK'),', '.join(f.get('symbols',[])[:8]))
