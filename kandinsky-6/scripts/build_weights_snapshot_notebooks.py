"""Build the two private Kaggle notebooks that save the pinned K6 Lite Diffusers snapshot as notebook output."""
from pathlib import Path
import ast
import json
import textwrap

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'kandinsky-6'
MODEL_ID='kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers'
MODEL_REVISION='4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81'
# Kaggle keeps at most 20 GB of notebook output, so the 24.9 GiB snapshot is split in two parts.
PARTS={
    'media':dict(number=17,title='K6 Lite Diffusers Weights Media',
                 summary='DiT, video VAE, audio VAE, vocoder, CLIP, tokenizers, scheduler (~9.4 GiB)'),
    'text':dict(number=18,title='K6 Lite Diffusers Weights Text',
                summary='Qwen2.5-VL text encoder (~15.5 GiB)'),
}

def slug(title):
    return title.lower().replace(' ','-')

def build_part(part,spec):
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(s).strip()+'\n'))
    def code(s):
        s=textwrap.dedent(s).strip()+'\n';ast.parse(s)
        cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s))
    other=[p for p in PARTS if p!=part][0]
    md(f'''# {spec['title']}

Приватная копия части `{part}` закреплённого Diffusers snapshot [{MODEL_ID}](https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}) @ `{MODEL_REVISION[:8]}`: {spec['summary']}. Kaggle сохраняет не больше 20 GB Output, поэтому snapshot (24,9 GiB) разделён на части `media` и `text`; вторая часть — notebook «{PARTS[other]['title']}».

Запуск: **Save Version → Save & Run All**, GPU не нужен, Internet On. Затем в notebook 16 добавьте обе части через **Add Input → Your Work → Notebook Output**: ячейка конфигурации найдёт их сама и подставит `LOCAL_MODEL_DIR`, повторные загрузки весов в каждой сессии исчезнут.

Лицензии: Kandinsky 6 — MIT, Qwen2.5-VL и CLIP — их собственные лицензии в исходных репозиториях. Notebook и Output приватные.

Учебные материалы отдельно: [MIT 6.S184](https://diffusion.csail.mit.edu/2026/index.html) · [конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Diffusers runner](https://github.com/vladislav-vasilenko/generative-ai-research/tree/main/kandinsky-6/generation).''')
    code(f'''PART={part!r}
MODEL_ID={MODEL_ID!r}
MODEL_REVISION={MODEL_REVISION!r}
OUTPUT_LIMIT_BYTES=19_500_000_000  # Kaggle keeps at most 20 GB of notebook output
''')
    code('''import json, os, shutil, time
from pathlib import Path
os.environ.update(HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_XET='1')
from huggingface_hub import HfApi, snapshot_download

# 'text' is the Qwen2.5-VL encoder alone; 'media' is everything else in the Diffusers layout.
PARTS = {'media': lambda path: not path.startswith('text_encoder/'),
         'text': lambda path: path.startswith('text_encoder/')}
OUT = Path('/kaggle/working/k6-lite-diffusers-'+PART)
files = [f for f in HfApi().list_repo_tree(MODEL_ID, revision=MODEL_REVISION, recursive=True)
         if getattr(f, 'size', None) is not None and PARTS[PART](f.path)]
need = sum(f.size for f in files)
free = shutil.disk_usage('/kaggle/working').free
print(f'{PART}: {len(files)} files, {need/2**30:.2f} GiB; free in /kaggle/working {free/2**30:.1f} GiB', flush=True)
if need > OUTPUT_LIMIT_BYTES:
    raise RuntimeError('This part does not fit into the Kaggle output limit')
if free < need*1.05:
    raise OSError('Not enough free space in /kaggle/working for this part')
''')
    code('''start = time.perf_counter()
snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=OUT, allow_patterns=[f.path for f in files])
download_seconds = time.perf_counter()-start
# snapshot_download keeps download metadata here; it is not part of the Diffusers layout.
shutil.rmtree(OUT/'.cache', ignore_errors=True)
bad = [f.path for f in files if not (OUT/f.path).is_file() or (OUT/f.path).stat().st_size != f.size]
if bad:
    raise RuntimeError(f'Missing or truncated files: {bad}')
manifest = dict(model_id=MODEL_ID, revision=MODEL_REVISION, part=PART, total_bytes=need,
    download_seconds=round(download_seconds, 1), files={f.path: f.size for f in files},
    lfs_sha256={f.path: f.lfs.sha256 for f in files if getattr(f, 'lfs', None)})
(OUT/'weights-manifest.json').write_text(json.dumps(manifest, indent=2)+'\\n')
print(f'Saved {len(files)} files, {need/2**30:.2f} GiB in {download_seconds:.0f} s to {OUT}')
''')
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),kaggle=dict(accelerator='none',isInternetEnabled=True),
        k6_deployment=dict(portable_navigation=True,target='kaggle_cpu_internet',
            code_layout='classic_python_cells',backend='huggingface_hub',full_run_verified=False)))
    for i,c in enumerate(cells):c['id']=f'k6-weights-{part}-{i:03d}'
    name=f"{spec['number']}_kaggle_k6_weights_{part}.ipynb"
    (BASE/'notebooks'/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    folder=BASE/'generation/weights'/part
    folder.mkdir(parents=True,exist_ok=True)
    meta=dict(id='applehawk/'+slug(spec['title']),title=spec['title'],code_file=f'../../../notebooks/{name}',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=True,
        dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Built',name,'->',meta['id'])

if __name__=='__main__':
    for part,spec in PARTS.items():build_part(part,spec)
