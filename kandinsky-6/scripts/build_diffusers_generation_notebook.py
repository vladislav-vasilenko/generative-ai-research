"""Build the standalone classic-cell Kaggle Diffusers T2V/I2V runner; no lessons or code-string payloads."""
from pathlib import Path
import ast
import json
import pprint
import textwrap

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'kandinsky-6'
PUBLIC='https://www.kaggle.com/code/applehawk/kandinsky-6-lite-diffusers-generation'

def build():
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(s).strip()+'\n'))
    def code(s):
        s=textwrap.dedent(s).strip()+'\n';ast.parse(s)
        cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s))
    md('''# Kandinsky 6 Lite — Diffusers Generation · T2V / I2V

Практический runner без ComfyUI и учебных разделов. GPU **T4 ×2**, Internet ON. Выберите preset / Smoke или Baseline, для I2V укажите reference из Kaggle Input; выполняйте ячейки по порядку.

Smoke: 25 сгенерированных → 24 экспортированных кадра, 1 s. Baseline: 121 → 120 кадров, 5 s, 24 FPS. Путь подготовлен по официальному API; полный pretrained запуск и two-GPU speedup ещё не измерены.

[Diffusers API](https://huggingface.co/docs/diffusers/en/api/pipelines/kandinsky6) · [Lite checkpoint](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers) · [README и presets](https://github.com/vladislav-vasilenko/generative-ai-research/tree/main/kandinsky-6/generation) · [Сохранённый вариант ComfyUI](https://www.kaggle.com/code/applehawk/15-kaggle-t4-lite-distill).

Учебные материалы отдельно: [MIT 6.S184](https://diffusion.csail.mit.edu/2026/index.html) · [конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [оглавление проекта](https://github.com/vladislav-vasilenko/generative-ai-research/blob/main/curriculum/CONTENTS.md).''')
    md('''## 1. Среда

В редакторе Kaggle включите **Internet → On**. Если `pypi.org` не разрешается, проверьте переключатель; при уже включённом Internet возможен временный DNS-сбой. Ячейка проверяет доступ к PyPI до pip и пропускает установку, когда версии уже подходят. Pillow не понижается: Pillow 13 из образа Kaggle подходит, поэтому на свежей сессии перезапуск ядра не нужен и «Save & Run All» проходит. pip может предупредить, что неиспользуемый здесь Gradio требует `pillow<13`; на запуск это не влияет. Если всё же изменилась уже импортированная библиотека, потребуется Restart Python kernel, затем повтор с первой ячейки.''')
    code('''import os, sys, subprocess, socket
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from packaging.requirements import Requirement
import torch

if not Path('/kaggle/working').is_dir() or torch.cuda.device_count() < 2:
    raise RuntimeError('Run in Kaggle with GPU T4 ×2. No installation or model downloads started.')
for i in range(2):
    print(i, torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory/2**30,2),'GiB')
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda)
SCRATCH=Path('/kaggle/temp/k6-diffusers')
RESULTS=Path('/kaggle/working/k6-diffusers-results')
SCRATCH.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True)
os.environ.update(HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',TOKENIZERS_PARALLELISM='false')
pins={'diffusers':'0.42.1','transformers':'5.22.1','accelerate':'1.13.0'}


def pending_dependencies(requirements, version_lookup=version):
    missing = []
    for item in requirements:
        requirement = Requirement(item)
        try:
            installed = version_lookup(requirement.name)
        except PackageNotFoundError:
            missing.append(item)
            continue
        if installed not in requirement.specifier:
            missing.append(item)
    return missing


def check_pypi_access():
    url = 'https://pypi.org/simple/diffusers/'
    try:
        with urlopen(Request(url, method='HEAD'), timeout=10):
            pass
    except HTTPError as error:
        raise RuntimeError(f'PyPI responds with HTTP {error.code}. Check Kaggle Internet access; pip has not started.') from error
    except URLError as error:
        cause = ('DNS cannot resolve pypi.org' if isinstance(error.reason, socket.gaierror)
                 else f'HTTPS connection failed: {error.reason}')
        raise RuntimeError(cause + '. In Kaggle editor set Internet to On, then retry this cell. '
                           'If Internet is already On, retry after the temporary network outage. pip has not started.') from error
    except (TimeoutError, OSError) as error:
        raise RuntimeError('PyPI network probe timed out/failed. Check Kaggle Internet and retry this cell. pip has not started.') from error


def check_loaded_dependency_versions():
    # A pip upgrade cannot replace modules already imported by this Python kernel.
    for distribution, module_name in [('pillow','PIL'),('diffusers','diffusers'),
                                      ('transformers','transformers'),('accelerate','accelerate')]:
        module = sys.modules.get(module_name)
        active = getattr(module, '__version__', None)
        if active is not None and str(active) != version(distribution):
            raise RuntimeError(f'{distribution}: kernel has {active}, installed files are {version(distribution)}. '
                               'Restart the Python kernel, then run this notebook again from cell 1.')


def ensure_runtime_dependencies(requirements, constraint_file):
    pending = pending_dependencies(requirements)
    if pending:
        print('Dependencies to install/update:', pending, flush=True)
        check_pypi_access()
        subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
            '--retries','1','--timeout','15','--upgrade-strategy','only-if-needed',
            '-c',str(constraint_file),*requirements],check=True)
    else:
        print('Compatible dependencies already installed; skipping installation and PyPI probe.',flush=True)
    check_loaded_dependency_versions()
    return bool(pending)


requirements = [name+'=='+v for name,v in pins.items()] + [
    'av>=18.0.0,<19', 'safetensors>=0.8.0', 'pillow>=10.4']
constraints=[]
for name in ['torch','torchvision','torchaudio']:
    try:constraints.append(name+'=='+version(name))
    except PackageNotFoundError:pass
constraint_file=SCRATCH/'torch-constraints.txt'
constraint_file.write_text('\\n'.join(constraints)+'\\n')
ensure_runtime_dependencies(requirements,constraint_file)
(RESULTS/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','--disable-pip-version-check','freeze'],text=True))
''')
    md('''## 2. Что генерировать

`orbit_woman`, `strange_tram`, `mechanical_bloom`, `room_assembly` — T2V. `mario_poster`, `mario_pipe_illustration`, `mario_kandinsky_growth` — I2V для приложенных reference. Сами картинки не опубликованы: загрузите их в Input и вставьте путь. `LOCAL_MODEL_DIR` принимает только полный **Diffusers layout**, не ComfyUI single-file checkpoint. Чтобы не скачивать ~25 GB весов в каждой сессии, один раз запустите приватные notebooks 17 (`media`) и 18 (`text`) и добавьте их Output через **Add Input → Your Work**: при `AUTO_WEIGHT_INPUTS=True` ячейка конфигурации найдёт обе части сама.

Для `mario_kandinsky_growth` сначала подготовьте один PNG/JPEG с логотипом уже на грибе: в metadata пресета есть отдельный prompt подготовки кадра. SVG не подаётся вторым reference. Точная песня берётся из вашего `MUSIC_PATH` при экспорте; модель генерирует оригинальный funk/new-wave фон.

Таймлайн — текстовое пожелание, не точные keyframes. Distill использует CFG=1: negative prompt не создаёт отдельную guidance-ветвь; существенные ограничения записаны и в положительном prompt.''')
    code('''PRESET_NAME='orbit_woman'
PROFILE='smoke'                 # 'baseline' for 5 s
SEED=42
IMAGE_PATH=None                # e.g. '/kaggle/input/my-references/images-4.jpeg'
ASPECT='auto'                  # auto / landscape / portrait / square
LOCAL_MODEL_DIR=None           # optional '/kaggle/input/.../full-diffusers-snapshot'
AUTO_WEIGHT_INPUTS=True        # use attached weight snapshots from notebooks 17 + 18 when both are found
AUDIO_MODE='generated'          # 'music' or 'mix' to use your own audio file
MUSIC_PATH=None                 # e.g. '/kaggle/input/my-audio/burning_down_the_house.aac'
MUSIC_OFFSET_SECONDS=0.0
MUSIC_GAIN=0.65
DIT_PLACEMENT='cuda:0'          # whole DiT on one T4; 'balanced' splits layers over 2 GPUs (measured: no speedup)
MEASURE_PLACEMENTS=[]           # optional extra denoise-only runs, e.g. [('baseline','balanced'),('baseline','cuda:0')]
PARALLEL_JOBS=[]                # optional clips, two at a time (one per GPU), e.g. [('orbit_woman','baseline'),('strange_tram','baseline')]
MODEL_ID='kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers'
MODEL_REVISION='4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81'
''')
    md('### Presets: редактируемые сценарии')
    code('PRESETS = '+pprint.pformat(json.loads((BASE/'generation/prompt_presets.json').read_text()),width=110,sort_dicts=False))
    md('''## 3. Функции запуска

Обычные Python-ячейки. Qwen/CLIP кодируют оба сценария и выгружаются. По умолчанию (`DIT_PLACEMENT='cuda:0'`) весь DiT стоит на одной T4: замер на Kaggle показал, что Lite Baseline помещается в одну карту (8,6 GiB), а разбиение `'balanced'` по слоям на две GPU не ускоряет ролик, потому что GPU работают поочерёдно. Вторая GPU полезна для второго ролика: раздел 8. Полный CPU-offload pipeline поверх этого device map не включается. Codecs используют FP32 и штатный Diffusers VAE tiling.''')
    source=(BASE/'generation/diffusers_runner.py').read_text()
    for section in source.split('# %% '):
        if section.startswith('"""'):continue
        _,body=section.split('\n',1)
        code(body)
    md('## 4. Конфигурация, reference и prompt')
    code('''if LOCAL_MODEL_DIR is None and AUTO_WEIGHT_INPUTS:
    LOCAL_MODEL_DIR=assemble_local_snapshot('/kaggle/input',SCRATCH/'local-snapshot',MODEL_REVISION)
preset=PRESETS[PRESET_NAME]
PROMPTS={name:compile_prompt(preset,name)[0] for name in PROFILES}
PROMPT,NEGATIVE=compile_prompt(preset,PROFILE)
REFERENCE,REFERENCE_SHA,WIDTH,HEIGHT=reference_and_size(preset,PROFILE,IMAGE_PATH,ASPECT)
CFG=dict(model_id=MODEL_ID,revision=MODEL_REVISION,local_model_dir=LOCAL_MODEL_DIR,
    scratch=str(SCRATCH),results=str(RESULTS),media_cache=str(SCRATCH/'media-cache'),
    packages=pins,max_sequence_length=1024,seed=SEED,profile=PROFILE,dit_placement=DIT_PLACEMENT,
    width=WIDTH,height=HEIGHT,**PROFILES[PROFILE])
CFG['job_key']=job_key_for(CFG,PROMPT,REFERENCE_SHA)
RUN_DIR=RESULTS/(PRESET_NAME+'-'+PROFILE+'-'+CFG['job_key'][:12])
RUN_DIR.mkdir(exist_ok=True)
LATENTS=RUN_DIR/'joint-latents.safetensors'
REPORTS={}
(RUN_DIR/'config.json').write_text(json.dumps(dict(CFG,prompt=PROMPT,negative_prompt=NEGATIVE,
    reference_sha256=REFERENCE_SHA),indent=2))
print(json.dumps(CFG,indent=2));print(PROMPT)
print('Qwen text phase needs about 18 GiB of free scratch unless weights come from Input.')
if REFERENCE is not None:
    from IPython.display import display
    display(REFERENCE)
if AUDIO_MODE in ('music','mix'):
    # Validate the uploaded track and preview the chosen excerpt before any model download.
    if not MUSIC_PATH or not Path(MUSIC_PATH).is_file():
        raise FileNotFoundError('Upload your AAC/audio file to Kaggle Input and set its actual MUSIC_PATH before generation.')
    MUSIC_PREVIEW,MUSIC_SOURCE=soundtrack_audio(torch.zeros(1,CFG['seconds']*44100),44100,
        MUSIC_PATH,MUSIC_OFFSET_SECONDS,'music',MUSIC_GAIN)
    (RUN_DIR/'music-source.json').write_text(json.dumps(MUSIC_SOURCE,indent=2))
    from IPython.display import Audio,display
    display(Audio(MUSIC_PREVIEW.numpy(),rate=44100,normalize=False))
    print('Soundtrack checked; preview uses the selected offset and gain.')
''')
    md('''## 5. Текст и joint sampling

Оба prompt-cache сохраняются в Output. Скачать text-веса потребуется снова только при новом/изменённом preset; private text cache удаляется после encoding. Самплирование сохраняет joint latents и фоновую трассировку `nvidia-smi`. При ошибке decoder повторяйте раздел 6: Qwen и DiT заново не нужны.''')
    code("CONDITIONING=measured('text',lambda:encode_texts(PROMPTS,CFG),REPORTS)")
    code("GPU_TRACE=RUN_DIR/('nvidia-smi-denoise-'+str(time.time_ns())+'.csv')\nSAMPLE=measured('sample',lambda:sample_joint(CFG,PROMPT,NEGATIVE,CONDITIONING[PROFILE],REFERENCE,LATENTS,GPU_TRACE),REPORTS)\n(RUN_DIR/'sample.json').write_text(json.dumps(dict(SAMPLE,stage=REPORTS['sample']),indent=2,default=str))\nprint(SAMPLE)")
    md('''### Загрузка GPU во время denoising

`nvidia-smi` опрашивается каждые 200 ms в фоне только на время вызова pipeline (без загрузки весов). Summary попадает в `sample.json` и `metrics.json`, CSV и PNG — в Output. `all_busy_fraction` — доля отсчётов, где обе GPU загружены ≥ 50 %: при `device_map='balanced'` и batch 1 ожидается, что GPU работают поочерёдно. Это `utilization.gpu` драйвера (доля времени с активным kernel), не FLOPs. При повторном использовании сохранённых latents трассировки нет.''')
    code('''if SAMPLE.get('nvidia_smi',{}).get('samples'):
    from IPython.display import display
    import matplotlib.pyplot as plt
    display(plot_nvidia_smi(GPU_TRACE,GPU_TRACE.with_suffix('.png')))
    plt.close('all')
    print(json.dumps(SAMPLE['nvidia_smi'],indent=2))
else:
    print('No nvidia-smi samples:',SAMPLE.get('nvidia_smi','latents reused, denoising skipped'))''')
    md('''### Дополнительные замеры размещений (необязательно)

Если `MEASURE_PLACEMENTS` не пуст, ячейка по очереди выполняет только denoising для каждой пары `(profile, placement)` того же preset и seed, с трассой `nvidia-smi`. Декодирование и MP4 для них не делаются. Нехватка памяти (OOM) записывается в `sample.json` как результат и не останавливает notebook. Подходит для «Save & Run All»: все варианты попадают в одну таблицу ниже.''')
    code('''if MEASURE_PLACEMENTS:
    MEASUREMENTS=measure_placements(CFG,preset,PRESET_NAME,MEASURE_PLACEMENTS,CONDITIONING,IMAGE_PATH,ASPECT)
else:
    print('MEASURE_PLACEMENTS is empty; no extra measurements.')''')
    md('''### Сравнение размещений DiT

Запустите разделы 4–5 с `DIT_PLACEMENT='balanced'`, затем с `'cuda:0'` при том же preset, profile и seed: placement входит в `job_key`, поэтому второй прогон заново выполняет denoising, а не берёт latents из кэша. `denoise_s` измеряет только вызов pipeline после синхронизации CUDA; `peak_alloc_gib` — пик PyTorch allocator за всю стадию sample, включая загрузку. Один прогон на вариант — ориентир, не статистика.''')
    code('''for row in compare_placements(RESULTS):
    print(json.dumps(row))''')
    md('## 6. Video / audio decode из сохранённых latents')
    code("import numpy as np\nFRAMES=measured('video',lambda:decode_video(CFG,LATENTS),REPORTS)\nnp.save(RUN_DIR/'frames.npy',FRAMES,allow_pickle=False)\nrelease_memory()")
    code("(WAVEFORM,SAMPLE_RATE)=measured('audio',lambda:decode_audio(CFG,LATENTS),REPORTS)\nnp.save(RUN_DIR/'waveform.npy',WAVEFORM.numpy(),allow_pickle=False)\nrelease_memory()")
    md('## 7. MP4 и измерения')
    code('''FRAMES=np.load(RUN_DIR/'frames.npy',allow_pickle=False)
WAVEFORM=torch.from_numpy(np.load(RUN_DIR/'waveform.npy',allow_pickle=False))
WAVEFORM,SOUNDTRACK=soundtrack_audio(WAVEFORM,SAMPLE_RATE,MUSIC_PATH,MUSIC_OFFSET_SECONDS,AUDIO_MODE,MUSIC_GAIN)
# A fresh name preserves the previous export when this cell is repeated.
OUTPUT=RUN_DIR/('clip-'+str(time.time_ns())+'.mp4')
EXPORTED=measured('export',lambda:export_clip(FRAMES,WAVEFORM,SAMPLE_RATE,CFG,OUTPUT),REPORTS)
(RUN_DIR/'metrics.json').write_text(json.dumps(dict(stages=REPORTS,sample=SAMPLE,export=EXPORTED,soundtrack=SOUNDTRACK),indent=2,default=str))
from IPython.display import Video,display
display(Video(str(OUTPUT),embed=True))
print(json.dumps(EXPORTED,indent=2))
''')
    md('''## 8. Два ролика параллельно (необязательно)

`PARALLEL_JOBS` — список пар `(preset, profile)`. Ролики одного profile идут парами: первый на `cuda:0`, второй на `cuda:1`, каждый целиком (denoising, video и audio decode, MP4) на своей GPU. Тексты кодируются один раз заранее, веса скачиваются до старта потоков, загрузка моделей идёт по очереди, вычисления — одновременно. Для I2V preset нужен `IMAGE_PATH`. Время одного ролика не уменьшается; выигрыш — число роликов в час. Нехватка памяти записывается в `parallel.json` как результат.''')
    code('''if PARALLEL_JOBS:
    PARALLEL_RUNS={}
    for profile in dict.fromkeys(p for _,p in PARALLEL_JOBS):
        PARALLEL_RUNS[profile]=[]
        for i,name in enumerate(n for n,p in PARALLEL_JOBS if p==profile):
            run=prepare_run(CFG,PRESETS[name],name,profile,f'cuda:{i%2}',IMAGE_PATH,ASPECT)
            run['key']=name+'-'+profile
            PARALLEL_RUNS[profile].append(run)
    JOB_CONDITIONING=measured('text-parallel',lambda:encode_texts(
        {r['key']:r['prompt'] for runs in PARALLEL_RUNS.values() for r in runs},CFG),REPORTS)
    prefetch_media_weights(CFG)
    PARALLEL={}
    for profile,runs in PARALLEL_RUNS.items():
        PARALLEL[profile]=run_parallel_clips(runs,JOB_CONDITIONING,
            RESULTS/('nvidia-smi-parallel-'+profile+'-'+str(time.time_ns())+'.csv'))
        print(profile,json.dumps({k:v for k,v in PARALLEL[profile].items() if k!='results'},default=str))
    (RESULTS/('parallel-'+str(time.time_ns())+'.json')).write_text(json.dumps(PARALLEL,indent=2,default=str))
else:
    print('PARALLEL_JOBS is empty; no parallel clips.')''')
    md('''Сохраните MP4, `config.json`, `metrics.json` и `joint-latents.safetensors` через Kaggle Save Version / Output. Временная среда и media cache в `/kaggle/temp` не являются архивом. Source-level/CPU проверки находятся в GitHub; успешный Quick Save на Kaggle не доказывает pretrained inference.''')
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),kaggle=dict(accelerator='nvidiaTeslaT4',isInternetEnabled=True),
        k6_deployment=dict(portable_navigation=True,target='2xT4_16GB_CPU_RAM_32GB',
            code_layout='classic_python_cells',backend='diffusers',full_gpu_run_verified=False)))
    for i,c in enumerate(cells):c['id']=f'k6-diffusers-{i:03d}'
    out=BASE/'notebooks/16_kaggle_diffusers_lite_generation.ipynb'
    out.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    meta=dict(id='applehawk/kandinsky-6-lite-diffusers-generation',
        title='Kandinsky 6 Lite Diffusers Generation',code_file='../notebooks/16_kaggle_diffusers_lite_generation.ipynb',
        language='python',kernel_type='notebook',is_private=False,enable_gpu=True,
        enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[])
    (BASE/'generation/kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    tree=ast.parse(source)
    nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name=='compile_prompt') or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROFILES' for t in n.targets))]
    namespace={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'prompt-compiler','exec'),namespace)
    prompts=json.loads((BASE/'generation/prompt_presets.json').read_text())
    text='# Prompts: Kandinsky 6 Lite T2V / I2V\n\n[Runner](README.md) · [JSON presets](prompt_presets.json). Smoke: 1 s, Baseline: 5 s. Таймлайн описательный; CFG=1 не использует negative guidance. Все звуковые описания внутри `<AUDCAP>` — пожелания генератору. Точную существующую запись добавляйте из своего аудиофайла при экспорте.\n\n'
    for key,preset in prompts.items():
        text+='## '+key+' — '+preset['metadata']['prompt_name']+'\n\n'
        if preset['metadata']['mode']=='i2v':
            text+='Reference: `'+preset['reference_filename_hint']+'` (файл не включён).\n\n'
        if 'reference_preparation_prompt' in preset['metadata']:
            text+='**Подготовка первого кадра (отдельный image editor):**\n\n```text\n'+textwrap.fill(preset['metadata']['reference_preparation_prompt'],100)+'\n```\n\n'
        for profile in ['smoke','baseline']:
            prompt,_=namespace['compile_prompt'](preset,profile)
            text+='### '+profile+'\n\n```text\n'+textwrap.fill(prompt,100)+'\n```\n\n'
    (BASE/'generation/PROMPTS.md').write_text(text)
    print('Built',out,'cells',len(cells),'code',sum(c['cell_type']=='code' for c in cells))

if __name__=='__main__':build()
