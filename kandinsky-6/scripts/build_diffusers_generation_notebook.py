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
    md('## 1. Среда')
    code('''import os, sys, subprocess
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
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
constraints=[]
for name in ['torch','torchvision','torchaudio']:
    try:constraints.append(name+'=='+version(name))
    except PackageNotFoundError:pass
constraint_file=SCRATCH/'torch-constraints.txt'
constraint_file.write_text('\\n'.join(constraints)+'\\n')
subprocess.run([sys.executable,'-m','pip','install','--upgrade-strategy','only-if-needed',
    '-c',str(constraint_file),*[name+'=='+v for name,v in pins.items()],
    'av>=18.0.0,<19','safetensors>=0.8.0','pillow'],check=True)
(RESULTS/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
''')
    md('''## 2. Что генерировать

`orbit_woman`, `strange_tram`, `mechanical_bloom`, `room_assembly` — T2V. `mario_poster`, `mario_pipe_illustration`, `mario_kandinsky_growth` — I2V для приложенных reference. Сами картинки не опубликованы: загрузите их в Input и вставьте путь. `LOCAL_MODEL_DIR` принимает только полный **Diffusers layout**, не ComfyUI single-file checkpoint.

Для `mario_kandinsky_growth` сначала подготовьте один PNG/JPEG с логотипом уже на грибе: в metadata пресета есть отдельный prompt подготовки кадра. SVG не подаётся вторым reference. Точная песня берётся из вашего `MUSIC_PATH` при экспорте; модель генерирует оригинальный funk/new-wave фон.

Таймлайн — текстовое пожелание, не точные keyframes. Distill использует CFG=1: negative prompt не создаёт отдельную guidance-ветвь; существенные ограничения записаны и в положительном prompt.''')
    code('''PRESET_NAME='orbit_woman'
PROFILE='smoke'                 # 'baseline' for 5 s
SEED=42
IMAGE_PATH=None                # e.g. '/kaggle/input/my-references/images-4.jpeg'
ASPECT='auto'                  # auto / landscape / portrait / square
LOCAL_MODEL_DIR=None           # optional '/kaggle/input/.../full-diffusers-snapshot'
AUDIO_MODE='generated'          # 'music' or 'mix' to use your own audio file
MUSIC_PATH=None                 # e.g. '/kaggle/input/my-audio/burning-down-the-house.wav'
MUSIC_OFFSET_SECONDS=0.0
MUSIC_GAIN=0.65
MODEL_ID='kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers'
MODEL_REVISION='4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81'
''')
    md('### Presets: редактируемые сценарии')
    code('PRESETS = '+pprint.pformat(json.loads((BASE/'generation/prompt_presets.json').read_text()),width=110,sort_dicts=False))
    md('''## 3. Функции запуска

Обычные Python-ячейки. Qwen/CLIP кодируют оба сценария и выгружаются; DiT размещается блоками на двух GPU. Это распределение памяти с передачей активаций, не обещание двукратного ускорения. Полный CPU-offload pipeline поверх этого device map не включается. Codecs используют FP32 и штатный Diffusers VAE tiling.''')
    source=(BASE/'generation/diffusers_runner.py').read_text()
    for section in source.split('# %% '):
        if section.startswith('"""'):continue
        _,body=section.split('\n',1)
        code(body)
    md('## 4. Конфигурация, reference и prompt')
    code('''preset=PRESETS[PRESET_NAME]
PROMPTS={name:compile_prompt(preset,name)[0] for name in PROFILES}
PROMPT,NEGATIVE=compile_prompt(preset,PROFILE)
REFERENCE,REFERENCE_SHA,WIDTH,HEIGHT=reference_and_size(preset,PROFILE,IMAGE_PATH,ASPECT)
CFG=dict(model_id=MODEL_ID,revision=MODEL_REVISION,local_model_dir=LOCAL_MODEL_DIR,
    scratch=str(SCRATCH),results=str(RESULTS),media_cache=str(SCRATCH/'media-cache'),
    packages=pins,max_sequence_length=1024,seed=SEED,profile=PROFILE,
    width=WIDTH,height=HEIGHT,**PROFILES[PROFILE])
CFG['job_key']=digest(dict(model=MODEL_ID,revision=MODEL_REVISION,packages=pins,prompt=PROMPT,
    seed=SEED,width=WIDTH,height=HEIGHT,num_frames=CFG['num_frames'],image=REFERENCE_SHA,
    steps=10,cfg=1.0,dtype='float16',placement='balanced-two-gpu'))
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
''')
    md('''## 5. Текст и joint sampling

Оба prompt-cache сохраняются в Output. Скачать text-веса потребуется снова только при новом/изменённом preset; private text cache удаляется после encoding. Самплирование сохраняет joint latents. При ошибке decoder повторяйте раздел 6: Qwen и DiT заново не нужны.''')
    code("CONDITIONING=measured('text',lambda:encode_texts(PROMPTS,CFG),REPORTS)")
    code("SAMPLE=measured('sample',lambda:sample_joint(CFG,PROMPT,NEGATIVE,CONDITIONING[PROFILE],REFERENCE,LATENTS),REPORTS)\n(RUN_DIR/'sample.json').write_text(json.dumps(SAMPLE,indent=2,default=str))\nprint(SAMPLE)")
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
