"""Build a complete navigation index of authored study materials, not vendored source trees."""
from pathlib import Path
import ast,json,os,re
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'curriculum/CONTENTS.md'

def link(path,label=None):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    rel=os.path.relpath(path,OUT.parent).replace(os.sep,'/')
    return f'[{label or path.name}]({rel})'

def nb_sections(path):
    nb=json.loads(path.read_text());headings=[]
    for c in nb['cells']:
        if c['cell_type']!='markdown':continue
        source=''.join(c['source']) if isinstance(c['source'],list) else c['source']
        for hashes,heading in re.findall(r'^\s*(#{1,3})\s+(.+)',source,re.M):
            text=re.sub(r'\[([^]]+)\]\([^)]*\)',r'\1',heading).replace('|','/').replace('`','')
            if re.match(r'(?i)(подготовка|перед началом|как работать|проверьте понимание|самостоятельное исследование|критерий понимания|упражнения|источники|sources|итог)',text):continue
            if len(hashes)==1 and not text.startswith(('Часть','Part')):continue
            headings.append((len(hashes),text))
    parts=[title for level,title in headings if level==1]
    sections=[title for level,title in headings if level==2]
    topics=parts or (sections if len(sections)>=2 else [title for level,title in headings if level==3]) or sections
    return '; '.join(topics[:6]) or 'См. пошаговые ячейки notebook'

text='''# Полное содержание и единый маршрут обучения

[Главная страница](../README.md) · [Подробная методика и критерии P0–P8](README.md) · [Самопроверка](PROGRESS.md).

Это оглавление всех авторских учебных материалов и точек входа в оригиналы. Основной скелет — [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html), [конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf). Сначала теория и небольшая реализация, затем архитектурный компонент, UI-наблюдение и отдельная исследовательская гипотеза. Исходные незаполненные MIT labs — задания; Run All применим к готовым авторским CPU-примерам, а не автоматически ко всем originals.

## Один маршрут через пять уровней

| Шаг / тема | Лекция и исходная lab | Python / архитектурный блок | ComfyUI | Результат перед переходом |
|---|---|---|---|---|
| 0. Вероятности и формы | Notes Appendix A | [Probability bridge](notebooks/00_probability_and_model_contracts.ipynb), [K6 system map](../kandinsky-6/notebooks/01_system_and_training_map.ipynb) | [C0: image dataflow](COMFYUI_LABS.md#c0-контроль-dataflow-до-нейросети) | covariance/score, shape и units |
| 1. ODE/SDE | [Lecture 1 / Lab 1](../lecture_01/README.md) | Euler, Brownian/OU, [K5 sampling](../kandinsky-5/notebooks/07_sampling.ipynb) | C1 reproducibility, C2 solver | ошибка при уменьшении шага, моменты распределения |
| 2. Flow Matching | [Lecture 2 / Lab 2](../lecture_02/README.md) | [K5 FM](../kandinsky-5/notebooks/06_dit_and_flow_matching.ipynb), [K6 joint FM](../kandinsky-6/notebooks/09_flow_matching_training.ipynb) | C2: steps/NFE на совместимом checkpoint | path/velocity/time conventions + held-out loss |
| 3. Score / conditioning / CFG | [Lecture 3 / Lab 2–3](../lecture_03/README.md) | [K5 text](../kandinsky-5/notebooks/01_text_embedders.ipynb), [K6 text](../kandinsky-6/notebooks/02_text_conditioning.ipynb), [K6 CFG](../kandinsky-6/notebooks/11_sampling_cfg_and_nfe.ipynb) | C3 CFG; Distill CFG=1 не universal sweep | masks, encoder/refiner, branch count |
| 4. VAE / codecs | [Lecture 4 / Lab 3](../lecture_04/README.md) | [K5 VAE theory](../kandinsky-5/notebooks/02_vae.ipynb), [K6 video](../kandinsky-6/notebooks/03_video_vae.ipynb), [K6 audio/vocoder](../kandinsky-6/notebooks/04_audio_vae_and_vocoder.ipynb) | C4 standalone reconstruction | ELBO/posterior/scale, реконструкция отдельно от генерации |
| 5. DiT / CrossDiT | Lecture 4, Lab 3 Part 3 | [K5 nn/DiT](../kandinsky-5/notebooks/03_nn_building_blocks.ipynb), [K6 DiT](../kandinsky-6/notebooks/07_dit_and_text_crossdit.ipynb), [K6 dual AV CrossDiT](../kandinsky-6/notebooks/08_dual_stream_audio_video_crossdit.ipynb) | C5 token budget, Lite graph | Q/K/V, gates, shapes, directional ablation |
| 6. Discrete branch | [Lecture 5](../lecture_05/README.md) | [CTMC lab](../lecture_05/05_ctmc_and_discrete_flow.ipynb) | C6 visualizes Python result | rates, positivity, exact law vs simulation |
| 7. Distillation / deployment | [Industrial map](INDUSTRY_BRIDGE.md) | [K6 π-Flow](../kandinsky-6/notebooks/10_piflow_distillation.ipynb), [runtime/MLX](../kandinsky-6/notebooks/13_runtime_memory_and_mlx.ipynb) | [K6 UI component lab](../kandinsky-6/notebooks/14_comfyui_component_experiments.ipynb) | policy/NFE distinction, resource baseline |
| 8. Separate research | [Research protocol](RESEARCH_PROTOCOL.md) | [K6 manual journal](../kandinsky-6/notebooks/00_manual_experiment_comfy_mlx_film.ipynb), FiLM/voice or own hypothesis | baseline + controlled A/B | data split, metrics, regression, supported conclusion |

Lab 2 связывает Lecture 2–3, Lab 3 — Lecture 3–4. CTMC — самостоятельная ветвь; continuous industrial video можно изучать после Lecture 4. Не требуется запускать все промышленные семейства: выберите один проверяемый case.

## Полное содержание лекций и базовых лабораторных

'''
for n in range(1,6):
    folder=ROOT/f'lecture_{n:02d}'
    text+=f"### Lecture {n}\n\n{link(folder/'README.md','Порядок, источники и задания')}\n\n"
    for p in sorted(folder.glob('*.md')):
        if p.name!='README.md':text+='- '+link(p)+' — авторский конспект.\n'
    for p in sorted(folder.glob('*.ipynb')):
        text+='- '+link(p)+' — '+nb_sections(p)+'.\n'
    for p in sorted((folder/'references').glob('*.ipynb')):
        text+='- '+link(p)+' — неизменённый исходный MIT notebook, содержит задания.\n'
    if n==1:text+='- '+link(folder/'references/README.md','Происхождение Lab 1')+'; '+link(folder/'outputs/lab_one_ru.html','сохранённый HTML-пример выполнения')+'.\n'
    text+='\n'
text+='## Общие авторские notebooks\n\n'
for p in sorted((ROOT/'curriculum/notebooks').glob('*.ipynb')):
    text+='- '+link(p)+' — '+nb_sections(p)+'.\n'
text+='- '+link('Diffusion_DDPM_Research.ipynb','DDPM / CIFAR-10')+' — дополнительная noise-prediction ветвь, отдельный dependency extra.\n\n'

for family,title in [('kandinsky-5','Kandinsky 5 Lite: 14 лабораторных'),('kandinsky-6','Kandinsky 6 Lite: компоненты и отдельное исследование')]:
    text+=f'## {title}\n\n'+link(f'{family}/README.md','Порядок, запуск и Colab')+'; '
    text+=link(f'{family}/COMPONENT_GUIDE.md','Карта компонентов и методов')+'.\n\n' if family=='kandinsky-6' else link(f'{family}/MODEL_CATALOG.md','Каталог моделей')+'.\n\n'
    text+='| Notebook | Содержание / последовательность внутри |\n|---|---|\n'
    for p in sorted((ROOT/family/'notebooks').glob('*.ipynb')):
        description=nb_sections(p)
        if p.name.startswith('00_manual'):description='Исследовательский журнал: диск → ComfyUI baseline → VCTK → parity → будущий FiLM/voice experiment; читать после component route'
        text+=f'| {link(p)} | {description} |\n'
    text+='\n'

text+='''## Промышленные семейства и архитектурные блоки

[Сводная industrial map с закреплёнными official sources](INDUSTRY_BRIDGE.md) — общий вход. Ниже это кейсы для переноса знаний; самостоятельных полноценных pretrained labs для каждого семейства пока нет.

| Семейство | Что искать после MIT | Материал стенда |
|---|---|---|
| HunyuanVideo | causal VAE, dual/single stream, flow scheduler | [Pinned source map](INDUSTRY_BRIDGE.md#tencent-hunyuanhunyuanvideo), K5 02 / K6 03, ComfyUI C4 |
| Wan 2.1 | T5, Wan-VAE, time modulation, FM solver | [Pinned source map](INDUSTRY_BRIDGE.md#wan-videowan21) |
| Wan 2.2 | A14B noise experts vs dense TI2V-5B, codec contracts | [Pinned source map](INDUSTRY_BRIDGE.md#wan-videowan22) |
| Kandinsky 5 | Qwen/CLIP, CrossDiT, VAE, generation_utils | [14 labs](../kandinsky-5/README.md) |
| Kandinsky 6 | asymmetric AV streams, cross gates, π-Flow, audio codec/vocoder | [14 component labs](../kandinsky-6/README.md), [source method index](../kandinsky-6/COMPONENT_GUIDE.md) |
| LTX-Video | causal codec / Transformer / RF | [Pinned source map](INDUSTRY_BRIDGE.md#lightricksltx-video) |
| Mochi 1 | asymmetric text/visual streams, VAE | [Pinned source map](INDUSTRY_BRIDGE.md#genmoaimochi) |
| Open-Sora | release-specific MMDiT, codec и sampler | [Pinned source map](INDUSTRY_BRIDGE.md#hpcaitechopen-sora) |

Для сравнения AV fusion в K6 08 читаем [Ovi](https://arxiv.org/abs/2510.01284) и [LTX-2](https://arxiv.org/abs/2601.03233). Это отдельные статьи; LTX-Video в общей карте не переименовывается в LTX-2.

## Все ComfyUI опыты и готовые workflows

[Инструкция C0–C6](COMFYUI_LABS.md): C0 dataflow → C1 reproducibility/cache → C2 solver/NFE → C3 CFG → C4 codec reconstruction → C5 tokens/resources → C6 границы UI и Python/custom nodes.

| Workflow | Что демонстрирует | Статус |
|---|---|---|
| [Image IO без моделей](workflows/00_image_io_no_models.json) | LoadImage→SaveImage, known pixels | Static checked; запуск вручную |
| [Hunyuan VAE reconstruction](workflows/04_hunyuan_vae_reconstruction.json) | Codec отдельно от DiT/text | Static checked; запуск вручную с имеющимся VAE |
| [K6 Lite Smoke](../kandinsky-6/workflows/k6_lite_distill_smoke.json) | 432×240 / 33 frames / seed42 / 10 steps | Static checked; pretrained generation отложена |
| [K6 Lite Baseline](../kandinsky-6/workflows/k6_lite_distill_baseline.json) | 864×480 / 121 frames / seed42 / 10 steps | Static checked; pretrained generation отложена |

[Test chart](assets/curriculum_test_chart.png), [RGB comparison script](scripts/compare_images.py), [K6 stage measurement script](../kandinsky-6/scripts/measure_comfy.py). Во время CPU-лабораторных workflow не отправляются на выполнение.

## Отдельное исследование: от гипотезы к выводу

1. [Шесть уровней доказательств и research protocol](RESEARCH_PROTOCOL.md).
2. [Карточка одного эксперимента](EXPERIMENT_TEMPLATE.md).
3. [Существующий K6 manual notebook](../kandinsky-6/notebooks/00_manual_experiment_comfy_mlx_film.ipynb).
4. [Пошаговый voice/ComfyUI/MLX protocol](../kandinsky-6/COMFYUI_AND_EXPERIMENT_RU.md).
5. [Текущее состояние реализации](../kandinsky-6/STATUS.md) — какие этапы действительно выполнены.
6. [Baseline measurement table](../kandinsky-6/reports/templates/baseline_runs.csv) и [voice evaluation table](../kandinsky-6/reports/templates/voice_evaluation.csv).
7. [VCTK preparation](../kandinsky-6/scripts/prepare_vctk.py), [PyTorch fixture](../kandinsky-6/scripts/torch_parity_reference.py), [MLX parity prototype](../kandinsky-6/scripts/mlx_parity.py).

Первое запланированное исследование: speaker-conditioned zero-init FiLM на замороженном K6 audio backbone, VCTK и held-out texts; baseline/regression проверяются отдельно. Это гипотеза, а не уже полученное voice cloning. Pretrained этапы остаются отложены до выбранного пользователем SSD gate 50 GB+.

## Все руководства, источники и результаты

'''
folders=[ROOT,ROOT/'curriculum',*sorted(ROOT.glob('lecture_*')),ROOT/'kandinsky-5',ROOT/'kandinsky-6']
extra=[ROOT/'curriculum/references/README.md',ROOT/'kandinsky-5/reference/course_materials/README.md',
       ROOT/'kandinsky-5/reports/VALIDATION.md',ROOT/'kandinsky-6/reports/VALIDATION.md',
       ROOT/'kandinsky-6/reports/COMPONENT_VALIDATION.md',ROOT/'kandinsky-6/reference/README.md']
paths=sorted({p for folder in folders for p in folder.glob('*.md')}|{p for p in extra if p.exists()})
for p in paths:
    if p==OUT:continue
    title=p.read_text().splitlines()[0].lstrip('# ').replace('|','/')
    text+='- '+link(p,str(p.relative_to(ROOT)))+' — '+title+'.\n'
text+='''
**Индексы источников:** [PDF/lab manifest](references/materials_manifest.json), [industrial revisions](references/industry_manifest.json), [дополнительные статьи/репозитории](references/additional_sources.json), [K6 source hashes](../kandinsky-6/reference/source-manifest.json).

**Численные отчёты:** [общий CPU execution](reports/validation.json), [дополнительные CPU контроли](reports/supplementary-experiments.json), [K6 component execution](../kandinsky-6/reports/component-validation.json), [K6 educational results](../kandinsky-6/reports/educational-results.json). Полные executed notebooks и рабочие outputs находятся в ignored artifacts/reports/executed, а не в учебном оглавлении.

## Инструменты и воспроизведение

Это служебные точки входа для воспроизведения и развития материалов. Для чтения курса они не являются дополнительными обязательными лабораторными. Third-party source trees навигируются через manifests/COMPONENT_GUIDE, а не смешиваются с авторскими файлами.

'''
for family in ['curriculum','lecture_01','kandinsky-5','kandinsky-6']:
    text+=f'### `{family}/scripts`\n\n'
    for p in sorted((ROOT/family/'scripts').glob('*.py')):
        doc=ast.get_docstring(ast.parse(p.read_text())) or ''
        summary=doc.splitlines()[0] if doc else 'См. аргументы и назначение в исходнике'
        text+='- '+link(p)+' — '+summary+'.\n'
    text+='\n'
text+='''## Как поддерживать содержание

`uv run --locked python curriculum/scripts/build_contents.py` собирает индекс из фактических authored notebooks/guide files. После изменений навигации выполните `sync_study_navigation.py --write`, затем `--check`. Порядок обучения задаёт эта карта вместе с curriculum README; автоиндекс не меняет original labs и пользовательские ответы.
'''
OUT.write_text(text)
print('Built',OUT)
