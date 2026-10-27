# DiT → CrossDiT → Conditioning → Denoise Loop → π-Flow

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../curriculum/README.md) · [Полное содержание](../curriculum/CONTENTS.md).
<!-- study-navigation:end -->

[Полное содержание стенда](../curriculum/CONTENTS.md) · [Конкретные задания MIT](MIT_BRIDGE.md) · [Каждый класс/метод исходников](SOURCE_MAP.md) · [Измеренные результаты](reports/VALIDATION.md).

**11 подробных пошаговых лабораторных**: теория → математический вывод → discrete/tensor representation → самостоятельный прогноз → original-code experiment → измерение/график → вывод и упражнения. Материалы на русском. Предпосылки — Gaussian expectation/covariance, basic PyTorch/autograd, matrix multiplication и ODE Euler. Теоретический скелет — [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html), [notes](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf).

В `vae/` изучается пространство latent representation; **`dit/` объясняет поле, conditioning и алгоритм генерации в этом пространстве**. Сначала маленькие модели и известные поля, затем K5/K6 source; промышленные многомиллиардные weights не нужны.

## Последовательность

| Lab | Тема | Формулы / что проверяется |
|---|---|---|
| [00](notebooks/00_probability_path_and_learning_route.ipynb) | Probability path, target, time convention | Conditional expectation, reverse-time sign, u→x0, Euler units |
| [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | DiT primitives | Patch indices, Fourier time, LN/RMSNorm/GELU, attention/RoPE, adaLN-Zero gradients, output layout |
| [02](notebooks/02_single_stream_crossdit_k5.ipynb) | One-stream CrossDiT K5 | SA→text CA→FFN; original K5/K6 small-block comparison with explicit weight/precision mapping |
| [03](notebooks/03_dual_stream_crossdit_k6.ipynb) | Dual Audio/Video CrossDiT K6 | Asymmetric Q/K/V, pre/post-text ordering, gates/time routing, manual parity, Jacobian/directional cuts |
| [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Text / audio / video conditioning | Frozen encoder vs trainable projections, token/pooled paths, mask polarity, cache, both conditioning helpers |
| [05](notebooks/05_visual_conditioning_references_and_rope.ipynb) | Visual references | pretrain/i2v/tail schemas, 2C+1 channels, batched/packed roles, projector, all 5 visual methods, semantic RoPE |
| [06](notebooks/06_denoise_loop_euler_cfg_and_partial_sampling.ipynb) | Native denoise loop | Euler convergence vs exact matrix exponential, CFG NFE, frozen modality, helper functions, RoPE A/B |
| [07](notebooks/07_piflow_math_dx_policy.ipynb) | π-Flow mathematics | DX clean anchors, warp inverse/Jacobian, grid interpolation error, polynomial coefficients, copy/detach, policy integration |
| [08](notebooks/08_piflow_grid_head_and_native_sampler.ipynb) | π-Flow execution | Expanded head, canonical/legacy layouts, shortened final interval, MultimodalDXPolicy and actual network/policy counts |
| [09](notebooks/09_miniature_policy_imitation_distillation.ipynb) | Tiny imitation experiment | Frozen analytic teacher, detached on-policy rollout, live policy loss, fixed held-out and trajectory measurements |
| [10](notebooks/10_comfyui_and_manual_research_protocol.ipynb) | ComfyUI / manual research | Graph→hidden code map, measurement units, cache/warmup, real experiment record without submitting jobs |

Переходите дальше, когда можете вывести формулу и предсказать output shape до запуска. [MIT_BRIDGE](MIT_BRIDGE.md) указывает questions/cell numbers. Text CrossDiT, AV fusion, reference-tail handling и π-Flow — дополнения к MIT, а не выдуманные исходные MIT задания.

## Запуск без pretrained weights

Из корня checkout:

```bash
uv sync --locked
uv run --locked jupyter lab dit/notebooks/00_probability_path_and_learning_route.ipynb
uv run --locked --offline python dit/scripts/validate_labs.py --execute
uv run --locked --offline python dit/scripts/summarize_labs.py
```

Выберите root Python 3.12 / `.venv`; каждому notebook нужен свежий kernel, особенно при переходе из других курсов с собственным `labkit`. Установкой занимается `uv sync`, notebooks не устанавливают пакеты и не обращаются к сети. Root lock достаточен; FlashAttention, Diffusers, CUDA/MLX и текстовые encoders для этих CPU experiments не требуются.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vladislav-vasilenko/generative-ai-research/blob/main/dit/notebooks/00_probability_path_and_learning_route.ipynb)

В Colab выберите CPU и добавьте перед первой cell checkout учебного кода:

```python
!git clone --depth 1 --branch main https://github.com/vladislav-vasilenko/generative-ai-research.git /content/generative-ai-research
%cd /content/generative-ai-research
```

Это загрузка кода, не weights/datasets. Colab runtime не был проверен авторским fresh-kernel run; воспроизводимый baseline — root lock. Ссылки и checkout направлены на основную ветку main.

## Точность утверждений

- **Analytic controls:** affine probability path, Gaussian/velocity algebra, Euler с exact matrix-exponential solution, interpolation bounds.
- **Native K6:** tensor-only original blocks/loops исполняются с существующим isolated dense-SDPA adapter; имена parameters и тела network/solver methods сохранены.
- **Native K5:** выбранные class bodies + явный FP32 helper/SDPA adapter, сняты CUDA decorators. Parity относится к matched dense tiny block, не полным checkpoints, BF16, NABLA или distributed runtime.
- **Random native tiny:** reduced widths/blocks; modulation gates открываются только для sensitivity probes. Это случайные веса, не Kandinsky Lite.
- **Conditioning codecs/text:** подписанные doubles/synthetic embeddings; real Qwen/CLIP/VAE не загружаются.
- **Toy π-ID:** 2D analytic teacher и MLP policy student. Реальная K6 distillation/Sim-LADD не воспроизводится.
- **Pretrained / ComfyUI / MPS / MLX:** runtime не запускался. На пользовательской M3 Max / 48 GB станции big-weight phase отложена до 50 GB+ SSD.

Все шесть приложенных `.py` совпали с pinned K6 snapshot. Две вставки `dit.py` эквивалентны по AST; различие — завершающий newline. Новые копии source не нужны; [sources.json](sources.json) фиксирует SHA, paths и revisions. Existing source snapshots и canonical MIT задания не изменяются.

## Пять различий, которые нужно уметь объяснить

1. **One-stream K5** — один генерируемый visual state с text context; **dual-stream K6** — два генерируемых AV states с bidirectional CA. Text refiner остаётся отдельным conditioning computation.
2. Q receiving-stream после text update, K/V из сохранённого pre-text state. Вариант с другими states — другая функция; lab03 показывает контрпример.
3. Внешний frozen encoder и trainable DiT text projection — разные модули. Cache projections не заменяет time-dependent refiners.
4. **Network NFE, grid G и cheap policy calls** — разные бюджеты. K6 policy density 128 распределяется по raw intervals; не умножайте её на число segments автоматически.
5. Student расширяет output head для policy coefficients; это не означает обучение только head и не сохраняет teacher output в точности. Это также не speculative decoding с accept/reject.

## Связи с остальным стендом

[Общий VAE курс](../vae/README.md), [K5 DiT/FM overview](../kandinsky-5/notebooks/06_dit_and_flow_matching.ipynb), [K6 DiT overview](../kandinsky-6/notebooks/07_dit_and_text_crossdit.ipynb), [K6 AV block overview](../kandinsky-6/notebooks/08_dual_stream_audio_video_crossdit.ipynb), [K6 π-Flow overview](../kandinsky-6/notebooks/10_piflow_distillation.ipynb), [общий research protocol](../curriculum/RESEARCH_PROTOCOL.md). Существующие paths сохраняются; новый root раздел углубляет и связывает их.

## Источники

[MIT notes](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf), [DiT](https://arxiv.org/abs/2212.09748), [Attention](https://arxiv.org/abs/1706.03762), [RoPE](https://arxiv.org/abs/2104.09864), [PixArt-α](https://arxiv.org/abs/2310.00426), [Flow Matching](https://arxiv.org/abs/2210.02747), [K5](https://arxiv.org/abs/2511.14993), [K6 §3.2/4.5](https://arxiv.org/html/2610.05608v1), [π-Flow §3.1/4](https://arxiv.org/html/2510.14974v1). Ссылки на papers не объявляют их взаимозаменяемыми и не доказывают первенство названия CrossDiT.

```text
dit/
  README.md, MIT_BRIDGE.md, SOURCE_MAP.md, sources.json
  notebooks/    00–10: теория, original code, эксперименты и упражнения
  labkit/       adapters и analytic controls; без pretrained loaders
  scripts/      generator, fresh-kernel validator, summary, optional math renderer
  reports/      measured JSON/CSV и выбранные PNG
  artifacts/    ignored: executed notebooks, previews и рабочие outputs
```

Переcборка: `uv run --locked python dit/scripts/build_notebooks.py`, затем `curriculum/scripts/sync_study_navigation.py --write`, validator и summary. Новые материалы не назначают чужим исходникам новую лицензию: [K6 notices](../kandinsky-6/THIRD_PARTY.md) и исходные LICENSE сохраняются. Это независимый учебный стенд, не официальный MIT/Kandinsky course.

## Проверка отображения формул

[Strict KaTeX report](reports/math-rendering.json) проверяет display formulas и фиксирует notebook SHA. Renderer — опциональная зависимость, не часть Python lab runtime:

```bash
npm install --prefix dit/artifacts/render-tools --ignore-scripts --no-audit --no-fund katex@0.16.25
node dit/scripts/check_math.cjs --katex-path=dit/artifacts/render-tools/node_modules/katex
```

Локальные previews и графики осмотрены. Это не обещание одинакового рендеринга GitHub/Colab во всех версиях.
