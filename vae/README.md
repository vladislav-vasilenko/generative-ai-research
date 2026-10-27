# VAE: от MIT Lab 3 до промышленного видеокодека

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../curriculum/README.md) · [Полное содержание](../curriculum/CONTENTS.md).
<!-- study-navigation:end -->

[Весь стенд](../README.md) · [Полное содержание](../curriculum/CONTENTS.md) · [MIT Lecture 4 / Lab 3](../lecture_04/README.md) · [Карта всех классов и методов](SOURCE_MAP.md) · [Проверки и измерения](reports/VALIDATION.md).

**[Связь с конкретными заданиями MIT Lab 3](MIT_LAB3_BRIDGE.md)** — что читать, какое упражнение выполнить и какую проверку показать перед переходом. Это общая ветвь VAE в корне проекта; HunyuanVideo — первый подробный промышленный case. Новые codecs можно добавлять в эту же карту без дублирования теории.

Девять пошаговых лабораторных по **теории VAE, tensor geometry, causal Conv3D, spatial/temporal tiling и free-VRAM planner**. Порядок внутри: объяснение → вывод формулы → shapes/units → прогноз → исполняемый код → измерение/график → вывод и упражнения. Предпосылки: Gaussian distribution, expectation/KL, tensor broadcasting, convolution и базовый PyTorch autograd. Если тема новая, сначала [probability bridge](../curriculum/notebooks/00_probability_and_model_contracts.ipynb) и [Lecture 4](../lecture_04/README.md).

Теоретическая основа — [MIT notes §6.2, стр. 46–52 и Appendix D, стр. 77–82](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf), [Lecture 4, slides 3–19](https://diffusion.csail.mit.edu/2026/docs/20260128_Lecture_04_edited.pdf) и [Lab 3 Part 4–5](../lecture_04/references/lab_three_original.ipynb). Для промышленного case источник — **[HunyuanVideo, arXiv:2412.03603v6](https://arxiv.org/pdf/2412.03603v6), §4.1, стр. 5–6, Fig. 6 и Eq. 1**. Использован предоставленный PDF v6. Приложенный `hunyuan_vae.py` побайтово совпал с уже закреплённым K6 source; SHA/revision указаны в [sources.json](sources.json). Статья объясняет tiling и training mismatch; выбор tiles по CUDA free bytes — отдельная особенность исследуемой адаптации.

## Последовательность

| Шаг | Лабораторная | Измеримый результат |
|---|---|---|
| 00 | [Маршрут и контракты](notebooks/00_route_and_source_contracts.ipynb) | Формы, scalar compression и provenance |
| 01 | [ELBO и вероятностный codec](notebooks/01_elbo_and_probabilistic_codec.ipynb) | Monte Carlo bound, Gaussian KL, gradient check, linear VAE training |
| 02 | [Causal Conv3D и геометрия](notebooks/02_causal_convolution_and_geometry.ipynb) | Native convolution, first-frame rule, mask, GroupNorm counterexample |
| 03 | [Архитектура, posterior и API](notebooks/03_architecture_posterior_and_api.ipynb) | Hooks всех stages, mode/sample, scaling и return contract |
| 04 | [Spatial tiling](notebooks/04_spatial_tiling_and_blending.ipynb) | Original blend/encode/decode, exact oracle и missing-edge example |
| 05 | [Temporal tiling](notebooks/05_temporal_tiling_and_first_frame.ipynb) | Original loops, frame-count sweep, context reset и error profile |
| 06 | [Free-VRAM planner](notebooks/06_free_vram_planner.ipynb) | Original heuristic, synthetic CUDA budget sweep, coverage/stride guards |
| 07 | [Контролируемый benchmark](notebooks/07_controlled_benchmark.ipynb) | 3 seeds × 4 policies, RGB MSE, seams/interior, CPU timing |
| 08 | [Pretrained и ComfyUI вручную](notebooks/08_pretrained_and_comfyui_protocol.ipynb) | Dry-run runner, measurement schema, ресурсный gate и критерии вывода |

Читать можно без запуска; для самостоятельного освоения сначала предскажите результат каждой проверки. Затем Run All и упражнения с одним изменяемым фактором. 00–07 выполняются на CPU без сети, весов и datasets; 08 по умолчанию исполняет только dry-run/help.

## Связь с уже существующими VAE лабораторными

| Codec / уровень | Где изучать | Что сопоставить |
|---|---|---|
| Учебный 2D VAE MIT | [Lab 3 Part 4–5](../lecture_04/references/lab_three_original.ipynb), [карта заданий](MIT_LAB3_BRIDGE.md) | Gaussian posterior, learned reconstruction variance, freeze перед latent DiT |
| Общая теория | [01 ELBO/KL/NLL](notebooks/01_elbo_and_probabilistic_codec.ipynb) | Одинаковые вероятностные формулы при разных tensor layouts |
| HunyuanVideo causal 3D VAE | [02–07 этого курса](notebooks/02_causal_convolution_and_geometry.ipynb) | Temporal lattice, native architecture, tiling и memory policy |
| Kandinsky 5 Lite video codec | [K5 lab 02](../kandinsky-5/notebooks/02_vae.ipynb) | Posterior/scaling и интерфейс генерации K5 |
| Kandinsky 6 video codecs | [K6 lab 03](../kandinsky-6/notebooks/03_video_vae.ipynb), [component guide](../kandinsky-6/COMPONENT_GUIDE.md) | Hunyuan/KVAE варианты; используемый class и compression заданы checkpoint/config |
| Kandinsky 6 audio codec / vocoder | [K6 lab 04](../kandinsky-6/notebooks/04_audio_vae_and_vocoder.ipynb) | Waveform, mel, continuous latents и vocoder — отдельные units/модели |

Существующие K5/K6 notebooks сохраняют пути; `vae/` объединяет маршрут и общую теорию, чтобы не ломать ссылки и не плодить копии. При обучении самого VAE encoder/decoder trainable; при рассматриваемом latent DiT training codec frozen. Конкретную стадию и checkpoint всегда проверяем по источнику.

## Запуск на M3 Max / обычном CPU

Из корня репозитория:

```bash
uv sync --locked
uv run --locked jupyter lab vae/notebooks/00_route_and_source_contracts.ipynb
# Тот же Python 3.12 / root .venv в kernel picker.
uv run --locked --offline python vae/scripts/validate_labs.py --execute
```

`uv sync` устанавливает зависимости; сами notebooks не устанавливают пакеты. Root lock уже содержит PyTorch, NumPy, Matplotlib, Jupyter и nbclient. Diffusers для этих CPU labs не нужен: оригинальные pure classes/methods извлекаются AST, а маленькие dependency adapters явно подписаны в [labkit](labkit/vae_lab.py).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vladislav-vasilenko/generative-ai-research/blob/main/vae/notebooks/00_route_and_source_contracts.ipynb)

Для Colab [откройте стартовую lab](https://colab.research.google.com/github/vladislav-vasilenko/generative-ai-research/blob/main/vae/notebooks/00_route_and_source_contracts.ipynb). После открытия выполните в **добавленной первой code cell** checkout учебных исходников (это загрузка кода, не model weights):

```python
!git clone --depth 1 --branch main https://github.com/vladislav-vasilenko/generative-ai-research.git /content/generative-ai-research
%cd /content/generative-ai-research
```

Выберите CPU runtime: GPU для этих опытов не нужен. Основные numerical packages в Colab обычно доступны; при отличиях версий запишите их. Авторская проверка сделана в root lock, не в Colab. Ссылки и checkout направлены на основную ветку main.

## Что именно измеряется

- **Аналитические проверки:** ELBO/KL, autograd vs finite difference, causality отдельной conv.
- **Index oracle:** original stitching на известной coordinate lattice; нулевая ошибка подтверждает indices только на проверенной grid.
- **Native tiny:** encoder/decoder из pinned source, widths 4/8/8/8, latent=2, layers=1, mid attention=False, random weights. Это не released codec с latent=16.
- **Planner:** original method с синтетическим CUDA budget в локальном namespace. Физический CUDA GPU не используется.
- **Pretrained:** ещё не запускался. [Runner](scripts/benchmark_local_codec.py) требует явный `--execute`, локальные веса и отдельное окружение Diffusers; это reference implementation, а не attached adaptive VAE.

Исходный snapshot не меняется. В source audit обнаруживаются конкретные ограничения: non-CUDA budget=∞, coverage gaps для некоторых размеров, зависимость normalization от всего tile, posterior вместо tensor на выходе encode. Каждый вывод ограничен конкретным источником, вызовом и входом; мы не объявляем весь upstream неисправным.

## Реальные веса и ComfyUI

На станции пользователя **M3 Max / 48 GB unified memory** pretrained этап отложен до освобождения **50 GB+ SSD**. Для VAE reconstruction не нужны DiT/Qwen/CLIP/SR/audio модели. Не загружайте полный pipeline ради одного кодека. Имеющийся ComfyUI checkpoint можно исследовать через его loader; Diffusers folder и single safetensors — разные форматы/контракты.

Начните с [Lab 08](notebooks/08_pretrained_and_comfyui_protocol.ipynb), [C4 codec workflow](../curriculum/COMFYUI_LABS.md#c4-vae-reconstruction-отдельно-от-генератора) и [JSON](../curriculum/workflows/04_hunyuan_vae_reconstruction.json). Workflow — image/F=1 контроль; temporal video требует соответствующих video nodes. Здесь ComfyUI jobs не отправляются и Desktop environment не меняется.

## Файлы и воспроизводимость

```text
vae/
  README.md, MIT_LAB3_BRIDGE.md, SOURCE_MAP.md, sources.json
  notebooks/   00–08: общая теория и подробный HunyuanVideo case
  labkit/      AST loader, tiny native codec, oracle, метрики
  scripts/     generator, validator, summary, ручной pretrained runner
  reports/     измеренные JSON/CSV и выбранные PNG
  artifacts/   ignored: executed notebooks и рабочие outputs
hunyuan-video/README.md   Вход со стороны промышленного семейства
```

Переcборка: `uv run --locked python vae/scripts/build_notebooks.py`, затем `curriculum/scripts/sync_study_navigation.py --write` и validator. Большие файлы/PDF/weights не коммитятся. Pinned third-party source и его notices лежат в K6 reference, не дублируются и не перелицензируются. Это независимые учебные материалы; точная provenance — в SOURCE_MAP.

## Дополнительная проверка отображения формул

[Strict KaTeX report](reports/math-rendering.json) содержит hashes notebooks и результат проверки 32 display formulas. Python labs не зависят от KaTeX; для опциональной повторной renderer-проверки используйте небольшое отдельное npm окружение в ignored artifacts:

```bash
npm install --prefix vae/artifacts/render-tools --ignore-scripts --no-audit --no-fund katex@0.16.25
node vae/scripts/check_math.cjs --katex-path=vae/artifacts/render-tools/node_modules/katex
```

Это проверяет математический синтаксис локально. GitHub/Colab могут применять собственные ограничения рендерера.
