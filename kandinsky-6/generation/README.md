# Kandinsky 6 Lite — генерация через Diffusers

[Главная](../../README.md) · [K6](../README.md) · [Все 14 промптов](PROMPTS.md) · [JSON presets](prompt_presets.json) · [Notebook 16](../notebooks/16_kaggle_diffusers_lite_generation.ipynb).

Отдельный runner для T2V/I2V, без учебных выводов, упражнений и ComfyUI. Использует официальный [Kandinsky6TI2VAPipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/kandinsky6) и **Lite Distill 5s**, FP16 DiT, CFG=1, 10 NFE, FP32 codecs. Подготовлен для Kaggle **2×T4 / 32 GB CPU RAM**. Pretrained запуск на этих двух GPU пока не проверен; tiny CPU API tests не являются замером качества/скорости модели.

| Вариант | Notebook / Kaggle | Назначение |
|---|---|---|
| ComfyUI deployment | [Notebook 15](../notebooks/15_kaggle_t4_lite_distill.ipynb) · [Kaggle](https://www.kaggle.com/code/applehawk/15-kaggle-t4-lite-distill) | Сохранённый программный повтор существующего ComfyUI workflow, распределение VAE tiles |
| Diffusers Generation | [Notebook 16](../notebooks/16_kaggle_diffusers_lite_generation.ipynb) · [Kaggle](https://www.kaggle.com/code/applehawk/kandinsky-6-lite-diffusers-generation) | Новый T2V/I2V runner с preset-подборкой и распределением блоков DiT |

Имена файлов и URL старого варианта сохранены для существующих ссылок. В навигации backend явно подписан. Дистиллированная модель не требует ComfyUI и не требует переобучения для обычного инференса через Diffusers. Single-file ComfyUI weights и исходный Diffusers layout — разные форматы загрузки; автоматической конвертации здесь нет.

## Запуск

1. Откройте notebook 16 в Kaggle, выберите GPU T4 ×2 и Internet ON. Первая ячейка проверяет устройства до установок/загрузок. Зависимости закреплены, preinstalled PyTorch защищён constraints.
2. В конфигурации задайте `PRESET_NAME`, `PROFILE='smoke'` и `SEED`. Для I2V загрузите reference в Kaggle Input и задайте `IMAGE_PATH`. `ASPECT='auto'` выбирает portrait/square/landscape по reference; resize/padding сохраняет всю композицию.
3. Выполняйте обычные Python-ячейки. `PRESETS` можно редактировать прямо в notebook. JSON timeline преобразуется в текстовый prompt и `<AUDCAP>`; это пожелания о порядке событий, не гарантированные keyframes.
4. Посмотрите Smoke. Измените `PROFILE='baseline'`, повторите конфигурацию, text cache lookup, sampling и codecs/export. Оба prompt-cache подготовлены при первой загрузке Qwen.
5. Сохраните MP4 и отчёты из Output. Для повторного decoder достаточно `CFG` и сохранённых latents: Qwen/DiT не нужны. Чтобы восстановить после потери kernel, сначала восстановите определения функций и прежнюю конфигурацию; файлы `/kaggle/temp` не переживают новую сессию.

| Режим | Native generated frames | Export frames / 24 FPS | Landscape | Portrait | Square |
|---|---|---|---|---|---|
| Smoke | 25 (`4k+1`) | 24 / 1.00 s | 432×256 | 256×432 | 320×320 |
| Baseline | 121 (`4k+1`) | 120 / 5.00 s | 864×480 | 480×864 | 640×640 |

Последний кадр обрезается только при экспорте; waveform также обрезается до 1/5 секунд. Проверяется фактическое декодирование video/audio streams. AAC padding может добавить небольшую погрешность длительности аудиопотока.

## Presets

| Ключ | Задача | Smoke | Baseline |
|---|---|---|---|
| `orbit_woman` | T2V: взрослая женщина в каменном дворе | Облёт около 15°, движение шарфа | Непрерывный облёт около 160°, ткань и parallax |
| `strange_tram` | T2V: сюрреалистическая retrofuturist комедия | Автомат высовывает механический язычок | Работа автомата, реакция существа, подъезд трамвая |
| `mechanical_bloom` | T2V: биомеханический цветок | Начало раскрытия и пыльца | Последовательное раскрытие с небольшим orbit |
| `room_assembly` | T2V: Scandinavian furniture ad | Открытие коробки, первые рейки | Сборка одной кровати и финальный hold |
| `mario_poster` | I2V: первая картинка | Blink, nod, secondary motion | Жест с кепкой, плечи, возвращение к позе |
| `mario_pipe_illustration` | I2V: второй рисунок | Blink и кивок цветка | Реакция на гриб, squash-and-stretch, settle |
| `mario_kandinsky_growth` | I2V: подготовленный кадр с логотипом на грибе | Укус и начало роста | Укус → рост вверх/вширь → финальный bounce |

Retrofuturist presets описывают общие визуальные признаки: practical effects, странные существа, старые промышленные декорации, muted teal/amber и film grain. Это оригинальные сцены, без заявления о точном повторе работ Gossip Goblin. Градусы orbit и тайминги — цели режиссуры, не измеренный результат. В односекундном Smoke полный круг и многоступенчатая сборка намеренно не запрашиваются.

## Логотип Kandinsky и Talking Heads

Для `mario_kandinsky_growth` нужен **один подготовленный reference**, где логотип из вашего SVG уже нанесён на шляпку гриба. В JSON/PROMPTS.md есть `reference_preparation_prompt` для image editor. Исходные картинки и SVG не копируются в репозиторий. Сам pipeline получает PNG/JPEG/PIL image, не SVG и не пару независимых reference; список images означает batch, а не второй logo-condition. Полное длинное начертание на маленьком грибе может быть нечитаемо на Smoke resolution; для него лучше крупный передний план. Даже с reference сохранение букв в движении требует визуальной проверки.

Звук генератора описан как оригинальный энергичный new-wave/funk instrumental. Чтобы в MP4 звучала именно **Talking Heads — Burning Down the House**, подключите ваш аудиофайл как Input и выберите:

```python
AUDIO_MODE = 'mix'      # generated effects + supplied recording; 'music' replaces generated audio
MUSIC_PATH = '/kaggle/input/my-audio/burning_down_the_house.aac'
MUSIC_OFFSET_SECONDS = 0.0
MUSIC_GAIN = 0.65
```

Функция читает нужный фрагмент, переводит его в mono, обрезает до длительности клипа, смешивает с эффектами и предотвращает clipping. Существующая запись не скачивается и не имитируется моделью. Офсет выбирается вручную; автоматического beat sync нет.

## Если первая ячейка не устанавливает зависимости

- `Failed to resolve pypi.org`: среда не может разрешить DNS-имя PyPI. Проверьте **Internet → On** в редакторе Kaggle и повторите ячейку. Если Internet включён, возможен временный сетевой сбой; новые попытки pip и смена dtype/GPU его не устраняют. Обновлённая ячейка проверяет HTTPS-доступ к PyPI до установки, различает DNS/HTTP/timeout и ограничивает повторы pip.
- `gradio ... requires pillow<13.0`: предупреждение pip о предустановленном Gradio, который notebook не использует. Pillow намеренно не понижается: ядро Kaggle импортирует PIL до первой ячейки, и понижение до 12.x требовало ручного перезапуска ядра, из-за чего «Save & Run All» падал на первой ячейке. Если ячейка 1 всё же сообщает о расхождении версии уже импортированной библиотеки, перезапустите Python kernel и повторите notebook с первой ячейки. Файлы результатов не удаляются.
- Если прямые зависимости уже совместимы, установка pip и проверка PyPI пропускаются. Удалённая загрузка весов всё равно требует Hugging Face; подключённый локальный snapshot задаётся отдельно.

Предоставленный AAC Talking Heads можно использовать напрямую: загрузите его в Kaggle Input, скопируйте **фактический** путь в `MUSIC_PATH` и задайте `AUDIO_MODE='mix'`. Имя файла можно оставить исходным, включая скобки, если путь записан Python-строкой. Песня добавляется при mux/mix после генерации, а не как conditioning модели. Путь и длительность фрагмента проверяются до загрузки моделей; notebook показывает предпрослушивание с выбранными offset/gain. Оригинальный аудиофайл не входит в GitHub или публичную версию notebook.

## Память и две GPU

Qwen/CLIP сначала кодируют оба prompt и полностью выгружаются. При удалённой загрузке text weights лежат в приватном временном каталоге, который очищается после encoding; нужно примерно 18 GiB свободного scratch. Media cache сохраняется отдельно. Полный read-only Diffusers snapshot из Input задаётся `LOCAL_MODEL_DIR` и не изменяется.

По умолчанию `DIT_PLACEMENT='cuda:0'`: весь DiT загружается на одну T4, runner проверяет фактическое размещение и печатает карту; CPU/disk spill останавливает запуск. `'balanced'` делит блоки DiT между двумя T4 (budget 10 GiB на карту) — это model sharding для **одного ролика**, не sequence parallelism: GPU работают поочерёдно. `pipe.to(...)` и full-pipeline `enable_model_cpu_offload()` поверх карты не вызываются. Для I2V video VAE кодирует reference на той же GPU, что и DiT; после sampling весь pipeline освобождается. В обычном прогоне video decode работает на GPU 1, audio decode на GPU 0.

Режимы `'balanced'`, `'cuda:0'` и `'cuda:1'` можно сравнить замером. Placement входит в `job_key`, поэтому смена режима заново запускает denoising. `sample.json` записывает `denoise_seconds` (только вызов pipeline) и сводку фонового `nvidia-smi` (200 ms); ячейка сравнения собирает эти прогоны в таблицу. `MEASURE_PLACEMENTS=[('baseline','balanced'),('baseline','cuda:0')]` выполняет такие замеры (только denoising, без MP4) в одном прогоне, в том числе в «Save & Run All»; нехватка памяти записывается в `sample.json` как результат. Замер на Kaggle 2×T4 (version 7, по одному прогону на вариант, [отчёт](../reports/dit-placement-kaggle-measurements.json)): denoising Smoke 12,5 s (`balanced`) против 13,4 s (`cuda:0`), Baseline 746 s против 762 s; разница в пределах разброса между прогонами. Baseline целиком помещается на одну T4: 8,6 GiB PyTorch, 9,5 GiB по `nvidia-smi` из 14,6. При `balanced` обе GPU одновременно заняты в 0–7 % отсчётов. Значит, split не ускоряет один ролик и для Lite не нужен по памяти; вторая GPU полезна для второго ролика параллельно. Два параллельных pipeline ещё не измерены.

**Веса без повторной загрузки.** Kaggle хранит до 20 GB Output, а snapshot весит 24,9 GiB, поэтому он сохраняется двумя приватными CPU-notebooks: [17 `media`](../notebooks/17_kaggle_k6_weights_media.ipynb) (DiT, VAE, аудио-кодек, vocoder, CLIP, токенизаторы, ~9,4 GiB) и [18 `text`](../notebooks/18_kaggle_k6_weights_text.ipynb) (Qwen2.5-VL, ~15,5 GiB). Каждый скачивает свою часть ревизии `4c487e8b`, сверяет размеры файлов с Hub и пишет `weights-manifest.json`. После «Save & Run All» обоих добавьте их Output в notebook 16 через **Add Input → Your Work → Notebook Output**: при `AUTO_WEIGHT_INPUTS=True` ячейка конфигурации найдёт обе части той же ревизии, соберёт из symlinks один Diffusers-каталог в `/kaggle/temp` и подставит `LOCAL_MODEL_DIR`. Если найдена одна часть или ни одной, notebook скачивает веса как раньше. Генератор: [`build_weights_snapshot_notebooks.py`](../scripts/build_weights_snapshot_notebooks.py).

**Два ролика параллельно (раздел 8 notebook).** `PARALLEL_JOBS=[('orbit_woman','baseline'),('strange_tram','baseline')]`: ролики одного profile идут парами, первый целиком на `cuda:0`, второй на `cuda:1` (denoising, decode, MP4). Тексты кодируются заранее одним проходом Qwen, веса скачиваются до старта потоков. Загрузка моделей сериализована: `low_cpu_mem_usage` в accelerate временно подменяет методы `nn.Module` глобально, и две одновременные загрузки в потоках могли бы помешать друг другу. Время одного ролика не уменьшается, растёт число роликов в час. Каждый ролик пишет `parallel.json`, сводка с `wall_seconds`, `clips_per_hour` и трассой `nvidia-smi` — в `parallel-*.json`. Почему для одновременной работы двух GPU нужен context parallelism и что для этого менять в `Kandinsky6Transformer3DModel` — [разбор в сравнении с LTX-2 и Wan](../CONTEXT_PARALLEL_RU.md). Video decode затем работает на GPU 1, audio decode на GPU 0.

## Исходники и проверка

`diffusers_runner.py` — обычные функции без загрузки моделей при импорте. [Генератор](../scripts/build_diffusers_generation_notebook.py) переносит их в классические notebook cells; внешние `.py`, `CODE='...'`, ComfyUI или checkout репозитория для запуска `.ipynb` не требуются. `PROMPTS.md` строится из того же JSON и compiler. [Source manifest](source_manifest.json), [CPU API report](../reports/diffusers-generation-validation.json).

```bash
python kandinsky-6/scripts/build_diffusers_generation_notebook.py
python kandinsky-6/scripts/validate_diffusers_generation.py
# Publish a code-only version; this does not start generation:
kaggle kernels push -p kandinsky-6/generation --no-run
```

Для CPU validator нужны закреплённые Diffusers/Transformers/Accelerate и PyTorch; запуск не скачивает веса. Исходные API: [Diffusers](https://huggingface.co/docs/diffusers/en/api/pipelines/kandinsky6), [Lite](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers). Учебные объяснения остаются в [MIT route](../../curriculum/README.md) и component labs.
