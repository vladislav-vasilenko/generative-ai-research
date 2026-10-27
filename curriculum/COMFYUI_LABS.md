# ComfyUI как наблюдаемый эксперимент

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](README.md) · [Полное содержание](CONTENTS.md).
<!-- study-navigation:end -->

Для каждой работы сначала выполняйте математический/PyTorch вариант, затем наблюдайте соответствующее поведение в graph. ComfyUI показывает dataflow и запускает inference. Он не заменяет вывод loss, доказательство marginalization или контроль gradients.

## Подготовленные workflow

| Файл | Что требует | Что проверено |
|---|---|---|
| [C0: image IO](workflows/00_image_io_no_models.json) | Только встроенные LoadImage/SaveImage; без weights | JSON и связи; runtime ещё не запускался |
| [C4: Hunyuan VAE reconstruction](workflows/04_hunyuan_vae_reconstruction.json) | Только уже имеющийся `hunyuan_video_vae_bf16.safetensors`; без Qwen/DiT | JSON/links и типы built-in nodes; runtime ещё не запускался |
| [K6 Smoke](../kandinsky-6/workflows/k6_lite_distill_smoke.json) | Полный минимальный комплект Lite+Qwen/CLIP+codecs | Статическая проверка; полный запуск отложен |
| [K6 Baseline](../kandinsky-6/workflows/k6_lite_distill_baseline.json) | Тот же комплект, больше frames/resolution | Статическая проверка; полный запуск отложен |

C0/C4 подготовлены для установленного ComfyUI 0.42.5. Это собственные учебные graphs, не официальные templates производителя. Они не отправляются в очередь автоматически. **Полный pretrained запуск и скачивание больших моделей остаются отложены до 50 GB+ SSD.** Сам по себе 50 GB gate не гарантирует достаточную RAM.

Как открыть: перетащите JSON на canvas, затем в LoadImage загрузите [тестовую PNG](assets/curriculum_test_chart.png) через upload. До выбора файла placeholder имени недостаточно: PNG должна появиться в списке input files Desktop. Проверьте имя VAE в loader. Сохраните свою рабочую копию workflow; исходные JSON не содержат ваших ответов.

## C0. Контроль dataflow до нейросети

**Связь:** P0, tensors и preprocessing. **Вопрос:** прошли ли через pipeline именно те pixels, которые мы думаем измерять?

1. Открыть image IO graph и загрузить chart.
2. Выполнить вручную, когда решите начать практику. Зафиксировать input/output filename и форму.
3. Сравнить RGB arrays; SHA всего PNG может отличаться из-за metadata даже при одинаковых pixels.
4. При различии на один уровень 8-bit проверить quantization, alpha/color handling, а не приписывать это модели.

Компаньон Python:

```bash
uv run --locked python curriculum/scripts/compare_images.py --original curriculum/assets/curriculum_test_chart.png --reconstruction /ПУТЬ/К/output.png
```

Это контроль файлового pipeline, не генеративная модель.

## C1. Seed, повторяемость и cache

**Связь:** P1–P2, initial random state. Используйте один установленный pipeline с известными semantics seed.

Сохраните prompt/model/geometry/precision. Повторите один seed в свежем процессе, затем измените только seed. Сохраните общий initial noise tensor, если хотите сравнивать разные samplers: один seed не гарантирует одинаковое потребление RNG в разных алгоритмах/frameworks.

Warm timings измеряйте на новых seeds (например 43/44/45), проверяя реальное исполнение sampler. Повтор неизменного graph может вернуть cache hit. Bitwise equality на разных backends не гарантируется. Отдельно измеряются deterministic reproducibility и visual similarity.

## C2. Шаг интеграции, solver и NFE

**Связь:** Lecture 1 + Lecture 2, Lab 2. На поддерживаемом **non-distilled** checkpoint выберите один solver и sweep steps, например 8/16/32, при общем initial noise. Запишите time grid/shift и фактические neural evaluations.

Нельзя считать число видимых steps универсальным NFE. CFG может добавить branch work; higher-order solver или policy sampler имеют другую структуру. Для K6 Distill видимые euler/simple selectors не определяют фактический PiFlow loop. Его основной baseline оставляем на официальном profile; перенос sweep — отдельный вариант с отдельным выводом.

Результат: таблица latency/NFE/ошибок и слепая визуальная оценка на нескольких prompts/seeds. Меньше steps не означает автоматически лучше или хуже; это проверяемый tradeoff.

## C3. Classifier-free guidance

**Связь:** Lecture 3B / Lab 3 Part 2. Используйте pipeline, обученный и реализованный с обычным CFG. При фиксированных initial noise/conditions сравните w=1 и несколько w>1; если поддержано — w=0.

Покажите соответствие prompt, diversity, artefacts и compute cost. Сначала проверьте в Python identity w=1 и label dropout. Negative prompt в UI может быть самостоятельным условием, а не truly unconditional branch.

**K6 Lite Distill с CFG=1 не используем как универсальную демонстрацию guidance.** Переход к SFT checkpoint требует других weights и явной конфигурации; автоматически их не скачиваем.

## C4. VAE reconstruction отдельно от генератора

**Связь:** Lecture 4 / Lab 3 Part 4, [K5 VAE lab](../kandinsky-5/notebooks/02_vae.ipynb).

Graph: `LoadImage → VAEEncode → VAEDecode → SaveImage`, второй SaveImage сохраняет исходник. В VAEEncode и VAEDecode приходит один и тот же VAE. Qwen, CLIP, DiT и sampler отсутствуют.

1. Начать с chart 256×256, затем собственное изображение со сторонами, кратными codec constraints.
2. Записать actual shape и cropping, filename/hash codec, encode mode/sample semantics, dtype, cold/warm time и observed memory.
3. Сравнить исходник с reconstruction: одинаковые shape, RGB `[0,1]`, MSE/PSNR; никаких скрытых resize в метрике.
4. Сопоставить со своим VAE-кодом: `logvar`, posterior sample/mode, latent scale/shift и decoder. Не ожидать идентичности между разными VAE.
5. Повторить на текстуре/градиенте/тексте/фото. Объяснить, где codec теряет детали.

Этот граф не показывает KL или ELBO и не обучает VAE. Reconstruction quality не равна качеству генерации latent prior/DiT. PSNR — pixel metric, а не общая эстетическая оценка.

## C5. Resolution, frames и attention budget

**Связь:** Lecture 4 / Lab 3 Part 3. В Python посчитать tokens после codec и patchify, затем в одном реальном pipeline менять только resolution или frames.

При удвоении H и W число spatial tokens растёт примерно в 4 раза, а naive full-attention score entries — в 16. Реальная память/latency зависит от kernels, caching, dtype и offload: не выдавайте эту формулу за измерение Mac. Не сравнивайте два профиля как один factor, если изменены одновременно H/W/T.

Сохранять cold/warm times, observed peak и swap. CPU/GPU Apple используют общую память; offload не увеличивает физический пул 48 GB.

## C6. Что ComfyUI не вычисляет автоматически

После [CTMC notebook](../lecture_05/05_ctmc_and_discrete_flow.ipynb) можно загрузить `curriculum/artifacts/ctmc_probability.png` в C0 graph как демонстрационный image. Расчёт CTMC остаётся в Python. Стандартный KSampler не является jump-process simulator.

После реализации custom CTMC node нужно отдельно проверить rates/probability conservation и output parity. Аналогично, готовый FiLM adapter требует совместимого sampler/custom node; обычный LoRA Loader не обязан его понимать.

## Общий журнал

Используйте [карточку опыта](EXPERIMENT_TEMPLATE.md). Каждый вывод сопровождайте config, individual runs и артефактами. Статусы `prepared`, `static_checked`, `executed`, `measured` различаются. В этом наборе ComfyUI graphs пока `static_checked`.
