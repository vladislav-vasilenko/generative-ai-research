# Kandinsky 6 Lite: читаемый код Kaggle pipeline

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../../curriculum/README.md) · [Полное содержание](../../curriculum/CONTENTS.md).
<!-- study-navigation:end -->

[Главная](../../README.md) · [K6](../README.md) · [Notebook 15](../notebooks/15_kaggle_t4_lite_distill.ipynb) · [Публичный Kaggle](https://www.kaggle.com/code/applehawk/15-kaggle-t4-lite-distill).

## Как читать notebook

Notebook состоит из обычных Markdown/Python-ячеек. Девять стандартных `%%writefile` ячеек содержат Python с подсветкой синтаксиса и сохраняют его в небольшие модули. Исходник не спрятан в `CODE = "..."`; для изменения функции редактируйте её обычный текст в соответствующей ячейке. После изменения выполните эту ячейку и повторите нужную стадию.

Отдельные процессы сохраняют исходное ограничение памяти: Qwen, DiT, video VAE и audio decoder загружаются по очереди, а память CUDA освобождается при завершении процесса. Notebook управляет запуском и показывает журналы. Импорт всех моделей в живущий Jupyter kernel изменил бы это поведение.

| Модуль | Назначение |
|---|---|
| `k6_common.py` | ComfyUI bootstrap, device, conditioning cache, timings и метрики |
| `k6_text.py` | Qwen + CLIP, положительное/отрицательное conditioning |
| `k6_sample.py` | Один совместный video/audio latent; DX π-Flow, подсчёт DiT NFE |
| `k6_video.py` | FP32 Hunyuan VAE, сохранение кадров |
| `kaggle_parallel_vae_tiles.py` | Два процесса декодируют тайлы одного video latent; штатная сборка ComfyUI |
| `k6_audio.py` | TOD-VAE, BigVGAN, сохранение waveform |
| `k6_export.py` | MP4 с видео/аудио, проверка декодирования |
| `run_stage.py` | Минимальная точка входа worker-процесса |
| `download_models.py` | Точечная загрузка и SHA256 проверка семи файлов |

## Что используют две GPU

`CFG["video_tile_gpus"] = [0, 1]` распределяет чётные и нечётные VAE-тайлы одного ролика между двумя физическими GPU. Координатор складывает результаты в том же порядке, с теми же feather masks и финальной нормализацией, что `VAEDecodeTiled`. Очередь ограничена двумя результатами. Smoke: 6 тайлов, baseline: 150 при исходных параметрах.

Это параллелизм video VAE. Joint DiT sampling остаётся на GPU 0; text encoding — на GPU 1 на предыдущей стадии. Две T4 не предоставляют единого буфера 32 GB. Для контроля задайте `[0]` и сравните время, пик памяти и совпадение кадров на тех же `joint-latents.safetensors`. Наличие двух процессов само по себе не доказывает ускорение: загрузка второй копии VAE и передача тайлов имеют цену.

## Как продолжить уже рассчитанный ролик

1. Сохраните `joint-latents.safetensors`, конфигурацию, `sample-metrics.json` и доступный MP4 из Output. Не выполняйте sampling заново ради decoder.
2. Выполните обновлённые `%%writefile` ячейки и ячейку определения `run_stage`. Дождитесь завершения уже запущенного worker перед заменой его файлов.
3. Оставьте prompts, seed, dimensions и `CFG["run_root"]` прежними. Установите `CFG["video_tile_gpus"] = [0, 1]`.
4. Выполните `VIDEO_METRICS = run_stage("video", CFG)`, затем audio/export, если эти стадии ещё не завершены. Удалённые после text stage веса Qwen/CLIP для этого не требуются.
5. Проверьте `tile_parallelism.tiles_per_worker`, журнал `VIDEO_TILE`, GPU monitor и MP4 streams. Для A/B сначала скопируйте исходные `frames.npy` и `video-metrics.json` под другими именами: повтор decoder перезапишет эти файлы.

Если kernel перезапущен, обычных переменных `CFG`, `PYTHON`, `SCRATCH`, `WORKER` и функции `run_stage` в памяти уже нет. Восстановите настройки и модули, затем загрузите сохранённую конфигурацию. Файлы в `/kaggle/temp` не переживают удаление сессии; храните нужные latents и отчёты в Output.

## Синхронизация и проверка

`build_kaggle_notebook.py` пишет одинаковые исходники в notebook и в эту папку. `kaggle_parallel_vae_tiles.py` — исходный адаптер; остальные `.py` и `manifest.json` генерируются. Вносите правки в генератор, для адаптера — в его `.py`, затем пересобирайте:

```bash
python kandinsky-6/scripts/build_kaggle_notebook.py
python kandinsky-6/scripts/validate_kaggle_notebook.py --comfy-root /path/to/ComfyUI
python kandinsky-6/scripts/validate_parallel_vae_tiles.py --comfy-root /path/to/ComfyUI
# Дополнительно: уже скачанный VAE, два CPU-процесса, маленький synthetic latent
python kandinsky-6/scripts/validate_parallel_vae_tiles.py --comfy-root /path/to/ComfyUI --vae-path /path/to/hunyuan_video_vae_bf16.safetensors
```

Проверки сопоставляют notebook, runtime-файлы и manifest; выполняют `%%writefile`; сравнивают tiler с native ComfyUI и проверяют ошибки worker/таймаут. Отдельный CPU integration test использует существующие pretrained VAE веса. Это проверка декодирования искусственного latent, не генерация Kandinsky. Результаты: [notebook](../reports/kaggle-t4-validation.json), [двухпроцессная очередь](../reports/parallel-vae-tiles-validation.json), [pretrained CPU VAE](../reports/parallel-vae-pretrained-cpu-validation.json). CUDA two-GPU parity и ускорение пока не измерены.

## Публикация через официальный Kaggle CLI

[Документация Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md). Установите `kaggle`, выполните `kaggle auth login`. Сначала сохраните текущую версию и её metadata через `kaggle kernels pull applehawk/15-kaggle-t4-lite-distill -p /tmp/k6-backup -m`. Проверьте [kernel-metadata.json](kernel-metadata.json): он указывает существующий public notebook и GPU T4 ×2. `kaggle kernels push -p kandinsky-6/kaggle_runtime --no-run` публикует код без запуска всех ячеек. При добавлении Dataset Input сначала обновите metadata через pull и сохраните dataset_sources. Обычный `push` запускает notebook заново.
