# Состояние эксперимента, 30 октября 2026

**Учебное расширение:** [14 компонентных CPU-лабораторных](README.md), отдельные [DiT](notebooks/07_dit_and_text_crossdit.ipynb) и [Dual-stream AV CrossDiT](notebooks/08_dual_stream_audio_video_crossdit.ipynb). [Численные проверки](reports/COMPONENT_VALIDATION.md), [полное содержание](../curriculum/CONTENTS.md). CPU контроли и проверенный Desktop AV baseline не являются результатом voice cloning или MLX parity.

## Решение пользователя

Первоначальный SSD gate 50 GB+ был пересмотрен явным разрешением пользователя 30 октября: необходимые Lite веса установлены, Baseline в Desktop успешно продолжен из Sampler cache и сохранён. MLX и voice adapter training остаются отдельными непроверенными этапами.

## Выполнено

- Сохранены официальные исходники и configs с revisions/SHA-256.
- В опубликованном Lite Distill safetensors header найдены 2399 tensors; 816 audio tensors, 574 096 016 параметров. Это именованное подмножество единой модели, не отдельный эквивалентный AV generator.
- Подготовлены два 12-node Lite ComfyUI workflow; проверены endpoints/types всех 14 links и отсутствие Pro/SR/beautifier metadata.
- Baseline выполнен в ComfyUI 0.43.1 / plugin 1.0.2; Torch 2.12.1, Python 3.13.12, Apple M3 Max / MPS. Ранее подготовка выполнялась на ComfyUI 0.42.5.
- Выполнен CPU reference маленького настоящего K6 класса со случайными весами; gates сделаны ненулевыми для будущего численного сравнения.
- Скачан VCTK pilot: 120 clips / четыре speaker. После глобальной группировки текстов: reference 16, train 63, validation 20, test 21. Два первоначальных пересечения устранены; это результат подготовки данных, не качества voice cloning.
- Повторная download audio subset остановлена с сохранением progress, чтобы использовать checkpoint ComfyUI после его загрузки.

- Подготовлен пошаговый notebook из 33 ячеек; результаты выполнения лёгких ячеек — в reports/VALIDATION.md.
- Lite Smoke/Baseline установлены в example_workflows расширения и user/default/workflows/K6 Research.

- Проверен MPS AV Baseline 864×480, 121 кадр, 24 FPS, H.264 + AAC 44100 Hz. Исправлен VAE allocation 75.16 GiB заменой на tiled decoder; продолжение 563.51 s использовало Sampler cache без повторного sampling. [Измерения и границы](reports/comfy-tiled-vae-validation.json).
- Подготовлен Kaggle notebook для 2×T4 / 32 GB RAM с отдельными процессами стадий, загрузкой pinned weights и измерениями.

## Не выполнено

- Pretrained Kaggle/T4 запуск нового [deployment notebook](notebooks/15_kaggle_t4_lite_distill.ipynb): код и native CPU API проверены, полного CUDA прогона пока нет.
- Pretrained PyTorch↔MLX parity: нужен полный локальный checkpoint.
- MLX numerical fixture: установка mlx-metal столкнулась с network timeout; повторная попытка остановлена по решению отложить запуск. Проверяйте фактическое наличие отчёта `reports/mlx-random_fixture-parity.json`, а не считайте файл скрипта доказательством запуска.
- Полный joint video/audio MLX pipeline и PiFlow sampler.
- Обучение FiLM, waveform generation с adapter, SECS/WER/FAD/no-regression evaluation и Comfy custom node для FiLM.

## Исправление предпосылок

В предложенном тексте смешаны основной отчёт K6, отдельная voice-cloning статья и новые гипотезы. Voice paper дообучает также audio backbone; полностью frozen backbone с одним FiLM — наш отдельный вариант. Zero-init проверяется при одинаковом контексте; добавление reference tokens само по себе меняет attention.

KVAE-Audio 48 kHz/64 channels и MMAudio 44.1 kHz/40 channels не являются взаимозаменяемыми. Stock distilled PiFlow требует оба потока. Обещания 30×, менее 2 GB, 10–20 записей/5–15 минут не являются измерениями этого проекта.

ComfyUI checkpoints можно использовать для PyTorch и исследования MLX без повторного скачивания. Quantized variants и переносы representation имеют отдельные checks. Оригиналы не изменяются; completion stamp защищает от случайной загрузки незавершённого sparse artifact.
