# Состояние эксперимента, 27 октября 2026

## Решение пользователя

Загрузку больших весов и генерацию отложили до освобождения **50 GB+** на диске. При подготовке было около 24.2 GiB свободно. Установка MLX остановлена; полные модели не запускались.

## Выполнено

- Сохранены официальные исходники и configs с revisions/SHA-256.
- В опубликованном Lite Distill safetensors header найдены 2399 tensors; 816 audio tensors, 574 096 016 параметров. Это именованное подмножество единой модели, не отдельный эквивалентный AV generator.
- Подготовлены два 12-node Lite ComfyUI workflow; проверены endpoints/types всех 14 links и отсутствие Pro/SR/beautifier metadata.
- Проверено окружение пользователя: ComfyUI 0.42.5 / plugin 1.0.2; Torch 2.12.1, Python 3.13.12. Малый Metal Tensor успешно вычислен.
- Выполнен CPU reference маленького настоящего K6 класса со случайными весами; gates сделаны ненулевыми для будущего численного сравнения.
- Скачан VCTK pilot: 120 clips / четыре speaker. После глобальной группировки текстов: reference 16, train 63, validation 20, test 21. Два первоначальных пересечения устранены; это результат подготовки данных, не качества voice cloning.
- Повторная download audio subset остановлена с сохранением progress, чтобы использовать checkpoint ComfyUI после его загрузки.

- Подготовлен пошаговый notebook из 33 ячеек; результаты выполнения лёгких ячеек — в reports/VALIDATION.md.
- Lite Smoke/Baseline установлены в example_workflows расширения и user/default/workflows/K6 Research.

## Не выполнено

- Полная генерация K6 в ComfyUI: в проверенных model folders пока найдены только CLIP и video VAE.
- Pretrained PyTorch↔MLX parity: нужен полный локальный checkpoint.
- MLX numerical fixture: установка mlx-metal столкнулась с network timeout; повторная попытка остановлена по решению отложить запуск. Проверяйте фактическое наличие отчёта `reports/mlx-random_fixture-parity.json`, а не считайте файл скрипта доказательством запуска.
- Полный joint video/audio MLX pipeline и PiFlow sampler.
- Обучение FiLM, waveform generation с adapter, SECS/WER/FAD/no-regression evaluation и Comfy custom node для FiLM.

## Исправление предпосылок

В предложенном тексте смешаны основной отчёт K6, отдельная voice-cloning статья и новые гипотезы. Voice paper дообучает также audio backbone; полностью frozen backbone с одним FiLM — наш отдельный вариант. Zero-init проверяется при одинаковом контексте; добавление reference tokens само по себе меняет attention.

KVAE-Audio 48 kHz/64 channels и MMAudio 44.1 kHz/40 channels не являются взаимозаменяемыми. Stock distilled PiFlow требует оба потока. Обещания 30×, менее 2 GB, 10–20 записей/5–15 минут не являются измерениями этого проекта.

ComfyUI checkpoints можно использовать для PyTorch и исследования MLX без повторного скачивания. Quantized variants и переносы representation имеют отдельные checks. Оригиналы не изменяются; completion stamp защищает от случайной загрузки незавершённого sparse artifact.
