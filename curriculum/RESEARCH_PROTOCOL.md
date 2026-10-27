# Переход от учебного задания к исследованию

## Шесть уровней доказательств

| Уровень | Что доказываем | Пример |
|---|---|---|
| 0: анализ | Понимаем inputs/outputs и формулы | Прочитали DiT/VAE, построили shape map |
| 1: unit experiment | Реализация совпадает с известным ответом | Euler convergence, Gaussian score, patch roundtrip |
| 2: toy learning | Сеть может выучить поставленную задачу | Held-out 2D Flow Matching, synthetic FiLM |
| 3: component parity | Перенесённый pretrained block вычисляет то же | PyTorch↔MLX с одинаковыми inputs/weights |
| 4: pipeline baseline | Реальная исходная модель запускается и измерена | ComfyUI video/audio output, latency/memory |
| 5: controlled research | Изменение даёт проверенный эффект | Adapter vs controls на held-out data и seed |

Ни один уровень не заменяет следующий. Статический workflow не доказывает inference; падающий toy loss не доказывает voice cloning.

## Минимальный эксперимент

1. Выберите один вопрос: codec reconstruction, sampler NFE/quality, conditioning, framework parity или adapter.
2. Зафиксируйте исходную реализацию, проверку корректности и конфигурацию. Сделайте baseline до вмешательства.
3. Разделите данные на train/validation/test до настройки. Выбор checkpoint и hyperparameters — по validation.
4. Задайте одну intervention и контроли. Для paired comparison используйте одинаковые target/reference/initial noise и измерительные модели.
5. Выполните несколько повторов; training seed и sampling seed записывайте раздельно. Сохраните индивидуальные результаты, не только среднее.
6. Оцените эффект с неопределённостью. Bootstrap группируйте по реальной независимой единице: speaker/clip source/subject, а не по всем кадрам как независимым людям.
7. Проверьте regression: исходные задачи без нового условия, качество/разборчивость/синхронизация и стоимость.
8. Сформулируйте ограниченный вывод. Test после повторного подбора уже не является независимым test.

## Рекомендуемые завершающие проекты

**R1, сначала:** один VAE component в Python и ComfyUI. Зафиксировать preprocessing, posterior mode/sample и latent scale; измерить reconstruction MSE/PSNR и runtime. Это доступнее full T2V, но ещё не сравнение генеративного качества.

**R2:** один поддерживаемый non-distilled pipeline, один solver, общий initial noise; NFE/latency/quality sweep. Отдельно считать CFG branch work. Визуальная шкала должна иметь описанный rubric; CLIP/CLAP не являются общей мерой эстетики или speaker identity.

**R3, после baseline/parity:** голосовой FiLM для K6. Использовать [существующий протокол](../kandinsky-6/COMFYUI_AND_EXPERIMENT_RU.md): B0/B1/B2/A1, VCTK pilot, held-out тексты, SECS/WER и no-reference regression. Frozen adapter-only — отдельная гипотеза; полноценный trainer/sampler пока не заявлен.

Для первого исследования выберите один проект. Полное обучение всех Hunyuan/Wan/Kandinsky на M3 Max не является необходимым шагом курса.
