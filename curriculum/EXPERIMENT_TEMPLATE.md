# Карточка учебного или исследовательского эксперимента

Скопируйте в собственный отчёт и заполните до запуска. Если запуск отложен, оставляйте статус `not_run`, а не предполагаемые числа.

- **ID / этап:** например P3-FM-schedule или C4-VAE-reconstruction.
- **Вопрос и H1:** какое одно изменение должно повлиять на какой измеримый результат?
- **Теория:** формула, time convention, shapes, область определения, предсказание.
- **Baseline / intervention:** что меняется; какие параметры/данные/weights остаются одинаковыми?
- **Данные:** источник/revision, split, единица наблюдения, preprocessing, отсутствие утечки.
- **Config:** model/component hash, precision, geometry, batch, solver, time grid, NFE, prompt/seed.
- **Среда:** commit, Python/Torch/MLX/Comfy/plugin versions, CPU/MPS/CUDA, свободный SSD.
- **Метрики:** имена, единицы, способ агрегации, ограничения инструмента.
- **Порог/критерий:** что заранее будет означать ошибку реализации, успех H1 или tradeoff?
- **Результаты:** таблица, графики, individual runs, error log; отдельно cold/warm/cache hit.
- **Интерпретация:** numerical error / Monte Carlo error / model error / resource limit.
- **Вывод:** что данные подтверждают, чего не показывают, какой следующий один эксперимент?

| run_id | variant | train_seed | inference_seed | split | value | units | cold/warm | status | artifact |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | not_run | |

Один удачный кадр — пример, не распределённая оценка. Несколько seed одного training run не заменяют несколько независимых trainings. Наблюдаемый peak RAM при опросе раз в секунду может быть ниже настоящего пика.
