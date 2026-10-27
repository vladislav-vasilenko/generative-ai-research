# Lecture 2 → Lab 2: от probability path к обучаемому полю

**До начала:** закрыты P0–P2 [общего маршрута](../curriculum/README.md). Вы умеете симулировать заданную ODE, но теперь нужно понять, откуда взять её поле. Источник: slides PDF pp.5–20,25–34; notes §3, PDF pp.14–24. [Официальные слайды](https://diffusion.csail.mit.edu/2026/docs/20260122_Lecture_02.pdf).

[Исходный Lab 2](references/lab_two_original.ipynb) — неизменённый notebook из вложения, **79 ячеек**. В нём есть `NotImplementedError`; это задания, а не готовый Run All. Сохраните свою рабочую копию, оригинал используйте для сравнения. [Происхождение и hashes](../curriculum/references/README.md).

| Шаг | Ячейки оригинала, индекс с 0 | Действие | Результат |
|---|---|---|---|
| 1 | Part 1: 8–10 | Прочитать interface `ConditionalProbabilityPath`; выписать shapes | Endpoint z, t, state x и внешнее условие y не смешиваются |
| 2 | Q2.1: 14–20 | Реализовать alpha/beta и derivatives | Проверены endpoints и interior derivatives |
| 3 | Q2.2: 21–24 | Sample conditional Gaussian | Среднее alpha*z, covariance beta² I |
| 4 | Q2.3: 25–30 | Conditional velocity | Проверить analytical transport и отсутствие division-by-zero |
| 5 | Q3.1: 38–49 | CFM loss и small training | Held-out loss + snapshots marginal samples; future clean z не передаётся sampler |
| 6 | Part 4: 64–78 | Linear path и arbitrary source/target | Gaussian source не является обязательным свойством любого FM |
| После Lecture 3A | Q2.4: 31–36; Q3.2–3.3: 50–63 | Score и conversion | Понять область определения, а не копировать формулу в endpoint |

Основная конструкция — marginal probability path:

$$
p_t(x) = \int p_t(x\mid z) p_{\mathrm{data}}(z) dz.
$$

Здесь `z` — чистый endpoint, а `x` — текущее состояние. Conditional fields усредняются с **posterior weights**, зависящими от наблюдаемого состояния:

$$
p(z\mid X_t=x).
$$

Это не неизменные prior weights. Регрессия на доступный conditional target даёт marginal predictor при достаточной модели и корректном обучении.

## Три опыта

1. **Schedule:** при фиксированных data/model/steps сравните выбранные alpha/beta. Не меняйте одновременно architecture и learning rate. Запишите target magnitude по t и ошибки возле endpoints.
2. **Learning vs integration:** сохраните одну trained сеть; уменьшайте шаг только sampler. Затем при одном solver сравните training checkpoints. Так различаются approximation error сети и discretization error.
3. **Arbitrary endpoints:** выполните Part 4.3 на двух 2D distributions; сравните mean/covariance и plots. Один scatter plot не доказывает совпадение плотностей.

В [вводном notebook](../curriculum/notebooks/00_probability_and_model_contracts.ipynb) отдельно показаны posterior marginalization и reverse time. В [ComfyUI C2](../curriculum/COMFYUI_LABS.md) этот навык переносится в контроль sampler/steps/NFE, но полная генерация пока отложена до 50 GB+ SSD.

**Готовность:** без кода объясните, почему обучение использует известный endpoint, а inference его не знает; выведите знак target при замене t на 1-t; покажите один held-out numerical result.

При этом порядке не запускайте весь незаполненный original через Run All: после Q2.3 переходите к Q3.1/Part 4, пропуская score-specific cells и их визуализации до Lecture 3A. В новой сессии сначала повторно выполните Part 0 и необходимые class definitions.
