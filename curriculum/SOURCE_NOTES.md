# Уточнения к исходным материалам

Оригиналы сохраняются побайтово; этот файл не исправляет их молча. Для источников даны SHA-256 и zero-based cell indexes в [manifest](references/materials_manifest.json). Ни одна исходная лабораторная автоматически не исполняется.

## Lab 2: endpoint в Q3.3

В markdown cell 59 заданы `alpha(t) = t` и `beta(t) = sqrt(1-t)`. Для interior interval:

$$
\alpha_t=t,\qquad \beta_t=\sqrt{1-t},\qquad 0\le t<1.
$$

Тогда denominator преобразования velocity → score равен:

$$
\beta_t^2\dot{\alpha}_t-\alpha_t\dot{\beta}_t\beta_t = 1-\frac{t}{2}.
$$

Предел при приближении к `t = 1` слева равен `1/2`, а не `0`. В исходнике показано отдельное значение 0 при endpoint; использовать его как алгебраическое продолжение нельзя. Производная beta сингулярна, conditional Gaussian вырождается. Используйте interior t и отдельно рассматривайте limit/marginal. Это не доказывает общую невозможность marginal score.

## Lab 3: обозначения и shapes

- **Markdown cell 24:** при фиксированном классе `y` речь о распределении данных `p_data(x ∣ y)`. Встречающееся `p_simple(x|y)` не должно превращать источник шума в data distribution.
- **Q3.2, cell 40:** число image tokens равно `(H / p) * (W / p)`. Проверяйте divisibility и точное обратное unpatchify.
- **Part 4, cell 55:** `logvar` означает логарифм variance, поэтому `std = exp(0.5 * logvar)`. В пояснении встречается log sigma; следуйте одной выбранной параметризации.

## Общие уточнения

- **Lecture 5.** CTMC state не следует обычной ODE/SDE в пространстве token IDs. Однако вектор вероятностей удовлетворяет ODE Колмогорова. Это разные уровни описания.
- **Нумерация/даты.** Файлы и страница курса относятся к 2026; на некоторых титульных слайдах осталась дата 2025. В маршруте используем filenames, хеши и реальные PDF pages, а не эту подпись.
- **Inference и training.** ComfyUI VAE reconstruction не измеряет ELBO/KL и не обучает VAE. CFG slider не добавляет unconditional training; distilled checkpoint может не поддерживать обычный guidance sweep.

Это целевые уточнения для маршрута, не утверждение о полном аудите всех решений/ячеек и не официальное errata MIT.
