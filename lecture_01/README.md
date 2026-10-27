# Lecture 1: ОДУ, СДУ и численное семплирование

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../curriculum/README.md) · [Полное содержание](../curriculum/CONTENTS.md).

**Связанные лекции:** [Lecture 1](https://diffusion.csail.mit.edu/2026/docs/20260120_Lecture_01.pdf).
<!-- study-navigation:end -->

Проект по [первой лекции MIT 6.S184, 2026](https://diffusion.csail.mit.edu/2026/docs/20260120_Lecture_01.pdf) и предоставленной лабораторной `lab_one.ipynb`.

## Место в общем курсе

Это первые два этапа [сквозного маршрута Lecture 1–5](../curriculum/README.md). Перед Lab 1 выполните небольшой [мост по многомерным вероятностям](../curriculum/notebooks/00_probability_and_model_contracts.ipynb). После ODE/SDE, Gaussian score и Langevin переходите к [Lecture 2 / Lab 2](../lecture_02/README.md), где поле уже не задаётся вручную, а обучается.

## Что открыть

- [Полный разбор](lab_one_ru.ipynb) — перевод, реализованные методы, ответы, графики и проверки. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vladislav-vasilenko/generative-ai-research/blob/main/lecture_01/lab_one_ru.ipynb)
- [Самостоятельная работа](lab_one_ru_exercises.ipynb) — тот же перевод: восемь методов и текстовые ответы нужно заполнить. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vladislav-vasilenko/generative-ai-research/blob/main/lecture_01/lab_one_ru_exercises.ipynb)
- [HTML для чтения](outputs/lab_one_ru.html) — выполненная лабораторная с графиками.
- [Конспект Lecture 1](LECTURE_01_RU.md) — основные идеи и формулы.
- [Происхождение материалов](references/README.md) — оригинал, атрибуция и изменения.

## Запуск локально

Команды выполняются **из корня репозитория**. Установите [uv](https://docs.astral.sh/uv/getting-started/installation/); Python 3.12 задан в `.python-version`, версии библиотек — в `uv.lock`.

```bash
uv sync --locked
uv run --locked jupyter lab lecture_01/lab_one_ru_exercises.ipynb
```

В Jupyter или VS Code выберите ядро Python из корневой `.venv`. В VS Code: **Select Kernel → Select Another Kernel → Python Environments → `.venv`**. Команда `import sys; print(sys.executable)` в ячейке должна показать `.venv/bin/python` (в Windows — `.venv\Scripts\python.exe`). Активация окружения не требуется. Все этапы доступны на CPU; GPU не требуется.

Выполняйте ячейки сверху вниз через **Shift+Enter**. Полный запуск: **Kernel → Restart Kernel and Run All Cells**. В разделах есть условие, инструкция, код, рисунок и разбор. После изменения класса заново выполните ячейки, создающие его экземпляры.

## Google Colab

Нажмите **Open in Colab** рядом с нужным ноутбуком выше и выполняйте ячейки по порядку. При отсутствии библиотек выполните `%pip install torch numpy matplotlib seaborn scipy tqdm` в отдельной ячейке.

## Маршрут по этапам

| Раздел | Что сделать | Что проверить | Что менять |
|---|---|---|---|
| 0 | Импорты и интерфейсы | `xt`: `(B, d)`; `t`, `h`: скаляры | Размерность и число частиц |
| 1.1 | Два метода `step` | Эйлер приближается к точному решению | Число узлов при постоянном времени |
| 2.1 | Нулевой снос, постоянная диффузия | Среднее около нуля; дисперсия `sigma**2 * T` | Только `sigma`, затем только `T` |
| 2.2 | Снос `-theta * xt` | Возврат и стационарная дисперсия `sigma**2/(2*theta)` | Сначала шум, затем силу возврата |
| 2.2, сетка D | Строки с общим D | Одинаковая стационарная дисперсия; время масштабировано | `ds`, `sigmas`, время |
| 3.1, плотности | `Density`, `Sampleable`, гауссиана и смесь | Различие `sample`, `log_density`, `score` | Средние, ковариации, веса |
| 3.1, Ланжевен | СДУ и снимки распределения | Снос содержит `sigma**2/2` | Шаг и длительность по отдельности |
| Анимация | `RUN_ANIMATION=True` | Встроенные кадры без ffmpeg | `animate_every` |
| 3.2 | Производная логарифма гауссианы | OU с теми же коэффициентами и стартом | Положительные `theta`, `sigma` |
| Контроль | Эксперимент на одной гауссиане | Среднее и дисперсия с учётом погрешности | Число частиц и шаг |

## Проверки и повторное получение результатов

```bash
uv run --locked pytest lecture_01/tests -q
uv run --locked python lecture_01/scripts/execute_notebook.py --animation
```

Первая команда проверяет аналитические решения, градиенты и полноту перевода. Вторая выполняет все ячейки в чистом ядре, сохраняет результаты и обновляет HTML. Без `--animation` запуск пропустит анимацию.

Для правок перевода есть [сборщик](scripts/build_notebooks.py). **Он перезаписывает обе русские версии и очищает их выводы**; свои ответы храните в отдельной копии. После правок:

```bash
uv run --locked python lecture_01/scripts/build_notebooks.py
uv run --locked python lecture_01/scripts/execute_notebook.py --animation
```

## Мини-отчёт

Для каждого задания запишите: **цель → параметры и seed → теоретическое ожидание → рисунок или числа → объяснение расхождений → вывод**. Различайте ошибку численного метода, конечное число частиц и недостаточное время перемешивания.
