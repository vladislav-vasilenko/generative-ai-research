# Generative AI Research

Учебный репозиторий по генеративным моделям: от численного решения ОДУ и СДУ до обучения DDPM на изображениях. Материалы снабжены русскими пояснениями, формулами и экспериментами.

## С чего начать

| Материал | Содержание | Подготовка |
|---|---|---|
| [Конспект Lecture 1](lecture_01/LECTURE_01_RU.md) | Выборки, векторные поля, Эйлер, броуновское движение | Вероятность и производные |
| [Lab One: полный разбор](lecture_01/lab_one_ru.ipynb) | Перевод всех 49 исходных ячеек, решения, графики и проверки | Python, базовый PyTorch; достаточно CPU |
| [Lab One: упражнения](lecture_01/lab_one_ru_exercises.ipynb) | Самостоятельная реализация тех же методов | После условий и конспекта |
| [Lab One: HTML](lecture_01/outputs/lab_one_ru.html) | Выполненная работа без запуска Python | Браузер; для формул может требоваться интернет |
| [DDPM на CIFAR-10](Diffusion_DDPM_Research.ipynb) | Добавление шума, MLP, простая CNN и обратная генерация | Для полноценного обучения желательно CUDA GPU |

Порядок: **конспект → Lab One с пояснениями → упражнения → DDPM**. Lecture 1 посвящена механике потоков и диффузий. Обучение DDPM идёт дальше первой лекции и служит отдельным продолжением.

## Установка

Рекомендуется Python **3.11 или 3.12**. Из корня репозитория:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

В Windows активация: `.venv\Scripts\activate`. Для одной Lab One достаточно `lecture_01/requirements.txt`. Для CUDA выбирайте совместимую пару PyTorch/torchvision по [официальной инструкции](https://pytorch.org/get-started/locally/).

В Jupyter выберите Python из `.venv` и выполняйте ячейки сверху вниз через **Shift+Enter**. Полная проверка: **Kernel → Restart Kernel and Run All Cells**. Тренировочные ячейки DDPM действительно обучают модели; для чтения теории выполнять их не нужно.

Ноутбуки можно загрузить в [Google Colab](https://colab.research.google.com/) через **File → Upload notebook**. Lab One не требует датасета. DDPM скачивает CIFAR-10 в `data/` при первом выполнении ячейки данных и сохраняет веса в текущем каталоге. Эти файлы исключены из Git.

## Lecture 1 / Lab One

[Инструкция по каждому этапу](lecture_01/README.md) описывает запуск, параметры и ожидаемые результаты:

1. Интерфейсы ОДУ и СДУ, формы тензоров.
2. Эйлер и Эйлер—Маруяма; почему используется `sqrt(h)`.
3. Броуновское движение: рост дисперсии.
4. OU: возврат к нулю и стационарность.
5. Плотности, выборки и скор-функции.
6. Ланжевен: преобразование распределения и анимация.
7. Вывод OU из Ланжевена и количественный контроль.

Основная версия содержит решения. `_exercises` намеренно останавливается на `NotImplementedError`, пока методы не реализованы. [Оригинал и атрибуция](lecture_01/references/README.md) позволяют свериться с предоставленной лабораторной MIT.

## DDPM: что реализовано

[Ноутбук](Diffusion_DDPM_Research.ipynb) разбирает нормализацию, расписание дисперсий, зашумление за один вызов, обучение предсказателя шума и обратную цепочку.

- **MLP** разворачивает изображение в 3072 числа.
- **ConvDiffusion** — простая CNN без уменьшающей/увеличивающей ветвей и skip-связей; это не полноценная U-Net.
- Метки классов не используются: генерация безусловная.
- Цель обучения — среднеквадратичная ошибка предсказания шума.
- Обратный шаг использует фиксированную дисперсию `beta_t`; отличие от апостериорной дисперсии объясняется в тексте.

Читайте пояснения над каждой ячейкой. Начните с небольшого запуска, затем увеличивайте число эпох. Снижение тренировочной ошибки само по себе не гарантирует качественные изображения. Малые модели здесь служат пониманию алгоритма.

## Проверка

```bash
python -m pytest lecture_01/tests tests -q
python lecture_01/scripts/execute_notebook.py --animation
```

Тесты используют аналитические формулы и синтетические данные, без полноценного обучения на CIFAR-10. Вторая команда выполняет Lab One в чистом ядре и обновляет результаты и HTML. [Отчёт о проверке](VALIDATION.md) описывает проверенную функциональность и ограничения.

## Структура

```text
Diffusion_DDPM_Research.ipynb    DDPM: комментарии и формулы
requirements.txt               Общее окружение
VALIDATION.md                  Результаты проверки
tests/                         Проверки DDPM
lecture_01/
  README.md                    Инструкция
  LECTURE_01_RU.md              Конспект
  lab_one_ru.ipynb              Выполненный разбор
  lab_one_ru_exercises.ipynb    Самостоятельная работа
  requirements.txt             Зависимости Lab One
  references/                  Оригинал и атрибуция
  scripts/                     Сборка перевода и выполнение
  tests/                       Проверки численных методов
  outputs/                     HTML
```

## Статьи, книги и дополнительные разборы

| Источник | Когда читать |
|---|---|
| [MIT, 2026](https://diffusion.csail.mit.edu/2026/) и [Lecture 1](https://diffusion.csail.mit.edu/2026/docs/20260120_Lecture_01.pdf) | Основной маршрут курса |
| [Ho, Jain, Abbeel — DDPM, 2020](https://arxiv.org/abs/2006.11239) | Уравнения (2), (4), (7), (11), (14), алгоритмы 1–2 |
| [Song et al. — Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) | Связь DDPM, score и непрерывных СДУ |
| [Lilian Weng — What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | Дополнительный разбор прямого/обратного процесса и потерь |
| [Блог Lilian Weng](https://lilianweng.github.io/) | Дальнейшее чтение о генеративных моделях |
| [Higham — Numerical Simulation of SDEs, 2001](https://epubs.siam.org/doi/10.1137/S0036144500378302) | После численных схем Lab One |
| [Øksendal — Stochastic Differential Equations](https://link.springer.com/book/10.1007/978-3-642-14394-6) | Строгая теория СДУ |
| [Goodfellow, Bengio, Courville — Deep Learning](https://www.deeplearningbook.org/) | Главы 3, 6, 8, 20: вероятность, сети, оптимизация, генеративные модели |
| [torchvision: CIFAR10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html) | Загрузка и параметры датасета |

Внешние материалы остаются работами их авторов. Адаптация Lab One обозначена отдельно и не является официальным переводом MIT.
