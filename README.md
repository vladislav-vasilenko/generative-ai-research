# Generative AI Research

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](curriculum/README.md).
<!-- study-navigation:end -->

Учебный и исследовательский стенд по генеративным моделям: **многомерные вероятности → ODE/SDE → Flow Matching и score → CFG → DiT/VAE → видео и аудио**. Материалы на русском; исходные MIT labs и официальный код моделей сохранены отдельно.

**Начните с [учебной карты](curriculum/README.md).** Она указывает страницы лекций, задания, эксперименты и критерии готовности. Для внешнего review сначала прочитайте этот README, затем [результаты проверок](curriculum/reports/VALIDATION.md) и [границы K6-эксперимента](kandinsky-6/STATUS.md).

## Быстрый запуск без больших весов

Нужны Python 3.12 и [uv](https://docs.astral.sh/uv/getting-started/installation/). Команды выполняются из корня checkout, содержащего `curriculum/`:

```bash
uv sync --locked
uv run --locked jupyter lab curriculum/notebooks/00_probability_and_model_contracts.ipynb
```

`uv sync` устанавливает Python-зависимости. Сам стартовый notebook работает на CPU, не скачивает датасеты/веса и не отправляет задания в ComfyUI. В Jupyter/VS Code выберите kernel из корневой `.venv`.

Проверка учебных примеров в свежих ядрах:

```bash
uv run --locked python curriculum/scripts/validate_curriculum.py --execute
uv run --locked python curriculum/scripts/check_markdown_math.py
uv run --locked python curriculum/scripts/sync_study_navigation.py --check
```

## Последовательность изучения

| Этап | Материалы | Что должно получиться |
|---|---|---|
| 0. Вероятности и shapes | [Вводный notebook](curriculum/notebooks/00_probability_and_model_contracts.ipynb) | Совместная Gaussian, score, marginalization, соглашения о времени |
| 1. ODE/SDE | [Lecture 1 / Lab 1](lecture_01/README.md) | Численные траектории и проверки моментов |
| 2. Flow Matching | [Lecture 2 / Lab 2](lecture_02/README.md) | Probability paths, velocity target и обучение поля |
| 3. Score и CFG | [Lecture 3 / Lab 2–3](lecture_03/README.md) | Score conversion, stochastic sampling, conditional generation |
| 4. Архитектуры и latents | [Lecture 4 / Lab 3](lecture_04/README.md) | DiT, VAE и latent generative model |
| 5. Дискретная ветвь | [Lecture 5 / CTMC](lecture_05/README.md) | Jump processes, rate matrix и categorical denoising |
| 6. Промышленные модели | [Карта промышленных моделей](curriculum/INDUSTRY_BRIDGE.md) | Найти те же контракты в официальном коде |
| 7. Исследование | [Протокол эксперимента](curriculum/RESEARCH_PROTOCOL.md) | Baseline, один фактор, измерения и ограниченный вывод |

Lab 2 связывает Lecture 2–3, Lab 3 — Lecture 3–4. CTMC — отдельная ветвь, не обязательная предпосылка непрерывных видеомоделей. **Формула → предсказание → код → измерение → объяснение** повторяются на каждом этапе. [Самопроверка](curriculum/PROGRESS.md) отделяет исполнение кода от освоения темы.

## Практические маршруты

- **[ComfyUI](curriculum/COMFYUI_LABS.md):** визуальные опыты, image IO без модели, отдельный VAE reconstruction, затем sampler/CFG/ресурсные эксперименты. JSON лежат в `curriculum/workflows/`.
- **[Kandinsky 5 Lite](kandinsky-5/README.md):** 14 лабораторных по компонентам, Flow Matching и inference. У курса отдельные `pyproject.toml`, `uv.lock` и окружение.
- **[Kandinsky 6 Lite](kandinsky-6/README.md):** ручной notebook, Smoke/Baseline workflow, подготовка VCTK и постановка будущего voice-conditioning эксперимента.
- **[DDPM на CIFAR-10](Diffusion_DDPM_Research.ipynb):** дополнительный пример noise prediction. Для него: `uv sync --locked --extra ddpm`.

## Что проверено, а что остаётся экспериментальным

| Компонент | Статус | Доказательства |
|---|---|---|
| Вводный и CTMC notebooks | Выполнены на CPU; оригинальные Lab 2/3 остаются заданиями | [Отчёт](curriculum/reports/VALIDATION.md) |
| K6 manual notebook | Лёгкий маршрут выполнен без pretrained weights | [Отчёт K6](kandinsky-6/reports/VALIDATION.md) |
| ComfyUI graphs | Статические проверки связей/типов; runtime не заявлен | [Инструкция](curriculum/COMFYUI_LABS.md) |
| MLX audio prototype | Подготовлен код; numerical/pretrained parity пока не пройден | [Статус K6](kandinsky-6/STATUS.md) |
| Voice adapter / полный AV inference | План и проверки подготовлены; сквозной результат не получен | [Ручной протокол](kandinsky-6/COMFYUI_AND_EXPERIMENT_RU.md) |

Синтетические loss/score проверки не являются результатом клонирования голоса. Скорость и память H100 не переносятся на Mac без измерения. Полный запуск на текущей M3 Max станции отложен до освобождения 50 GB+ на SSD; это локальное ограничение, не универсальное требование моделей.

## Структура проекта

```text
curriculum/       Общая учебная карта, протоколы и малые опыты
  notebooks/      Вводная исполняемая лабораторная
  workflows/      Учебные ComfyUI JSON
  scripts/        Проверки, подготовка источников и опыты
  references/     Происхождение, revisions, hashes
  reports/        Измеренные результаты и графики
lecture_01/       ODE/SDE: разбор, упражнения, конспект
lecture_02/       Flow Matching и канонический original Lab 2
lecture_03/       Score/CFG и связи Lab 2–3
lecture_04/       DiT/VAE/LDM и канонический original Lab 3
lecture_05/       Дополнительная CTMC-лабораторная
kandinsky-5/      Курс K5 с отдельным окружением
kandinsky-6/      Экспериментальный K6 стенд
pyproject.toml    Зависимости основного курса
uv.lock          Зафиксированные версии
```

`reference/` в курсах содержит закреплённые сторонние исходники/configs, а не новую авторскую реализацию. Активный исследовательский код лежит в `labkit/` и `scripts/`. Большие веса, datasets, virtualenv, извлечённые тексты и выполненные рабочие копии notebooks исключены из Git; локальные результаты сохраняются в соответствующем `artifacts/`. Временные каталоги не нужны для запуска checkout.

## Источники и атрибуция

Основной теоретический маршрут — [MIT 6.S184, 2026](https://diffusion.csail.mit.edu/2026/index.html). [Индекс предоставленных материалов](curriculum/references/README.md) объясняет происхождение originals и адаптаций. Это независимый учебный проект, не официальный курс MIT и не официальный релиз Kandinsky/Hunyuan/Wan.

Для архитектур используются ссылки на конкретные official repository revisions; лицензии стороннего кода сохраняются. Поясняющие обзоры дополняют первичные статьи и код, не заменяя их.

[Дополнительные чтения по месту в курсе](curriculum/SUPPLEMENTARY_READING.md): Rectified Flow, обзоры diffusion/video и небольшие CPU-контроли.
