# Задание Claude: отполировать Generative AI Studying Stand

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](README.md) · [Полное содержание](CONTENTS.md).
<!-- study-navigation:end -->

[Главная](../README.md) · [Полное содержание](CONTENTS.md) · [Методика маршрута](README.md).

## Задача

Ты — методолог обучения генеративным моделям, технический редактор и исследователь PyTorch/ODE/SDE. Проведи содержательное и редакторское улучшение репозитория **generative-ai-research**. Стенд должен быть понятен студенту и пригоден для внешнего просмотра командой Kandinsky: ясный вход, последовательное накопление знаний, проверяемые примеры и аккуратная навигация. Сохрани выполненную работу; не заменяй курс общим обзором.

Основной маршрут: **многомерные вероятности → ODE/SDE → Flow Matching → score/conditioning/CFG → DiT/VAE → промышленные video/audio модели → ComfyUI → отдельное измеряемое исследование**.

Твоя задача — улучшить материалы и проверить результат, а не только написать рекомендации. Сначала изучи актуальную рабочую ветку, git diff, README, CONTENTS и отчёты. Существующие авторские документы — материал для анализа, а не доказательство истинности каждого утверждения.

## 1. Источники и педагогический каркас

Основной теоретический скелет — [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html), [lecture notes](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf). Lecture 1 ↔ Lab 1; Lecture 2–3 ↔ Lab 2; Lecture 3–4 ↔ Lab 3. Lecture 5 / CTMC — самостоятельная дискретная ветвь; она не обязательная предпосылка continuous video.

Основной архитектурный источник K6 — [Hugging Face Papers: 2610.05608](https://huggingface.co/papers/2610.05608), [полный отчёт](https://arxiv.org/html/2610.05608v1). Проверяй архитектурные утверждения по статье, pinned source и конкретному checkpoint config. Обзорный abstract/автоматическое резюме HF не заменяет текст статьи.

Первоисточники для углубления:

- [DiT](https://arxiv.org/abs/2212.09748): latent patches, time/class conditioning, adaLN-Zero.
- [Kandinsky 5](https://arxiv.org/abs/2511.14993): предшествующий text/video CrossDiT.
- [Flow Matching](https://arxiv.org/abs/2210.02747), [Rectified Flow](https://arxiv.org/abs/2209.03003): path, velocity и regression target.
- [π-Flow](https://arxiv.org/abs/2510.14974): network-free policy и distillation.
- [Ovi](https://arxiv.org/abs/2510.01284), [LTX-2](https://arxiv.org/abs/2601.03233): сравнение AV fusion; не считать их буквальными реализациями K6.
- [MMAudio](https://arxiv.org/abs/2412.15322), [BigVGAN](https://arxiv.org/abs/2206.04658): audio components.

Дополнительные чтения уже привязаны к темам в [SUPPLEMENTARY_READING](SUPPLEMENTARY_READING.md). Они дополняют MIT, а не создают ещё один обязательный syllabus. Для HunyuanVideo, Wan 2.1/2.2, LTX-Video, Mochi 1 и Open-Sora используй [INDUSTRY_BRIDGE](INDUSTRY_BRIDGE.md) и official revisions. LTX-Video и LTX-2 — разные выбранные семейства.

## 2. Файлы и зоны работы

| Материал | Что улучшить |
|---|---|
| [Корневой README](../README.md) | Быстрый ответ: что это, с чего начать, как запустить, какой статус |
| [CONTENTS](CONTENTS.md) | Полное оглавление, точные ссылки, понятные переходы между всеми пятью уровнями |
| [curriculum README](README.md) | Предпосылки, порядок чтения/исполнения и критерии перехода |
| `lecture_01`…`lecture_05` | Согласованные обозначения, source links, связь с original labs и следующими темами |
| [K5 README](../kandinsky-5/README.md), `kandinsky-5/notebooks` | Компонентный курс, глубокие VAE/FM, связь с K6; без устаревших инструкций |
| [K6 README](../kandinsky-6/README.md), [COMPONENT_GUIDE](../kandinsky-6/COMPONENT_GUIDE.md) | Training map, shapes, class/method index, роль каждого компонента |
| `kandinsky-6/notebooks/01`…`14` | Самостоятельные учебные лабораторные по компонентам |
| [K6 07 DiT](../kandinsky-6/notebooks/07_dit_and_text_crossdit.ipynb) | Особенно глубокое объяснение DiT → text-conditioned CrossDiT |
| [K6 08 Dual-stream CrossDiT](../kandinsky-6/notebooks/08_dual_stream_audio_video_crossdit.ipynb) | Особенно глубокое объяснение асимметричных Audio/Video streams и cross-modal exchange |
| [ComfyUI labs](COMFYUI_LABS.md), workflows | Связать node wiring с кодом и измеряемым опытом |
| [Research protocol](RESEARCH_PROTOCOL.md), [K6 manual](../kandinsky-6/COMFYUI_AND_EXPERIMENT_RU.md) | Гипотеза → baseline → данные → обучение → оценка → ограниченный вывод |
| Reports / validation scripts | Достоверные измерения, воспроизводимость, точные границы проверок |

Не создавай дублирующие README с конкурирующими маршрутами. Корневой README — компактный вход; полное оглавление — CONTENTS; методика — curriculum README; подробности компонентов — тематические notebooks/guide.

## 3. Методический стандарт каждой лабораторной

Перед кодом должны быть вопрос, необходимые знания, схема вход/выход и простое объяснение операции. Затем маленький разобранный пример, математическая запись, исполняемый Python/PyTorch опыт, измерение/график и интерпретация. В конце — упражнения, критерий понимания, следующий шаг и ссылки на основной источник/оригинальный код.

Проверяй, что:

- Каждая переменная и ось тензора объяснена при первом использовании.
- Density, probability, sample, batch, conditional endpoint и external condition не смешаны.
- Time convention указана: MIT noise→data и сравниваемое Kandinsky data→noise различаются знаком velocity.
- Gaussian covariance, `logvar`/std, KL reduction, likelihood/MSE и latent scaling имеют ясный смысл.
- Объяснение показывает причинную связь: зачем операция нужна, что она меняет, что измеряет опыт.
- Численный контроль имеет аналитический ответ, invariant или содержательное сравнение.
- Ошибка solver отделена от ошибки модели; NFE отделено от visible steps/grid/substeps.
- Упражнение предлагает проверяемую гипотезу, один изменяемый фактор и форму отчёта.
- Synthetic/random-weight опыт не преподносится как pretrained качество.

Не увеличивай объём ради объёма. Устрани повторения, канцелярит и бессодержательные обещания; сложные идеи объясняй через малые случаи.

## 4. Специальный аудит DiT / Dual-stream CrossDiT

Проверь source-level соответствие notebooks 07/08 и `core/components/dit.py`:

1. Patch embedding и output projection: axes, число tokens, physical channels и model width.
2. Time sinusoid/MLP, pooled condition, отдельные video/audio text-refiners.
3. Self-attention, text cross-attention, межмодальное attention и FFN — различимые ветви.
4. Q задаёт destination output: video Q + audio K/V обновляет video, обратная ветвь обновляет audio. Не делай вывод о направлении только из имени `va`/`av`.
5. Точный порядок операций и сохранённые `vis_pre_ca` / `aud_pre_ca`; не подменяй это последовательной схемой с уже обновлённым cross output.
6. Modulation, zero-init, gate dimensions, `cross_gates`, `fix_modulation`, `ca_rope`.
7. Почему gradients и perturbation cuts подтверждают wiring, но ещё не lip-sync.
8. Отличие unimodal forward, joint forward, partial sampling и декодирования только одного output.
9. Обычный velocity head против expanded grid-head PiFlow. HF config уже расширен; wrapper умножает per-grid width сам.
10. Чёткая связь с DiT/K5/K6/Ovi/LTX-2 arXiv; термин CrossDiT не путать с image-classification CrossViT.

Сохраняй необычные детали оригинального кода при parity, в частности double normalization audio output. Если находишь предполагаемую ошибку upstream, сначала воспроизведи её, проверь caller reachability/config/version и отдели bug от intentional contract. Не объявляй ошибкой одно лишь необычное выражение.

## 5. Navigation / Markdown / визуальная проверка

- С любой лекции/лабы должен быть переход к CONTENTS, маршруту и основному источнику.
- Соседние K6 labs должны иметь previous/next links.
- В оглавлении должны быть все authored notebooks, guides, ComfyUI workflows, research entry points и evidence reports.
- Canonical original labs, рабочие ответы студента, reference source и reports должны быть различимы по назначению.
- Проверяй links и anchors; избегай локальных абсолютных путей там, где материал предназначен для GitHub/Colab.
- Для GitHub math используй отдельные display blocks или `math` fences. Не экранируй underscore как Markdown внутри формулы; не используй запрещённый `operatorname` и legacy `rm`.
- Проверяй реальный rendered view. Успех regex/style checker не равен визуальному успеху конкретного renderer.
- Mermaid labels со скобками/сложными символами заключай в кавычки. Проверь diagrams на GitHub-compatible renderer.
- Графики должны иметь axes/units/legend и подпись, что именно измерено. Не называй логический размер dense score matrix фактической peak memory FlashAttention.
- Избегай лишних badges и огромных общих таблиц в корневом README; подробности вынеси по ссылкам.

## 6. Ограничения текущего стенда

Рабочая станция: Apple M3 Max, 40 GPU cores, 48 GB общей памяти. Полные pretrained-загрузки и генерация **отложены пользователем до освобождения 50 GB+ SSD**. Не запускай их автоматически, не устанавливай MLX/CUDA stack и не меняй окружение ComfyUI Desktop для редакторской задачи. Малые CPU-контроли можно выполнять в зафиксированном root окружении.

В Git уже могут быть незакоммиченные пользовательские ответы:

- `lecture_01/lab_one_ru.ipynb`
- `lecture_01/lab_one_ru_exercises.ipynb`

Сохрани их byte-for-byte и не включай в свой commit. Не применяй глобальные форматтеры или `git reset/clean/stash` к пользовательской работе. Canonical originals и third-party snapshots не редактируй для красивого отчёта; изменения в adapter/пояснениях должны быть явными.

Pretrained AV inference, полный joint MLX parity, реальное voice-FiLM обучение/SECS/WER/FAD пока не объявлены выполненными. Frozen-FiLM — отдельная исследовательская гипотеза; статья по voice adaptation обучает также audio backbone. Не обещай 10–20 клипов, 5–15 минут или определённую latency без измерения.

## 7. Проверки и воспроизведение

Из корня checkout:

```bash
uv run --locked --offline python curriculum/scripts/validate_curriculum.py
uv run --locked --offline python curriculum/scripts/check_markdown_math.py
uv run --locked --offline python curriculum/scripts/sync_study_navigation.py --check
uv run --locked --offline python kandinsky-6/scripts/validate_component_notebooks.py
```

Если меняешь исполняемый код или его существенные предпосылки, выполни соответствующие notebooks в свежих kernels:

```bash
uv run --locked --offline python kandinsky-6/scripts/validate_component_notebooks.py --execute
uv run --locked --offline python curriculum/scripts/validate_curriculum.py --execute
```

При изменении только Markdown не нужно повторять training benchmarks. Проверяй ровно затронутое поведение; не создавай tests, которые просто повторяют реализацию.

Генераторы `build_component_notebooks.py`, `build_component_guide.py` и `build_contents.py` должны соответствовать публикуемым файлам: правка generated artifact без правки generator потеряется. После пересборки — `sync_study_navigation.py --write`, затем `--check`. Смотри режимы validator: static report не должен перезаписывать evidence execution. Сохраняй версии, source hashes и labels analytic/synthetic/native/pretrained.

## 8. Приёмка и финальный отчёт

Работа готова, если новый читатель за пару минут находит старт, понимает предпосылки и может пройти одну законченную цепочку lecture→code→component→UI→research. Все ссылки ведут на существующие материалы. DiT и Dual-stream CrossDiT объясняются самостоятельно, с source-level деталями и проверяемыми опытами. README, CONTENTS и notebooks не противоречат друг другу. Нет лишних скачиваний, испорченных оригиналов, staged student answers или выдуманных результатов.

В результате предоставь:

1. Краткое объяснение, что изменилось и почему учиться стало последовательнее.
2. Таблицу затронутых материалов и исправленных проблем.
3. Реально выполненные проверки и их результат; оставшиеся ограничения отдельно.
4. Один рекомендуемый следующий учебный шаг.
5. Аккуратный commit/PR с конкретным описанием; merge выполняется только при соответствующем запросе пользователя.

## Короткий промпт для начала работы

> Прочитай `curriculum/CLAUDE_REVIEW_BRIEF.md` и выполни указанную редакторскую и методическую полировку актуального checkout. Используй MIT 6.S184 как основной маршрут, K6 report 2610.05608 и pinned official code для архитектуры. Особенно проверь labs 07 DiT и 08 Dual-stream Audio+Video CrossDiT. Улучшай файлы, проверяй результат и сохраняй пользовательские ответы, оригиналы и честные границы измерений. Начни с git status и существующих README/CONTENTS/reports.
