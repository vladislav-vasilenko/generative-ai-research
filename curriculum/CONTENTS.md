# Полное содержание и единый маршрут обучения

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](README.md) · [Полное содержание](CONTENTS.md).
<!-- study-navigation:end -->

[Главная страница](../README.md) · [Подробная методика и критерии P0–P8](README.md) · [Самопроверка](PROGRESS.md).

Это оглавление всех авторских учебных материалов и точек входа в оригиналы. Основной скелет — [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html), [конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf). Сначала теория и небольшая реализация, затем архитектурный компонент, UI-наблюдение и отдельная исследовательская гипотеза. Исходные незаполненные MIT labs — задания; Run All применим к готовым авторским CPU-примерам, а не автоматически ко всем originals.

## Один маршрут через пять уровней

| Шаг / тема | Лекция и исходная lab | Python / архитектурный блок | ComfyUI | Результат перед переходом |
|---|---|---|---|---|
| 0. Вероятности и формы | Notes Appendix A | [Probability bridge](notebooks/00_probability_and_model_contracts.ipynb), [K6 system map](../kandinsky-6/notebooks/01_system_and_training_map.ipynb) | [C0: image dataflow](COMFYUI_LABS.md#c0-контроль-dataflow-до-нейросети) | covariance/score, shape и units |
| 1. ODE/SDE | [Lecture 1 / Lab 1](../lecture_01/README.md) | Euler, Brownian/OU, [K5 sampling](../kandinsky-5/notebooks/07_sampling.ipynb) | C1 reproducibility, C2 solver | ошибка при уменьшении шага, моменты распределения |
| 2. Flow Matching | [Lecture 2 / Lab 2](../lecture_02/README.md) | [K5 FM](../kandinsky-5/notebooks/06_dit_and_flow_matching.ipynb), [K6 joint FM](../kandinsky-6/notebooks/09_flow_matching_training.ipynb) | C2: steps/NFE на совместимом checkpoint | path/velocity/time conventions + held-out loss |
| 3. Score / conditioning / CFG | [Lecture 3 / Lab 2–3](../lecture_03/README.md) | [K5 text](../kandinsky-5/notebooks/01_text_embedders.ipynb), [K6 text](../kandinsky-6/notebooks/02_text_conditioning.ipynb), [K6 CFG](../kandinsky-6/notebooks/11_sampling_cfg_and_nfe.ipynb) | C3 CFG; Distill CFG=1 не universal sweep | masks, encoder/refiner, branch count |
| 4. VAE / codecs | [Lecture 4 / Lab 3](../lecture_04/README.md) | [VAE MIT → video](../vae/README.md), [K5 VAE theory](../kandinsky-5/notebooks/02_vae.ipynb), [K6 video](../kandinsky-6/notebooks/03_video_vae.ipynb), [K6 audio/vocoder](../kandinsky-6/notebooks/04_audio_vae_and_vocoder.ipynb) | C4 standalone reconstruction | ELBO/posterior/scale, реконструкция отдельно от генерации |
| 5. DiT / CrossDiT | Lecture 4, Lab 3 Part 3 | [Общий DiT route](../dit/README.md), [K5 nn/DiT](../kandinsky-5/notebooks/03_nn_building_blocks.ipynb), [K6 DiT](../kandinsky-6/notebooks/07_dit_and_text_crossdit.ipynb), [K6 dual AV CrossDiT](../kandinsky-6/notebooks/08_dual_stream_audio_video_crossdit.ipynb) | C5 token budget, Lite graph | Q/K/V, gates, shapes, directional ablation |
| 6. Discrete branch | [Lecture 5](../lecture_05/README.md) | [CTMC lab](../lecture_05/05_ctmc_and_discrete_flow.ipynb) | C6 visualizes Python result | rates, positivity, exact law vs simulation |
| 7. Distillation / deployment | [Industrial map](INDUSTRY_BRIDGE.md) | [K6 π-Flow](../kandinsky-6/notebooks/10_piflow_distillation.ipynb), [runtime/MLX](../kandinsky-6/notebooks/13_runtime_memory_and_mlx.ipynb) | [K6 UI component lab](../kandinsky-6/notebooks/14_comfyui_component_experiments.ipynb) | policy/NFE distinction, resource baseline |
| 8. Separate research | [Research protocol](RESEARCH_PROTOCOL.md) | [K6 manual journal](../kandinsky-6/notebooks/00_manual_experiment_comfy_mlx_film.ipynb), FiLM/voice or own hypothesis | baseline + controlled A/B | data split, metrics, regression, supported conclusion |

Lab 2 связывает Lecture 2–3, Lab 3 — Lecture 3–4. CTMC — самостоятельная ветвь; continuous industrial video можно изучать после Lecture 4. Не требуется запускать все промышленные семейства: выберите один проверяемый case.

## Полное содержание лекций и базовых лабораторных

### Lecture 1

[Порядок, источники и задания](../lecture_01/README.md)

- [LECTURE_01_RU.md](../lecture_01/LECTURE_01_RU.md) — авторский конспект.
- [lab_one_ru.ipynb](../lecture_01/lab_one_ru.ipynb) — Часть 0. Введение; Часть 1. Численные методы моделирования ОДУ и СДУ; Часть 2. Визуализация решений СДУ; Часть 3. Преобразование распределений с помощью СДУ.
- [lab_one_ru_exercises.ipynb](../lecture_01/lab_one_ru_exercises.ipynb) — Часть 0. Введение; Часть 1. Численные методы моделирования ОДУ и СДУ; Часть 2. Визуализация решений СДУ; Часть 3. Преобразование распределений с помощью СДУ.
- [lab_one_original.ipynb](../lecture_01/references/lab_one_original.ipynb) — неизменённый исходный MIT notebook, содержит задания.
- [Происхождение Lab 1](../lecture_01/references/README.md); [сохранённый HTML-пример выполнения](../lecture_01/outputs/lab_one_ru.html).

### Lecture 2

[Порядок, источники и задания](../lecture_02/README.md)

- [lab_two_original.ipynb](../lecture_02/references/lab_two_original.ipynb) — неизменённый исходный MIT notebook, содержит задания.

### Lecture 3

[Порядок, источники и задания](../lecture_03/README.md)


### Lecture 4

[Порядок, источники и задания](../lecture_04/README.md)

- [lab_three_original.ipynb](../lecture_04/references/lab_three_original.ipynb) — неизменённый исходный MIT notebook, содержит задания.

### Lecture 5

[Порядок, источники и задания](../lecture_05/README.md)

- [05_ctmc_and_discrete_flow.ipynb](../lecture_05/05_ctmc_and_discrete_flow.ipynb) — 1. Rate matrix — не transition probability matrix; 2. Ошибка шага и нарушение positivity; 3. Gillespie: точные времена скачков для постоянного Q; 4. Factorized masking path для зависимых tokens; 5. Сверка с Bayes posterior и построение локальных rates.

## Общие авторские notebooks

- [00_probability_and_model_contracts.ipynb](notebooks/00_probability_and_model_contracts.ipynb) — 1. Один объект, batch и совместная вероятность; 2. Два time conventions описывают один путь; 3. Conditional endpoint не равен marginal predictor; 4. CFG: две ветви — не две разные архитектуры; 5. Patchify и стоимость tokens; 6. VAE: std, KL и заморозка.
- [DDPM / CIFAR-10](../Diffusion_DDPM_Research.ipynb) — дополнительная noise-prediction ветвь, отдельный dependency extra.

## Kandinsky 5 Lite: 14 лабораторных

[Порядок, запуск и Colab](../kandinsky-5/README.md); [Каталог моделей](../kandinsky-5/MODEL_CATALOG.md).

| Notebook | Содержание / последовательность внутри |
|---|---|
| [00_start_and_training_map.ipynb](../kandinsky-5/notebooks/00_start_and_training_map.ipynb) | 1. Модель — это система; 2. Читаем настоящий рецепт; 3. Параметры и память; 4. Размер латентного видео; 5. Frozen и trainable — проверяем на маленькой сети; 6. Где заканчиваются подтверждённые факты |
| [01_text_embedders.ipynb](../kandinsky-5/notebooks/01_text_embedders.ipynb) | 1. Токен — индекс, embedding — вектор; 2. Две ветви conditioning; 3. Prompt template и crop; 4. Почему mask и padding важны; 5. Настоящие токенизаторы без больших весов; 6. Настоящие embeddings отдельно от DiT |
| [02_vae.ipynb](../kandinsky-5/notebooks/02_vae.ipynb) | Часть A. Теория VAE: от сжатия к вероятностной модели; Часть B. Базовые опыты и связь с исходниками; Часть C. Более строгий учебный VAE: сравниваем β, prior и posterior; Часть C3. Дополнение по материалам: что именно регуляризует VAE; Часть D. Прямой разбор настоящего vae.py Kandinsky |
| [03_nn_building_blocks.ipynb](../kandinsky-5/notebooks/03_nn_building_blocks.ipynb) | 1. Загружаем классы без CUDA compilation; 2. VisualEmbeddings: патч — маленький кубик; 3. TextEmbeddings и TimeEmbeddings; 4. RoPE1D: поворот вместо прибавления позиции; 5. RoPE3D: три координаты; 6. Modulation: время меняет обработку признаков |
| [04_attention.ipynb](../kandinsky-5/notebooks/04_attention.ipynb) | 1. Query, Key, Value — вопрос, адрес, содержимое; 2. Воспроизводим SDPA; 3. Mask: запрещённые связи; 4. Query chunking без потери ключей; 5. Масштаб видео; 6. SelfAttentionEngine: выбор реализации |
| [05_utils_and_nabla.ipynb](../kandinsky-5/notebooks/05_utils_and_nabla.ipynb) | 1. Что здесь имеет веса; 2. local_patching — сложить соседей рядом; 3. fractal_flatten: dense и block ветви; 4. Простой block selector; 5. STA добавляет локальных соседей; 6. Где применяется настоящий NABLA |
| [06_dit_and_flow_matching.ipynb](../kandinsky-5/notebooks/06_dit_and_flow_matching.ipynb) | Часть A. Что значит Flow Matching «внутри DiT»; Часть B. Базовые опыты и связь с исходниками; Часть C. Разложим настоящий Kandinsky forward на этапы; Часть D. Полный маленький conditional flow: учим два распределения; Часть E. Что точно делает Kandinsky 5.0 inference |
| [07_sampling.ipynb](../kandinsky-5/notebooks/07_sampling.ipynb) | 1. Одна строка sampler; 2. Shifted schedule; 3. Нелинейная velocity и ошибка Euler; 4. CFG требует двух вызовов; 5. Distillation не уменьшает архитектуру; 6. Связываем с кодом |
| [08_parallelize.ipynb](../kandinsky-5/notebooks/08_parallelize.ipynb) | 1. Data, tensor и sequence parallel; 2. ColwiseParallel — делим выходные признаки; 3. RowwiseParallel — складываем частичные суммы; 4. Sequence parallel и глобальный контекст; 5. LoRA меняет имена вложенных модулей; 6. parallelize_seq в настоящем коде |
| [09_local_m3_max.ipynb](../kandinsky-5/notebooks/09_local_m3_max.ipynb) | 1. Сначала проверка MPS; 2. Forward настоящих классов на Metal; 3. Скачивание только выбранных компонентов; 4. Проверим файлы до запуска; 5. Первый небольшой запуск; 6. Финальный 5s маршрут |
| [10_colab_cuda_distill.ipynb](../kandinsky-5/notebooks/10_colab_cuda_distill.ipynb) | Какие defaults скрывает короткая ячейка; Mac или Colab?; 1. Выбираем runtime; 2. Подготовка официальной среды; 3. Только distill checkpoint и shared компоненты; 4. Конфигурация с абсолютными путями |
| [11_kvae_separate.ipynb](../kandinsky-5/notebooks/11_kvae_separate.ipynb) | 1. Два смысла слова tokenizer; 2. Почему shape compatibility недостаточно; 3. Что сообщает настоящая конфигурация; 4. Настоящие классы KVAE без больших весов; 5. Mode reconstruction с настоящими весами |
| [12_generation_utils.ipynb](../kandinsky-5/notebooks/12_generation_utils.ipynb) | Как именно запускаем исходные функции; Карта потока и соглашение о формах; 1. get_sparse_params: подготовка NABLA mask; 2. adaptive_mean_std_normalization: ограниченная поправка статистик; 3. normalize_first_frame: первые четыре latent frames; 4. get_velocity: один DiT forward или CFG из двух |
| [13_normalization_and_vae_contract_audit.ipynb](../kandinsky-5/notebooks/13_normalization_and_vae_contract_audit.ipynb) | 1. Какой именно main мы проверяем; 2. Сначала договоримся о значении слова «кадр»; 3. Положительный контроль: что функция делает при нормальном входе; 4. Математика adaptive_mean_std_normalization; 5. Матрица граничных случаев: длина, reference_frames, clipping, dtype; 6. Где эти функции действительно вызываются |

## Kandinsky 6 Lite: компоненты и отдельное исследование

[Порядок, запуск и Colab](../kandinsky-6/README.md); [Карта компонентов и методов](../kandinsky-6/COMPONENT_GUIDE.md).

| Notebook | Содержание / последовательность внутри |
|---|---|
| [00_manual_experiment_comfy_mlx_film.ipynb](../kandinsky-6/notebooks/00_manual_experiment_comfy_mlx_film.ipynb) | Исследовательский журнал: диск → ComfyUI baseline → VCTK → parity → будущий FiLM/voice experiment; читать после component route |
| [01_system_and_training_map.ipynb](../kandinsky-6/notebooks/01_system_and_training_map.ipynb) | 1. Модель из 3B параметров — это ещё не весь pipeline; 2. Считать параметры можно без загрузки весов; 3. Связь с ресурсами |
| [02_text_conditioning.ipynb](../kandinsky-6/notebooks/02_text_conditioning.ipynb) | 1. Четыре разных представления текста; 2. Маска ограничивает доступ к K/V; 3. Что можно кэшировать, а что зависит от времени |
| [03_video_vae.ipynb](../kandinsky-6/notebooks/03_video_vae.ipynb) | 1. Зачем распределение вместо одной точки; 2. Миниатюрное обучение с известными данными; 3. Causal Conv3d и нормализация латентов |
| [04_audio_vae_and_vocoder.ipynb](../kandinsky-6/notebooks/04_audio_vae_and_vocoder.ipynb) | 1. Три пространства и три разные задачи; 2. Сколько audio tokens соответствует видео; 3. Настоящий postprocess с явно учебными decoder/vocoder |
| [05_nn_patches_and_modulation.ipynb](../kandinsky-6/notebooks/05_nn_patches_and_modulation.ipynb) | 1. Из локальной ячейки latent в visual token; 2. AdaLN — condition управляет обработкой state; 3. OutLayerAudio и маленькие на вид особенности |
| [06_attention_and_rope.ipynb](../kandinsky-6/notebooks/06_attention_and_rope.ipynb) | 1. Attention — выбор источников информации; 2. Вращение сохраняет норму; позиции меняют сравнение; 3. Почему video attention дорого |
| [07_dit_and_text_crossdit.ipynb](../kandinsky-6/notebooks/07_dit_and_text_crossdit.ipynb) | 0. Вопрос, вход и выход; 1. Что означает DiT и почему K6 использует два канала conditioning; 2. Полный forward одной формулой; 3. adaLN-Zero: свежая модель не видит ни текст, ни время; 4. Text CrossDiT: visual query, text K/V; 5. Velocity head и π-Flow grid head |
| [08_dual_stream_audio_video_crossdit.ipynb](../kandinsky-6/notebooks/08_dual_stream_audio_video_crossdit.ipynb) | 0. Вопрос, вход и выход; 1. Направление определяется по query; 2. Точный порядок внутри FusedTransformerDecoderBlock; 3. Флаги: какое время управляет fusion; 4. Zero-init invariance, затем причинное отключение связей; 5. Gradient показывает путь обучения, не качество синхронизации |
| [09_flow_matching_training.ipynb](../kandinsky-6/notebooks/09_flow_matching_training.ipynb) | 1. Сначала соглашение о времени; 2. Данные с известной общей причиной; 3. Почему шумы и времена требуют отдельного контроля |
| [10_piflow_distillation.ipynb](../kandinsky-6/notebooks/10_piflow_distillation.ipynb) | 1. Distillation меняет контракт головы; 2. Из grid prediction в DX policy; 3. Native sampler + настоящий случайный tiny grid-DiT |
| [11_sampling_cfg_and_nfe.ipynb](../kandinsky-6/notebooks/11_sampling_cfg_and_nfe.ipynb) | 1. Solver интегрирует поле, но не обучает его; 2. Guidance — комбинация двух predictions; 3. Что замерять в промышленном опыте |
| [12_latents_conditioning_and_output.ipynb](../kandinsky-6/notebooks/12_latents_conditioning_and_output.ipynb) | 1. Shape сам по себе не задаёт смысл; 2. Tail conditioning и token roles; 3. Проверяем encode wrapper без выдуманного VAE API; 4. Последний этап — mux |
| [13_runtime_memory_and_mlx.ipynb](../kandinsky-6/notebooks/13_runtime_memory_and_mlx.ipynb) | 1. Offload, quantization и cache делают разные вещи; 2. Измеряем только то, что действительно запущено; 3. Parity gates: по слоям, затем по pipeline |
| [14_comfyui_component_experiments.ipynb](../kandinsky-6/notebooks/14_comfyui_component_experiments.ipynb) | 1. UI показывает wiring, а не все внутренние слои; 2. Последовательность опытов и измерений; 3. Как перейти к исследованию голоса |
| [15_kaggle_t4_lite_distill.ipynb](../kandinsky-6/notebooks/15_kaggle_t4_lite_distill.ipynb) | 1. Подготовка Kaggle и порядок запуска; 2. Почему память освобождается между стадиями; 3. Код и зависимости: отдельная среда, существующий CUDA PyTorch; 4. Все модели: точные файлы, revisions, размеры и контрольные суммы; 5. Настройки эксперимента и контракты тензоров; 6. Общая часть worker: ComfyUI loaders и безопасные промежуточные файлы |

## VAE: MIT Lab 3 → HunyuanVideo → tiling

[Порядок и запуск](../vae/README.md); [Связь с MIT заданиями](../vae/MIT_LAB3_BRIDGE.md); [Все классы и методы](../vae/SOURCE_MAP.md); [CPU измерения и ограничения](../vae/reports/VALIDATION.md).

| Notebook | Теория, эксперимент и упражнения |
|---|---|
| [00_route_and_source_contracts.ipynb](../vae/notebooks/00_route_and_source_contracts.ipynb) | 1. Четыре разных объекта исследования; 2. Геометрия и единицы; 3. План накопления знаний |
| [01_elbo_and_probabilistic_codec.ipynb](../vae/notebooks/01_elbo_and_probabilistic_codec.ipynb) | 1. Почему нельзя просто интегрировать decoder; 2. Gaussian posterior: вывод KL по координатам; 3. Как gradient проходит через sample; 4. От likelihood к loss HunyuanVideo; 5. Возвращаемся в MIT Q4.7: reconstruction с learned variance |
| [02_causal_convolution_and_geometry.ipynb](../vae/notebooks/02_causal_convolution_and_geometry.ipynb) | 1. Дискретная причинная свёртка; 2. Проверяем условия реализации, а не имя класса; 3. Почему special first frame меняет inverse geometry; 4. Receptive field и GroupNorm: где теряется локальность |
| [03_architecture_posterior_and_api.ipynb](../vae/notebooks/03_architecture_posterior_and_api.ipynb) | 1. Полный путь encoder и decoder; 1b. Mid attention: из feature volume в Q/K/V; 2. Return contract: encode не возвращает готовый tensor; 3. Scaling: смена единиц перед DiT; 4. Dispatch и stateful настройки |
| [04_spatial_tiling_and_blending.ipynb](../vae/notebooks/04_spatial_tiling_and_blending.ipynb) | 1. Зачем overlap и как устроен blend; 2. Exact coordinate oracle: сначала доказываем индексы; 3. Покрытие правого/нижнего края — отдельное условие; 3b. Точный halo+crop на локальной свёртке; 4. Почему overlap не гарантирует full equivalence |
| [05_temporal_tiling_and_first_frame.ipynb](../vae/notebooks/05_temporal_tiling_and_first_frame.ipynb) | 1. Temporal lattice; 2. Frame-count audit: F=4k+1 ещё недостаточно; 3. Проверяем decoder отдельно; 4. Reset history и временные швы |
| [06_free_vram_planner.ipynb](../vae/notebooks/06_free_vram_planner.ipynb) | 1. Что на самом деле моделирует heuristic; 2. От числа tiles к двум осям; 3. Memory guard ≠ OOM guarantee |
| [07_controlled_benchmark.ipynb](../vae/notebooks/07_controlled_benchmark.ipynb) | 1. Гипотеза, контроль и метрики; 2. Графики и диагностическая локализация; 3. Overlap work и retained buffers |
| [08_pretrained_and_comfyui_protocol.ipynb](../vae/notebooks/08_pretrained_and_comfyui_protocol.ipynb) | 1. Исследовательская карточка; 2. Команда ручного запуска после resource gate; 3. ComfyUI как визуальный контроль; 4. Как сделать вывод, когда появятся реальные числа |

## DiT / CrossDiT / Conditioning / Denoise Loop / π-Flow

[11 labs: порядок и запуск](../dit/README.md); [Конкретные задания MIT](../dit/MIT_BRIDGE.md); [Каждый класс и метод](../dit/SOURCE_MAP.md); [CPU измерения](../dit/reports/VALIDATION.md).

| Notebook | Теория → code → experiment |
|---|---|
| [00_probability_path_and_learning_route.ipynb](../dit/notebooks/00_probability_path_and_learning_route.ipynb) | 1. От распределения к траектории; 2. Почему solver не равен DiT; 3. Порядок нового маршрута; Измерение → вывод; Самостоятельные опыты; Первичные источники |
| [01_dit_primitives_and_tensor_geometry.ipynb](../dit/notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | 1. Patchification — перестановка индексов плюс проекция; 2. Time embedding и adaLN; 2b. LayerNorm, RMSNorm и FFN без скрытых формул; 3. Attention от скалярных произведений к tensor axes; 4. OutLayer — learned prediction, не обратная embedding matrix; Измерение → вывод |
| [02_single_stream_crossdit_k5.ipynb](../dit/notebooks/02_single_stream_crossdit_k5.ipynb) | 1. Что означает «один поток»; 2. Что изменено в execution adapter; 3. Cross-attention и условная зависимость; Измерение → вывод; Самостоятельные опыты; Первичные источники |
| [03_dual_stream_crossdit_k6.ipynb](../dit/notebooks/03_dual_stream_crossdit_k6.ipynb) | 1. Два генерируемых состояния и разные widths; 2. Порядок внутри FusedTransformerDecoderBlock; 3. Cross gates и два времени; 4. Измеряем cross dependence градиентом; Измерение → вывод; Самостоятельные опыты |
| [04_conditioning_text_and_clean_modalities.ipynb](../dit/notebooks/04_conditioning_text_and_clean_modalities.ipynb) | 1. Условная плотность и три маршрута информации; 2. Mask polarity: boolean keep-mask не равна float 0/1 bias; 3. Что можно кэшировать; 4. Null conditions и wrapper boundaries; Измерение → вывод; Самостоятельные опыты |
| [05_visual_conditioning_references_and_rope.ipynb](../dit/notebooks/05_visual_conditioning_references_and_rope.ipynb) | 1. Три схемы visual condition; 2. Append / restore / generated mask; 3. Semantic time и storage index различаются; 4. Constraint projection и solver; Измерение → вывод; Самостоятельные опыты |
| [06_denoise_loop_euler_cfg_and_partial_sampling.ipynb](../dit/notebooks/06_denoise_loop_euler_cfg_and_partial_sampling.ipynb) | 1. Дискретизация интеграла и ошибка; 2. CFG — две оценки поля, одна Euler траектория; 3. Partial sampling: modality остаётся в input; 4. Helpers, RoPE caching и настоящее tiny forward; Измерение → вывод; Самостоятельные опыты |
| [07_piflow_math_dx_policy.ipynb](../dit/notebooks/07_piflow_math_dx_policy.ipynb) | 1. Почему clean endpoint связан с velocity; 2. Raw τ и shifted σ: вывод inverse и Jacobian; 3. Grid interpolation, error и endpoint amplification; 4. Polynomial mode: coefficients, не grid samples; 5. Native policy rollout: discretization vs representation; 6. Разная длина rollout в одном batch |
| [08_piflow_grid_head_and_native_sampler.ipynb](../dit/notebooks/08_piflow_grid_head_and_native_sampler.ipynb) | 1. Physical channels и expanded head; 2. Почему final segment короче; 3. MultimodalDXPolicy: независимые policies, совместно предсказанные параметры; 4. Полный native sampler с random tiny grid-DiT; 5. Ошибка compute budget и корректный итог; Измерение → вывод |
| [09_miniature_policy_imitation_distillation.ipynb](../dit/notebooks/09_miniature_policy_imitation_distillation.ipynb) | 1. FM regression и imitation — разные targets; 2. Train → measure, без утечки held-out target; 3. От velocity loss к trajectory error: что требует доказательства; 4. Что в этом эксперименте не воспроизведено; Измерение → вывод; Самостоятельные опыты |
| [10_comfyui_and_manual_research_protocol.ipynb](../dit/notebooks/10_comfyui_and_manual_research_protocol.ipynb) | 1. Где блоки находятся в ComfyUI graph; 2. Ручной baseline после resource gate; 3. Карточка одного исследования; Измерение → вывод; Самостоятельные опыты; Первичные источники |

## Промышленные семейства и архитектурные блоки

[Сводная industrial map с закреплёнными official sources](INDUSTRY_BRIDGE.md) — общий вход. Ниже это кейсы для переноса знаний; самостоятельных полноценных pretrained labs для каждого семейства пока нет.

| Семейство | Что искать после MIT | Материал стенда |
|---|---|---|
| HunyuanVideo | causal VAE, dual/single stream, flow scheduler | [Pinned source map](INDUSTRY_BRIDGE.md#tencent-hunyuanhunyuanvideo), [9 VAE labs](../vae/README.md), K5 02 / K6 03, ComfyUI C4 |
| Wan 2.1 | T5, Wan-VAE, time modulation, FM solver | [Pinned source map](INDUSTRY_BRIDGE.md#wan-videowan21) |
| Wan 2.2 | A14B noise experts vs dense TI2V-5B, codec contracts | [Pinned source map](INDUSTRY_BRIDGE.md#wan-videowan22) |
| Kandinsky 5 | Qwen/CLIP, CrossDiT, VAE, generation_utils | [14 labs](../kandinsky-5/README.md) |
| Kandinsky 6 | asymmetric AV streams, cross gates, π-Flow, audio codec/vocoder | [14 component labs](../kandinsky-6/README.md), [source method index](../kandinsky-6/COMPONENT_GUIDE.md) |
| LTX-Video | causal codec / Transformer / RF | [Pinned source map](INDUSTRY_BRIDGE.md#lightricksltx-video) |
| Mochi 1 | asymmetric text/visual streams, VAE | [Pinned source map](INDUSTRY_BRIDGE.md#genmoaimochi) |
| Open-Sora | release-specific MMDiT, codec и sampler | [Pinned source map](INDUSTRY_BRIDGE.md#hpcaitechopen-sora) |

Для сравнения AV fusion в K6 08 читаем [Ovi](https://arxiv.org/abs/2510.01284) и [LTX-2](https://arxiv.org/abs/2601.03233). Это отдельные статьи; LTX-Video в общей карте не переименовывается в LTX-2.

## Все ComfyUI опыты и готовые workflows

[Инструкция C0–C6](COMFYUI_LABS.md): C0 dataflow → C1 reproducibility/cache → C2 solver/NFE → C3 CFG → C4 codec reconstruction → C5 tokens/resources → C6 границы UI и Python/custom nodes.

| Workflow | Что демонстрирует | Статус |
|---|---|---|
| [Image IO без моделей](workflows/00_image_io_no_models.json) | LoadImage→SaveImage, known pixels | Static checked; запуск вручную |
| [Hunyuan VAE reconstruction](workflows/04_hunyuan_vae_reconstruction.json) | Codec отдельно от DiT/text | Static checked; запуск вручную с имеющимся VAE |
| [K6 Lite Smoke](../kandinsky-6/workflows/k6_lite_distill_smoke.json) | 432×256 / 33 frames / seed42 / 10 steps | Static checked; отдельный полный запуск Smoke не зафиксирован |
| [K6 Lite Baseline](../kandinsky-6/workflows/k6_lite_distill_baseline.json) | 864×480 / 121 frames / seed42 / 10 steps | MPS: сохранён MP4 со звуком, [tiled decode validation](../kandinsky-6/reports/comfy-tiled-vae-validation.json) |
| [K6 SR ×2](../kandinsky-6/workflows/k6_sr_standalone_2x.json) | Готовый MP4 → SR; расчёт 1728×960 для Baseline | Граф и CPU-контракты проверены; pretrained SR не запускался |
| [K6 SR ×2,25](../kandinsky-6/workflows/k6_sr_standalone_2_25x.json) | Тот же источник → SR; расчёт 1952×1088 после grid alignment | Нужен отдельный SR bundle 17,45 GB; [инструкция](../kandinsky-6/SUPER_RESOLUTION_RU.md) |

[Test chart](assets/curriculum_test_chart.png), [RGB comparison script](scripts/compare_images.py), [K6 stage measurement script](../kandinsky-6/scripts/measure_comfy.py). Во время CPU-лабораторных workflow не отправляются на выполнение.

## Отдельное исследование: от гипотезы к выводу

1. [Шесть уровней доказательств и research protocol](RESEARCH_PROTOCOL.md).
2. [Карточка одного эксперимента](EXPERIMENT_TEMPLATE.md).
3. [Существующий K6 manual notebook](../kandinsky-6/notebooks/00_manual_experiment_comfy_mlx_film.ipynb).
4. [Пошаговый voice/ComfyUI/MLX protocol](../kandinsky-6/COMFYUI_AND_EXPERIMENT_RU.md).
5. [Текущее состояние реализации](../kandinsky-6/STATUS.md) — какие этапы действительно выполнены.
6. [Baseline measurement table](../kandinsky-6/reports/templates/baseline_runs.csv) и [voice evaluation table](../kandinsky-6/reports/templates/voice_evaluation.csv).
7. [VCTK preparation](../kandinsky-6/scripts/prepare_vctk.py), [PyTorch fixture](../kandinsky-6/scripts/torch_parity_reference.py), [MLX parity prototype](../kandinsky-6/scripts/mlx_parity.py).

Первое запланированное исследование: speaker-conditioned zero-init FiLM на замороженном K6 audio backbone, VCTK и held-out texts; baseline/regression проверяются отдельно. Это гипотеза, а не уже полученное voice cloning. Desktop AV Baseline с tiled VAE проверен 30 октября; Kaggle/T4, voice adapter и MLX остаются отдельными этапами с собственными критериями.

## Все руководства, источники и результаты

- [CLAUDE.md](../CLAUDE.md) — Working on this study stand.
- [README.md](../README.md) — Generative AI Research.
- [VALIDATION.md](../VALIDATION.md) — Проверка проекта (исторический отчёт).
- [curriculum/CLAUDE_REVIEW_BRIEF.md](CLAUDE_REVIEW_BRIEF.md) — Задание Claude: отполировать Generative AI Studying Stand.
- [curriculum/COMFYUI_LABS.md](COMFYUI_LABS.md) — ComfyUI как наблюдаемый эксперимент.
- [curriculum/EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) — Карточка учебного или исследовательского эксперимента.
- [curriculum/INDUSTRY_BRIDGE.md](INDUSTRY_BRIDGE.md) — Как читать промышленный pipeline после Lab 1–3.
- [curriculum/PROGRESS.md](PROGRESS.md) — Самопроверка студента.
- [curriculum/README.md](README.md) — От вероятностей к исследованию генеративных видео- и аудиомоделей.
- [curriculum/RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) — Переход от учебного задания к исследованию.
- [curriculum/SOURCE_NOTES.md](SOURCE_NOTES.md) — Уточнения к исходным материалам.
- [curriculum/SUPPLEMENTARY_READING.md](SUPPLEMENTARY_READING.md) — Дополнительные материалы к основному курсу.
- [curriculum/references/README.md](references/README.md) — Источники и границы адаптации.
- [curriculum/reports/VALIDATION.md](reports/VALIDATION.md) — Проверка учебного маршрута, 27 октября 2026.
- [dit/MIT_BRIDGE.md](../dit/MIT_BRIDGE.md) — MIT 6.S184 → DiT/CrossDiT и π-Flow: точное соответствие.
- [dit/README.md](../dit/README.md) — DiT → CrossDiT → Conditioning → Denoise Loop → π-Flow.
- [dit/SOURCE_MAP.md](../dit/SOURCE_MAP.md) — Источники и карта каждого архитектурного блока.
- [dit/reports/VALIDATION.md](../dit/reports/VALIDATION.md) — DiT / CrossDiT / π-Flow: измеренные CPU результаты.
- [hunyuan-video/README.md](../hunyuan-video/README.md) — HunyuanVideo: архитектурные лабораторные.
- [kandinsky-5/MODEL_CATALOG.md](../kandinsky-5/MODEL_CATALOG.md) — Каталог моделей Kandinsky Lab.
- [kandinsky-5/README.md](../kandinsky-5/README.md) — KandinskyLab — учебные Jupyter-лабораторные по Kandinsky 5 Lite.
- [kandinsky-5/THIRD_PARTY.md](../kandinsky-5/THIRD_PARTY.md) — Сторонние материалы.
- [kandinsky-5/reference/course_materials/README.md](../kandinsky-5/reference/course_materials/README.md) — Материалы MIT и дополнения к лабораторным Kandinsky.
- [kandinsky-5/reports/VALIDATION.md](../kandinsky-5/reports/VALIDATION.md) — Статус проверки KandinskyLab.
- [kandinsky-6/COMFYUI_AND_EXPERIMENT_RU.md](../kandinsky-6/COMFYUI_AND_EXPERIMENT_RU.md) — Kandinsky 6: ручной запуск и протокол исследования.
- [kandinsky-6/COMPONENT_GUIDE.md](../kandinsky-6/COMPONENT_GUIDE.md) — K6: карта архитектурных компонентов и оригинального кода.
- [kandinsky-6/README.md](../kandinsky-6/README.md) — Kandinsky 6 Lite — по-компонентный учебный курс.
- [kandinsky-6/STATUS.md](../kandinsky-6/STATUS.md) — Состояние эксперимента, 30 октября 2026.
- [kandinsky-6/SUPER_RESOLUTION_RU.md](../kandinsky-6/SUPER_RESOLUTION_RU.md) — Kandinsky 6 SR: готовый Baseline MP4 → видео большего разрешения.
- [kandinsky-6/THIRD_PARTY.md](../kandinsky-6/THIRD_PARTY.md) — Source provenance.
- [kandinsky-6/reference/README.md](../kandinsky-6/reference/README.md) — Закреплённые сторонние исходники.
- [kandinsky-6/reports/COMPONENT_VALIDATION.md](../kandinsky-6/reports/COMPONENT_VALIDATION.md) — K6 component labs: проверка 27 октября 2026.
- [kandinsky-6/reports/VALIDATION.md](../kandinsky-6/reports/VALIDATION.md) — Проверка K6 материалов — 27 октября 2026.
- [lecture_01/LECTURE_01_RU.md](../lecture_01/LECTURE_01_RU.md) — Lecture 1: от генерации к движению случайных точек.
- [lecture_01/README.md](../lecture_01/README.md) — Lecture 1: ОДУ, СДУ и численное семплирование.
- [lecture_02/README.md](../lecture_02/README.md) — Lecture 2 → Lab 2: от probability path к обучаемому полю.
- [lecture_03/README.md](../lecture_03/README.md) — Lecture 3A / 3B: score matching, stochastic sampling и CFG.
- [lecture_04/README.md](../lecture_04/README.md) — Lecture 4 → Lab 3: DiT, VAE и обучение в latent space.
- [lecture_05/README.md](../lecture_05/README.md) — Lecture 5: CTMC и дискретный Flow Matching.
- [vae/MIT_LAB3_BRIDGE.md](../vae/MIT_LAB3_BRIDGE.md) — MIT Lab 3 → VAE → промышленный видеокодек.
- [vae/README.md](../vae/README.md) — VAE: от MIT Lab 3 до промышленного видеокодека.
- [vae/SOURCE_MAP.md](../vae/SOURCE_MAP.md) — HunyuanVideo VAE: карта источников, компонентов и границ выводов.
- [vae/reports/VALIDATION.md](../vae/reports/VALIDATION.md) — HunyuanVideo VAE: проверка и измеренные результаты.

**Индексы источников:** [PDF/lab manifest](references/materials_manifest.json), [industrial revisions](references/industry_manifest.json), [дополнительные статьи/репозитории](references/additional_sources.json), [K6 source hashes](../kandinsky-6/reference/source-manifest.json).

**Численные отчёты:** [общий CPU execution](reports/validation.json), [дополнительные CPU контроли](reports/supplementary-experiments.json), [K6 component execution](../kandinsky-6/reports/component-validation.json), [K6 educational results](../kandinsky-6/reports/educational-results.json). Полные executed notebooks и рабочие outputs находятся в ignored artifacts/reports/executed, а не в учебном оглавлении.

## Инструменты и воспроизведение

Это служебные точки входа для воспроизведения и развития материалов. Для чтения курса они не являются дополнительными обязательными лабораторными. Third-party source trees навигируются через manifests/COMPONENT_GUIDE, а не смешиваются с авторскими файлами.

### `curriculum/scripts`

- [build_bridge_notebooks.py](scripts/build_bridge_notebooks.py) — Build two small teaching notebooks. Never touches original labs or user answers.
- [build_comfy_workflows.py](scripts/build_comfy_workflows.py) — Build model-free and VAE-only study graphs. Never queues jobs or downloads weights.
- [build_contents.py](scripts/build_contents.py) — Build a complete navigation index of authored study materials, not vendored source trees.
- [check_markdown_math.py](scripts/check_markdown_math.py) — Check portable math style in authored Markdown guides; no notebook execution.
- [compare_images.py](scripts/compare_images.py) — Compare aligned RGB pixels; no hidden resize and no model inference.
- [index_additional_sources.py](scripts/index_additional_sources.py) — Pin supplementary reading/code metadata, never install or execute downloaded code.
- [index_industry_sources.py](scripts/index_industry_sources.py) — Pin public architecture entry points. Downloads text/metadata only, never weights.
- [index_materials.py](scripts/index_materials.py) — Index user-provided teaching sources without executing notebook instructions.
- [supplementary_experiments.py](scripts/supplementary_experiments.py) — Small CPU experiments for supplementary reading. No external package/model download.
- [sync_study_navigation.py](scripts/sync_study_navigation.py) — Keep MIT-first navigation in authored study docs; canonical originals/user answers untouched.
- [validate_curriculum.py](scripts/validate_curriculum.py) — Validate curriculum structure and optionally execute only its two small new labs.

### `lecture_01/scripts`

- [build_notebooks.py](../lecture_01/scripts/build_notebooks.py) — Собрать перевод и учебные дополнения, сохранив соответствие исходным ячейкам.
- [execute_notebook.py](../lecture_01/scripts/execute_notebook.py) — Выполнить Lab One в чистом ядре, сохранить результаты и HTML для чтения.

### `kandinsky-5/scripts`

- [add_colab_launch.py](../kandinsky-5/scripts/add_colab_launch.py) — Add reproducible Colab entry points without changing local lab behavior.
- [add_inference_example.py](../kandinsky-5/scripts/add_inference_example.py) — См. аргументы и назначение в исходнике.
- [build_generation_lab.py](../kandinsky-5/scripts/build_generation_lab.py) — Build a complete, output-free lab for every pinned generation utility.
- [build_normalization_audit.py](../kandinsky-5/scripts/build_normalization_audit.py) — Build a reproducible source, numerical and call-site audit, without weights.
- [build_notebooks.py](../kandinsky-5/scripts/build_notebooks.py) — См. аргументы и назначение в исходнике.
- [check_native_port.py](../kandinsky-5/scripts/check_native_port.py) — Meaningful component contract checks, no pretrained downloads.
- [deepen_labs.py](../kandinsky-5/scripts/deepen_labs.py) — Add reproducible theory chapters to the two existing labs after base generation.
- [extend_from_materials.py](../kandinsky-5/scripts/extend_from_materials.py) — Rebuild the source-grounded MIT additions in labs 02 and 06.
- [package_colab.py](../kandinsky-5/scripts/package_colab.py) — См. аргументы и назначение в исходнике.
- [verify_notebooks.py](../kandinsky-5/scripts/verify_notebooks.py) — См. аргументы и назначение в исходнике.

### `kandinsky-6/scripts`

- [build_comfy_baselines.py](../kandinsky-6/scripts/build_comfy_baselines.py) — Derive minimal Lite workflows from the pinned official graph.
- [build_component_guide.py](../kandinsky-6/scripts/build_component_guide.py) — Build README navigation and an AST-based component/source index.
- [build_component_notebooks.py](../kandinsky-6/scripts/build_component_notebooks.py) — Build the K6 component course. Only source generation, no network/weights.
- [build_kaggle_notebook.py](../kandinsky-6/scripts/build_kaggle_notebook.py) — Build the standalone, staged Kaggle 2xT4 Kandinsky 6 Lite deployment notebook.
- [build_manual_notebook.py](../kandinsky-6/scripts/build_manual_notebook.py) — Build the lightweight manual experiment notebook; no model downloads.
- [build_sr_workflows.py](../kandinsky-6/scripts/build_sr_workflows.py) — Derive standalone x2/x2.25 Kandinsky SR workflows from an installed official template.
- [extract_audio_local.py](../kandinsky-6/scripts/extract_audio_local.py) — Copy original audio tensor bytes from a local ComfyUI checkpoint.
- [inspect_comfy_models.py](../kandinsky-6/scripts/inspect_comfy_models.py) — Read model paths and safetensors headers, without importing ComfyUI or loading weights.
- [install_comfy_templates.py](../kandinsky-6/scripts/install_comfy_templates.py) — Install research templates locally; default is a dry run, no model download.
- [measure_comfy.py](../kandinsky-6/scripts/measure_comfy.py) — Record read-only ComfyUI memory snapshots during a manually queued generation.
- [mlx_parity.py](../kandinsky-6/scripts/mlx_parity.py) — Compare MLX against a saved PyTorch reference and record actual measurements.
- [prepare_vctk.py](../kandinsky-6/scripts/prepare_vctk.py) — Download a small, pinned VCTK pilot through remote ZIP byte ranges.
- [summarize_components.py](../kandinsky-6/scripts/summarize_components.py) — Publish a concise evidence report and selected figures from successful local executions.
- [torch_parity_reference.py](../kandinsky-6/scripts/torch_parity_reference.py) — Generate a PyTorch numerical reference; never label random fixtures pretrained.
- [validate_component_notebooks.py](../kandinsky-6/scripts/validate_component_notebooks.py) — Validate K6 teaching labs; --execute uses fresh CPU kernels, no downloads.
- [validate_kaggle_notebook.py](../kandinsky-6/scripts/validate_kaggle_notebook.py) — Validate the Kaggle deployment notebook without model downloads; optional native CPU API checks.
- [validate_manual_notebook.py](../kandinsky-6/scripts/validate_manual_notebook.py) — Validate notebook cells; --execute runs only its default lightweight route.
- [validate_sr_workflows.py](../kandinsky-6/scripts/validate_sr_workflows.py) — Check standalone SR graphs against a running ComfyUI, without submitting a prompt or loading weights.

### `vae/scripts`

- [benchmark_local_codec.py](../vae/scripts/benchmark_local_codec.py) — Dry-run by default; manually benchmark a LOCAL Diffusers Hunyuan VAE, no downloads.
- [build_notebooks.py](../vae/scripts/build_notebooks.py) — Generate the sequential HunyuanVideo VAE course; preserve notebook code in this source.
- [summarize_labs.py](../vae/scripts/summarize_labs.py) — Publish compact measured CPU results, CSV and selected plots after successful execution.
- [validate_labs.py](../vae/scripts/validate_labs.py) — Check all Hunyuan VAE labs; --execute runs fresh CPU kernels, no weights/downloads.
- [check_math.cjs](../vae/scripts/check_math.cjs) — optional strict KaTeX parse/render; dependency setup in vae README.

### `dit/scripts`

- [build_notebooks.py](../dit/scripts/build_notebooks.py) — Build a MIT-linked course from DiT primitives to K5/K6 CrossDiT and native π-Flow.
- [summarize_labs.py](../dit/scripts/summarize_labs.py) — Publish measured DiT/CrossDiT/π-Flow results, compact CSVs and selected plots.
- [validate_labs.py](../dit/scripts/validate_labs.py) — Validate DiT/CrossDiT/π-Flow labs; --execute uses fresh CPU kernels, no downloads.
- [check_math.cjs](../dit/scripts/check_math.cjs) — optional strict KaTeX parse/render; dependency setup in vae README.

## Как поддерживать содержание

`uv run --locked python curriculum/scripts/build_contents.py` собирает индекс из фактических authored notebooks/guide files. После изменений навигации выполните `sync_study_navigation.py --write`, затем `--check`. Порядок обучения задаёт эта карта вместе с curriculum README; автоиндекс не меняет original labs и пользовательские ответы.
