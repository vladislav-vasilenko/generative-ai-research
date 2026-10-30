# Project: LLM Inference Optimization Lab

**Роль:** Senior ML Systems Engineer / Deep Learning Infrastructure Researcher.

**Цель:** Создать и выполнить воспроизводимые эксперименты с vLLM и SGLang, исследовать механизмы высокопроизводительного инференса Transformer-моделей и количественно оценить влияние оптимизаций на latency, throughput и потребление памяти.

Проект должен стать технически обоснованным R&D-кейсом с исходным кодом, экспериментальными данными и отчётом.

## Task 0. Hardware Discovery & Compatibility

Перед реализацией:

1. Определи ОС, CPU, GPU, архитектуру, объём RAM/VRAM, доступные вычислительные бэкенды.
2. Проверь версии Python, PyTorch, MLX, CUDA/Metal и доступные зависимости.
3. Изучи актуальные требования vLLM и SGLang.
4. Построй матрицу совместимости оптимизаций и платформ.
5. Выбери воспроизводимую конфигурацию.

Если используется Apple Silicon:
- Рассмотри vLLM-Metal и MLX-бэкенд SGLang.
- Учитывай, что CUDA-специфичные реализации и оптимизации могут быть недоступны.
- Не используй CPU fallback как эквивалент GPU-инференса.

Если доступны NVIDIA GPU:
- Используй нативные CUDA-бэкенды vLLM и SGLang.
- Предусмотри исследование Tensor Parallelism при наличии нескольких GPU.

Не устанавливай глобально конфликтующие зависимости. Используй изолированные окружения.

**Deliverable:** `docs/hardware_compatibility.md`

## Task 1. Deployment & API Serving

Разверни два независимых inference-сервера:

- vLLM
- SGLang

Выбери одну открытую Transformer-модель, совместимую с обоими бэкендами. Предпочтительно небольшую модель семейства Qwen, подходящую для доступной памяти.

Обеспечь:
- OpenAI-compatible API.
- Streaming и non-streaming inference.
- Health checks.
- Корректную токенизацию и chat templates.
- Контроль параметров генерации.
- Логирование конфигурации, версий и ошибок.
- Проверку корректности генерируемых ответов.

Для сравнительных тестов используй одинаковые веса модели, precision, tokenizer и generation settings.

**Deliverables:** конфигурации запуска, инструкции и smoke tests.

## Task 2. Inference Benchmark Harness

Разработай универсальный Python benchmarking framework для обоих серверов.

Поддерживаемые workloads:

| Workload | Input | Output |
|---|---|---|
| Short chat | 128 tokens | 64 tokens |
| Standard QA | 512 tokens | 128 tokens |
| Long context | 2048 tokens | 256 tokens |

Исследуй concurrency: 1, 2, 4, 8, 16 — если позволяет оборудование.

Реализуй сбор метрик:

- TTFT (Time To First Token).
- TPOT (Time Per Output Token).
- Inter-Token Latency.
- End-to-end latency (p50/p95/p99).
- Request throughput (req/s).
- Output throughput (tokens/s).
- Total token throughput.
- RAM/VRAM utilization.
- Error rate.
- Goodput при заданных latency SLO.

Для каждого эксперимента:
- Warmup.
- Достаточное число запросов.
- Не менее трёх независимых повторов для ключевых конфигураций.
- Фиксированный random seed.
- Контроль prefix-cache contamination.
- Сохранение исходных результатов JSON/CSV.

Сохраняй фактическое количество токенов, а не только запрошенный `max_tokens`.

**Deliverable:** `benchmarks/` с воспроизводимыми нагрузочными тестами.

## Task 3. Chunked Prefill & KV-cache

Исследуй влияние механизмов:

1. Chunked prefill enabled/disabled.
2. Prefill token budget.
3. Prefix caching enabled/disabled.
4. Maximum concurrent sequences.
5. KV-cache memory allocation.

Проверь гипотезы:

- Улучшает ли chunked prefill TTFT при смешанной prefill/decode-нагрузке?
- Как изменяется TPOT при увеличении concurrency?
- Как общий prefix влияет на throughput?
- При каких настройках возникают memory pressure и очереди запросов?

Используй one-variable-at-a-time ablations.

**Deliverable:** таблица результатов и графики зависимости throughput/latency от конфигурации.

## Task 4. Scheduling & Continuous Batching

Исследуй:

- Continuous batching.
- Request admission.
- Queueing.
- Scheduling policies, если поддерживаются.
- Head-of-line blocking.
- Различия между низкой и высокой конкурентной нагрузкой.

Создай два сценария:

**A. Homogeneous workload:** запросы приблизительно одинаковой длины.

**B. Heterogeneous workload:** короткие и длинные запросы поступают одновременно.

Измерь изменение TTFT, TPOT, p95 latency, throughput и goodput.

Определи, при каких условиях scheduler становится ограничивающим фактором.

**Deliverable:** `experiments/scheduling.md`

## Task 5. Speculative Decoding

Проверь возможность применения speculative decoding на доступном оборудовании.

Исследуй совместимые методы:

- Draft-model speculation.
- N-gram speculation.
- MTP, если поддерживается моделью и бэкендом.

Выполни baseline vs speculative decoding при одинаковой нагрузке.

Измерь:

- Decode throughput.
- TPOT.
- End-to-end latency.
- Speculative token acceptance rate.
- Дополнительный memory overhead.

Не обещай ускорение заранее. Исследуй также режимы, когда speculative decoding ухудшает производительность.

Если функциональность недоступна, зафиксируй причину и не выдавай эксперимент за выполненный.

**Deliverable:** `experiments/speculative_decoding.md`

## Task 6. Distributed Inference (Conditional)

Выполняется при наличии подходящего multi-GPU оборудования.

Исследуй Tensor Parallelism:

- TP=1 vs TP=2.
- Throughput scaling.
- Communication overhead.
- Memory distribution.
- Latency при разных уровнях concurrency.

При невозможности локального запуска:

- Подготовь отдельный воспроизводимый CUDA benchmark.
- Объясни архитектуру TP/PP и ограничения FSDP.
- Отметь эксперимент как NOT EXECUTED.

Не приравнивай distributed training к distributed inference.

## Task 7. Research Report & Recruiter Summary

Сформируй итоговый отчёт:

`reports/INFERENCE_OPTIMIZATION_REPORT.md`

Структура:

1. Hardware/software environment.
2. Model and inference engines.
3. Experimental methodology.
4. Baseline performance.
5. Parameter ablation results.
6. Latency/throughput trade-offs.
7. Memory bottlenecks.
8. Comparison: vLLM vs SGLang.
9. Unsupported experiments and limitations.
10. Engineering conclusions.

Построй графики:
- Throughput vs concurrency.
- TTFT/TPOT vs concurrency.
- Latency vs throughput.
- KV-cache/memory utilization.
- Baseline vs optimized configuration.

Дополнительно сформируй `reports/RECRUITER_ANSWER_RU.md` — краткий профессиональный ответ на вопрос GigaRecruiter:

«Какие сценарии инференса вы реализовывали в SGLang/vLLM, какие параметры тюнили и какой прирост latency/throughput получили?»

Ответ должен использовать только фактически полученные результаты.

Не выдумывай ускорения, production-scale нагрузки, commercial experience или результаты неисполненных экспериментов.

## Definition of Done

Проект считается завершённым, если:

- Оба inference-движка протестированы либо документированы реальные причины невозможности запуска.
- Baseline измерен.
- Проведены минимум два поддерживаемых оптимизационных эксперимента.
- Результаты воспроизводимы.
- Сохранены все конфигурации и сырые измерения.
- Построены сравнительные графики.
- Написан технический отчёт.
- Подготовлен честный ответ рекрутеру.

**Приоритет:** сначала работающий end-to-end benchmark с реальными измерениями, затем параметрические исследования. Не усложняй архитектуру до получения baseline. Не заменяй выполненные эксперименты теоретическими рассуждениями.