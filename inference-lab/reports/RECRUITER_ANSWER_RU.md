# Ответ рекрутеру

Версия 31 октября 2026. Prefix-cache цифры — из завершённой дополнительной проверки с 64 запросами; chunked-prefill наблюдения — из отдельного пилота.

> Мой практический опыт с vLLM и SGLang — исследовательский стенд для autoregressive LLM serving. Развернул Qwen2.5-0.5B-Instruct в FP16 на одной NVIDIA T4, настроил streaming/non-streaming API и написал общий нагрузочный benchmark. Проверял короткие, длинные и смешанные запросы при concurrency 1–16. Тюнил chunked prefill ON/OFF, batch token budget в vLLM и chunk size в SGLang — 4096→512, а также prefix caching.
>
> На синтетическом сценарии с общим префиксом — 2048 input / 256 output tokens, concurrency=8 — prefix caching дал около **+38% output throughput в vLLM** и **+145% в SGLang**. TTFT p50 снизился примерно **1,11→0,10 с** и **9,05→0,81 с** соответственно. Это три парных повтора по 64 запроса на точку, с 16 warmup той же длины; оба движка запускались последовательно на одной физической GPU.
>
> Для chunked prefill результаты зависели от метрики. В пилоте смешанной нагрузки SGLang при C=8 уменьшение chunk 4096→512 снизило TTFT p50 коротких запросов примерно **1,86→0,106 с**. Это небольшая подгруппа — по 16 коротких запросов на повтор, три повтора. Изменения общего throughput были небольшими или нестабильными.
>
> Все измерения лабораторные, в eager-режиме. Исследовал поведение continuous batching на смешанных длинах; scheduling policy оставалась фиксированной. Speculative decoding и multi-GPU scaling в этой работе пока не проверял.

Точные paired throughput changes для cache: vLLM +38.16% (диапазон +33.62…+38.18%), SGLang +145.15% (+142.07…+148.75%). Это медианы **парных относительных изменений**; абсолютные значения — отдельные медианы per-repeat метрик. Диапазоны трёх повторов не являются confidence intervals.

Exact decoded-output parity и сохранение общего качества ответов не установлены. В дополнительной проверке exact text совпал в 85/192 cache OFF/ON парах у vLLM и 31/192 у SGLang. Метрика производительности относится к фиксированной длине выхода, а не QA accuracy.

[Методика и ограничения](INFERENCE_OPTIMIZATION_REPORT.md) · [Дополнительная проверка: JSON](prefix-validation-2026-10-31/comparisons.json) · [CSV](prefix-validation-2026-10-31/summary.csv) · [Raw audit](prefix-validation-2026-10-31/measurement-audit.json) · [Chunked-prefill subgroup audit](cuda-2026-10-30/short-chunk-audit.json).
