# Ответ рекрутеру — только по выполненным измерениям

В учебном R&D-проекте развернул vLLM 0.31.0 и SGLang 0.5.22 с Qwen/Qwen2.5-0.5B-Instruct в FP16 на одной NVIDIA T4. Настроил OpenAI-compatible streaming и non-streaming chat serving; написал health/correctness checks и общий closed-loop benchmark. Проверял короткие, длинные и смешанные запросы при concurrency [1, 2, 4, 8, 16]. Это лабораторный опыт.

Настраивал chunked prefill и его token budget, prefix caching; отдельно исследовал поведение continuous batching при неоднородных длинах. Ниже медианы относительных изменений по трём или более повторам на заранее выбранных сценариях; фактические tokens взяты из server usage.

- vllm, baseline → chunked4096, mixed, C=8: throughput -1.4% (диапазон повторов -2.0…+0.7%), E2E p95 +1.3%, TTFT p50 -1.8%.
- vllm, chunked4096 → chunked512, mixed, C=8: throughput +1.9% (диапазон повторов +1.5…+4.6%), E2E p95 -1.6%, TTFT p50 +11.4%.
- vllm, baseline → prefix, shared, C=8: throughput +35.4% (диапазон повторов +33.5…+39.7%), E2E p95 -32.6%, TTFT p50 -89.7%.
- sglang, baseline → chunked4096, mixed, C=8: throughput -0.0% (диапазон повторов -0.1…+1.3%), E2E p95 -0.2%, TTFT p50 -11.1%.
- sglang, chunked4096 → chunked512, mixed, C=8: throughput +5.5% (диапазон повторов -6.1…+6.7%), E2E p95 -6.5%, TTFT p50 -49.4%.
- sglang, baseline → prefix, shared, C=8: throughput +133.6% (диапазон повторов +120.2…+134.3%), E2E p95 -38.6%, TTFT p50 -91.1%.

Сценарий shared использует общий длинный prefix; эти результаты нельзя переносить на произвольный чат. Политики scheduler, speculative decoding и multi-GPU scaling в этот прогон не входили. Конфигурации, per-request JSONL, ошибки, замеры памяти и версии сохранены.
