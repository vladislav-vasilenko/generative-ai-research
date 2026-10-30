# Ответ рекрутеру — только по выполненным измерениям

В учебном R&D-проекте развернул vLLM 0.31.0 и SGLang 0.5.22 с Qwen/Qwen2.5-0.5B-Instruct в FP16 на одной NVIDIA T4. Реализовал OpenAI-compatible streaming и non-streaming chat API, health/correctness checks и общий closed-loop benchmark. Проверял shared-prefix запросы 2048→256, concurrency=8; по 64 запроса и три перезапуска, 16 warmup той же длины. Оба движка последовательно на одной GPU. Это лабораторный опыт.

В этой проверке менял только prefix caching ON/OFF. Ниже медианы относительных изменений по трём или более повторам на заранее выбранных сценариях; фактические tokens взяты из server usage.

- vllm, baseline → prefix, shared, C=8: throughput +38.2% (диапазон повторов +33.6…+38.2%), E2E p95 -29.2%, TTFT p50 -90.8%.
- sglang, baseline → prefix, shared, C=8: throughput +145.1% (диапазон повторов +142.1…+148.7%), E2E p95 -37.6%, TTFT p50 -91.1%.

Сценарий shared использует общий длинный prefix; эти результаты нельзя переносить на произвольный чат. Политики scheduler, speculative decoding и multi-GPU scaling в этот прогон не входили. Конфигурации, per-request JSONL, ошибки, замеры памяти и версии сохранены.
