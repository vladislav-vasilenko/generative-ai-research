# Детали для технического интервью

Основа ответа — собственный R&D-стенд для autoregressive LLM serving. Текущие численные выводы и границы находятся в [отчёте](../reports/INFERENCE_OPTIMIZATION_REPORT.md), готовый текст — в [ответе рекрутеру](../reports/RECRUITER_ANSWER_RU.md).

## Что именно настраивалось

| Intervention | vLLM | SGLang |
|---|---|---|
| Chunked prefill OFF/ON | `--no-enable-chunked-prefill` / `--enable-chunked-prefill` | `--chunked-prefill-size -1` / `4096` |
| Размер/бюджет | `--max-num-batched-tokens 4096` → `512` | `--chunked-prefill-size 4096` → `512`; `--max-prefill-tokens 4096` фиксирован |
| Prefix cache OFF/ON | `--no-enable-prefix-caching` / `--enable-prefix-caching` | `--disable-radix-cache` / отсутствие этого флага |
| Параллельные запросы | Closed-loop client concurrency 1/2/4/8/16 | Та же нагрузка и transport |

Размер chunk у SGLang и batch token budget у vLLM имеют разную внутреннюю семантику. Сравнение intervention проводится внутри движка. Eager, FP16, TP=1, context 4096, max sequences 32 и memory fraction 0.70 фиксированы. Scheduling policy не менялась; у SGLang явно задан FCFS. Continuous batching остаётся частью штатного runtime.

## Что реализовано поверх движков

Изолированные среды, закреплённый model snapshot, подготовка точных templated prompt lengths, запуск/остановка серверов, readiness и semantic smoke, stdlib HTTP/SSE client с persistent connections, общий benchmark, сбор usage/resources/Prometheus и отчёты. OpenAI-compatible serving предоставляет сам движок. Non-streaming проверен smoke-тестом; timed нагрузка использует streaming. Новая архитектура модели в движки не портировалась.

## Как считать эффект

Для каждого repeat: `100 × (output_tok_s_on / output_tok_s_off − 1)`. Итог — медиана трёх парных изменений; min/max показывают наблюдаемый разброс, а не confidence interval. Абсолютные TTFT/throughput в таблицах — отдельные медианы per-repeat метрик. Это может отличаться от процента, посчитанного отношением двух медиан.

Output throughput считается по фактическому `usage.completion_tokens` и полному wall time точки. Ошибки не дают успешных tokens, но остаются во времени и error rate. Startup, smoke и warmup исключены. TTFT — первый непустой text delta на клиенте. SSE может объединять tokens, поэтому TPOT обозначен как оценка, а inter-chunk gaps не называются token ITL.

## Почему prefix caching дал большой эффект

Shared workload специально построен с одинаковым длинным контекстом и различающимся ID в конце. Cache начинает заполняться во время измеряемых запросов; warmup имеет другие IDs в начале prompt. Большая часть общего prefill затем переиспользуется, что видно в дельтах cache counters. Это сценарий высокой повторяемости prefix, а не прогноз ускорения произвольного чата.

Больший относительный эффект одного движка не изолирует преимущество его cache algorithm: различаются исходная скорость, scheduler/backend и PyTorch stack. В пилоте использованы разные Kaggle allocations; дополнительная проверка запускает оба движка на одной физической T4. Строгая причина различий между реализациями отдельно не установлена.

## Как проверялась корректность

На каждом рестарте проверялись model identity, natural-stop ответ «Paris», streaming и non-streaming. Timed нагрузка синтетическая, с `ignore_eos=True`, чтобы фиксировать вычислительный объём. Реальные prompt/output counts, IDs, warmup, timestamps и summaries проверяются [отдельным аудитом](../scripts/audit_measurements.py). Результаты не являются QA accuracy benchmark. Exact decoded outputs расходятся даже между некоторыми повторами; сохранение общего качества отдельно не доказано.

## Какие проблемы пришлось диагностировать

Первый prepare run выявил несовместимость с default return type `apply_chat_template`: `len(BatchEncoding)` считал поля. Исправлено явным `return_dict=False` и проверкой плоского списка token IDs.

SGLang startup первоначально завершался SIGKILL. Log дочернего процесса показал FlashInfer RMSNorm JIT link failure `cannot find -lcuda`. Runner находит существующий driver `libcuda.so.1`, создаёт scratch-ссылку `libcuda.so` и добавляет её каталог в `LIBRARY_PATH`. После этого serving и полный benchmark прошли. System CUDA и вычислительные kernels не менялись. Начальные ошибки сохранены в provenance.

## Граница практического опыта

В этой работе выполнены single-GPU AR serving и controlled ablations. Speculative decoding, смена scheduling policy, TP=2/PP/FSDP, BF16/FP8 и внешние workers пока не проверены. Video inference относится к отдельному треку; запуск Qwen через vLLM/SGLang не является практикой запуска video pipeline через эти движки.
