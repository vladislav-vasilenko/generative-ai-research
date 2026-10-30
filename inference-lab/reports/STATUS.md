# Статус inference lab

Дата: 31 октября 2026. Пилот и дополнительная проверка завершены. [Приватный Kaggle notebook пилота](https://www.kaggle.com/code/applehawk/llm-inference-vllm-vs-sglang-t4-lab).

| Этап | Подтверждено |
|---|---|
| Локальное hardware discovery | M3 Max, 40 GPU cores, 48 GB unified RAM; [снимок](hardware-local-2026-10-30.json) |
| Transport / metrics | Transport-тесты, включая настоящий loopback HTTP/SSE, missing usage, truncated stream, HTTP error и concurrency; отдельный тест парных процентов и отказа формировать ответ по неполному baseline |
| Qwen tokenizer | Реальный tokenizer/config, full revision `7ae557604adf67be50417f59c2c2f167def9a775`; все шесть workloads подготовлены с точными templated lengths; локально без весов |
| Kaggle v1 | Обе среды установлены, CUDA 13.0 доступна, T4 ×2 подтверждены; ошибка подготовки inputs до serving |
| Kaggle v2 | Повтор завершился до serving из-за syntax error; добавлены AST-проверки исходников и notebook cells |
| Kaggle v3 | vLLM завершён: 69 точек, 2208/2208 timed requests успешны, 356352 output tokens; actual token totals проверены по raw JSONL. SGLang в этой старой среде падал при startup |
| SGLang compatibility v1 | Причина подтверждена: FlashInfer RMSNorm JIT link failure `cannot find -lcuda`; SIGKILL был следствием crash дочернего процесса |
| SGLang compatibility v2 | Завершён, все 69 точек и 2208 timed requests успешны; raw данные проверены |
| Baseline / ablations | Оба пилота объединены: 138 точек, 4416 успешных requests, 712704 output tokens; проверены manifests, workloads, harness и token totals |
| Video inference / TP=2 / speculative decoding | В первую AR-кампанию не входят |
| Focused validation | Notebook 03 / версия 1 COMPLETE: 12 точек, 768/768 успешных timed requests, 196608 output tokens; 64 requests, C8, три повтора, 16 shape-matched warmup, оба engines на одном GPU; [raw audit](prefix-validation-2026-10-31/measurement-audit.json) |

V1 выявил изменение default return type в Transformers: `len(BatchEncoding)` считал поля контейнера, а не tokens. Исправление проверено на настоящем Qwen tokenizer в отдельной существующей среде Transformers 5.22.1. [Подлинный prepare log v1](diagnostics/kaggle-v1/prepare.log) и [versions](diagnostics/kaggle-v1/vllm/freeze.txt) сохранены. Ни тестовый HTTP fixture, ни успешная установка движков не являются результатом инференса.

[SGLang linker evidence](diagnostics/sglang-linker-v1/sglang/r0-baseline/server.log) показывает реальную build-ошибку, а не OOM. Изменение KV token cap её не устранило. Для следующего запуска добавлен путь к реальной driver library через scratch-ссылку, без изменения system CUDA и без подмены вычислительных kernels.

[Технический отчёт](INFERENCE_OPTIMIZATION_REPORT.md), [обновлённый ответ рекрутеру](RECRUITER_ANSWER_RU.md), [проверенный dataset пилота](cuda-2026-10-30/) и [отдельная validation](prefix-validation-2026-10-31/) сохранены. Протоколы и оценки двух кампаний не смешиваются.

В дополнительной shared 2048→256 / C8 проверке медиана парного прироста output throughput с prefix caching составила +38.16% у vLLM и +145.15% у SGLang; TTFT p50 −90.81% и −91.07%. Это synthetic/eager/closed-loop измерения. Exact decoded text baseline/cache-on совпал в 85/192 и 31/192 парах; сохранение общего качества не установлено. Фактическая длина выхода совпала во всех парах.
