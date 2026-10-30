# Inference Optimization Lab

Лаборатория воспроизводимого инференса под задачи ML Systems Engineer. Сначала — измеренный baseline, затем одно изменение и проверка результата. Численные ускорения появляются только после реальных запусков.

**Начинаем в `generative-ai-research`.** Здесь уже есть K6, Diffusers, ComfyUI, DiT/VAE и журнал запусков. У лаборатории отдельные код, результаты и среды движков; основной `pyproject.toml` не меняется. Выделять самостоятельный `inference-optimization-lab` разумно после baseline двух движков, двух ablations и законченного отчёта, когда содержимое уже воспроизводится отдельно от учебного курса.

## Два разных исследования

| Трек | Что запускаем | Что измеряем | Связь с вакансией |
|---|---|---|---|
| LLM serving, первый end-to-end | Одна Qwen через CUDA vLLM и SGLang | TTFT, TPOT, throughput, goodput, ошибки, память; chunked prefill и prefix caching | Практический ответ рекрутеру про serving, очереди и scheduler |
| Video inference, следующий основной кейс | Общий поддерживаемый video checkpoint в Diffusers / SGLang Diffusion / vLLM-Omni; далее FastVideo / ComfyUI | Text encoding, denoising, VAE отдельно; E2E, VRAM, качество, offload, step cache, SP/CP | Прямой опыт переноса и ускорения diffusion/video пайплайна |

SGLang Diffusion и vLLM-Omni исполняют генеративный видеопайплайн, включая DiT и decode. Использование vLLM для autoregressive Qwen само по себе не доказывает опыт video serving или реализации новой архитектуры модели. Метрики autoregressive KV-cache и speculative decoding не переносятся на diffusion denoising автоматически.

## Первый запуск: Kaggle T4 ×2

**[Приватный Kaggle notebook](https://www.kaggle.com/code/applehawk/llm-inference-vllm-vs-sglang-t4-lab)** · [Локальный notebook](notebooks/01_kaggle_vllm_sglang_t4.ipynb).

**SGLang после исправления Kaggle linker:** [отдельная кампания](https://www.kaggle.com/code/applehawk/llm-inference-sglang-t4-compatibility) · [Notebook 02](notebooks/02_kaggle_sglang_t4_compatibility.ipynb). У версии 1 диагностический режим, у версии 2 — полная SGLang матрица. Первая ошибка: FlashInfer RMSNorm JIT не находил `-lcuda`; [подлинный server log](reports/diagnostics/sglang-linker-v1/sglang/r0-baseline/server.log). Новый runner создаёт scratch-ссылку на настоящий `libcuda.so.1` и добавляет только linker search path. Это не изменение kernels или системной CUDA.

[Notebook 03: focused prefix validation](notebooks/03_kaggle_prefix_cache_validation.ipynb): 64 запроса, C=8, три повтора, 16 warmup 2048→256, оба движка на одном GPU. [Версия 1 завершена](https://www.kaggle.com/code/applehawk/llm-inference-prefix-cache-validation-t4): 12 точек, 768 успешных timed requests, 196608 output tokens. [Проверенные данные](reports/prefix-validation-2026-10-31/) сохраняются отдельно от 32-request пилота.

1. В Kaggle: GPU T4 ×2, Internet ON. Notebook самодостаточен; локальный checkout не требуется.
2. В первом сравнении используется GPU 0, движки последовательно. GPU 1 остаётся для отдельного TP-эксперимента.
3. Выполнить ячейки по порядку. Среды vLLM и SGLang создаются в `/kaggle/temp`, Python kernel не переустанавливается.
4. Сохранить Version вместе с Output. После первого успешного пилота увеличить requests для исследования tail latency.

Модель: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae5576` с записью полного resolved SHA и хешей файлов. FP16, eager, TP=1, context 4096. Это малая модель для проверки методики; выводов о масштабном production-инференсе из неё не делаем.

Пилот: inputs 128/512/2048 tokens **включая chat template**, outputs 64/128/256, concurrency 1/2/4/8/16, 32 запроса на точку, три перезапуска. Отдельно mixed lengths и shared prefix. SLO заданы до измерения: TTFT ≤ 1 s, TPOT ≤ 100 ms, E2E ≤ 30 s. Это лабораторные пороги, не требования работодателя.

| Arm | Chunked prefill | Budget | Prefix cache | Сравнение |
|---|---|---:|---|---|
| baseline | OFF | 4096 | OFF | Все homogeneous workloads; mixed C=8/16; shared C=8 |
| chunked4096 | ON | 4096 | OFF | С baseline на mixed: включение механизма |
| chunked512 | ON | 512 | OFF | С chunked4096: изменение бюджета |
| prefix | OFF | 4096 | ON | С baseline на shared: повторное использование общего prefix |

У SGLang chunk size и prefill batch cap — отдельные параметры; в протоколе зафиксированы оба. Названия похожих настроек двух движков не означают идентичную внутреннюю семантику. Сравнение эффектов выполняется прежде всего внутри каждого движка.

## Код и результаты

- [Hardware compatibility](docs/hardware_compatibility.md) и [локальный снимок](reports/hardware-local-2026-10-30.json).
- [Технический отчёт: пилот и дополнительная проверка](reports/INFERENCE_OPTIMIZATION_REPORT.md) и [готовый ответ рекрутеру](reports/RECRUITER_ANSWER_RU.md).
- [План и связь с вакансией](docs/RESEARCH_PLAN_RU.md); [исходное пользовательское ТЗ](docs/LLM_TASK_SPEC_RU.md).
- [Детали для технического интервью](docs/INTERVIEW_NOTES_RU.md): точные флаги, формулы, диагностика и границы практического опыта.
- [Benchmark harness](benchmarks/harness.py): persistent HTTP connections, SSE, actual server usage, raw requests, ошибки и ресурсы.
- [Подготовка входов](scripts/prepare_workloads.py): одна revision, tokenizer/chat-template contract, точные длины, SHA-256.
- [Kaggle campaign](scripts/run_kaggle.py): независимые среды, рестарты, smoke, конфигурации, логи.
- [Report builder](scripts/summarize.py): CSV, графики, парные эффекты и ответ рекрутеру из измерений.
- [Объединение кампаний](scripts/merge_campaigns.py): проверяет weights/tokenizer/workload/harness hashes, параметры и все фактические tokens перед соединением результатов двух Kaggle sessions.
- [Независимый аудит](scripts/audit_measurements.py): сверяет raw request IDs, длины, warmup, streaming timestamps, throughput и перцентили; [аудит пилота](reports/cuda-2026-10-30/measurement-audit.json).

Output каждого запуска содержит `campaign.json`, `workloads/`, `source/`, `freeze.txt`, hardware/environment, `launch.json`, `server.log`, `requests.jsonl`, `resources.jsonl`, Prometheus snapshots и summaries. Большие веса и virtualenv не входят в результаты.

`RECRUITER_ANSWER_RU.md` содержит численный ответ только после baseline обоих движков и минимум трёх парных повторов всех запланированных ablations. Отсутствующие результаты и ошибки не заменяются оценками. Speculative decoding, изменение scheduling policy и TP=2 в первую кампанию не входят.

TTFT — время до первого непустого text delta на клиенте. TPOT — оценка по первым/последним text events и фактическому output count; SSE может объединять несколько токенов. **Token ITL пока недоступна**, inter-chunk gaps сохраняются под своим названием. С 32 запросами p99 остаётся описательной статистикой пилота. Closed-loop concurrency sweep не измеряет устойчивый production RPS или полную arrival queue.

Перед каждой точкой выполняются четыре warmup-запроса 512→64, вне измерений. Это ограниченный warmup: отдельного exhaustive shape/JIT warmup для всех concurrency и длин нет. Оба движка работают в eager; результаты относятся к этому режиму. `scheduling.csv` отдельно сопоставляет одни и те же short request IDs в homogeneous и mixed нагрузке, чтобы aggregate latency длинных запросов не скрывала эффект на коротких.

## Локальные проверки без моделей

Из корня репозитория, в существующей locked-среде:

```bash
.venv/bin/python inference-lab/scripts/discover_hardware.py
.venv/bin/python -m pytest -q inference-lab/tests
.venv/bin/python inference-lab/scripts/build_kaggle_notebook.py
```

Тесты открывают временный loopback HTTP server. Они проверяют transport и метрики, а не скорость модели. Для локального JSON-отчёта:

```bash
.venv/bin/python inference-lab/scripts/summarize.py /absolute/path/to/downloaded/campaign
```

Для Notebook 02: `build_kaggle_notebook.py --sglang-only`; короткий startup diagnostic: `--sglang-diagnostic`. Обычный вызов пересобирает Notebook 01. Статус и stdout можно читать штатными `kaggle kernels status owner/slug/version` и `kaggle kernels logs owner/slug/version --follow` в отдельно настроенной среде Kaggle CLI.

Для Notebook 03: `build_kaggle_notebook.py --prefix-validation`. Его `warmup`, `cells` и точные версии критичных packages записываются в `campaign.json`. Завершённая проверка и её raw audit сохраняются отдельно от пилота.

Редактируйте Python sources и пересобирайте notebook; его metadata содержит SHA-256 копий исходников. Пользовательские изменения в лекционных notebooks не затрагиваются.
