# План исследования под вакансию ML Systems Engineer

## Где вести работу

Сейчас — в отдельном `inference-lab/` этого репозитория. Основа уже есть: [K6](../../kandinsky-6/README.md), [результаты Desktop/Kaggle](../../kandinsky-6/STATUS.md), [DiT](../../dit/README.md), [VAE](../../vae/README.md), [ComfyUI](../../curriculum/COMFYUI_LABS.md). Лаборатория не добавляет inference engines в окружение MIT labs.

После двух измеренных движков и контролируемых ablations можно извлечь её в самостоятельный portfolio repository: отдельный entrypoint, README, revisions, raw data и отчёт уже определены. Создание второго пустого репозитория сейчас не добавляет экспериментальных доказательств.

## Какой опыт покажет каждый этап

| Требование вакансии | Проверяемый кейс | Что этот кейс ещё не доказывает |
|---|---|---|
| SGLang / vLLM | AR chat serving, streaming, ошибки, нагрузка и cache/prefill ablations | Реализацию новой архитектуры модели или production-scale опыт |
| Diffusers / FastVideo / ComfyUI | Один video checkpoint, component contracts и воспроизводимый pipeline | Parity разных default checkpoints/samplers |
| torch.distributed / TP / SP | TP=1/2 для AR; затем DiT sequence/context parallel с профилем коммуникаций | Суммирование VRAM, `device_map` placement или training FSDP как эквивалент inference parallelism |
| BF16 / FP8 | Отдельная hardware-supported конфигурация, quality/latency/memory контроль | Ускорение на T4 за счёт отсутствующих native dtype kernels |
| Model implementation | Перенос одного component, loader/config, numerical parity и integration tests | Простую установку готового движка как авторский model port |
| Workers | Контракт job API, readiness, очередь, timeout/cancel, output storage | Реальное размещение на fal.ai/Arena.ai без deployment и проверки provider contract |

## Этап A: AR serving — первый измеренный baseline

1. Подтвердить окружение двух CUDA движков, сохранить freeze и логи. Один Qwen snapshot, dtype FP16, одинаковые inputs и generation settings.
2. Проверить `/health`, `/v1/models`, streaming/non-streaming и простой semantic smoke. Проверить совпадение server prompt-token count с токенизатором и chat template.
3. Baseline: homogeneous 128→64 / 512→128 / 2048→256, closed-loop concurrency 1/2/4/8/16; mixed lengths и отдельный shared-prefix workload.
4. Три независимых arm/repeat, restart server, warmup отдельно. Cache OFF в baseline; в prefix ON запросы не дублируют целый prompt, общий context задан намеренно.
5. Ablations: OFF→ON chunking с неизменным budget; затем budget 4096→512; отдельно prefix OFF→ON. Сохранять каждый запрос, фактические tokens, ошибку, текстовые SSE timestamps и память.
6. После пилота увеличить число запросов и добавить open-loop arrival-rate нагрузку. Разделять client, transport, server queue и GPU execution; для причин scheduler bottleneck нужен server trace, не только график latency.

Первая кампания **не** меняет scheduling policy: у SGLang FCFS, у vLLM фиксируется его resolved config. Continuous batching работает в движках. После baseline можно отдельно проверить подходящую policy SGLang на mixed нагрузке и admission cap в каждом движке.

Speculative decoding — следующий conditional experiment: сначала совместимость N-gram/draft и модели, затем acceptance, memory overhead и latency при низкой/высокой concurrency. Для Qwen2.5-0.5B MTP capability не предполагается. При отсутствии запуска писать NOT EXECUTED с конкретной причиной; «не вошло в кампанию» не означает «не поддерживается».

## Этап B: video inference — ближе к задаче команды

SGLang Diffusion и vLLM-Omni поддерживают diffusion/video pipelines, а не только serving текстовых энкодеров. Здесь главные границы измерения — `text conditioning → denoising DiT → video/audio VAE → export`. Оптимизация text encoder — самостоятельный компонентный опыт с проверкой hidden states/embeddings; chat completion API не является заменой conditioning interface.

**Compatibility gate до загрузок:** проверить конкретные Kandinsky 6 Lite checkpoint/config и implementation в выбранных pinned revisions обоих движков. Официальная страница с Pro не доказывает поддержку Lite. Если общей конфигурации нет, зафиксировать unsupported вариант и выбрать небольшой общий video checkpoint по первичным support matrices; никаких фиктивных запусков Pro на T4.

В сохранённых [K6 vLLM-Omni instructions](../../kandinsky-6/reference/kandinsky-6/docs/quickstart/quick_start_vllm.md) default — Pro / Euler; в [SGLang instructions](../../kandinsky-6/reference/kandinsky-6/docs/quickstart/quick_start_sglang.md) — Pro-distill / PiFlow. Сравнение этих defaults меняет одновременно model, objective и число шагов. Его нельзя называть framework speedup.

Для сравнения зафиксировать один checkpoint и revision, precision, prompt conditioning, scheduler/timesteps, CFG, число кадров, resolution, FPS, seed/initial latents и NFE. Одинаковый seed в разных engines не гарантирует одинаковые initial latents. При невозможности общей sampling semantics описать разные режимы и сравнивать их как самостоятельные конфигурации с quality evaluation.

Сначала один clip, затем:

- Component timings и memory peaks: text encoder, DiT, video VAE, audio VAE/vocoder, export. Отдельно cold/warm, compile/load time.
- Offload baseline против одного изменения; отдельно VAE tiling. Не менять geometry и batch одновременно.
- Поддерживаемый denoising step cache: latency вместе с quality regression и фиксированными inputs. Это другой механизм, чем autoregressive prefix/KV cache.
- SP/CP на двух GPU только при реальной поддержке: compute vs communication profile. Ранее измеренный `device_map` placement не является TP/SP speedup.
- API/worker: очередь задач, progress, cancel/timeout, bounded concurrency, ошибки, artifact URLs. Provider deployment — после проверки рабочего локального контракта.

Источники: [SGLang Kandinsky 6](https://docs.sglang.io/cookbook/diffusion/Kandinsky/Kandinsky6), [vLLM-Omni video examples](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/text_to_video/), [vLLM-Omni K6 scheduler](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/models/kandinsky6/scheduling_kandinsky6/). Актуальная web-документация и локальные pinned snapshots могут описывать разные revisions; runtime gate обязателен.

## Этап C: ответ рекрутеру

Вопрос: «Какие сценарии инференса реализовывали в SGLang/vLLM, какие параметры тюнили и какой прирост latency/throughput получили?»

Каждый ответ содержит: учебный/рабочий контекст, model+GPU+версии, реализованный сценарий, baseline, одно изменение, нагрузку, число повторов, конкретный эффект и его границы. Для mixed/shared-prefix workloads эти условия называются явно. Негативный результат — допустимый инженерный вывод.

До выполнения не писать «получил X%», «production high-load», «реализовал модель внутри движка» или «ускорил видеогенерацию». Если выполнен только AR-кейс, сказать это прямо. Speculative decoding/TP/policy tuning не включать в опыт, пока они не исполнены. Автоматическая заготовка ответа создаётся из реальных парных результатов; перед отправкой работодателю проверить формулировки по raw data.
