# Hardware discovery and compatibility

Проверено 30 октября 2026. [JSON локального обнаружения](../reports/hardware-local-2026-10-30.json) получен read-only скриптом без установки пакетов и загрузки весов.

## Наблюдаемая локальная среда

| Поле | Значение |
|---|---|
| Host | macOS 27.0.1, Darwin arm64 |
| SoC | Apple M3 Max, CPU 16 cores, GPU 40 cores |
| Память | 48 GB unified RAM; отдельного дискретного VRAM пула нет |
| Metal | `system_profiler` сообщает поддержку Metal |
| Python | Root `.venv`: 3.12.14; системный `python3`: 3.14.7 |
| PyTorch | 2.14.1, CUDA runtime отсутствует, MPS built=true |
| GPU visibility | В sandbox-процессе `MPS available=false`; это не вывод о работоспособности Metal на host |
| MLX / MLX-LM / vLLM / SGLang | В root `.venv` не установлены |
| Свободный SSD | Меняется: первое наблюдение около 6 GiB, JSON-снимок — 16.98 GiB |

Не используем CPU fallback как замену GPU-inference. Установка MLX/Metal будет в отдельной среде; root `.venv` и ComfyUI Desktop для неё не меняются. Сначала необходимы доступ к Metal из процесса запуска и достаточный диск под версии/веса/кеш.

## Выбранный основной стенд

Пользователь подтвердил Kaggle **2× NVIDIA T4**. Первый удалённый запуск подтвердил: Linux x86_64, Python 3.12.3, две Tesla T4 по 15360 MiB, SM75, driver 580.178.04, CUDA runtime 13.0; topology — PHB, без NVLink. Данные: [vLLM environment](../reports/diagnostics/kaggle-v1/vllm/environment.json), [SGLang environment](../reports/diagnostics/kaggle-v1/sglang/environment.json). Это успешная проверка установки/CUDA visibility, а не serving baseline: v1 остановился при подготовке входов.

Для первого AR baseline: Qwen2.5-0.5B-Instruct, общие weights/tokenizer, FP16, один физический GPU. Другой GPU не занимает параллельный serving процесс. T4 ×2 не является общей 32 GB памятью: TP требует явного sharding и коммуникаций.

| Возможность | M3 Max / Metal | Kaggle T4 / CUDA | Как использовать |
|---|---|---|---|
| vLLM AR serving | vLLM-Metal: отдельный hardware plugin с MLX backend | Native CUDA vLLM; T4 имеет SM75 | Основное сравнение на CUDA, Metal — самостоятельная ветвь |
| SGLang AR serving | Apple/MLX backend: проверять выбранную версию и модель | Native CUDA, выбран Triton attention | Smoke до нагрузки; feature parity не предполагается |
| FP16 | Проверить backend и checkpoint | Базовый dtype | Один dtype в обоих CUDA движках |
| BF16/FP8 Tensor Core compute | Не переносить CUDA-поддержку на Metal | T4 не имеет native BF16/FP8 Tensor Core compute | Для этого нужен другой CUDA GPU; FP8 storage/KV — отдельная возможность |
| FlashAttention 2/3 | CUDA kernels не применимы напрямую | FA2 не является Turing backend; FA3 ориентирован на Hopper | Здесь выбран Triton, фактический backend проверить в server log |
| Chunked prefill / prefix caching | Наличие/ограничения зависят от plugin/backend | Доступность выбранных flags подтверждается запуском | Не заявлять поддержку только по hardware matrix |
| Speculative decoding | Зависит от backend/model/method | Возможна проверка N-gram/draft; MTP требует подходящей модели | В первом прогоне NOT EXECUTED |
| TP=2 / PP / SP | GPU cores внутри SoC не являются отдельными distributed ranks | Две GPU — возможность TP; topology/overhead проверяем отдельно | TP=1 baseline обязателен; speedup не гарантирован |
| Video inference | Отдельно проверять конкретный pipeline; K6 Desktop результаты уже в проекте | SGLang Diffusion / vLLM-Omni / FastVideo: собственные support/memory gates | Наличие T4 в AR matrix не доказывает поддержку video pipeline |

## Закрепление software

Первая кандидатная пара: `vllm==0.31.0`, `sglang==0.5.22`, Python 3.12, независимые virtualenv. Установка и CUDA visibility подтверждены v1; **успешный serving и корректность kernels требуют отдельного запуска**. В vLLM-среде установились PyTorch 2.13.0 / Transformers 5.17.0, в SGLang — PyTorch 2.14.1 / Transformers 5.19.0. Binary wheels несут собственные требования CUDA/PyTorch; пакет не подменяется произвольной host-версией torch. Если driver или kernel несовместим — сохраняем ошибку и выбираем новую закреплённую пару явно, с новым manifest.

Отдельные `freeze.txt` фиксируют транзитивные версии после установки. Сравнение относится к двум software stacks, а не к scheduler algorithms при идентичном PyTorch. Для повторения конкретного результата нужны его source snapshot, freeze, driver и model SHA.

## Первичные источники

- [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) — Linux/CUDA requirements, SM75+, изоляция среды и Metal plugin.
- [vLLM-Metal](https://github.com/vllm-project/vllm-metal) — отдельный путь Apple Silicon/MLX.
- [SGLang releases](https://github.com/sgl-project/sglang/releases) и [Apple backend roadmap](https://github.com/sgl-project/sglang/issues/19137) — проверять актуальные изменения отдельно от исторического статуса в issue.
- [vLLM 0.31 Triton backend](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/attention/backends/triton_attn.py), [SGLang 0.5.22 package requirements](https://github.com/sgl-project/sglang/blob/v0.5.22/python/pyproject.toml).
- [FlashAttention official repository](https://github.com/Dao-AILab/flash-attention) — hardware-specific attention support.
- [NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) — характеристики GPU; это не замер текущей Kaggle-сессии.
