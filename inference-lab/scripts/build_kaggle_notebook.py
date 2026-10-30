"""Build a private, self-contained Kaggle notebook from the reviewable Python sources."""
import ast
import argparse
import hashlib
import json
import pprint
import textwrap
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def build(diagnostic=False, sglang_only=False, prefix_validation=False):
    sources = {p.name: p.read_text() for p in [
        BASE / "benchmarks/harness.py", BASE / "scripts/discover_hardware.py",
        BASE / "scripts/prepare_workloads.py", BASE / "scripts/run_kaggle.py",
        BASE / "scripts/summarize.py"]}
    for name, source in sources.items():
        ast.parse(source, filename=name)
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    cells = []

    def normalized(source):
        first, separator, rest = source.partition("\n")
        return (first.lstrip() + ("\n" + textwrap.dedent(rest) if separator else "")).strip() + "\n"

    def md(source):
        source = normalized(source)
        cells.append(dict(cell_type="markdown", metadata={}, source=source,
                          id=hashlib.sha256(source.encode()).hexdigest()[:12]))

    def code(source):
        source = normalized(source)
        ast.parse(source)
        cells.append(dict(cell_type="code", metadata={}, source=source, execution_count=None, outputs=[],
                          id=hashlib.sha256(source.encode()).hexdigest()[:12]))

    md("""# Inference Lab: vLLM / SGLang · Kaggle T4

    **Первый кейс — autoregressive LLM serving**, не video inference. Отдельный видеокейс:
    SGLang Diffusion / vLLM-Omni / FastVideo; K6 baseline находится в основном репозитории.

    GPU **T4 ×2**, Internet ON. Для первого сравнения используется **только GPU 0**;
    движки запускаются последовательно. Модель: Qwen2.5-0.5B-Instruct, общие веса FP16.
    Среды движков изолированы от Python kernel. Никаких изменений ComfyUI или основного курса.

    Полный запуск устанавливает CUDA-зависимости и расходует GPU quota. Это exploratory benchmark:
    32 запроса на ячейку, три независимых перезапуска, concurrency 1/2/4/8/16.
    Для устойчивого tail latency увеличьте число запросов после первого успешного baseline.
    Численные результаты не заявлены до выполнения ячеек.
    """)
    md("""## 1. Проверить оборудование до установки

    T4: 16 GB на устройство, а не единый 32 GB пул. В этом прогоне FP16; нативные BF16/FP8
    Tensor Core операции не исследуются. Фактические driver, CUDA runtime и GPU записываются в Output.
    """)
    code("""from pathlib import Path
    import json, shutil, subprocess, sys
    if not Path('/kaggle/working').is_dir() or not shutil.which('nvidia-smi'):
        raise RuntimeError('Use Kaggle with GPU T4 ×2. No installation or weights downloaded.')
    print(subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version,compute_cap',
                                   '--format=csv,noheader'], text=True))
    print('Scratch free GiB:', round(shutil.disk_usage('/kaggle/working').free / 2**30, 1))
    """)
    md("""## 2. Восстановить закреплённые исходники лаборатории

    Самодостаточная копия пяти Python-файлов. Их SHA-256 хранится в metadata notebook;
    исходники сохраняются рядом с результатами. Эта ячейка только записывает код в scratch.
    В репозитории редактируйте `.py`, затем пересоберите notebook генератором.
    """)
    code("SOURCES = " + pprint.pformat(sources, width=110, sort_dicts=False) + "\n\n"
         "SOURCE_DIR = Path('/kaggle/temp/inference-lab-source')\n"
         "SOURCE_DIR.mkdir(parents=True, exist_ok=True)\n"
         "for name, source in SOURCES.items():\n"
         "    (SOURCE_DIR / name).write_text(source)\n"
         "sys.path.insert(0, str(SOURCE_DIR))\n"
         "print('Prepared:', ', '.join(SOURCES))\n")
    md("""## 3. Параметры кампании

    Baseline: prefix cache OFF, chunked prefill OFF, eager, FP16, TP=1, FCFS (SGLang).
    Сначала включаем chunked prefill с бюджетом 4096, затем отдельно уменьшаем бюджет до 512.
    Prefix cache ON сравнивается с baseline на одинаковых shared-prefix запросах.
    Между arm/repeat — новый сервер. Warmup использует другие request IDs и исключён из метрик.

    Число запросов в ячейке должно быть не меньше максимальной concurrency. Для репетиций
    сохраняются одни и те же входы. `ignore_eos=True` применяется только к synthetic нагрузке;
    semantic smoke проверяет естественную остановку и оба режима API.
    """)
    code("""import copy
    from datetime import datetime, timezone
    from harness import run_benchmark, smoke_server
    from run_kaggle import DEFAULT_CONFIG, run_campaign
    from summarize import build_report

    CONFIG = copy.deepcopy(DEFAULT_CONFIG)
    # Увеличивать после успешного первого baseline, например до 128/256 requests.
    CONFIG['requests'] = 32
    CONFIG['repeats'] = 3
    OUTPUT = Path('/kaggle/working') / ('inference-results-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    print(json.dumps(CONFIG, indent=2))
    print('Output:', OUTPUT)
    """)
    md("""## 4. Установка → smoke → измерения → отчёт

    Оба движка получают один snapshot/tokenizer, его полный revision и SHA-256 файлов записываются.
    Серверы доступны только по loopback. Неуспешные установки/запуски сохраняются в logs и `errors.json`;
    они не превращаются в нулевую latency или «ускорение» другого движка.

    TTFT здесь — первый непустой текстовый delta на клиенте; TPOT — оценка по фактическому числу
    generated tokens. **Интервалы SSE не выдаются за token ITL.**
    """)
    code("""RESULTS = run_campaign(CONFIG, OUTPUT, run_benchmark, smoke_server, build_report, SOURCE_DIR)
    print('Saved:', RESULTS)
    """)
    md("""## 5. Результаты для технического отчёта и рекрутера

    Сохраните Version с Output: `INFERENCE_OPTIMIZATION_REPORT.md`, `RECRUITER_ANSWER_RU.md`,
    `summary.csv`, `comparisons.json`, `plots/`, `workloads/`, `source/` и логи всех arms.
    Ответ с приростом появится только при наличии трёх повторов и baseline обоих движков.
    Speculative decoding, TP=2 и видеогенерация в эту первую кампанию не входят.
    """)
    code("""from IPython.display import display, Markdown, Image
    display(Markdown((OUTPUT / 'RECRUITER_ANSWER_RU.md').read_text()))
    for plot in sorted((OUTPUT / 'plots').glob('*.png')):
        display(Image(filename=str(plot)))
    """)
    if diagnostic or sglang_only:
        cells[0]["source"] = ("# SGLang T4 compatibility: startup diagnostic\n\n"
            "Приватный диагностический запуск SGLang 0.5.22 на одной T4. "
            "В основной кампании наблюдался SIGKILL при startup. Здесь ограничиваем "
            "KV pool через `--max-total-tokens 65536`, сохраняя остальные serving flags. "
            "Первый шаг — health + semantic/streaming smoke + 2 коротких запроса, один запуск. "
            "При ошибке выводится server.log, дальнейших повторов нет. "
            "Это не performance comparison и не основание для ответа о приросте. "
            "Ниже сохранён общий протокол лаборатории; фактические overrides видны в CONFIG.\n")
        for cell in cells:
            if cell["cell_type"] == "code" and "CONFIG = copy.deepcopy(DEFAULT_CONFIG)" in cell["source"]:
                cell["source"] = cell["source"].replace("OUTPUT = Path('/kaggle/working')",
                    "CONFIG.update(engines={'sglang': '0.5.22'}, requests=2, repeats=1, concurrency=[1], "
                    "arms=['baseline'], smoke_only=True, abort_on_server_failure=True, sglang_max_total_tokens=65536)\n"
                    "OUTPUT = Path('/kaggle/working')")
        if sglang_only:
            cells[0]["source"] = ("# SGLang T4: full serving campaign after JIT linker fix\n\n"
                "Предыдущий startup failure: FlashInfer RMSNorm JIT не находил `-lcuda`. "
                "В scratch создаётся ссылка `libcuda.so` на настоящий driver `libcuda.so.1`; "
                "путь добавляется в LIBRARY_PATH. Системные библиотеки не меняются. "
                "Здесь один SGLang 0.5.22, общий pinned Qwen, FP16, T4 GPU 0, "
                "32 запроса на ячейку, три перезапуска и полный протокол baseline/prefill/cache ниже. "
                "Стартовая ошибка останавливает кампанию и печатает server.log. "
                "Для итогового сравнения соединить Output с проверенными vLLM измерениями.\n")
            for cell in cells:
                if cell["cell_type"] == "code" and "CONFIG.update(engines=" in cell["source"]:
                    cell["source"] = cell["source"].replace(
                        "CONFIG.update(engines={'sglang': '0.5.22'}, requests=2, repeats=1, concurrency=[1], "
                        "arms=['baseline'], smoke_only=True, abort_on_server_failure=True, sglang_max_total_tokens=65536)",
                        "CONFIG.update(engines={'sglang': '0.5.22'}, abort_on_server_failure=True, cuda_driver_linker_fix=True)")
    if prefix_validation:
        cells[0]["source"] = ("# Prefix cache validation: vLLM / SGLang on one T4\n\n"
            "Прицельная проверка: общая Qwen2.5-0.5B-Instruct FP16, input 2048 / output 256, "
            "concurrency=8, 64 запроса на точку, три независимых server restarts на вариант. "
            "Единственная intervention — prefix cache OFF/ON. Движки работают последовательно "
            "на одном физическом GPU 0, в отдельных закреплённых средах. "
            "Перед каждым замером 16 отдельных warmup-запросов той же длины (две полные волны). "
            "Warmup IDs стоят в начале prompt и не priming-уют длинный измеряемый prefix. "
            "Общий chat header может повторно использоваться. p99 не является целью этой проверки.\n")
        for cell in cells:
            if cell["cell_type"] == "code" and "CONFIG = copy.deepcopy(DEFAULT_CONFIG)" in cell["source"]:
                cell["source"] = cell["source"].replace("OUTPUT = Path('/kaggle/working')",
                    "CONFIG.update(requests=64, repeats=3, concurrency=[8], arms=['baseline', 'prefix'], "
                    "measurement_scope='prefix_validation', abort_on_server_failure=True, "
                    "cells={'baseline': [['shared', 8]], 'prefix': [['shared', 8]]}, "
                    "warmup={'requests': 16, 'input_tokens': 2048, 'output_tokens': 256}, "
                    "engine_constraints={'vllm': ['torch==2.13.0', 'transformers==5.17.0', 'tokenizers==0.23.3'], "
                    "'sglang': ['torch==2.14.1', 'transformers==5.19.0', 'tokenizers==0.23.2']})\n"
                    "OUTPUT = Path('/kaggle/working')")
            if cell["cell_type"] == "markdown" and "## 3. Параметры кампании" in cell["source"]:
                cell["source"] = ("## 3. Параметры focused validation\n\n"
                    "Измеряются только `baseline/shared/C8` и `prefix/shared/C8`. Число запросов 64 "
                    "означает восемь concurrency waves. Kernel warmup — 16 запросов 2048→256; "
                    "target prefix начинает заполняться при измерениях. Cache state сбрасывается рестартом "
                    "между arm/repeat. Все templates, длины и фактические server tokens проверяются.\n")
    for cell in cells:
        cell["id"] = hashlib.sha256(cell["source"].encode()).hexdigest()[:12]
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python"), inference_lab_source_sha256=hashes,
        kaggle=dict(accelerator="nvidiaTeslaT4", isInternetEnabled=True)))
    path = BASE / ("notebooks/03_kaggle_prefix_cache_validation.ipynb" if prefix_validation else
                   "notebooks/02_kaggle_sglang_t4_compatibility.ipynb" if diagnostic or sglang_only else
                   "notebooks/01_kaggle_vllm_sglang_t4.ipynb")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    metadata = dict(id="applehawk/llm-inference-vllm-vs-sglang-t4-lab", title="LLM inference: vLLM vs SGLang T4 lab",
                    code_file=path.name, language="python", kernel_type="notebook", is_private=True,
                    enable_gpu=True, enable_internet=True, machine_shape="NvidiaTeslaT4",
                    dataset_sources=[], competition_sources=[], kernel_sources=[], model_sources=[])
    if diagnostic or sglang_only:
        metadata.update(id="applehawk/llm-inference-sglang-t4-compatibility",
                        title="LLM inference SGLang T4 compatibility")
    if prefix_validation:
        metadata.update(id="applehawk/llm-inference-prefix-cache-validation-t4",
                        title="LLM inference prefix cache validation T4")
    metadata_path = BASE / ("kaggle/kernel-metadata-prefix-validation.json" if prefix_validation else
                            "kaggle/kernel-metadata-sglang.json" if diagnostic or sglang_only else "kaggle/kernel-metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-diagnostic", action="store_true")
    parser.add_argument("--sglang-only", action="store_true")
    parser.add_argument("--prefix-validation", action="store_true")
    args = parser.parse_args()
    print(build(args.sglang_diagnostic, args.sglang_only, args.prefix_validation))
