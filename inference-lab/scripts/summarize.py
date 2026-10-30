"""Build a report only from actual per-request measurements and paired repeats."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def build_report(output):
    output = Path(output)
    groups = defaultdict(list)
    rows = []
    for path in sorted(output.glob("*/r*-*/*/summary.json")):
        row = json.loads(path.read_text())
        row["artifact"] = str(path.relative_to(output))
        rows.append(row)
        if row["error_rate"] == 0:
            groups[(row["engine"], row["arm"], row["workload"], row["concurrency"])].append(row)
    config_path = output / "campaign.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    errors_path = output / "errors.json"
    errors = json.loads(errors_path.read_text()) if errors_path.exists() else []
    table = ["| Engine | Arm | Workload | C | Repeats | Output tok/s | TTFT p50 ms | TPOT p50 ms | E2E p95 ms |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    aggregates = []

    def median(items, key, quantile=None):
        values = [(item[key][quantile] if quantile else item[key]) for item in items]
        values = [value for value in values if value is not None]
        return statistics.median(values) if values else None

    def number(value):
        return f"{value:.2f}" if value is not None else "N/A"

    for (engine, arm, workload, concurrency), items in sorted(groups.items()):
        agg = dict(engine=engine, arm=arm, workload=workload, concurrency=concurrency,
                   repeats=len(items), output_tps=median(items, "output_tokens_per_second"),
                   ttft_p50_ms=median(items, "ttft_ms", "p50"),
                   tpot_p50_ms=median(items, "tpot_ms", "p50"),
                   e2e_p95_ms=median(items, "e2e_ms", "p95"))
        aggregates.append(agg)
        table.append(f"| {engine} | {arm} | {workload} | {concurrency} | {len(items)} | " +
                     " | ".join(number(agg[key]) for key in ("output_tps", "ttft_p50_ms", "tpot_p50_ms", "e2e_p95_ms")) + " |")
    (output / "summaries.json").write_text(json.dumps(rows, indent=2) + "\n")
    if aggregates:
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(aggregates)
    comparisons = []
    for engine in config.get("engines", {}):
        for base, arm, workload, concurrency in (
                ("baseline", "chunked4096", "mixed", 8),
                ("chunked4096", "chunked512", "mixed", 8),
                ("baseline", "prefix", "shared", 8)):
            before = {row["repeat"]: row for row in groups.get((engine, base, workload, concurrency), [])}
            after = {row["repeat"]: row for row in groups.get((engine, arm, workload, concurrency), [])}
            paired = sorted(before.keys() & after.keys())
            if len(paired) < 3:
                continue
            throughput = [100 * (after[i]["output_tokens_per_second"] / before[i]["output_tokens_per_second"] - 1) for i in paired]
            latency = [100 * (after[i]["e2e_ms"]["p95"] / before[i]["e2e_ms"]["p95"] - 1) for i in paired]
            ttft = [100 * (after[i]["ttft_ms"]["p50"] / before[i]["ttft_ms"]["p50"] - 1) for i in paired]
            comparisons.append(dict(engine=engine, baseline=base, arm=arm, workload=workload,
                                    concurrency=concurrency, paired_repeats=len(paired),
                                    output_tps_change_pct=statistics.median(throughput),
                                    output_tps_change_range_pct=[min(throughput), max(throughput)],
                                    e2e_p95_change_pct=statistics.median(latency),
                                    ttft_p50_change_pct=statistics.median(ttft)))
    (output / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    focused = config.get("measurement_scope") == "prefix_validation"
    baseline_cells = config.get("cells", {}).get("baseline") or [
        (kind, c) for kind in ("short", "standard", "long") for c in config.get("concurrency", [1, 2, 4, 8, 16])]
    required_baselines = [(engine, "baseline", kind, c) for engine in ("vllm", "sglang")
                          for kind, c in baseline_cells]
    both = all(len(groups.get(key, [])) >= 3 for key in required_baselines)
    status = "MEASURED — see scope and repeat counts" if both else "INCOMPLETE — no validated two-engine baseline yet"
    report = f"# Inference optimization report\n\nStatus: **{status}**.\n\n"
    report += f"Model: `{config.get('model', 'unknown')}`; FP16; CUDA_VISIBLE_DEVICES={config.get('gpu_devices', '?')}; TP={config.get('tensor_parallel', '?')}.\n\n"
    report += f"Requests per cell: {config.get('requests', 'unknown')}; planned independent repeats: {config.get('repeats', 'unknown')}.\n\n"
    shapes = ("2048-token inputs and forced 256-token outputs" if focused else
              "exact templated inputs 128/512/2048 tokens and forced outputs 64/128/256")
    report += (f"Synthetic closed-loop chat load, {shapes}. "
               "Identical model snapshot and workload files are reused by both engines. Environments are separate; "
               "this compares packaged serving stacks, including their PyTorch/tokenizer versions.\n\n"
               "Servers restart for every arm/repeat. Startup, semantic smoke and warmup are excluded. "
               "Tables show the median across per-repeat metrics, not a pooled percentile. "
               "Tail percentiles are descriptive, without a production p99 or production-scale claim.\n\n")
    warmup = config.get("warmup", dict(requests=4, input_tokens=512, output_tokens=64))
    report += (f"Warmup: {warmup['requests']} separate {warmup['input_tokens']}-input/"
               f"{warmup['output_tokens']}-output requests per point. CUDA graphs are disabled.\n\n")
    if focused:
        report += ("Focused validation: only shared-prefix 2048→256 at concurrency=8 is measured. "
                   "Warmup matches these lengths and covers two full concurrency waves. Warmup IDs occur early "
                   "in the prompt and differ from measured prompts, leaving the measured shared context unprimed "
                   "apart from the common chat header. Both engines run sequentially on one physical GPU. "
                   "64 requests give eight concurrency waves; no production p99 claim is made.\n\n")
    else:
        report += "The pilot does not establish exhaustive shape/JIT warmup; results are exploratory eager serving measurements.\n\n"
    report += "\n".join(table) + "\n\n## Paired ablations\n\n"
    report += "Relative changes use matched repeats; a negative latency change means lower latency.\n\n"
    for comparison in comparisons:
        report += f"- `{comparison['engine']}` {comparison['baseline']} → {comparison['arm']}, {comparison['workload']} C=8: "
        report += f"output throughput {comparison['output_tps_change_pct']:+.1f}%, E2E p95 {comparison['e2e_p95_change_pct']:+.1f}%, "
        report += f"TTFT p50 {comparison['ttft_p50_change_pct']:+.1f}%; {comparison['paired_repeats']} paired repeats.\n"
    if not comparisons:
        report += "No three-repeat ablation result available.\n"
    report += ("\n## Measurement boundaries\n\nClient TTFT = first non-empty text SSE delta. TPOT is a client estimate "
               "from first/last text event and actual server completion_tokens. Token ITL is unavailable over "
               "text-only SSE; raw inter-chunk intervals are stored separately. Errors contribute to wall time "
               "and error rate, but not successful token throughput. SLO goodput uses the fixed campaign thresholds.\n\n"
               "VRAM is sampled per physical GPU by nvidia-smi; allocated memory includes reserved engine pools. "
               "RAM RSS sums may double-count shared pages. Raw Prometheus snapshots are included; no peak KV "
               "occupancy is inferred from endpoint snapshots.\n\n"
               "Correctness gates cover semantic smoke, API/stream completion and actual prompt/output token counts. "
               "These synthetic load tests do not establish general answer quality or lossless decoded-output parity.\n\n"
               "Continuous batching remains active. Concurrency and heterogeneous lengths test scheduler behavior; "
               "no scheduling-policy speedup is claimed. Speculative decoding and TP=2 are NOT EXECUTED in this "
               "first campaign. BF16/FP8 are not tested on T4. Diffusion/video and external workers are separate tracks.\n\n")
    report += f"Recorded failures in selected measurement campaigns: {len(errors)}. See `errors.json`, per-engine `install.log`, `freeze.txt`, `environment.json`, and per-arm `server.log`.\n"
    if config.get("separate_kaggle_sessions"):
        report += f"Earlier failed attempts retained in provenance: {len(config.get('historical_error_records', []))}. These are not successful timed requests.\n"
        report += ("\n## Separate-session provenance\n\n"
                   "vLLM and SGLang were measured in separate Kaggle sessions. Model/weight hashes, tokenizer/chat "
                   "template, workloads, transport code and campaign parameters were validated before merging. "
                   "Within-engine ablations use matched independent repeats. Cross-engine comparisons describe "
                   "complete stacks on separate T4 allocations, not paired runs on one physical GPU. Original "
                   "sources/configurations and earlier failures are retained under provenance/. SGLang's JIT link "
                   "fix exposes the existing driver library through a scratch link, without changing CUDA kernels.\n")
    (output / "INFERENCE_OPTIMIZATION_REPORT.md").write_text(report)
    recruiter = "# Ответ рекрутеру — только по выполненным измерениям\n\n"
    expected_effects = 2 * sum(arm in config.get("arms", []) for arm in ("chunked4096", "chunked512", "prefix"))
    if not both or not expected_effects or len(comparisons) < expected_effects:
        recruiter += ("Ответ с численным приростом пока не готов: нужны baseline обоих движков и три повтора "
                      "контролируемых ablations. Подготовка стенда и transport-тесты не подтверждают практический "
                      "результат инференса. См. технический отчёт и логи.\n")
    else:
        recruiter += (f"В учебном R&D-проекте развернул vLLM {config['engines']['vllm']} и SGLang {config['engines']['sglang']} "
                      f"с {config['model']} в FP16 на одной NVIDIA T4. Настроил OpenAI-compatible streaming и "
                      "non-streaming chat serving; написал health/correctness checks и общий closed-loop benchmark. Проверял "
                      + ("shared-prefix запросы 2048→256, concurrency=8; по 64 запроса и три перезапуска, "
                         "16 warmup той же длины. Оба движка последовательно на одной GPU. Это лабораторный опыт.\n\n"
                         if focused else f"короткие, длинные и смешанные запросы при concurrency {config.get('concurrency')}. Это лабораторный опыт.\n\n")
                      + ("В этой проверке менял только prefix caching ON/OFF. " if focused else
                         "Настраивал chunked prefill и его token budget, prefix caching; отдельно исследовал поведение continuous batching при неоднородных длинах. ")
                      + "Ниже медианы относительных изменений по трём или более повторам на заранее выбранных сценариях; фактические tokens взяты из server usage.\n\n")
        for comparison in comparisons:
            lo, hi = comparison["output_tps_change_range_pct"]
            recruiter += (f"- {comparison['engine']}, {comparison['baseline']} → {comparison['arm']}, "
                          f"{comparison['workload']}, C=8: throughput {comparison['output_tps_change_pct']:+.1f}% "
                          f"(диапазон повторов {lo:+.1f}…{hi:+.1f}%), E2E p95 {comparison['e2e_p95_change_pct']:+.1f}%, "
                          f"TTFT p50 {comparison['ttft_p50_change_pct']:+.1f}%.\n")
        recruiter += ("\nСценарий shared использует общий длинный prefix; эти результаты нельзя переносить на "
                      "произвольный чат. Политики scheduler, speculative decoding и multi-GPU scaling в этот "
                      "прогон не входили. Конфигурации, per-request JSONL, ошибки, замеры памяти и версии сохранены.\n")
    (output / "RECRUITER_ANSWER_RU.md").write_text(recruiter)
    build_scheduling_report(output)
    if aggregates:
        if focused:
            plot_prefix_validation(output, aggregates)
        else:
            plot_results(output, aggregates)
    return dict(status=status, measured_cells=len(rows), comparisons=comparisons, errors=len(errors))


def build_scheduling_report(output):
    """Compare the same short request IDs alone and beside long requests."""
    cells = []
    short_ids = {}
    for path in output.glob("*/r*-*/*/summary.json"):
        summary = json.loads(path.read_text())
        requests_path = path.parent / "requests.jsonl"
        if summary["error_rate"] or summary["workload"] not in ("short", "mixed") or not requests_path.exists():
            continue
        if summary["concurrency"] not in (8, 16):
            continue
        requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
        cells.append((summary, requests))
        if summary["arm"] == "baseline" and summary["workload"] == "mixed":
            key = (summary["engine"], summary["repeat"], summary["concurrency"])
            short_ids[key] = {row["id"].rsplit("-", 1)[-1] for row in requests if row["expected_prompt_tokens"] == 128}
    buckets = []
    for summary, requests in cells:
        cases = []
        if summary["workload"] == "mixed":
            cases = [("mixed-short", [row for row in requests if row["expected_prompt_tokens"] == 128]),
                     ("mixed-long", [row for row in requests if row["expected_prompt_tokens"] == 2048])]
        elif summary["arm"] == "baseline":
            ids = short_ids.get((summary["engine"], summary["repeat"], summary["concurrency"]))
            if ids:
                cases = [("homogeneous-short", [row for row in requests if row["id"].rsplit("-", 1)[-1] in ids])]
        for case, selected in cases:
            if not selected:
                continue
            bucket = {key: summary[key] for key in ("engine", "arm", "repeat", "concurrency")}
            bucket.update(case=case, n=len(selected))
            for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
                values = [row[metric] for row in selected if row[metric] is not None]
                bucket[metric + "_p50"] = statistics.median(values) if values else None
                bucket[metric + "_p95"] = (statistics.quantiles(values, n=100, method="inclusive")[94]
                                             if len(values) > 1 else (values[0] if values else None))
            buckets.append(bucket)
    if not buckets:
        return []
    with (output / "scheduling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(buckets[0]))
        writer.writeheader()
        writer.writerows(buckets)
    comparisons = []
    for engine in ("vllm", "sglang"):
        for base, arm in (("baseline", "chunked4096"), ("chunked4096", "chunked512")):
            for concurrency in (8, 16):
                before = {r["repeat"]: r for r in buckets if r["engine"] == engine and r["arm"] == base
                          and r["case"] == "mixed-short" and r["concurrency"] == concurrency}
                after = {r["repeat"]: r for r in buckets if r["engine"] == engine and r["arm"] == arm
                         and r["case"] == "mixed-short" and r["concurrency"] == concurrency}
                paired = sorted(before.keys() & after.keys())
                if len(paired) < 3:
                    continue
                changes = [100 * (after[i]["ttft_ms_p50"] / before[i]["ttft_ms_p50"] - 1) for i in paired]
                comparisons.append(dict(engine=engine, baseline=base, arm=arm, concurrency=concurrency,
                                        paired_repeats=len(paired), case="short requests inside mixed load",
                                        ttft_p50_change_pct=statistics.median(changes),
                                        ttft_change_range_pct=[min(changes), max(changes)]))
    (output / "scheduling-comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    report = ("# Scheduling / heterogeneous load\n\n"
              "`scheduling.csv` separates short and long requests. Homogeneous short latency uses the same request "
              "IDs as the short half of the mixed workload, with the same 128-token input and 64-token output. "
              "This measures the effect of competing long requests under closed-loop load. It does not isolate "
              "scheduler CPU overhead from prefill GPU work or prove a specific head-of-line mechanism. "
              "No scheduling policy is changed. Each short bucket has only 16 requests per pilot repeat; "
              "tail estimates are exploratory.\n\n"
              "## Paired chunking effects on short-request TTFT\n\n")
    for item in comparisons:
        report += (f"- {item['engine']}, {item['baseline']} → {item['arm']}, C={item['concurrency']}: "
                   f"short-request TTFT p50 {item['ttft_p50_change_pct']:+.1f}%, "
                   f"{item['paired_repeats']} paired repeats.\n")
    if not comparisons:
        report += "No three-repeat short-request comparison available yet.\n"
    (output / "SCHEDULING_REPORT.md").write_text(report)
    return comparisons


def plot_prefix_validation(output, aggregates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    rows = json.loads((output / "summaries.json").read_text())
    config = json.loads((output / "campaign.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for index, engine in enumerate(("vllm", "sglang")):
        for column, key, label in ((0, "output_tps", "Output tokens/s"), (1, "ttft_p50_ms", "Client TTFT p50, ms")):
            axis = axes[index, column]
            raw = {}
            for x, arm in enumerate(("baseline", "prefix")):
                selected = next((r for r in aggregates if r["engine"] == engine and r["arm"] == arm), None)
                if selected:
                    axis.bar(x, selected[key], width=0.55, color=("#64748b", "#0891b2")[x])
                    raw[arm] = {r["repeat"]: r["output_tokens_per_second"] if column == 0 else r["ttft_ms"]["p50"]
                                for r in rows if r["engine"] == engine and r["arm"] == arm}
                    axis.scatter([x] * len(raw[arm]), list(raw[arm].values()), s=22,
                                 color="#0f172a", edgecolors="white", linewidths=0.6, zorder=4)
                    axis.annotate(f"{selected[key]:,.1f}", (x, selected[key]), xytext=(0, 9),
                                  textcoords="offset points", ha="center", fontsize=9)
            for repeat in raw.get("baseline", {}).keys() & raw.get("prefix", {}).keys():
                axis.plot([0, 1], [raw["baseline"][repeat], raw["prefix"][repeat]],
                          color="#334155", alpha=0.35, linewidth=0.8, zorder=3)
            axis.set(xticks=[0, 1], xticklabels=["Cache off", "Cache on"], ylabel=label,
                     title=f"{engine}: shared prefix, C=8")
            axis.margins(y=0.18)
            axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Qwen2.5-0.5B / T4 / FP16 eager / 2048 input + 256 output\n"
                 f"{config['requests']} requests per cell; {config['repeats']} server restarts per arm", fontsize=11)
    fig.text(0.5, 0.015, "Bars: medians of per-restart metrics. Dots/lines: individual matched restarts.",
             ha="center", fontsize=9, color="#334155")
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(plots / "prefix-validation.png", dpi=160)
    plt.close(fig)


def plot_results(output, aggregates):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        (output / "PLOTS_UNAVAILABLE.txt").write_text("Install matplotlib in an isolated reporting environment.\n")
        return
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, kind in zip(axes, ("short", "standard", "long")):
        for engine in ("vllm", "sglang"):
            selected = sorted([row for row in aggregates if row["arm"] == "baseline" and row["workload"] == kind
                               and row["engine"] == engine], key=lambda row: row["concurrency"])
            axis.plot([r["concurrency"] for r in selected], [r["output_tps"] for r in selected], "o-", label=engine)
        axis.set(title=kind, xlabel="Concurrency", ylabel="Output tokens/s")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "throughput-concurrency.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, key, label in zip(axes, ("ttft_p50_ms", "tpot_p50_ms", "e2e_p95_ms"),
                               ("TTFT p50, ms", "TPOT p50 estimate, ms", "E2E p95, ms")):
        for engine in ("vllm", "sglang"):
            selected = sorted([row for row in aggregates if row["arm"] == "baseline" and row["workload"] == "long"
                               and row["engine"] == engine], key=lambda row: row["concurrency"])
            axis.plot([r["concurrency"] for r in selected], [r[key] for r in selected], "o-", label=engine)
        axis.set(xlabel="Concurrency", ylabel=label, title="Long context")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "latency-concurrency.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 4))
    for engine in ("vllm", "sglang"):
        selected = sorted([row for row in aggregates if row["arm"] == "baseline" and row["workload"] == "long"
                           and row["engine"] == engine], key=lambda row: row["concurrency"])
        axis.plot([r["output_tps"] for r in selected], [r["e2e_p95_ms"] for r in selected], "o-", label=engine)
    axis.set(xlabel="Output tokens/s", ylabel="E2E p95, ms", title="Latency / throughput: long context")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "latency-throughput.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row_index, workload, arms in ((0, "mixed", ("baseline", "chunked4096", "chunked512")),
                                      (1, "shared", ("baseline", "prefix"))):
        for column, metric, ylabel in ((0, "output_tps", "Output tokens/s"),
                                       (1, "e2e_p95_ms", "E2E p95, ms")):
            axis = axes[row_index, column]
            for index, engine in enumerate(("vllm", "sglang")):
                selected = {r["arm"]: r for r in aggregates if r["engine"] == engine
                            and r["workload"] == workload and r["concurrency"] == 8}
                present = [(i, arm) for i, arm in enumerate(arms) if arm in selected]
                axis.bar([i + (index - 0.5) * 0.36 for i, _ in present],
                         [selected[arm][metric] for _, arm in present], width=0.36, label=engine)
            axis.set(xticks=list(range(len(arms))), xticklabels=arms, ylabel=ylabel,
                     title=f"{workload}, concurrency=8")
            axis.grid(axis="y", alpha=0.2)
            axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "baseline-ablations.png", dpi=160)
    plt.close(fig)
    memory = defaultdict(list)
    for summary_path in output.glob("*/r*-baseline/long-c*/summary.json"):
        summary = json.loads(summary_path.read_text())
        resource_path = summary_path.parent / "resources.jsonl"
        if not resource_path.exists():
            continue
        peaks = {}
        for line in resource_path.read_text().splitlines():
            for gpu in json.loads(line).get("gpu", []):
                try:
                    value = float(gpu["memory_used_mib"])
                except (ValueError, KeyError):
                    continue
                peaks[gpu["index"]] = max(peaks.get(gpu["index"], 0), value)
        for index, peak in peaks.items():
            memory[(summary["engine"], summary["concurrency"], index)].append(peak)
    if memory:
        fig, axis = plt.subplots(figsize=(7, 4))
        for engine, index in sorted({(engine, index) for engine, _, index in memory}):
            concurrencies = sorted(c for e, c, i in memory if e == engine and i == index)
            axis.plot(concurrencies, [statistics.median(memory[(engine, c, index)]) for c in concurrencies],
                      "o-", label=f"{engine} GPU {index}")
        axis.set(xlabel="Concurrency", ylabel="Sampled peak resident VRAM, MiB",
                 title="Long context: median across repeats, including reserved pools")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plots / "memory-concurrency.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(build_report(parser.parse_args().output), indent=2))
