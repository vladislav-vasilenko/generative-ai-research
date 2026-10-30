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
    required_baselines = [(engine, "baseline", kind, c) for engine in ("vllm", "sglang")
                          for kind in ("short", "standard", "long")
                          for c in config.get("concurrency", [1, 2, 4, 8, 16])]
    both = all(len(groups.get(key, [])) >= 3 for key in required_baselines)
    status = "MEASURED — see scope and repeat counts" if both else "INCOMPLETE — no validated two-engine baseline yet"
    report = f"# Inference optimization report\n\nStatus: **{status}**.\n\n"
    report += f"Model: `{config.get('model', 'unknown')}`; FP16; CUDA_VISIBLE_DEVICES={config.get('gpu_devices', '?')}; TP={config.get('tensor_parallel', '?')}.\n\n"
    report += f"Requests per cell: {config.get('requests', 'unknown')}; planned independent repeats: {config.get('repeats', 'unknown')}.\n\n"
    report += ("Synthetic closed-loop chat load, exact templated inputs 128/512/2048 tokens, forced outputs 64/128/256. "
               "Identical model snapshot and workload files are reused by both engines. Environments are separate; "
               "this compares packaged serving stacks, including their PyTorch/tokenizer versions.\n\n"
               "Servers restart for every arm/repeat. Startup, semantic smoke and warmup are excluded. "
               "Tables show the median across per-repeat metrics, not a pooled percentile. "
               "A 32-request pilot does not estimate production p99 reliably. No production-scale claim.\n\n")
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
               "Continuous batching remains active. Concurrency and heterogeneous lengths test scheduler behavior; "
               "no scheduling-policy speedup is claimed. Speculative decoding and TP=2 are NOT EXECUTED in this "
               "first campaign. BF16/FP8 are not tested on T4. Diffusion/video and external workers are separate tracks.\n\n")
    report += f"Recorded failures: {len(errors)}. See `errors.json`, per-engine `install.log`, `freeze.txt`, `environment.json`, and per-arm `server.log`.\n"
    (output / "INFERENCE_OPTIMIZATION_REPORT.md").write_text(report)
    recruiter = "# Ответ рекрутеру — только по выполненным измерениям\n\n"
    if not both or len(comparisons) < 6:
        recruiter += ("Ответ с численным приростом пока не готов: нужны baseline обоих движков и три повтора "
                      "контролируемых ablations. Подготовка стенда и transport-тесты не подтверждают практический "
                      "результат инференса. См. технический отчёт и логи.\n")
    else:
        recruiter += (f"В учебном R&D-проекте развернул vLLM {config['engines']['vllm']} и SGLang {config['engines']['sglang']} "
                      f"с {config['model']} в FP16 на одной NVIDIA T4. Реализовал OpenAI-compatible streaming и "
                      "non-streaming chat API, health/correctness checks и общий closed-loop benchmark. Проверял "
                      f"короткие, длинные и смешанные запросы при concurrency {config.get('concurrency')}. Это лабораторный опыт.\n\n"
                      "Настраивал chunked prefill и его token budget, prefix caching; отдельно исследовал поведение "
                      "continuous batching при неоднородных длинах. Ниже медианы относительных изменений по трём "
                      "или более повторам на заранее выбранных сценариях; фактические tokens взяты из server usage.\n\n")
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
    if aggregates:
        plot_results(output, aggregates)
    return dict(status=status, measured_cells=len(rows), comparisons=comparisons, errors=len(errors))


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
