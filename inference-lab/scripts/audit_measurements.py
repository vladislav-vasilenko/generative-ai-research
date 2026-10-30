"""Independently audit complete campaign artifacts; no inference is performed."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from run_kaggle import campaign_cells


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8), label)


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def counter(path, name, labels=None):
    """Sum a counter's series, ignoring its *_created companion and gauges."""
    if not path.exists():
        return None
    values = []
    for line in path.read_text().splitlines():
        match = re.fullmatch(re.escape(name) + r'(\{.*\})?\s+([\d.eE+\-]+)', line)
        if match and all(f'{key}="{value}"' in (match[1] or "") for key, value in (labels or {}).items()):
            values.append(float(match[2]))
    return sum(values) if values else None


def cache_delta(cell, engine):
    names = (("vllm:prefix_cache_queries_total", None, "vllm:prefix_cache_hits_total", None)
             if engine == "vllm" else
             ("sglang:prompt_tokens_total", None, "sglang:cached_tokens_total", {"cache_source": "device"}))
    changes = []
    for name, labels in ((names[0], names[1]), (names[2], names[3])):
        before = counter(cell / "metrics-before.prom", name, labels)
        after = counter(cell / "metrics-after.prom", name, labels)
        changes.append(after - before if before is not None and after is not None else None)
    return dict(queried_prompt_tokens=changes[0], cached_prompt_tokens=changes[1],
                cached_fraction=changes[1] / changes[0] if all(value is not None for value in changes) and changes[0] else None,
                semantics="engine-specific counters; endpoint snapshots, not peak occupancy")


def audit_campaign(root):
    root = Path(root)
    config = read_json(root / "campaign.json")
    manifest = read_json(root / "workloads/model.json")
    require(config["revision"] == manifest["resolved_revision"], "Model revision mismatch")
    require(not read_json(root / "errors.json"), "Selected campaign contains failures")
    warm_config = config.get("warmup", dict(requests=4, input_tokens=512, output_tokens=64))
    workloads = {p.stem: {r["id"]: r for r in read_rows(p)} for p in (root / "workloads").glob("*.jsonl")}
    require(all((r["prompt_tokens"], r["output_tokens"]) ==
                (warm_config["input_tokens"], warm_config["output_tokens"])
                for r in workloads["warmup"].values()), "Warmup shape differs from campaign")
    expected_paths, cells, agreement = set(), [], []
    for engine, version in config["engines"].items():
        environment = read_json(root / engine / "environment.json")
        require(environment["packages"][engine] == version, f"Installed version mismatch: {engine}")
        for repeat in range(config["repeats"]):
            paired_rows = {}
            for arm in config["arms"]:
                run = root / engine / f"r{repeat}-{arm}"
                launch = read_json(run / "launch.json")
                require(launch["status"] == "MEASURED" and launch["dtype"] == "float16", f"Incomplete launch: {run}")
                require(launch["model_revision"] == config["revision"], f"Launch revision mismatch: {run}")
                for kind, concurrency in campaign_cells(arm, config):
                    cell = run / f"{kind}-c{concurrency}"
                    expected_paths.add(cell / "summary.json")
                    summary, settings = read_json(cell / "summary.json"), read_json(cell / "config.json")
                    rows, warmup = read_rows(cell / "requests.jsonl"), read_json(cell / "warmup.json")
                    require(len(rows) == config["requests"], f"Request count mismatch: {cell}")
                    require(len({r["id"] for r in rows}) == len(rows) and {r["id"] for r in rows} == set(workloads[kind]), f"Request IDs mismatch: {cell}")
                    require(len(warmup) == warm_config["requests"], f"Warmup count mismatch: {cell}")
                    require({r["id"] for r in warmup} == set(workloads["warmup"]), f"Warmup IDs mismatch: {cell}")
                    require(not ({r["id"] for r in rows} & {r["id"] for r in warmup}), f"Warmup overlaps measurement: {cell}")
                    require(settings["concurrency"] == concurrency and settings["streaming"] and settings["load_model"] == "closed_loop", f"Load settings mismatch: {cell}")
                    labels = dict(engine=engine, arm=arm, repeat=repeat, workload=kind)
                    require(settings["labels"] == labels and all(summary[k] == v for k, v in labels.items()), f"Cell labels mismatch: {cell}")
                    for row in rows + warmup:
                        source = workloads["warmup" if row in warmup else kind][row["id"]]
                        require(row["success"] and not row["error"] and row["text"].strip() and row["finish_reason"], f"Failed request: {cell}/{row['id']}")
                        require(row["usage"]["prompt_tokens"] == source["prompt_tokens"] and row["usage"]["completion_tokens"] == source["output_tokens"], f"Actual length mismatch: {cell}/{row['id']}")
                        require(row["expected_prompt_tokens"] == source["prompt_tokens"] and row["requested_output_tokens"] == source["output_tokens"], f"Requested length mismatch: {cell}/{row['id']}")
                        offsets = row["text_event_offsets_s"]
                        require(offsets and offsets == sorted(offsets) and 0 <= offsets[0] <= offsets[-1] <= row["e2e_ms"] / 1000, f"Invalid timestamps: {cell}/{row['id']}")
                        close(row["ttft_ms"], offsets[0] * 1000, f"TTFT mismatch: {cell}")
                        if len(offsets) > 1:
                            close(row["tpot_ms"], (offsets[-1] - offsets[0]) * 1000 / (source["output_tokens"] - 1), f"TPOT proxy mismatch: {cell}")
                    require(summary["attempted"] == summary["successful"] == len(rows) and summary["error_rate"] == 0, f"Success count mismatch: {cell}")
                    prompt = sum(r["usage"]["prompt_tokens"] for r in rows)
                    generated = sum(r["usage"]["completion_tokens"] for r in rows)
                    require((prompt, generated) == (summary["actual_prompt_tokens"], summary["actual_output_tokens"]), f"Token totals mismatch: {cell}")
                    close(summary["output_tokens_per_second"], generated / summary["wall_seconds"], f"Throughput mismatch: {cell}")
                    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
                        values = [r[metric] for r in rows if r[metric] is not None]
                        for p in (50, 95, 99):
                            close(summary[metric][f"p{p}"], quantile(values, p / 100), f"Percentile mismatch: {cell}/{metric}/p{p}")
                    cells.append(dict(path=str(cell.relative_to(root)), requests=len(rows), warmup_requests=len(warmup), prompt_tokens=prompt, output_tokens=generated, cache=cache_delta(cell, engine)))
                    if kind == "shared":
                        paired_rows[arm] = {r["id"]: r for r in rows}
            if "baseline" in paired_rows and "prefix" in paired_rows:
                before, after = paired_rows["baseline"], paired_rows["prefix"]
                require(before.keys() == after.keys(), f"Unpaired shared request IDs: {engine}/{repeat}")
                agreement.append(dict(engine=engine, repeat=repeat, pairs=len(before), exact_text_matches=sum(before[k]["text"] == after[k]["text"] for k in before)))
    require(set(root.glob("*/r*-*/*/summary.json")) == expected_paths, "Unexpected or missing measured cells")
    files = [root / "campaign.json", root / "workloads/model.json"] + sorted(expected_paths)
    return dict(status="VALIDATED", model_revision=config["revision"], measured_cells=len(cells),
                timed_requests=sum(c["requests"] for c in cells), output_tokens=sum(c["output_tokens"] for c in cells),
                excluded_warmup_requests=sum(c["warmup_requests"] for c in cells), cells=cells,
                decoded_text_agreement=agreement, quality_parity_established=False,
                evidence_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    result = audit_campaign(args.campaign)
    (args.campaign / "measurement-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "measured_cells", "timed_requests", "output_tokens", "excluded_warmup_requests")}, indent=2))
