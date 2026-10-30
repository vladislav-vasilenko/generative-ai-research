"""Validate and combine vLLM and SGLang evidence from separate Kaggle sessions."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from run_kaggle import campaign_cells
from summarize import build_report


def load_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge_campaigns(vllm, sglang, output):
    sources = {"vllm": Path(vllm), "sglang": Path(sglang)}
    configs = {engine: load_json(path / "campaign.json") for engine, path in sources.items()}
    common = ("model", "revision", "requests", "repeats", "concurrency", "seed", "tensor_parallel",
              "gpu_devices", "memory_fraction", "max_model_len", "max_sequences", "arms", "slo",
              "cells", "warmup", "measurement_scope")
    for field in common:
        if configs["vllm"].get(field) != configs["sglang"].get(field):
            raise ValueError(f"Incompatible campaign setting: {field}")
    manifests = {engine: load_json(path / "workloads/model.json") for engine, path in sources.items()}
    for field in ("resolved_revision", "chat_template_sha256", "files"):
        if manifests["vllm"][field] != manifests["sglang"][field]:
            raise ValueError(f"Incompatible model/tokenizer manifest: {field}")
    for name in sorted(path.name for path in (sources["vllm"] / "workloads").glob("*.jsonl")):
        if file_hash(sources["vllm"] / "workloads" / name) != file_hash(sources["sglang"] / "workloads" / name):
            raise ValueError(f"Workloads differ: {name}")
    if file_hash(sources["vllm"] / "source/harness.py") != file_hash(sources["sglang"] / "source/harness.py"):
        raise ValueError("Benchmark transport/metrics implementations differ")
    for engine, source in sources.items():
        config = configs[engine]
        environment = load_json(source / engine / "environment.json")
        if environment["packages"][engine] != config["engines"][engine]:
            raise ValueError(f"Installed {engine} version does not match the campaign")
        for arm in config["arms"]:
            for repeat in range(config["repeats"]):
                launch = load_json(source / engine / f"r{repeat}-{arm}" / "launch.json")
                if launch["status"] != "MEASURED" or launch["dtype"] != "float16":
                    raise ValueError(f"Incomplete/invalid arm: {engine}/{repeat}/{arm}")
                for kind, concurrency in campaign_cells(arm, config):
                    cell = source / engine / f"r{repeat}-{arm}" / f"{kind}-c{concurrency}"
                    summary = load_json(cell / "summary.json")
                    requests = [json.loads(line) for line in (cell / "requests.jsonl").read_text().splitlines()]
                    if summary["error_rate"] or len(requests) != config["requests"] or not all(r["success"] for r in requests):
                        raise ValueError(f"Invalid requests in {cell}")
                    if len({r["id"] for r in requests}) != len(requests):
                        raise ValueError(f"Duplicate requests in {cell}")
                    for key, total in (("prompt_tokens", "actual_prompt_tokens"), ("completion_tokens", "actual_output_tokens")):
                        if sum(r["usage"][key] for r in requests) != summary[total]:
                            raise ValueError(f"Token-count mismatch in {cell}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    merged = copy.deepcopy(configs["vllm"])
    merged["engines"] = {engine: config["engines"][engine] for engine, config in configs.items()}
    merged["separate_kaggle_sessions"] = True
    merged["historical_error_records"] = [
        dict(source_engine=engine, **record) for engine, source in sources.items()
        if (source / "errors.json").exists() for record in load_json(source / "errors.json")]
    (output / "campaign.json").write_text(json.dumps(merged, indent=2) + "\n")
    shutil.copytree(sources["vllm"] / "workloads", output / "workloads")
    for engine, source in sources.items():
        shutil.copytree(source / engine, output / engine)
        provenance = output / "provenance" / engine
        provenance.mkdir(parents=True)
        shutil.copytree(source / "source", provenance / "source")
        for name in ("campaign.json", "errors.json", "cuda-linker.json"):
            if (source / name).exists():
                shutil.copy2(source / name, provenance / name)
    (output / "errors.json").write_text("[]\n")
    analysis = output / "analysis-source"
    analysis.mkdir()
    for name in ("summarize.py", "merge_campaigns.py", "run_kaggle.py"):
        shutil.copy2(Path(__file__).parent / name, analysis / name)
    result = build_report(output)
    (output / "merge-validation.json").write_text(json.dumps(dict(
        validated_fields=list(common), harness_sha256=file_hash(sources["vllm"] / "source/harness.py"),
        model_revision=manifests["vllm"]["resolved_revision"], complete_request_validation=True,
        separate_sessions=True), indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_campaigns(args.vllm, args.sglang, args.output), indent=2))
