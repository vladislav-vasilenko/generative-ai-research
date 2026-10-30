"""Run inside an engine environment: pin one snapshot and materialize chat inputs."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare_workloads(model_id, revision, cache_dir, output, requests, warmup_requests=4,
                      warmup_input_tokens=512, warmup_output_tokens=64):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = Path(snapshot_download(model_id, revision=revision, cache_dir=cache_dir,
                                      ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx"]))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)

    def item(kind, index, target, output_tokens, shared=False):
        identity = f"Request {index:06d}. "
        question = " Explain inference optimization as a numbered list."
        prefix = "Read the following context. " if shared else identity + "Read the context. "
        suffix = (identity if shared else "") + question
        padding = target
        for _ in range(12):
            messages = [{"role": "user", "content": prefix + " data" * max(0, padding) + suffix}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 return_dict=False)
            if not isinstance(ids, list) or not all(isinstance(token, int) for token in ids):
                raise TypeError("Expected a flat token ID list from explicit return_dict=False")
            if len(ids) == target:
                return dict(id=f"{kind}-{index:06d}", workload=kind, messages=messages,
                            prompt_tokens=len(ids), output_tokens=output_tokens)
            padding += target - len(ids)
        raise RuntimeError(f"Cannot construct exact {target}-token chat input: got {len(ids)}")

    workloads = {}
    for kind, length, generated in (("short", 128, 64), ("standard", 512, 128), ("long", 2048, 256)):
        workloads[kind] = [item(kind, i, length, generated) for i in range(requests)]
    workloads["mixed"] = [item("mixed", i, 128 if i % 2 == 0 else 2048,
                               64 if i % 2 == 0 else 256) for i in range(requests)]
    workloads["shared"] = [item("shared", i, 2048, 256, True) for i in range(requests)]
    workloads["warmup"] = [item("warmup", 100000 + i, warmup_input_tokens, warmup_output_tokens)
                           for i in range(warmup_requests)]
    for kind, rows in workloads.items():
        (output / f"{kind}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    files = {}
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            files[str(path.relative_to(snapshot))] = dict(bytes=path.stat().st_size,
                                                        sha256=digest.hexdigest())
    manifest = dict(model_id=model_id, requested_revision=revision, resolved_revision=snapshot.name,
                    snapshot_path=str(snapshot), files=files,
                    chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                    synthetic=True, exact_prompt_lengths_include_chat_template=True)
    (output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--warmup-input-tokens", type=int, default=512)
    parser.add_argument("--warmup-output-tokens", type=int, default=64)
    args = parser.parse_args()
    prepare_workloads(args.model, args.revision, args.cache, args.output, args.requests,
                      args.warmup_requests, args.warmup_input_tokens, args.warmup_output_tokens)
