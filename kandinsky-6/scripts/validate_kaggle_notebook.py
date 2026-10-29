"""Validate the Kaggle deployment notebook without model downloads; optional native CPU API checks."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote

import nbformat

CHECK = r'''
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

worker_path, download_path, comfy_root, check_root = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("k6_worker_check", worker_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cfg = dict(comfy_root=str(comfy_root), model_root=str(comfy_root / "models"),
           output_root=str(check_root / "videos"), run_root=str(check_root / "run"),
           cpu_threads=2, width=64, height=64, frames=9, fps=24.0, run_name="synthetic-api-check")
Path(cfg["output_root"]).mkdir(parents=True)
Path(cfg["run_root"]).mkdir(parents=True)
torch, nodes, mm = module.bootstrap(cfg, "check", cpu=True)
from kandinsky6.nodes import Kandinsky6EmptyLatent
latent, fps = Kandinsky6EmptyLatent().build(432, 256, 33, 24.0, 1)
video, audio = latent["samples"].unbind()
assert tuple(video.shape) == (1, 16, 9, 32, 54), video.shape
assert tuple(audio.shape) == (1, math.ceil(33 / 24 * 44100 / 1024), 40), audio.shape
assert fps == 24.0
source = [[torch.arange(24, dtype=torch.float32).reshape(1, 3, 8),
           {"pooled_output": torch.ones(1, 4), "attention_mask": torch.tensor([[True, True, False]]),
            "k6_audio_context": torch.arange(24).reshape(1, 3, 8).float(), "flag": None, "scale": 1.0}]]
path = check_root / "condition.safetensors"
module.save_conditioning(path, source, "prompt-A")
result = module.load_conditioning(path, "prompt-A")
assert torch.equal(result[0][0], source[0][0])
assert torch.equal(result[0][1]["attention_mask"], source[0][1]["attention_mask"])
assert result[0][1]["flag"] is None and result[0][1]["scale"] == 1.0
try:
    module.load_conditioning(path, "prompt-B")
except ValueError:
    pass
else:
    raise AssertionError("Stale conditioning accepted")
try:
    module.finite(torch.tensor([float("nan")]), "negative control")
except FloatingPointError:
    pass
else:
    raise AssertionError("NaN accepted")

import numpy as np
run = Path(cfg["run_root"])
frames = np.linspace(0, 1, 9 * 64 * 64 * 3, dtype=np.float32).reshape(9, 64, 64, 3)
samples = round(9 / 24 * 44100)
waveform = (0.01 * np.sin(2 * np.pi * 440 * np.arange(samples) / 44100)).astype(np.float32)[None, None]
np.save(run / "frames.npy", frames, allow_pickle=False)
np.save(run / "waveform.npy", waveform, allow_pickle=False)
metrics = {}
module.export_stage(cfg, metrics)
assert metrics["decoded_video_frames"] == 9 and metrics["decoded_audio_samples"] > 0
assert Path(metrics["mp4"]).is_file()

# Exercise reuse/cleanup with tiny fixtures: the mounted Input must survive cleanup.
download_spec = importlib.util.spec_from_file_location("k6_download_check", download_path)
download = importlib.util.module_from_spec(download_spec)
download_spec.loader.exec_module(download)
input_root = check_root / "input"
model_root = check_root / "models"
input_root.mkdir()
original = input_root / "qwen.safetensors"
original.write_bytes(b"fixture for path/ownership checks")
import hashlib
digest = hashlib.sha256(original.read_bytes()).hexdigest()
item = dict(path="text_encoders/qwen.safetensors", group="text", size=original.stat().st_size,
            sha256=digest, repo="not-used/not-used", revision="not-used", filename="not-used")
settings = check_root / "download-config.json"
settings.write_text(json.dumps(dict(model_root=str(model_root), input_model_dirs=[str(input_root)], models=[item])))
sys.argv = [str(download_path), str(settings), "text"]
download.main()
linked = model_root / item["path"]
assert linked.is_symlink() and linked.resolve() == original.resolve()
sys.argv[-1] = "purge-text"
download.main()
assert linked.is_file() and original.is_file(), "Input reference was removed"
owned = model_root / "text_encoders/owned.safetensors"
owned.write_bytes(original.read_bytes())
owned_item = dict(item, path="text_encoders/owned.safetensors")
settings.write_text(json.dumps(dict(model_root=str(model_root), input_model_dirs=[], models=[owned_item])))
(model_root / "download-receipts.json").write_text(json.dumps({owned_item["path"]: digest}))
download.main()
assert not owned.exists() and original.is_file()
print("NATIVE_CPU_CHECKS_OK", json.dumps({"video_latent_shape": list(video.shape),
      "audio_latent_shape": list(audio.shape), "conditioning_roundtrip": True,
      "stale_cache_rejected": True, "nan_rejected": True, "synthetic_mp4": metrics,
      "dataset_not_modified": True, "owned_temporary_cleanup": True,
      "pretrained_models_loaded": False, "torch_version": torch.__version__,
      "comfy_revision": __import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=comfy_root,text=True).strip()}))
'''


def validate(root: Path, comfy_root: Path | None, python: str | None):
    path = root / "kandinsky-6/notebooks/15_kaggle_t4_lite_distill.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    sources = {}
    models = None
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            tree = ast.parse(cell.source, filename=f"{path.name}:cell-{index}")
            assert not cell.outputs and cell.execution_count is None, "Authored notebook must remain clean"
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    key = node.targets[0].id
                    if key.endswith("_CODE"):
                        sources[key] = ast.literal_eval(node.value)
                        ast.parse(sources[key], filename=key)
                    if key == "MODELS":
                        models = ast.literal_eval(node.value)
        else:
            for link in re.findall(r"\]\(([^\s)]+)\)", cell.source):
                if link.startswith(("http:", "https:", "#", "mailto:")):
                    continue
                target = path.parent / unquote(link.split("#")[0])
                assert target.exists(), f"Broken local link: {link}"
    order = ["COMMON_CODE", "TEXT_CODE", "SAMPLE_CODE", "VIDEO_CODE", "AUDIO_CODE", "EXPORT_CODE", "ENTRY_CODE"]
    worker = "\n\n".join(sources[name] for name in order)
    ast.parse(worker)
    assert len(models) == 7
    assert sum(m["size"] for m in models) == 25416128714
    assert set(m["group"] for m in models) == {"text", "media"}
    assert len({m["path"] for m in models}) == 7
    for item in models:
        assert re.fullmatch(r"[0-9a-f]{40}", item["revision"])
        assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        assert not Path(item["path"]).is_absolute() and ".." not in Path(item["path"]).parts
    report = dict(notebook=str(path.relative_to(root)), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  cells=len(notebook.cells), code_cells=sum(c.cell_type == "code" for c in notebook.cells),
                  worker_sha256=hashlib.sha256(worker.encode()).hexdigest(), all_cells_and_embedded_scripts_parsed=True,
                  local_links_checked=True, pinned_model_files=7, pinned_model_bytes=sum(m["size"] for m in models),
                  kaggle_execution=False, cuda_t4_inference=False, pretrained_downloads_during_validation=False,
                  pretrained_generation_during_validation=False, native_cpu_checks=False)
    if comfy_root:
        with tempfile.TemporaryDirectory(prefix="k6-kaggle-check-") as temp:
            temp = Path(temp)
            (temp / "worker.py").write_text(worker)
            (temp / "download.py").write_text(sources["DOWNLOAD_CODE"])
            (temp / "check.py").write_text(CHECK)
            executable = python or str(comfy_root / ".venv/bin/python")
            result = subprocess.run([executable, str(temp / "check.py"), str(temp / "worker.py"),
                                     str(temp / "download.py"), str(comfy_root), str(temp)],
                                    cwd=comfy_root, text=True, capture_output=True, timeout=120)
            if result.returncode:
                print(result.stdout)
                print(result.stderr)
                raise RuntimeError("Native CPU API checks failed")
            line = next(line for line in result.stdout.splitlines() if line.startswith("NATIVE_CPU_CHECKS_OK "))
            checks = json.loads(line.removeprefix("NATIVE_CPU_CHECKS_OK "))
            checks["synthetic_mp4"].pop("mp4", None)
            report["native_cpu_checks"] = checks
    destination = root / "kandinsky-6/reports/kaggle-t4-validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--comfy-root", type=Path)
    parser.add_argument("--python")
    options = parser.parse_args()
    validate(options.root, options.comfy_root, options.python)
