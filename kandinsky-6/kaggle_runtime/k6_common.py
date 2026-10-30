import json
import logging
import os
import resource
import sys
import time
from pathlib import Path


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def bootstrap(cfg, stage, cpu=False):
    # Only this short-lived process imports ComfyUI and initializes CUDA.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.path.insert(0, cfg["comfy_root"])
    sys.path.insert(0, str(Path(cfg["comfy_root"]) / "custom_nodes/kandinsky6"))
    sys.argv = [__file__, "--models-directory", cfg["model_root"],
                "--disable-dynamic-vram", "--disable-xformers",
                "--use-pytorch-cross-attention", "--reserve-vram", "2",
                "--fp16-unet", "--fp32-vae",
                "--disable-metadata"]
    if cpu:
        sys.argv.append("--cpu")
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import folder_paths
    import nodes
    import comfy.model_management as mm
    from kandinsky6.register import register
    register()
    torch.set_num_threads(cfg["cpu_threads"])
    folder_paths.set_output_directory(cfg["output_root"])
    device = mm.get_torch_device()
    print("STAGE_DEVICE", json.dumps({"stage": stage, "comfy_device": str(device),
          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
          "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
          "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
          "text_device_requested": cfg.get("text_device")}), flush=True)
    if not cpu:
        if device.type != "cuda":
            raise RuntimeError("GPU stage needs an available CUDA GPU")
        torch.cuda.reset_peak_memory_stats(device)
    return torch, nodes, mm


def finite(tensor, name):
    import torch
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{name}: NaN/Inf; stop and inspect dtype/logs")


def timed(report, name, fn):
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    report[name + "_seconds"] = time.perf_counter() - start
    return result


def save_conditioning(path, conditioning, cache_key):
    import torch
    from safetensors.torch import save_file
    if len(conditioning) != 1:
        raise ValueError("This experiment uses one unscheduled conditioning row")
    context, extra = conditioning[0]
    tensors = {"context": context.detach().cpu().contiguous().clone()}
    scalars = {}
    for key, value in extra.items():
        if torch.is_tensor(value):
            finite(value, key)
            tensors["extra." + key] = value.detach().cpu().contiguous().clone()
        elif value is None or isinstance(value, (str, bool, int, float)):
            scalars[key] = value
        else:
            raise TypeError(f"Unsupported conditioning metadata: {key}")
    finite(context, "text context")
    save_file(tensors, str(path), metadata={"cache_key": cache_key,
              "scalars": json.dumps(scalars), "schema": "k6-conditioning-v1"})


def load_conditioning(path, cache_key):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata()
        if meta.get("cache_key") != cache_key or meta.get("schema") != "k6-conditioning-v1":
            raise ValueError("Conditioning is stale: rerun text encoding for this prompt/version")
        context = f.get_tensor("context")
        extra = json.loads(meta["scalars"])
        extra.update({k[6:]: f.get_tensor(k) for k in f.keys() if k.startswith("extra.")})
    return [[context, extra]]


def finish_report(cfg, stage, report, start):
    import torch
    import comfy.model_management as mm
    report["comfy_device"] = str(mm.get_torch_device())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        report.update(gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      gpu_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    report.update(stage=stage, elapsed_seconds=time.perf_counter() - start,
                  worker_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                  torch=torch.__version__, torch_cuda=torch.version.cuda,
                  pretrained=True)
    save_json(Path(cfg["run_root"]) / (stage + "-metrics.json"), report)
    print("STAGE_OK", json.dumps(report), flush=True)
