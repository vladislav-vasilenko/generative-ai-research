"""Build the standalone, staged Kaggle 2xT4 Kandinsky 6 Lite deployment notebook."""
from __future__ import annotations

import argparse
import ast
import json
import pprint
from pathlib import Path
from textwrap import dedent

COMFY_REV = "086099f5d8886163ef77dc354f8382cb7431582a"
K6_REV = "7ceeefad7fb625011f7db602c43893a989f19c15"
MODEL_MANIFEST = [
    dict(role="dit", group="media", path="diffusion_models/Kandinsky-6.0-Lite-distill-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors", repo="kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers", revision="4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81", filename="transformer/diffusion_pytorch_model.safetensors", size=6381599096, sha256="b1b52668561d9138d93bb4da13c9999e615bf569be9f8cf65732d54652447f53"),
    dict(role="qwen", group="text", path="text_encoders/qwen_2.5_vl_7b.safetensors", repo="Comfy-Org/Qwen-Image_ComfyUI", revision="1f12b17be14c89b026c51a91d67c32f84bb047bc", filename="split_files/text_encoders/qwen_2.5_vl_7b.safetensors", size=16584415576, sha256="cfafd739459bc86257397259f612a9aee88e5b98e85b5c0d0d1717e898b3463a"),
    dict(role="clip", group="text", path="text_encoders/clip_l.safetensors", repo="Comfy-Org/HunyuanVideo_repackaged", revision="533a43813f01517d9e624f9e824a85bb94094120", filename="split_files/text_encoders/clip_l.safetensors", size=246144152, sha256="660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd"),
    dict(role="video_vae", group="media", path="vae/hunyuan_video_vae_bf16.safetensors", repo="Comfy-Org/HunyuanVideo_repackaged", revision="533a43813f01517d9e624f9e824a85bb94094120", filename="split_files/vae/hunyuan_video_vae_bf16.safetensors", size=492984198, sha256="e8f8553275406d84ccf22e7a47601650d8f98bdb8aa9ccfdd6506b57a9701aed"),
    dict(role="audio_vae", group="media", path="audio_vae/v1-44.pth", repo="hkchengrex/MMAudio", revision="eb13a1a98fdbec91753775c57b074ccdfc60587c", filename="ext_weights/v1-44.pth", size=1221942998, sha256="ab6cc15dc31947675f75c950c41f4dcfd0d6d1817555ac871f809ec388e4651a"),
    dict(role="vocoder", group="media", path="audio_vae/bigvgan_vocoder/bigvgan_generator.pt", repo="nvidia/bigvgan_v2_44khz_128band_512x", revision="95a9d1dcb12906c03edd938d77b9333d6ded7dfb", filename="bigvgan_generator.pt", size=489041291, sha256="d9fe7ec6bd0b44ed9d66973d5012d8181c1570b01e5c72df51973e241dccd357"),
    dict(role="vocoder_config", group="media", path="audio_vae/bigvgan_vocoder/config.json", repo="nvidia/bigvgan_v2_44khz_128band_512x", revision="95a9d1dcb12906c03edd938d77b9333d6ded7dfb", filename="config.json", size=1403, sha256="65b7f487bfaf15256056a75f6667ef5f640214c981922a111927873705ea4ddb"),
]

COMMON = r'''
import argparse
import gc
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def bootstrap(cfg, stage, cpu=False):
    # Only this short-lived process imports ComfyUI and initializes CUDA.
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
'''

TEXT = r'''
def text_stage(cfg, report):
    torch, nodes, mm = bootstrap(cfg, "text", cpu=cfg["text_device"] == "cpu")
    from kandinsky6.nodes import Kandinsky6TextEncode
    clip, = timed(report, "model_load", lambda: nodes.DualCLIPLoader().load_clip(
        "qwen_2.5_vl_7b.safetensors", "clip_l.safetensors", "kandinsky5",
        device=cfg["text_device"]))
    report["text_load_device"] = str(clip.patcher.load_device)
    report["text_storage_dtypes"] = sorted(str(d) for d in clip.cond_stage_model.dtypes)
    with torch.inference_mode():
        positive, = timed(report, "positive_encode", lambda: Kandinsky6TextEncode().encode(
            clip, cfg["video_caption"], cfg["audio_caption"]))
        save_conditioning(Path(cfg["run_root"]) / "positive.safetensors", positive, cfg["text_cache_key"])
        report["context_shape"] = list(positive[0][0].shape)
        report["pooled_shape"] = list(positive[0][1]["pooled_output"].shape)
        report["context_dtype"] = str(positive[0][0].dtype)
        report["pooled_dtype"] = str(positive[0][1]["pooled_output"].dtype)
        negative, = timed(report, "negative_encode", lambda: Kandinsky6TextEncode().encode(
            clip, cfg["negative_caption"], ""))
        save_conditioning(Path(cfg["run_root"]) / "negative.safetensors", negative, cfg["text_cache_key"])
    # The worker exits next; references in Comfy's caches also disappear.
'''

SAMPLE = r'''
def sample_stage(cfg, report):
    torch, nodes, mm = bootstrap(cfg, "sample")
    from safetensors.torch import save_file
    from kandinsky6.nodes import Kandinsky6EmptyLatent, Kandinsky6Sampler
    run = Path(cfg["run_root"])
    positive = load_conditioning(run / "positive.safetensors", cfg["text_cache_key"])
    negative = load_conditioning(run / "negative.safetensors", cfg["text_cache_key"])
    model, = timed(report, "model_load", lambda: nodes.UNETLoader().load_unet(
        cfg["transformer_name"], "default"))
    inference_dtype = model.model.get_dtype_inference()
    if inference_dtype != torch.float16:
        raise RuntimeError(f"Expected FP16 on T4, got {inference_dtype}")
    if model.model.diffusion_model.n_grid != 10:
        raise ValueError("Expected Lite distilled 10-grid checkpoint")
    calls = [0]
    def count_call(module, args):
        calls[0] += 1
    handle = model.model.diffusion_model.register_forward_pre_hook(count_call)
    with torch.inference_mode():
        latent, _ = Kandinsky6EmptyLatent().build(
            cfg["width"], cfg["height"], cfg["frames"], cfg["fps"], 1)
        try:
            sampled, = timed(report, "sampling", lambda: Kandinsky6Sampler().sample(
                model=model, seed=cfg["seed"], steps=10, cfg=1.0,
                sampler_name="euler", scheduler="simple", positive=positive,
                negative=negative, latent_image=latent, denoise=1.0))
        finally:
            handle.remove()
        video, audio = sampled["samples"].unbind()
        finite(video, "sampled video latent")
        finite(audio, "sampled audio latent")
        if calls[0] != 10:
            raise AssertionError(f"Expected 10 DiT calls, measured {calls[0]}")
        save_file({"video": video.detach().cpu().contiguous(),
                   "audio": audio.detach().cpu().contiguous()}, str(run / "joint-latents.safetensors"))
    report.update(video_latent_shape=list(video.shape), audio_latent_shape=list(audio.shape),
                  dit_nfe=calls[0], inference_dtype=str(inference_dtype), sampler="DX PiFlow",
                  guidance_scale=1.0)
'''

VIDEO = r'''
def video_stage(cfg, report):
    torch, nodes, mm = bootstrap(cfg, "video")
    import numpy as np
    from safetensors.torch import load_file
    latent = load_file(str(Path(cfg["run_root"]) / "joint-latents.safetensors"))["video"]
    vae, = timed(report, "model_load", lambda: nodes.VAELoader().load_vae("hunyuan_video_vae_bf16.safetensors"))
    with torch.inference_mode():
        if cfg["tiled_decode"]:
            frames, = timed(report, "decode", lambda: nodes.VAEDecodeTiled().decode(
                vae, {"samples": latent}, tile_size=cfg["tile_size"], overlap=cfg["tile_overlap"],
                temporal_size=cfg["temporal_size"], temporal_overlap=cfg["temporal_overlap"]))
        else:
            frames, = timed(report, "decode", lambda: nodes.VAEDecode().decode(vae, {"samples": latent}))
    finite(frames, "decoded RGB")
    expected = (cfg["frames"], cfg["height"], cfg["width"], 3)
    if tuple(frames.shape) != expected:
        raise ValueError(f"Wrong RGB shape: {tuple(frames.shape)} vs {expected}")
    np.save(Path(cfg["run_root"]) / "frames.npy", frames.cpu().float().numpy(), allow_pickle=False)
    report.update(frames_shape=list(frames.shape), tiled_decode=cfg["tiled_decode"],
                  vae_dtype=str(vae.vae_dtype), rgb_min=float(frames.min()), rgb_max=float(frames.max()))
'''

AUDIO = r'''
def audio_stage(cfg, report):
    torch, nodes, mm = bootstrap(cfg, "audio")
    import numpy as np
    from safetensors.torch import load_file
    from kandinsky6.nodes import Kandinsky6AudioVAELoader, Kandinsky6AudioVAEDecode
    latent = load_file(str(Path(cfg["run_root"]) / "joint-latents.safetensors"))["audio"]
    vae, = timed(report, "model_load", lambda: Kandinsky6AudioVAELoader().load(
        "v1-44.pth", "bigvgan_vocoder", "44k", 0.417))
    with torch.inference_mode():
        audio, = timed(report, "decode", lambda: Kandinsky6AudioVAEDecode().decode(vae, {"samples": latent}))
    wave = audio["waveform"].cpu().float()
    finite(wave, "decoded waveform")
    if wave.ndim != 3 or wave.shape[:2] != (1, 1) or audio["sample_rate"] != 44100:
        raise ValueError("Expected mono waveform [1,1,N] at 44100 Hz")
    np.save(Path(cfg["run_root"]) / "waveform.npy", wave.numpy(), allow_pickle=False)
    report.update(waveform_shape=list(wave.shape), sample_rate=audio["sample_rate"],
                  audio_seconds=wave.shape[-1] / audio["sample_rate"], max_abs=float(wave.abs().max()))
'''

EXPORT = r'''
def export_stage(cfg, report):
    torch, nodes, mm = bootstrap(cfg, "export", cpu=True)
    import av
    import numpy as np
    from comfy_extras.nodes_video import CreateVideo, SaveVideo
    run = Path(cfg["run_root"])
    frames = torch.from_numpy(np.load(run / "frames.npy", allow_pickle=False))
    waveform = torch.from_numpy(np.load(run / "waveform.npy", allow_pickle=False))
    video = CreateVideo.execute(images=frames, fps=cfg["fps"],
              audio={"waveform": waveform, "sample_rate": 44100},
              bit_depth="auto", color_space="sRGB", codec="none").result[0]
    output_root = Path(cfg["output_root"])
    prefix = cfg["run_name"]
    before = set(output_root.glob(prefix + "_*.mp4"))
    timed(report, "mux", lambda: SaveVideo.execute(video=video, filename_prefix=prefix,
          format="mp4", codec={"codec": "h264"}))
    created = set(output_root.glob(prefix + "_*.mp4")) - before
    if len(created) != 1:
        raise RuntimeError(f"Expected exactly one new MP4, got {len(created)}")
    path, = created
    decoded_frames = 0
    with av.open(str(path)) as f:
        if len(f.streams.video) != 1 or len(f.streams.audio) != 1:
            raise ValueError("MP4 must contain both video and audio")
        vs, aus = f.streams.video[0], f.streams.audio[0]
        if (vs.width, vs.height) != (cfg["width"], cfg["height"]):
            raise ValueError("MP4 geometry differs from request")
        if abs(float(vs.average_rate) - cfg["fps"]) > 0.01 or aus.sample_rate != 44100:
            raise ValueError("MP4 FPS/sample rate differ from request")
        report.update(video_codec=vs.codec_context.name, audio_codec=aus.codec_context.name,
                      audio_sample_rate=aus.sample_rate, fps=float(vs.average_rate))
        for frame in f.decode(video=0):
            decoded_frames += 1
    if decoded_frames != cfg["frames"]:
        raise ValueError(f"Decoded frame count: {decoded_frames}, expected {cfg['frames']}")
    # Verify audio packets decode too, rather than trusting stream metadata alone.
    with av.open(str(path)) as f:
        audio_samples = sum(frame.samples for frame in f.decode(audio=0))
    expected_seconds = cfg["frames"] / cfg["fps"]
    if audio_samples == 0 or abs(audio_samples / 44100 - expected_seconds) > 0.15:
        raise ValueError("Decoded audio duration differs from video by more than 0.15 s")
    report.update(mp4=str(path), output_bytes=path.stat().st_size,
                  decoded_video_frames=decoded_frames, decoded_audio_samples=audio_samples)
    save_json(run / "result.json", report)
'''

ENTRY = r'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["text", "sample", "video", "audio", "export"], required=True)
    arguments = parser.parse_args()
    configuration = json.loads(Path(arguments.config).read_text())
    started = time.perf_counter()
    measurements = {}
    globals()[arguments.stage + "_stage"](configuration, measurements)
    finish_report(configuration, arguments.stage, measurements, started)
'''


def build(root: Path):
    cells = []
    def clean(source):
        if 'DOWNLOAD_CODE = r"""' in source:
            prefix, rest = source.split('r"""', 1)
            body, suffix = rest.split('"""', 1)
            return prefix.strip() + ' r"""' + body + '"""\n' + dedent(suffix).lstrip("\n")
        if source.startswith("\n"):
            return dedent(source).strip() + "\n"
        first, separator, rest = source.partition("\n")
        return first.strip() + ("\n" + dedent(rest) if separator else "")
    def md(source):
        cells.append(dict(cell_type="markdown", metadata={}, source=clean(source).strip() + "\n"))
    def code(source):
        source = clean(source).strip() + "\n"
        # Keep worker source legible as real multiline Python in the notebook.
        parsed = ast.parse(source)
        lines = source.splitlines(keepends=True)
        offsets = [sum(map(len, lines[:i])) for i in range(len(lines))]
        replacements = []
        for node in parsed.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODELS" for t in node.targets):
                value = node.value
                start = offsets[value.lineno - 1] + value.col_offset
                end = offsets[value.end_lineno - 1] + value.end_col_offset
                replacements.append((start, end, pprint.pformat(ast.literal_eval(value), width=110, sort_dicts=False)))
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                if any(isinstance(t, ast.Name) and t.id.endswith("_CODE") for t in node.targets):
                    value = node.value
                    start = offsets[value.lineno - 1] + value.col_offset
                    end = offsets[value.end_lineno - 1] + value.end_col_offset
                    replacements.append((start, end, 'r"""\n' + value.value.strip() + '\n"""'))
        for start, end, value in sorted(replacements, reverse=True):
            source = source[:start] + value + source[end:]
        ast.parse(source)
        cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=source))

    md(r'''# Kandinsky 6 Lite Distill в Kaggle: 2 × T4 16 GB + 32 GB RAM

    <!-- study-navigation:start -->
    **Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../../curriculum/README.md) · [Полное содержание](../../curriculum/CONTENTS.md).

    **Связанные лекции:** [Lecture 1](https://diffusion.csail.mit.edu/2026/docs/20260120_Lecture_01.pdf) · [Lecture 2](https://diffusion.csail.mit.edu/2026/docs/20260122_Lecture_02.pdf) · [Lecture 3](https://diffusion.csail.mit.edu/2026/docs/20260123_Lecture_03.pdf) · [Lecture 4](https://diffusion.csail.mit.edu/2026/docs/20260128_Lecture_04_edited.pdf).
    <!-- study-navigation:end -->

    Самостоятельный notebook: импортируйте **этот один `.ipynb`** в Kaggle. Код внутри создаёт все необходимые вспомогательные файлы. Запуск не требует копии учебного репозитория, ComfyUI Desktop или туннеля к Mac.

    **Цель:** повторить операции нашего ComfyUI Lite workflow программно: Qwen + CLIP → joint Audio/Video Lite DiT → DX π-Flow → HunyuanVideo VAE → TOD-VAE + BigVGAN → CreateVideo / SaveVideo → MP4 с аудио.

    Две T4 — это **два отдельных адресных пространства по ~15 GiB**, не одна GPU на 32 GB. CPU RAM также отдельна. В этом notebook нет tensor parallelism, DDP или обучения: стадии запускаются последовательно, завершают процесс и передают небольшие результаты через файлы.

    **Граница проверки:** код подготовлен по закреплённым ComfyUI/K6 исходникам. Сквозной pretrained запуск именно на Kaggle T4 ещё должен дать ваши реальные замеры; успешная локальная проверка кода не доказывает качество или вместимость 5s режима на T4.''')
    md('''## 1. Подготовка Kaggle и порядок запуска

    1. Создайте Notebook → **File → Import Notebook** → загрузите этот файл.
    2. В Settings выберите **GPU T4 ×2**, включите **Internet**. Доступность зависит от аккаунта; фактические устройства проверит код.
    3. Выполняйте ячейки сверху вниз. Сначала получится smoke **432×256, 33 кадра, 24 FPS**; затем можно включить baseline **864×480, 121 кадр** (~5.04 s).
    4. `/kaggle/temp` используется для среды, кода и весов; итоговые видео и отчёты сохраняются в `/kaggle/working/k6-results`.

    Выходы `/kaggle/working` можно сохранить через Save Version; временные файлы не являются постоянным хранилищем. При возобновлении сессии проверьте наличие файлов. Не сохраняйте десятки GB весов как outputs notebook.

    Если веса уже подключены как Kaggle Dataset, задайте `INPUT_MODEL_DIRS` ниже: будут созданы ссылки на read-only файлы без второй копии. После успешного text stage по умолчанию удаляются **только скачанные этим notebook временные text-веса**; исходный Dataset не изменяется.''')
    code('''import os, sys, json, shutil, subprocess, hashlib, time, re
    from pathlib import Path
    import torch
    import psutil

    if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Запустите notebook в Kaggle с включённым GPU; на Mac эта ячейка останавливается до установки/загрузок.")
    GPU_COUNT = torch.cuda.device_count()
    GPUS = [{"index": i, "name": torch.cuda.get_device_name(i),
             "vram_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
             "capability": list(torch.cuda.get_device_capability(i))} for i in range(GPU_COUNT)]
    print(json.dumps(GPUS, indent=2))
    def host_memory():
        vm = psutil.virtual_memory()
        total, available = vm.total, vm.available
        for limit, usage in [("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                             ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")]:
            if Path(limit).is_file() and Path(usage).is_file():
                value = Path(limit).read_text().strip()
                if value.isdigit() and int(value) < total:
                    total = int(value)
                    available = min(available, max(0, total - int(Path(usage).read_text())))
        return {"total_gib": total / 2**30, "available_gib": available / 2**30}

    print("CPU RAM / cgroup:", host_memory())
    print("Kernel torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
    if min(g["vram_gib"] for g in GPUS) < 13:
        raise RuntimeError("Этот план рассчитан на GPU класса T4 16GB или больше.")
    print("Worker text GPU:", min(1, GPU_COUNT - 1), "sampling/codec GPU: 0")
    # No model or CUDA tensor is allocated in the notebook's parent process.
    ''')
    md('''## 2. Почему память освобождается между стадиями

    | Стадия | Устройство | Тяжёлые веса одновременно | Результат |
    |---|---|---|---|
    | Text | GPU 1 + CPU offload | Qwen + CLIP | conditioning `.safetensors` |
    | Sampling | GPU 0 + CPU offload при необходимости | Lite DiT | два совместно сэмплированных латента |
    | Video decode | GPU 0 | video VAE, FP32 + tiles | RGB frames `.npy` |
    | Audio decode | GPU 0 | TOD-VAE + BigVGAN, FP32 | mono waveform `.npy` |
    | Export | CPU | кодек MP4 | H.264 + AAC |

    Все стадии **последовательны**. Вторая T4 используется text worker; это распределение стадий, а не ускорение одного DiT между GPU. При одной T4 text worker также может использовать GPU 0.

    Память весов первого приближения: число параметров × байт на параметр. Но пик включает activations, временные cast buffers, CPU copies и CUDA allocator. Поэтому номинальный размер файла не доказывает, что модель поместится; ниже записываются реальные peak counters и внешние замеры процесса.

    Для T4 задаём **FP16 для DiT**, а VAE/vocoder — FP32. Текстовый путь остаётся штатным ComfyUI: Qwen factory сохраняет checkpoint dtype BF16, а `SDClipModel` передаёт FP32 embeddings в forward и manual-cast operations приводят веса к dtype входа. Поэтому нельзя обещать FP16 text compute одним флагом `--fp16-text-enc`: factory может переопределить storage dtype. Записываем фактические storage/output dtypes; BF16 Tensor Cores для этого FP32 text path не требуются. FP16 DiT отличается от рекомендованного BF16 режима на новых GPU и требует проверки finite/качества. Не устанавливаем FlashAttention-2 или SageAttention: используется PyTorch SDPA с доступным backend.''')
    code('''SCRATCH = Path("/kaggle/temp/k6-lite")
    RESULTS = Path("/kaggle/working/k6-results")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    COMFY = SCRATCH / "ComfyUI"
    K6_SOURCE = SCRATCH / "kandinsky-6"
    MODEL_ROOT = SCRATCH / "models"
    VENV = SCRATCH / "venv"
    INPUT_MODEL_DIRS = []  # Example: ["/kaggle/input/my-k6-models"]
    KEEP_TEXT_WEIGHTS = False  # Set True only if scratch has room for all ~25.4 GB.
    TEXT_DEVICE = "default"  # Automatic GPU + CPU offload. "cpu" is a slower fallback.
    TEXT_GPU = min(1, GPU_COUNT - 1)
    MEDIA_GPU = 0
    CPU_THREADS = min(4, os.cpu_count() or 1)
    print("Scratch free GiB:", round(shutil.disk_usage(SCRATCH).free / 2**30, 2))
    ''')
    md('''## 3. Код и зависимости: отдельная среда, существующий CUDA PyTorch

    Закреплены точные revisions исходников, соответствующие исследованному расширению. Среда worker создаётся с `--without-pip --system-site-packages`: она видит уже установленный Kaggle PyTorch, а дополнительные зависимости устанавливаются в свою папку. Kernel не импортирует ComfyUI и не держит веса.

    Установка не вызывает `ensurepip`: доступный pip ≥22.3 управляет целевой средой через `--python`. Частично созданная папка после ошибки восстанавливается повторным выполнением этой ячейки без удаления скачанных исходников/весов. Установщик не запускается в глобальную среду. [Механизм pip](https://pip.pypa.io/en/stable/topics/python-option/).

    Ограничения для `torch`, `torchvision`, `torchaudio` предотвращают их незаметную замену установщиком. Если resolver сообщает несовместимость, не удаляйте ограничения наугад: сохраните лог и используйте совместимый Kaggle image/runtime. Фактический набор библиотек будет записан в `worker-environment.txt`; это не заявление, что все будущие Kaggle images эквивалентны.''')
    code(f'''COMFY_REV = {COMFY_REV!r}
    K6_REV = {K6_REV!r}

    def checkout(repo, revision, destination):
        destination = Path(destination)
        if not (destination / ".git").exists():
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", str(destination)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", repo], check=True)
            subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision], check=True)
            subprocess.run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], check=True)
        actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"Existing checkout differs: {{destination}}; {{actual}}")

    checkout("https://github.com/Comfy-Org/ComfyUI.git", COMFY_REV, COMFY)
    checkout("https://github.com/kandinskylab/kandinsky-6.git", K6_REV, K6_SOURCE)
    extension = COMFY / "custom_nodes/kandinsky6"
    if not extension.exists():
        shutil.copytree(K6_SOURCE / "comfyui", extension)
    # No ComfyUI server, Manager, remote tunnel, or full-bundle downloader is started.
    def prepare_worker_environment(base_python, venv_path):
        venv_path = Path(venv_path)
        # Run on every retry: an ensurepip failure can leave bin/python behind.
        subprocess.run([str(base_python), "-m", "venv", "--without-pip",
                        "--system-site-packages", str(venv_path)], check=True)
        worker_python = str(venv_path / "bin/python")
        candidates = [[str(base_python), "-m", "pip"]]
        for name in ("pip", "pip3"):
            executable = shutil.which(name)
            if executable:
                candidates.append([executable])
        for candidate in candidates:
            probe = subprocess.run(candidate + ["--help"], text=True, capture_output=True)
            if probe.returncode == 0 and "--python" in probe.stdout:
                pip_command = candidate + ["--python", worker_python]
                subprocess.run(pip_command + ["--version"], check=True)
                return worker_python, pip_command
        raise RuntimeError("Нужен доступный pip >=22.3 с опцией --python. Сохраните лог и проверьте Kaggle runtime; ensurepip здесь не требуется.")

    PYTHON, PIP = prepare_worker_environment(sys.executable, VENV)
    subprocess.run([PYTHON, "-c", "import sys, torch; print('Worker:', sys.executable); print('Inherited torch:', torch.__version__, 'CUDA:', torch.version.cuda)"], check=True)
    from importlib.metadata import version, PackageNotFoundError
    constraints = []
    for name in ["torch", "torchvision", "torchaudio"]:
        try:
            constraints.append(name + "==" + version(name))
        except PackageNotFoundError:
            pass
    (SCRATCH / "torch-constraints.txt").write_text("\\n".join(constraints) + "\\n")
    subprocess.run(PIP + ["install", "--disable-pip-version-check",
                    "--upgrade-strategy", "only-if-needed", "-c", str(SCRATCH / "torch-constraints.txt"),
                    "-r", str(COMFY / "requirements.txt"), "-r", str(extension / "requirements.txt"),
                    "imageio-ffmpeg>=0.6.0,<0.7"], check=True)
    freeze = subprocess.check_output(PIP + ["freeze"], text=True)
    (RESULTS / "worker-environment.txt").write_text(freeze)
    subprocess.run([PYTHON, "-c", "import torch, av, safetensors; print(torch.__version__, torch.version.cuda); print('Dependencies import OK')"], check=True)
    dependency_probe = "import sys; from pathlib import Path; root=Path(sys.argv[1]); sys.path[:0]=[str(root),str(root/'custom_nodes/kandinsky6')]; sys.argv=['probe','--cpu','--disable-dynamic-vram']; import comfy.options; comfy.options.enable_args_parsing(); import nodes; from kandinsky6.nodes import Kandinsky6Sampler, Kandinsky6AudioVAELoader; from comfy_extras.nodes_video import CreateVideo, SaveVideo; print('ComfyUI/K6 API imports OK; no weights loaded')"
    subprocess.run([PYTHON, "-c", dependency_probe, str(COMFY)], cwd=str(COMFY), check=True)
    ''')
    md('''## 4. Все модели: точные файлы, revisions, размеры и контрольные суммы

    Используются те же repackaged веса, что в нашем ComfyUI graph. Их нельзя подменить целой папкой Diffusers без изменения загрузчиков. Lite transformer объединяет аудио- и видеоветки; отдельный audio DiT здесь не скачивается.

    Семь файлов вместе занимают примерно **25.4 GB / 23.7 GiB**. При staged загрузке text и media группы не хранятся одновременно, если веса скачаны этим notebook. Загружать целиком Pro, SR, Qwen3.5 или HF repositories не требуется.

    SHA256 проверяет полные байты файла, а не только его имя. JSON BigVGAN обязателен: без него недостаточно скачать один `bigvgan_generator.pt`.''')
    code("MODELS = " + repr(MODEL_MANIFEST) + '''
    for item in MODELS:
        print(f"{item['group']:5s} {item['size']/1e9:6.3f} GB  {item['path']}")
    print("Total GB:", round(sum(m["size"] for m in MODELS) / 1e9, 3))
    (RESULTS / "model-manifest.json").write_text(json.dumps(MODELS, indent=2) + "\\n")
    ''')
    md('''### Загрузка только нужной группы и повторное использование Kaggle Input

    Download helper работает в отдельной среде, где установлен `huggingface_hub`. Официальные публичные файлы запрашиваются с `token=False`; секреты не нужны. `.staging` находится на том же диске: перенос в финальный путь не создаёт вторую полную копию. Отключён Xet chunk cache; HTTP-загрузки Hub могут возобновлять partials.

    В начале каждой группы проверяется свободное место для ещё отсутствующих байтов плюс запас 1 GiB. Наличие mounted Input файла проверяется по размеру и SHA256 перед созданием ссылки.''')
    code(r'''
    DOWNLOAD_CODE = r"""
import argparse, hashlib, json, os, shutil
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("group", choices=["text", "media", "purge-text"])
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    root = Path(c["model_root"])
    root.mkdir(parents=True, exist_ok=True)
    receipts_path = root / "download-receipts.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    if a.group == "purge-text":
        for item in c["models"]:
            dest = root / item["path"]
            if item["group"] == "text" and item["path"] in receipts and not dest.is_symlink() and dest.is_file():
                if not dest.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Refusing cleanup outside notebook model root")
                if dest.stat().st_size != item["size"] or receipts[item["path"]] != item["sha256"]:
                    raise ValueError("Refusing cleanup of an unrecognized file")
                dest.unlink()
                receipts.pop(item["path"])
                print("Released scratch text weights:", dest.name, flush=True)
        receipts_path.write_text(json.dumps(receipts, indent=2))
        return
    items = [i for i in c["models"] if i["group"] == a.group]
    for item in items:
        dest = root / item["path"]
        if dest.exists():
            continue
        candidates = []
        for source in map(Path, c["input_model_dirs"]):
            if not source.is_dir():
                raise FileNotFoundError(f"Kaggle Input folder missing: {source}")
            candidates.extend(p for p in source.rglob(Path(item["path"]).name)
                              if p.is_file() and p.stat().st_size == item["size"])
        for candidate in candidates:
            if sha256(candidate) == item["sha256"]:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(candidate.resolve())
                print("Reused Input:", item["path"], flush=True)
                break
    missing = [i for i in items if not (root / i["path"]).exists()]
    needed = 0
    for item in missing:
        staging = root / ".staging" / item["repo"].replace("/", "--")
        complete = staging / item["filename"]
        downloads = staging / ".cache/huggingface/download" / Path(item["filename"]).parent
        partials = list(downloads.glob(Path(item["filename"]).name + ".*.incomplete"))
        existing_bytes = max([p.stat().st_size for p in partials] +
                             ([complete.stat().st_size] if complete.is_file() else [0]))
        needed += max(0, item["size"] - existing_bytes)
    if shutil.disk_usage(root).free < needed + 2**30:
        raise OSError(f"Need {needed / 2**30:.2f} GiB + 1 GiB free. Attach weights as Kaggle Input or change scratch storage.")
    from huggingface_hub import hf_hub_download
    for item in items:
        dest = root / item["path"]
        if not dest.exists():
            downloaded = Path(hf_hub_download(repo_id=item["repo"], filename=item["filename"],
                revision=item["revision"], token=False,
                local_dir=root / ".staging" / item["repo"].replace("/", "--")))
            if downloaded.stat().st_size != item["size"] or sha256(downloaded) != item["sha256"]:
                raise ValueError(f"Downloaded artifact mismatch: {item['path']}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            downloaded.replace(dest)
            receipts[item["path"]] = item["sha256"]
            receipts_path.write_text(json.dumps(receipts, indent=2))
        if dest.stat().st_size != item["size"] or sha256(dest) != item["sha256"]:
            raise ValueError(f"Existing artifact mismatch: {item['path']}")
        print("VERIFIED", item["path"], flush=True)

if __name__ == "__main__":
    main()
"""
    DOWNLOAD_SCRIPT = SCRATCH / "download_models.py"
    DOWNLOAD_SCRIPT.write_text(DOWNLOAD_CODE)
    download_config = SCRATCH / "download-config.json"
    download_config.write_text(json.dumps({"model_root": str(MODEL_ROOT), "models": MODELS,
                                         "input_model_dirs": INPUT_MODEL_DIRS}, indent=2))
    worker_env = os.environ.copy()
    worker_env.update(HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1",
                      HF_HOME=str(SCRATCH / "hf-cache"), TOKENIZERS_PARALLELISM="false")

    def ensure_models(group):
        subprocess.run([PYTHON, str(DOWNLOAD_SCRIPT), str(download_config), group],
                       env=worker_env, check=True)

    ensure_models("text")
    ''')
    md('''## 5. Настройки эксперимента и контракты тензоров

    Smoke — проверка всей цепочки на меньшей задаче; её время/качество нельзя выдавать за полноценный 5s benchmark. Для обоих профилей оставляем **10 вызовов DiT, CFG=1, denoise=1, audio scaling=0.417, seed=42**. Видимые `euler/simple` передаются для совместимости узла; distilled checkpoint выбирает DX π-Flow внутри `Kandinsky6Sampler`.

    Видео: `[B,16,Tz,H/8,W/8]`, где `Tz=(T−1)/4+1`. Аудио: `[B,A,40]`, где `A=ceil((T/FPS)·44100/1024)` для закреплённого K6 contract. В checkpoint один joint transformer. Положительная строка собирается как `video_caption <AUDCAP>audio_caption<ENDAUDCAP>`; обе модальности получают её conditioning. Conditioning cache зависит от prompt, encoder revisions и dtype, но не от разрешения или seed.

    Связь с MIT: [Lecture 2 / FM](../../lecture_02/README.md), [conditioning / Lecture 3](../../lecture_03/README.md); детали компонент — [K6 text](02_text_conditioning.ipynb), [dual-stream CrossDiT](08_dual_stream_audio_video_crossdit.ipynb), [π-Flow](10_piflow_distillation.ipynb), [VAE tiling](../../vae/README.md).''')
    code('''VIDEO_CAPTION = "A person speaks clearly towards a static camera in a quiet room. <S>Today we test the voice model.<E> Natural lighting, medium shot."
    AUDIO_CAPTION = "Clean English speech, one speaker, no music, no background noise."
    NEGATIVE_CAPTION = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"
    PROFILES = {"smoke": {"width": 432, "height": 256, "frames": 33},
                "baseline": {"width": 864, "height": 480, "frames": 121}}
    text_contract = {"video_caption": VIDEO_CAPTION, "audio_caption": AUDIO_CAPTION,
                     "negative_caption": NEGATIVE_CAPTION, "k6_revision": K6_REV,
                     "comfy_revision": COMFY_REV, "dtype": "native-comfy-text-fp32-forward", "text_device": TEXT_DEVICE,
                     "models": [m for m in MODELS if m["group"] == "text"]}
    TEXT_CACHE_KEY = hashlib.sha256(json.dumps(text_contract, sort_keys=True).encode()).hexdigest()

    def make_config(profile="smoke", seed=42):
        cfg = dict(PROFILES[profile], run_name=f"{profile}-seed{seed}", seed=seed, fps=24.0,
                   comfy_root=str(COMFY), model_root=str(MODEL_ROOT), output_root=str(RESULTS / "videos"),
                   run_root=str(RESULTS / f"{profile}-seed{seed}"), transformer_name=MODELS[0]["path"].removeprefix("diffusion_models/"),
                   video_caption=VIDEO_CAPTION, audio_caption=AUDIO_CAPTION, negative_caption=NEGATIVE_CAPTION,
                   text_device=TEXT_DEVICE, cpu_threads=CPU_THREADS, text_cache_key=TEXT_CACHE_KEY,
                   tiled_decode=True, tile_size=256, tile_overlap=64, temporal_size=16, temporal_overlap=4,
                   comfy_revision=COMFY_REV, k6_revision=K6_REV)
        Path(cfg["run_root"]).mkdir(parents=True, exist_ok=True)
        Path(cfg["output_root"]).mkdir(parents=True, exist_ok=True)
        Path(cfg["run_root"], "config.json").write_text(json.dumps(cfg, indent=2))
        return cfg

    CFG = make_config()
    print(json.dumps(CFG, indent=2))
    ''')
    md('''## 6. Общая часть worker: ComfyUI loaders и безопасные промежуточные файлы

    Следующие ячейки — исполняемый исходник стадий. Вызов `write_text` собирает его в файл, который Python запускает в изолированном процессе. Это даёт возможность измерить и освободить весь процесс, включая внутренние model caches.

    `safetensors` используется для conditioning и joint latents. Формат conditioning сохраняет все tensor extras, включая pooled и K6 audio fields. Несовместимые metadata и устаревший prompt-cache останавливают запуск. Ни одного `torch.load` для промежуточных пользовательских объектов нет; официальные TOD/BigVGAN веса штатный загрузчик открывает с `weights_only=True`.''')
    code("COMMON_CODE = " + repr(dedent(COMMON).strip() + "\n"))
    md('''### 6.1. Qwen + CLIP и conditioning для Audio/Video

    `DualCLIPLoader(type="kandinsky5")` — тот же shared encoder path, что в K6 ComfyUI. `Kandinsky6TextEncode` задаёт K6 template и выбирает final normalized Qwen hidden state. Не подменяем его последним произвольным hidden layer из Transformers. Negative conditioning вычисляем для полноты исходного графа; distilled sampler с CFG=1 его не использует.''')
    code("TEXT_CODE = " + repr(dedent(TEXT).strip() + "\n"))
    md('''### 6.2. Joint sampling: два латента, десять вызовов DiT

    Применяем штатный `Kandinsky6Sampler`. Hook только считает вызовы модели; он не меняет её выход. π-policy substeps выполняются внутри sampler и не прибавляются к NFE. Сохраняем **оба** совместно полученных латента — аудио не генерируется независимой второй моделью.

    FP16 — явный compute choice для T4. Проверяем фактический dtype и `n_grid=10`; если детектор не распознал нужный checkpoint, запуск останавливается до длительного sampling.''')
    code("SAMPLE_CODE = " + repr(dedent(SAMPLE).strip() + "\n"))
    md('''### 6.3. Video VAE: tiled decode

    На отдельном процессе уже нет DiT и Qwen. Штатный `VAEDecodeTiled` получает tile size в **пикселях** и сам переводит его в latent units с учётом spatial/temporal compression. Tiling уменьшает peak, но overlap/blending могут менять результат относительно full decode. Такой прогон нельзя обозначать как bitwise parity с untiled BF16 baseline.''')
    md(r'''#### Почему на Mac появился Invalid buffer size: 75.16 GiB

    В приложенном отчёте baseline уже дошёл до video decoder: 121 RGB кадр 864×480 превращаются в latent grid 31×60×108. В `normal_attention` выбранного VAE оси времени и пространства объединяются: N=200880. Одна полная матрица attention scores в BF16 занимает:

    $$
    M = N^2\cdot 2\ \text{bytes} = 200880^2\cdot 2\ \text{bytes} \approx 75.16\ \text{GiB}.
    $$

    Это размер одного промежуточного тензора, а не весов модели. В закреплённом `slice_attention` размер среза возвращается к N, если N не делится на выбранное число steps. В отчёте само значение steps не записано; нельзя утверждать, какой путь его выбрал. Но full score allocation точно согласуется с ошибкой. `Invalid buffer size` также не обязательно распознаётся как recoverable OOM для автоматического retry.

    Штатный tiled decode ограничивает T/H/W **до attention**. При параметрах ниже один latent tile имеет не более 4×32×32 позиций; оценка одной FP32 score matrix — 64 MiB. Это не оценка полной peak VRAM: convolutions, normalization, blending и output buffers добавляют свою память.

    Разбор и код: [VAE attention / slice_attention](https://github.com/Comfy-Org/ComfyUI/blob/086099f5d8886163ef77dc354f8382cb7431582a/comfy/ldm/modules/diffusionmodules/model.py), [native tiled node](https://github.com/Comfy-Org/ComfyUI/blob/086099f5d8886163ef77dc354f8382cb7431582a/nodes.py).''')
    code('''full_positions = 31 * 60 * 108
    tile_positions = (16 // 4) * (256 // 8) * (256 // 8)
    print("Full BF16 attention score matrix GiB:", full_positions**2 * 2 / 2**30)
    print("One FP32 tiled score matrix MiB:", tile_positions**2 * 4 / 2**20)

    # This reproduces the pinned tiled_scale_multidim position rule.
    # It stops at length-overlap, not length; that difference matters at the edge.
    from math import prod
    def tile_starts(length, tile, overlap):
        return list(range(0, length - overlap, tile - overlap)) if length > tile else [0]

    for profile, latent_shape in [("smoke", (9, 32, 54)), ("baseline", (31, 60, 108))]:
        starts = [tile_starts(n, t, o) for n, t, o in zip(latent_shape, (4, 32, 32), (1, 8, 8))]
        counts = [len(x) for x in starts]
        total = prod(counts)
        assert total == {"smoke": 6, "baseline": 150}[profile]
        print(profile, "time × height × width:", counts, "VAE decode calls:", total)
    ''')
    code("VIDEO_CODE = " + repr(dedent(VIDEO).strip() + "\n"))
    md('''### 6.4. TOD-VAE + BigVGAN: audio decode

    `Kandinsky6AudioVAELoader` получает TOD checkpoint, **каталог** BigVGAN с `.pt` и `.json`, mode `44k` и scaling `0.417`. Native decoder переводит latent в mel, затем waveform и применяет K6 output normalization. Сохраняем Comfy AUDIO shape `[1,1,N]` и 44100 Hz.''')
    code("AUDIO_CODE = " + repr(dedent(AUDIO).strip() + "\n"))
    md('''### 6.5. CreateVideo / SaveVideo и проверка контейнера

    Используем те же встроенные узлы ComfyUI. Проверяем не только появление `.mp4`, но и фактическое декодирование RGB кадров и аудио: число кадров, размер, FPS, sample rate и длительность. Это проверка технической цепочки; понятность речи, тембр и lip-sync оцениваются отдельно при просмотре.''')
    code("EXPORT_CODE = " + repr(dedent(EXPORT).strip() + "\n") + "\nENTRY_CODE = " + repr(dedent(ENTRY).strip() + "\n") + dedent('''
        WORKER = SCRATCH / "run_stage.py"
        WORKER.write_text("\\n\\n".join([COMMON_CODE, TEXT_CODE, SAMPLE_CODE, VIDEO_CODE, AUDIO_CODE, EXPORT_CODE, ENTRY_CODE]))
        compile(WORKER.read_text(), str(WORKER), "exec")
        print("Worker written:", WORKER)
        '''))
    md('''## 7. Запуск стадии и сбор измерений

    Устанавливаем `CUDA_VISIBLE_DEVICES` **до импорта PyTorch в worker**. Внутри процесса физическая GPU 1 видна как `cuda:0` — это нормальная перенумерация. Notebook kernel не вызывает `.to("cuda")`.

    Перед запуском проверяются наличие и размер файлов именно в `cfg["model_root"]`; при отсутствии выводятся полные пути. `run_stage` получает актуальный CFG при каждом вызове. Если вы заменили CFG, старое значение не сохраняется в default argument. Явный вызов `run_stage("text", CFG)` удобен для ручного восстановления.

    Монитор опрашивает RSS дерева worker и доступную RAM. Worker дополнительно записывает peak PyTorch allocated/reserved и OS peak RSS. Allocator counters не равны полной VRAM, которую видит `nvidia-smi`; поэтому сохраняется также sample общей памяти GPU. Частота опроса может пропустить очень короткий пик; точная частота и raw samples остаются в JSON.''')
    code(r'''
    def validate_stage_models(stage, cfg):
        roles = {"text": {"qwen", "clip"}, "sample": {"dit"},
                 "video": {"video_vae"}, "audio": {"audio_vae", "vocoder", "vocoder_config"},
                 "export": set()}[stage]
        model_root = Path(cfg["model_root"])
        problems = []
        for item in MODELS:
            if item["role"] not in roles:
                continue
            location = model_root / item["path"]
            if not location.is_file():
                target = f" (broken symlink -> {location.resolve()})" if location.is_symlink() else ""
                problems.append(f"Missing: {location}{target}")
            elif location.stat().st_size != item["size"]:
                problems.append(f"Size mismatch: {location}; expected {item['size']}, got {location.stat().st_size}")
        if problems:
            group = "text" if stage == "text" else "media"
            raise FileNotFoundError(
                "\n".join(problems) +
                f"\nStage model_root: {model_root}. Align CFG['model_root'] with MODEL_ROOT, "
                f"run ensure_models({group!r}), then retry run_stage({stage!r}, CFG). "
                "The model worker has not been started."
            )

    def run_stage(stage, cfg=None):
        cfg = CFG if cfg is None else cfg
        validate_stage_models(stage, cfg)
        config_path = Path(cfg["run_root"]) / "config.json"
        config_path.write_text(json.dumps(cfg, indent=2))
        gpu = TEXT_GPU if stage == "text" else MEDIA_GPU
        env = worker_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = "" if stage == "export" or (stage == "text" and cfg["text_device"] == "cpu") else str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        # Bundle a ffmpeg binary only if the image lacks it; the worker sees this PATH.
        if not shutil.which("ffmpeg"):
            binary = subprocess.check_output([PYTHON, "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"], text=True).strip()
            bindir = SCRATCH / "bin"
            bindir.mkdir(exist_ok=True)
            ffmpeg = bindir / "ffmpeg"
            if not ffmpeg.exists():
                ffmpeg.symlink_to(binary)
            env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
        log_path = Path(cfg["run_root"]) / (stage + ".log")
        samples = []
        start = time.monotonic()
        with log_path.open("w") as log:
            child = subprocess.Popen([PYTHON, str(WORKER), "--config", str(config_path), "--stage", stage],
                                     cwd=str(COMFY), env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    try:
                        process = psutil.Process(child.pid)
                        rss = process.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True) if p.is_running())
                        sample = {"seconds": time.monotonic() - start, "worker_tree_rss_gib": rss / 2**30,
                                  "host_available_gib": host_memory()["available_gib"]}
                        if shutil.which("nvidia-smi"):
                            try:
                                smi = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                                                     capture_output=True, text=True, timeout=5)
                                sample["nvidia_smi_mib"] = smi.stdout.strip()
                            except subprocess.TimeoutExpired:
                                sample["nvidia_smi_timeout"] = True
                        samples.append(sample)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    if len(samples) % 15 == 0:
                        print(f"{stage}: {time.monotonic()-start:.0f} s elapsed; log: {log_path}", flush=True)
                    time.sleep(2)
                returncode = child.wait()
            except BaseException:
                # An interrupted notebook must not leave a model worker running.
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                raise
        monitor = {"poll_interval_seconds": 2, "samples": samples, "returncode": returncode,
                   "parent_wall_seconds": time.monotonic() - start}
        (Path(cfg["run_root"]) / (stage + "-monitor.json")).write_text(json.dumps(monitor, indent=2))
        print(log_path.read_text()[-7000:])
        if returncode != 0:
            raise RuntimeError(f"{stage} failed (exit {returncode}). Inspect {log_path}; -9 often indicates host RAM exhaustion.")
        metrics = json.loads((Path(cfg["run_root"]) / (stage + "-metrics.json")).read_text())
        return metrics

    def reuse_text(source_cfg, target_cfg):
        # Same captions, encoder versions and dtype; resolution/seed may differ.
        start = time.perf_counter()
        for name in ["positive.safetensors", "negative.safetensors"]:
            target = Path(target_cfg["run_root"]) / name
            shutil.copy2(Path(source_cfg["run_root"]) / name, target)
        elapsed = time.perf_counter() - start
        run = Path(target_cfg["run_root"])
        # Record cache-copy cost; never copy old encoder timings into a new run.
        (run / "text-metrics.json").write_text(json.dumps({"stage": "text", "cache_reused": True,
            "source_run_root": source_cfg["run_root"], "elapsed_seconds": elapsed,
            "worker_peak_rss_gib": 0.0, "cuda_visible_devices": ""}, indent=2))
        (run / "text-monitor.json").write_text(json.dumps({"parent_wall_seconds": elapsed,
            "samples": [], "returncode": 0, "cache_reused": True}, indent=2))
    ''')
    md('''## 8. Smoke: сначала получаем измеримый сквозной результат

    Выполняйте пять ячеек последовательно. После возврата каждой функции worker уже завершён: его веса и CUDA context освобождены. При ошибке продолжать к следующей стадии не нужно. Промежуточные файлы позволяют исправить decoder/export без повторного Qwen/sampling.''')
    code('''# Successful text stages may have purged temporary encoder weights.
    # Restore them on retry only when files are absent; avoid another full hash pass otherwise.
    if any(not (Path(CFG["model_root"]) / item["path"]).is_file()
           for item in MODELS if item["group"] == "text"):
        if Path(CFG["model_root"]).resolve() != MODEL_ROOT.resolve():
            raise ValueError("CFG model_root differs from MODEL_ROOT; recreate CFG before restoring text weights")
        ensure_models("text")
    TEXT_METRICS = run_stage("text", CFG)
    assert Path(CFG["run_root"], "positive.safetensors").is_file()
    assert Path(CFG["run_root"], "negative.safetensors").is_file()
    if not KEEP_TEXT_WEIGHTS:
        ensure_models("purge-text")
    print("Text stage finished; worker memory released.")
    ''')
    code('''ensure_models("media")
    SAMPLE_METRICS = run_stage("sample")
    print("Measured DiT NFE:", SAMPLE_METRICS["dit_nfe"])
    ''')
    code('''VIDEO_METRICS = run_stage("video")
    print("Decoded frames:", VIDEO_METRICS["frames_shape"])
    ''')
    code('''AUDIO_METRICS = run_stage("audio")
    print("Audio duration:", AUDIO_METRICS["audio_seconds"], "seconds")
    ''')
    code('''EXPORT_METRICS = run_stage("export")
    from IPython.display import Video, display
    display(Video(EXPORT_METRICS["mp4"], embed=True))
    ''')
    md('''## 9. Собираем таблицу и график фактических измерений

    Каждая цифра берётся из выполненного worker. Sampling time отдельно от model load, text encode, codec и mux. Здесь fresh processes, поэтому это **cold staged pipeline**, не latency постоянно резидентной модели.

    Не складывайте peak VRAM между стадиями и не называйте сумму памяти двух GPU памятью одной модели. Для общего wall time используйте сумму parent stage timings; отдельно учитывайте скачивание/установку. График строится только после успешных измерений.''')
    code('''import csv
    import matplotlib.pyplot as plt

    def summarize(cfg):
        run = Path(cfg["run_root"])
        rows = []
        for stage in ["text", "sample", "video", "audio", "export"]:
            metric = json.loads((run / f"{stage}-metrics.json").read_text())
            monitor = json.loads((run / f"{stage}-monitor.json").read_text())
            rows.append({"stage": stage, "wall_seconds": monitor["parent_wall_seconds"],
                         "worker_seconds": metric["elapsed_seconds"],
                         "worker_peak_rss_gib": metric["worker_peak_rss_gib"],
                         "gpu_peak_allocated_gib": metric.get("gpu_peak_allocated_gib", 0),
                         "gpu_peak_reserved_gib": metric.get("gpu_peak_reserved_gib", 0),
                         "physical_gpu": metric["cuda_visible_devices"],
                         "dit_nfe": metric.get("dit_nfe", 0), "cache_reused": metric.get("cache_reused", False)})
        with (run / "stage-summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (run / "reproducibility.json").write_text(json.dumps({"config": cfg, "models": MODELS,
                    "gpu_inventory": GPUS, "kernel_torch": torch.__version__,
                    "cpu_ram": host_memory()}, indent=2))
        for row in rows:
            print(row)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        labels = [r["stage"] for r in rows]
        axes[0].bar(labels, [r["wall_seconds"] for r in rows]); axes[0].set_ylabel("Stage wall time, s")
        axes[1].bar(labels, [r["gpu_peak_allocated_gib"] for r in rows]); axes[1].set_ylabel("Worker CUDA allocated peak, GiB")
        fig.suptitle(cfg["run_name"] + " — measured staged pipeline")
        fig.tight_layout(); fig.savefig(run / "stage-measurements.png", dpi=140); plt.show()
        print("Total cold stage wall seconds:", round(sum(r["wall_seconds"] for r in rows), 2))
        return rows

    SMOKE_SUMMARY = summarize(CFG)
    ''')
    md('''## 10. Опционально: baseline 5s и повторные seeds

    После просмотра smoke установите `RUN_BASELINE=True`. Для 121 кадра spatial/temporal attention и decode существенно тяжелее; успешный smoke не гарантирует вместимость baseline. Conditioning можно переиспользовать, поскольку текст не меняется. Стадии всё ещё последовательны; VRAM не суммируется.

    Для исследования запустите минимум три seed при фиксированных настройках. Сравнивайте FP16/tiled режим с тем же режимом, а не с опубликованным H100 BF16 временем. Не называйте первый сгенерированный голос voice cloning: в этом pipeline нет speaker reference или обученного voice adapter.''')
    code('''RUN_BASELINE = False
    if RUN_BASELINE:
        BASELINE_CFG = make_config("baseline", seed=42)
        reuse_text(CFG, BASELINE_CFG)
        for stage in ["sample", "video", "audio", "export"]:
            run_stage(stage, BASELINE_CFG)
        BASELINE_SUMMARY = summarize(BASELINE_CFG)
        baseline_result = json.loads((Path(BASELINE_CFG["run_root"]) / "result.json").read_text())
        display(Video(baseline_result["mp4"], embed=True))
    else:
        print("Baseline skipped. Сначала проверьте smoke, затем включите RUN_BASELINE.")
    ''')
    md('''## 11. Ошибки, восстановление и вывод

    | Признак | Действие |
    |---|---|
    | `ensurepip` завершился с ошибкой при создании venv | Замените setup cell обновлённой версией и выполните её снова: `--without-pip` и внешний `pip --python` восстанавливают частичную среду; перезапуск kernel и удаление весов не нужны |
    | `FileNotFoundError` для Qwen/CLIP | Проверьте `CFG["model_root"]` и наличие файлов в `text_encoders`. После staged cleanup повторный text stage требует восстановления encoder weights; обновлённая ячейка text это делает. Для старой версии синхронизируйте корень, выполните `ensure_models("text")`, затем `run_stage("text", CFG)` |
    | Предупреждение `cu130` | Относится к accelerated backend ComfyUI; само по себе не является `FileNotFoundError`. Не меняйте Torch ради исправления пути к модели; сохраняйте штатный Kaggle CUDA-стек и разберите отдельный traceback, если позднее возникнет CUDA error |
    | Internet / Hugging Face 403 | Проверьте Internet/доступ аккаунта и URL конкретного файла; не запускайте общий Pro/SR downloader |
    | No space left | Оставьте staged text cleanup, подключите веса как Input или выберите scratch с достаточным местом; пользовательские datasets не удаляются |
    | Text worker exit −9 | Проверьте RAM graph/monitor; не запускайте стадии параллельно. Закройте другие notebook workloads. При persistent OOM нужен меньший/quantized text checkpoint с отдельной проверкой conditioning |
    | Text CUDA OOM | `TEXT_DEVICE="cpu"` в конфигурации и повтор text stage; CPU вариант существенно медленнее. Изменение этой настройки требует пересоздать CFG/cache key |
    | Sampling OOM | Сначала только smoke. Уменьшайте resolution/frame count, сохраняя spatial multiples 16 и `T=4k+1`; это новая задача, запишите настройки |
    | NaN/Inf в FP16 | Сохраните лог, остановитесь. FP16 на T4 не эквивалентен рекомендованному BF16; увеличение steps проблему dtype не исправляет |
    | Decoder OOM | Уменьшите tile size/temporal size в CFG и повторите только video stage; сравнивайте возможные seams/temporal artefacts |
    | MPS Invalid buffer size: 75.16 GiB в Desktop | Замените node 9 VAEDecode на VAEDecodeTiled: 256 / 64 / 16 / 4. Здесь tiled decode уже выбран; не отключайте его для первого baseline |
    | Не хватает config.json | Повторите media download cell: JSON — седьмой обязательный файл |
    | Error после обновления Kaggle image | Сохраните `worker-environment.txt`, CUDA/driver inventory и traceback; закреплённый source commit не фиксирует всё CUDA окружение |

    После неудачной стадии процесс уже завершён; перед повтором измените нужные поля `CFG` и перезапустите соответствующую `run_stage`. Изменение prompts, encoder versions или dtype требует заново выполнить ячейку конфигурации и text stage, а при staged cleanup — ещё и загрузку text weights. Повторный video/audio/export использует сохранённые joint latents.

    **Запишите собственный вывод:**

    1. Завершились ли все пять стадий, присутствуют ли проверенные video/audio streams?
    2. Каков peak VRAM каждой стадии, какой процесс достиг максимального RSS? Были ли offload/OOM?
    3. Сколько заняли text encode, 10 DiT calls, video/audio decode, MP4 export и холодный pipeline в целом?
    4. Как изменились время и память при 33→121 кадрах? Видны ли tile seams, артефакты, разборчива ли речь?
    5. Что измерено, а что осталось гипотезой? Не делайте вывод о voice cloning или parity с BF16 по одному ролику.

    Скачайте из Output `videos/*.mp4`, CSV, график, конфигурацию, JSON metrics/monitors, logs, manifest и environment. Для архива всех результатов можно вручную выполнить `shutil.make_archive('/kaggle/working/k6-results', 'zip', RESULTS)` — веса в эту папку не входят.

    ## Источники и следующий шаг

    - [K6 Lite Distill model card](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers): выбранный checkpoint и режим distill.
    - [Официальная ComfyUI интеграция K6](https://github.com/kandinskylab/kandinsky-6/tree/7ceeefad7fb625011f7db602c43893a989f19c15/comfyui), [sampler](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/comfyui/kandinsky6/sampling.py), [nodes](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/comfyui/kandinsky6/nodes.py).
    - [ComfyUI pinned source](https://github.com/Comfy-Org/ComfyUI/tree/086099f5d8886163ef77dc354f8382cb7431582a), [model management](https://github.com/Comfy-Org/ComfyUI/blob/086099f5d8886163ef77dc354f8382cb7431582a/comfy/model_management.py): dtype/offload policy.
    - [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks), [Kaggle runtime images](https://github.com/Kaggle/docker-python).
    - [MIT course](https://diffusion.csail.mit.edu/2026/index.html), [K6 по компонентам](../README.md), [подробные π-Flow labs](../../dit/README.md).

    Использование компонентов регулируется их собственными licenses/model cards: [TOD/MMAudio](https://huggingface.co/hkchengrex/MMAudio), [BigVGAN](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x), [THIRD_PARTY](../THIRD_PARTY.md). K6 MIT не заменяет лицензии всех сторонних весов.''')
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                            language_info=dict(name="python", version="3.11"),
                            kaggle=dict(accelerator="nvidiaTeslaT4", isInternetEnabled=True, language="python", sourceType="notebook"),
                            k6_deployment=dict(target="2xT4_16GB_CPU_RAM_32GB", full_gpu_run_verified=False,
                                               comfy_revision=COMFY_REV, k6_revision=K6_REV)), cells=cells)
    for index, cell in enumerate(cells):
        cell["id"] = f"k6-kaggle-{index:03d}"
    dest = root / "kandinsky-6/notebooks/15_kaggle_t4_lite_distill.ipynb"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print("Built", dest, "cells", len(cells), "code", sum(c["cell_type"] == "code" for c in cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    build(parser.parse_args().root)
