import json
from pathlib import Path
from k6_common import bootstrap, timed, save_conditioning

def text_stage(cfg, report):
    torch, nodes, mm = bootstrap(cfg, "text", cpu=cfg["text_device"] == "cpu")
    from kandinsky6.nodes import Kandinsky6TextEncode
    clip, = timed(report, "model_load", lambda: nodes.DualCLIPLoader().load_clip(
        "qwen_2.5_vl_7b.safetensors", "clip_l.safetensors", "kandinsky5",
        device=cfg["text_device"]))
    report["text_load_device"] = str(clip.patcher.load_device)
    report["text_offload_device"] = str(clip.patcher.offload_device)
    print("TEXT_PLACEMENT", json.dumps({"load_device": report["text_load_device"],
          "offload_device": report["text_offload_device"]}), flush=True)
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
