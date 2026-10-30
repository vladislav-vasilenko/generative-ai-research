from pathlib import Path
from k6_common import bootstrap, timed, finite

def video_stage(cfg, report):
    torch, nodes, mm = bootstrap(cfg, "video")
    import numpy as np
    from safetensors.torch import load_file
    latent = load_file(str(Path(cfg["run_root"]) / "joint-latents.safetensors"))["video"]
    vae, = timed(report, "model_load", lambda: nodes.VAELoader().load_vae("hunyuan_video_vae_bf16.safetensors"))
    with torch.inference_mode():
        if len(cfg.get("video_tile_gpus", [0])) == 2:
            if not cfg["tiled_decode"]:
                raise ValueError("Two-GPU video decode requires tiled_decode=True")
            from kaggle_parallel_vae_tiles import decode_two_gpus
            frames = timed(report, "decode", lambda: decode_two_gpus(cfg, vae, latent, report))
        elif cfg["tiled_decode"]:
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
