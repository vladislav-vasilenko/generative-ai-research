from pathlib import Path
from k6_common import bootstrap, timed, load_conditioning, finite

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
