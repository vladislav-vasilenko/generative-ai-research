from pathlib import Path
from k6_common import bootstrap, timed, finite

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
