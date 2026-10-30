from pathlib import Path
from k6_common import bootstrap, timed, save_json

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
