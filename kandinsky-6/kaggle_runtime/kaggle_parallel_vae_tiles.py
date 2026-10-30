"""Decode one video's Hunyuan VAE tiles on two devices; stitch once with native ComfyUI.

The coordinator retains ComfyUI's tile order, feather masks and process_output.
The second process computes odd tiles ahead of it. Raw tile outputs cross a
bounded, atomic NumPy-file queue, avoiding CUDA IPC and /dev/shm allocations.
This is VAE tile parallelism, not DiT tensor parallelism.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def tile_parameters(vae, cfg):
    """Mirror the pixel-to-latent conversion of the pinned VAEDecodeTiled node."""
    size, overlap = cfg["tile_size"], cfg["tile_overlap"]
    temporal, border = cfg["temporal_size"], cfg["temporal_overlap"]
    if size < 4 * overlap:
        overlap = size // 4
    if temporal < 2 * border:
        border //= 2
    factor = vae.temporal_compression_decode()
    if factor is None:
        raise ValueError("Two-device tile decoding requires a temporal video VAE")
    temporal = max(2, temporal // factor)
    border = max(1, min(temporal // 2, border // factor))
    spatial = vae.spacial_compression_decode()
    return dict(tile_t=temporal, tile_x=size // spatial, tile_y=size // spatial,
                overlap=(border, overlap // spatial, overlap // spatial))


def iter_tiles(samples, params):
    """Enumerate views in the exact batch/T/H/W order used by tiled_scale_multidim."""
    sizes = (params["tile_t"], params["tile_x"], params["tile_y"])
    overlaps = params["overlap"]
    index = 0
    for batch in range(samples.shape[0]):
        sample = samples[batch:batch + 1]
        if all(length <= size for length, size in zip(sample.shape[2:], sizes)):
            yield index, sample
            index += 1
            continue
        positions = [range(0, length - border, size - border) if length > size else [0]
                     for length, size, border in zip(sample.shape[2:], sizes, overlaps)]
        for position in itertools.product(*positions):
            view = sample
            for axis, (start, size, border) in enumerate(zip(position, sizes, overlaps), 2):
                start = max(0, min(sample.shape[axis] - border, start))
                view = view.narrow(axis, start, min(size, sample.shape[axis] - start))
            yield index, view
            index += 1


def watch_parent(parent_pid):
    """Release the remote CUDA process even if the coordinator is forcibly killed."""
    def watch():
        while True:
            try:
                os.kill(parent_pid, 0)
            except ProcessLookupError:
                os._exit(130)
            time.sleep(0.25)
    threading.Thread(target=watch, daemon=True).start()


def produce_odd_tiles(samples, params, decode, directory, prefetch=2):
    import numpy as np
    directory = Path(directory)
    indices, seconds = [], 0.0
    for index, tile in iter_tiles(samples, params):
        if index % 2 == 0:
            continue
        while len(list(directory.glob("tile-*.json"))) >= prefetch:
            time.sleep(0.02)
        start = time.perf_counter()
        output = decode(tile).detach().cpu().float().numpy()
        seconds += time.perf_counter() - start
        partial = directory / f"tile-{index:06d}.incomplete"
        with partial.open("wb") as stream:
            np.save(stream, output, allow_pickle=False)
        partial.replace(directory / f"tile-{index:06d}.npy")
        meta = dict(index=index, input_shape=list(tile.shape), output_shape=list(output.shape))
        partial.write_text(json.dumps(meta))
        partial.replace(directory / f"tile-{index:06d}.json")
        indices.append(index)
    return dict(tile_indices=indices, tiles=len(indices), raw_decode_seconds=seconds)


class DiskTileWorker:
    def __init__(self, command, env, directory, timeout=300):
        self.directory = Path(directory)
        self.timeout = timeout
        self.log_path = self.directory / "remote.log"
        self.log = self.log_path.open("w")
        self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT)

    def receive(self, index, input_shape):
        import numpy as np
        import torch
        metadata = self.directory / f"tile-{index:06d}.json"
        deadline = time.monotonic() + self.timeout
        while not metadata.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f"Remote VAE worker exited before tile {index}:\n" + self.log_path.read_text(errors="replace")[-5000:])
            if time.monotonic() > deadline:
                raise TimeoutError(f"Remote VAE tile {index} exceeded {self.timeout}s")
            time.sleep(0.02)
        meta = json.loads(metadata.read_text())
        if meta["index"] != index or meta["input_shape"] != list(input_shape):
            raise ValueError("Remote tile order/input shape differs from the native tiler")
        path = metadata.with_suffix(".npy")
        data = np.load(path, allow_pickle=False)
        if list(data.shape) != meta["output_shape"]:
            raise ValueError("Incomplete remote tile payload")
        path.unlink()
        metadata.unlink()
        return torch.from_numpy(data)

    def finish(self):
        status = self.process.wait(timeout=self.timeout)
        if status:
            raise RuntimeError(f"Remote VAE worker failed ({status}):\n" + self.log_path.read_text(errors="replace")[-5000:])

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()


def dispatch_vae(vae, decode):
    """Override only the tiler's callback; retain native load/crop/blend/normalization."""
    import comfy.utils

    class DispatchVAE(type(vae)):
        def decode_tiled_3d(self, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)):
            params = dict(tile_t=tile_t, tile_x=tile_x, tile_y=tile_y, overlap=overlap)
            count = sum(1 for _ in iter_tiles(samples, params))
            output = comfy.utils.tiled_scale_multidim(
                samples, decode, tile=(tile_t, tile_x, tile_y), overlap=overlap,
                upscale_amount=self.upscale_ratio, out_channels=self.output_channels,
                index_formulas=self.upscale_index_formula, output_device=self.output_device,
                pbar=comfy.utils.ProgressBar(count))
            return self.process_output(output)

    result = DispatchVAE.__new__(DispatchVAE)
    result.__dict__.update(vae.__dict__)
    return result


def decode_two_gpus(cfg, vae, latent, report, *, cpu_validation=False):
    """Distribute raw tiles; CPU validation never represents a two-CUDA-device run."""
    import nodes
    if vae.handles_tiling or vae.latent_dim != 3:
        raise ValueError("This adapter targets the HunyuanVideo 3D VAE's external tiler")
    devices = cfg["video_tile_gpus"]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("video_tile_gpus must contain two different physical GPU indices")
    params = tile_parameters(vae, cfg)
    total = sum(1 for _ in iter_tiles(latent, params))
    if total < 2:
        raise ValueError("A one-tile video cannot be distributed between two tile workers")
    kwargs = dict(tile_size=cfg["tile_size"], overlap=cfg["tile_overlap"],
                  temporal_size=cfg["temporal_size"], temporal_overlap=cfg["temporal_overlap"])
    timeout = cfg.get("video_tile_timeout_seconds", 300)
    with tempfile.TemporaryDirectory(prefix="video-tile-ipc-", dir=cfg["run_root"]) as directory:
        directory = Path(directory)
        remote_cfg = dict(cfg, run_root=str(directory), tile_parent_pid=os.getpid(),
                          video_latent_path=str(Path(cfg["run_root"]) / "joint-latents.safetensors"),
                          tile_parameters=params, tile_ipc_directory=str(directory),
                          tile_worker_cpu_validation=cpu_validation)
        config = directory / "remote-config.json"
        config.write_text(json.dumps(remote_cfg))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "" if cpu_validation else str(devices[1])
        env["PYTHONUNBUFFERED"] = "1"
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config)]
        remote = DiskTileWorker(command, env, directory, timeout)
        index, counts = 0, [0, 0]
        labels = ["cpu:coordinator", "cpu:worker"] if cpu_validation else [f"GPU {d}" for d in devices]
        print(f"VIDEO_TILE_PARALLEL: {total} tiles; {labels[0]} even / {labels[1]} odd", flush=True)
        def decode(tile):
            nonlocal index
            current = index
            index += 1
            rank = current % 2
            counts[rank] += 1
            if rank:
                result = remote.receive(current, tile.shape)
            else:
                result = vae.first_stage_model.decode(tile.to(dtype=vae.vae_dtype, device=vae.device)).to(dtype=vae.vae_output_dtype()).cpu()
            print(f"VIDEO_TILE {current + 1}/{total}: device={labels[rank]}", flush=True)
            return result
        try:
            adapted = dispatch_vae(vae, decode)
            frames, = nodes.VAEDecodeTiled().decode(adapted, {"samples": latent}, **kwargs)
            remote.finish()
            remote_metrics = json.loads((directory / "video_tiles-metrics.json").read_text())
            if remote_metrics["tiles"] != counts[1] or sum(counts) != total:
                raise ValueError("Not all native VAE tiles were consumed exactly once")
            report["tile_parallelism"] = dict(mode="cpu_validation" if cpu_validation else "two_process_vae_tiles",
                physical_gpus=None if cpu_validation else devices,
                tiles_per_worker=counts, total_tiles=total, remote=remote_metrics,
                native_stitch_order=True, dit_tensor_parallel=False)
            return frames
        finally:
            remote.close()


def remote_main(config):
    cfg = json.loads(Path(config).read_text())
    watch_parent(cfg["tile_parent_pid"])
    from k6_common import bootstrap, timed, finish_report
    started, report = time.perf_counter(), {}
    torch, nodes, mm = bootstrap(cfg, "video_tiles", cpu=cfg.get("tile_worker_cpu_validation", False))
    from safetensors.torch import load_file
    latent = load_file(cfg["video_latent_path"])["video"]
    vae, = timed(report, "model_load", lambda: nodes.VAELoader().load_vae("hunyuan_video_vae_bf16.safetensors"))
    p = cfg["tile_parameters"]
    shape = vae._tile_bounded_shape(latent.shape, p["tile_x"], p["tile_y"], p["tile_t"])
    mm.load_models_gpu([vae.patcher], memory_required=vae.memory_used_decode(shape, vae.vae_dtype), force_full_load=vae.disable_offload)
    with torch.inference_mode(), mm.cuda_device_context(vae.device):
        def decode(tile):
            return vae.first_stage_model.decode(tile.to(dtype=vae.vae_dtype, device=vae.device)).to(dtype=vae.vae_output_dtype())
        report.update(produce_odd_tiles(latent, p, decode, cfg["tile_ipc_directory"]))
    finish_report(cfg, "video_tiles", report, started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    options = parser.parse_args()
    remote_main(options.config)
