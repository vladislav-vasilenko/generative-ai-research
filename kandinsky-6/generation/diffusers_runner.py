"""Native Diffusers generation helpers; no ComfyUI imports or model loading at import time."""
# %% Imports
import contextlib
import gc
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from diffusers import (
    AutoencoderKLHunyuanVideo, Kandinsky6TI2VAPipeline,
    Kandinsky6Transformer3DModel, MMAudioVAE, MMAudioVocoder, PiflowScheduler,
)
from diffusers.utils import encode_video
from diffusers.video_processor import VideoProcessor
from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

# %% Prompt compiler and profiles
PROFILES = {
    'smoke': {'num_frames': 25, 'export_frames': 24, 'seconds': 1},
    'baseline': {'num_frames': 121, 'export_frames': 120, 'seconds': 5},
}
SIZES = {'landscape': {'smoke': (432, 256), 'baseline': (864, 480)},
         'portrait': {'smoke': (256, 432), 'baseline': (480, 864)},
         'square': {'smoke': (320, 320), 'baseline': (640, 640)}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compile_prompt(preset, profile):
    spec = preset['profiles'][profile]
    duration = PROFILES[profile]['seconds']
    previous = 0
    visual, sounds = [], []
    for event in spec['timeline']:
        start, end = event['start'], event['end']
        if start != previous or not start < end <= duration:
            raise ValueError(f'Invalid {profile} timeline: {event}')
        previous = end
        visual.append(f"From {start:g} to {end:g} seconds: {event['action']}")
        sounds.append(event['audio_cue'])
    if previous != duration:
        raise ValueError('Timeline must cover the selected export duration')
    scene = preset['scene_setup']
    prompt = ' '.join([preset['metadata']['base_style'], scene['room_description'],
        scene['lighting'], spec['camera_setup'],
        'Initial state: ' + ' '.join(preset['elements']['initial_state']),
        f'One uninterrupted shot, about {duration} seconds.', *visual,
        '<AUDCAP>' + ' '.join(sounds) + '<ENDAUDCAP>'])
    return prompt, ', '.join(preset['negative_prompts'])


def reference_and_size(preset, profile, image_path=None, aspect='auto'):
    image, image_sha = None, None
    if preset['metadata']['mode'] == 'i2v':
        if not image_path or not Path(image_path).is_file():
            raise FileNotFoundError('Set IMAGE_PATH to the reference in /kaggle/input; image files are not bundled')
        image_sha = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        if aspect == 'auto':
            ratio = image.width / image.height
            aspect = 'portrait' if ratio < 0.8 else ('square' if ratio <= 1.25 else 'landscape')
    elif image_path:
        raise ValueError('Select an i2v preset when using a reference image')
    aspect = 'landscape' if aspect == 'auto' else aspect
    width, height = SIZES[aspect][profile]
    if image is not None:
        image = ImageOps.pad(image, (width, height), method=Image.Resampling.LANCZOS, color=image.getpixel((0, 0)))
    return image, image_sha, width, height

# %% Timing and checkpoint helpers

def finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f'{name} contains NaN/Inf; inspect the FP16 path before continuing')


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def measured(name, operation, reports):
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
        torch.cuda.reset_peak_memory_stats(i)
    start = time.perf_counter()
    result = operation()
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
    # ru_maxrss is a process-lifetime high-water mark, not a per-stage RAM peak.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    reports[name] = dict(seconds=time.perf_counter()-start,
        process_lifetime_peak_rss_gib=rss/(2**30 if sys.platform == 'darwin' else 2**20),
        gpu=[dict(index=i, allocated_peak_gib=torch.cuda.max_memory_allocated(i)/2**30,
                  reserved_peak_gib=torch.cuda.max_memory_reserved(i)/2**30)
             for i in range(torch.cuda.device_count())])
    print(name, json.dumps(reports[name]), flush=True)
    return result


def model_kwargs(cfg, cache_dir):
    source = cfg['local_model_dir'] or cfg['model_id']
    kwargs = dict(cache_dir=str(cache_dir), local_files_only=bool(cfg['local_model_dir']))
    if not cfg['local_model_dir']:
        kwargs['revision'] = cfg['revision']
    return source, kwargs


def text_key(prompt, cfg):
    return digest(dict(prompt=prompt, revision=cfg['revision'], model=cfg['model_id'],
                       dtype='float16', max_sequence_length=cfg['max_sequence_length'], packages=cfg['packages']))


def compact_conditioning(encoded):
    tokens, pooled, mask = encoded[:3]
    if tokens.shape[0] != 1:
        raise ValueError('This runner encodes one prompt per call')
    # The public pipeline call has no prompt_attention_mask argument. Remove padding
    # after Qwen encoding so cached inputs contain only the valid token sequence.
    if mask is not None:
        tokens = tokens[:, mask[0].bool()]
    finite(tokens, 'Qwen conditioning')
    finite(pooled, 'CLIP conditioning')
    return {'prompt_embeds': tokens.detach().cpu().contiguous().clone(),
            'pooled_prompt_embeds': pooled.detach().cpu().contiguous().clone()}

# %% Background nvidia-smi utilization trace
NVIDIA_SMI_FIELDS = ('timestamp', 'index', 'utilization.gpu', 'memory.used', 'power.draw')


def summarize_nvidia_smi(path, interval_ms, busy_percent=50):
    # nvidia-smi prints one row per GPU in index order for every polling tick.
    samples, current = [], {}
    for line in Path(path).read_text().splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != len(NVIDIA_SMI_FIELDS):
            continue
        try:
            index, util, memory = int(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        if index in current:
            samples.append(current)
            current = {}
        current[index] = (util, memory)
    if current:
        samples.append(current)
    gpus = sorted({index for sample in samples for index in sample})
    samples = [sample for sample in samples if len(sample) == len(gpus)]
    if not samples:
        return dict(available=True, samples=0, interval_ms=interval_ms)
    busy_counts = [sum(sample[i][0] >= busy_percent for i in gpus) for sample in samples]
    share = lambda count: sum(c == count for c in busy_counts)/len(samples)
    return dict(available=True, samples=len(samples), interval_ms=interval_ms, busy_percent=busy_percent,
        gpu=[dict(index=i,
                  mean_util_percent=sum(s[i][0] for s in samples)/len(samples),
                  max_util_percent=max(s[i][0] for s in samples),
                  busy_fraction=sum(s[i][0] >= busy_percent for s in samples)/len(samples),
                  max_memory_used_mib=max(s[i][1] for s in samples)) for i in gpus],
        all_busy_fraction=share(len(gpus)), one_busy_fraction=share(1), none_busy_fraction=share(0))


class NvidiaSmiTrace:
    """Poll per-GPU utilization in a child nvidia-smi process while the wrapped block runs."""

    def __init__(self, path, interval_ms=200):
        self.path, self.interval_ms = Path(path), interval_ms
        self.process, self.summary = None, dict(available=False)

    def __enter__(self):
        self.file = self.path.open('w')
        try:
            self.process = subprocess.Popen(['nvidia-smi', '--query-gpu='+','.join(NVIDIA_SMI_FIELDS),
                '--format=csv,noheader,nounits', '-lms', str(self.interval_ms)],
                stdout=self.file, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print('nvidia-smi not found; GPU utilization is not traced', flush=True)
        return self

    def __exit__(self, *exc):
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.file.close()
        if self.process is not None:
            self.summary = summarize_nvidia_smi(self.path, self.interval_ms)
        return False


def plot_nvidia_smi(path, output_path, interval_ms=200):
    import matplotlib.pyplot as plt
    series = {}
    for line in Path(path).read_text().splitlines():
        parts = [part.strip() for part in line.split(',')]
        try:
            series.setdefault(int(parts[1]), []).append(float(parts[2]))
        except (IndexError, ValueError):
            continue
    figure, axis = plt.subplots(figsize=(10, 3))
    for index, values in sorted(series.items()):
        axis.plot([k*interval_ms/1000 for k in range(len(values))], values, label=f'cuda:{index}')
    axis.set(xlabel='seconds since denoising start', ylabel='GPU utilization, %', ylim=(0, 105))
    axis.legend(loc='upper right')
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    return figure

# %% Qwen + CLIP, then unload before loading the DiT
@torch.inference_mode()
def encode_texts(prompts, cfg):
    directory = Path(cfg['results'])/'conditioning'
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory/(text_key(prompt, cfg)+'.safetensors') for name,prompt in prompts.items()}
    missing = {name: prompt for name,prompt in prompts.items() if not paths[name].is_file()}
    if not missing:
        print('Reused saved conditioning for Smoke and Baseline')
        return paths
    if not cfg['local_model_dir'] and shutil.disk_usage(cfg['scratch']).free < 18*2**30:
        raise OSError('Text encoding needs 18 GiB free scratch; attach a full Diffusers snapshot as Input or free scratch')
    # Only this temporary directory owns downloaded text weights. Mounted Input is read-only.
    with tempfile.TemporaryDirectory(prefix='text-weights-', dir=cfg['scratch']) as cache:
        source, kwargs = model_kwargs(cfg, cache)
        device = torch.device('cuda:1')
        qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(source, subfolder='text_encoder',
            dtype=torch.float16, device_map='auto', max_memory={1: '11GiB', 'cpu': '18GiB'},
            low_cpu_mem_usage=True, attn_implementation='sdpa', **kwargs)
        clip = CLIPTextModel.from_pretrained(source, subfolder='text_encoder_2',
                                            dtype=torch.float32, **kwargs).to(device)
        pipe = Kandinsky6TI2VAPipeline(transformer=None, vae=None, audio_vae=None, vocoder=None,
            text_encoder=qwen, text_encoder_2=clip,
            tokenizer=Qwen2_5_VLProcessor.from_pretrained(source, subfolder='tokenizer', **kwargs),
            tokenizer_2=CLIPTokenizer.from_pretrained(source, subfolder='tokenizer_2', **kwargs),
            scheduler=PiflowScheduler.from_pretrained(source, subfolder='scheduler', **kwargs))
        try:
            print('Qwen device map:', qwen.hf_device_map, flush=True)
            for name,prompt in missing.items():
                encoded = pipe.encode_prompt(prompt, do_classifier_free_guidance=False,
                    device=device, dtype=torch.float16, max_sequence_length=cfg['max_sequence_length'])
                tensors = compact_conditioning(encoded)
                save_file(tensors, str(paths[name]), metadata={'key': text_key(prompt,cfg)})
                print('Encoded', name, list(tensors['prompt_embeds'].shape), flush=True)
                del encoded, tensors
        finally:
            del pipe, qwen, clip
            release_memory()
    return paths

# %% DiT placement: two-GPU layer split or one whole GPU
DIT_PLACEMENTS = ('balanced', 'cuda:0', 'cuda:1')


def dit_load_options(placement):
    # 'balanced' splits blocks for memory; a single device keeps the whole DiT on one T4.
    if placement == 'balanced':
        return dict(device_map='balanced', max_memory={0:'10GiB',1:'10GiB'}), {0,1}
    if placement in DIT_PLACEMENTS:
        return dict(device_map=placement), {int(placement.split(':')[1])}
    raise ValueError(f'DIT_PLACEMENT must be one of {DIT_PLACEMENTS}, got {placement!r}')


def mapped_gpu_indices(device_map):
    devices = set()
    for value in device_map.values():
        if isinstance(value, int):
            devices.add(value)
        else:
            value = str(value)
            if value.startswith('cuda:'):
                devices.add(int(value.split(':')[1]))
    return devices


def load_sampling_pipeline(cfg, has_image):
    source, kwargs = model_kwargs(cfg, cfg['media_cache'])
    placement = cfg['dit_placement']
    options, expected = dit_load_options(placement)
    transformer = Kandinsky6Transformer3DModel.from_pretrained(source, subfolder='transformer',
        torch_dtype=torch.float16, low_cpu_mem_usage=True, **options, **kwargs)
    if transformer.config.model_dim != 1792 or transformer.config.num_visual_blocks != 32:
        raise ValueError('Expected the Lite architecture, not Pro')
    # A single-device load may skip accelerate dispatch and leave no hf_device_map.
    device_map = getattr(transformer, 'hf_device_map', None) or {'': str(transformer.device)}
    if mapped_gpu_indices(device_map) != expected or any(str(v) in ('cpu','disk') for v in device_map.values()):
        raise RuntimeError(f'Expected DiT on GPU(s) {sorted(expected)}, no CPU/disk: {device_map}')
    print('DiT device map:', json.dumps(device_map, default=str), flush=True)
    # Encoder overrides prevent the pipeline from downloading Qwen a second time.
    pipe = Kandinsky6TI2VAPipeline.from_pretrained(source, transformer=transformer,
        text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None,
        torch_dtype={'default':torch.float32}, **kwargs)
    if not isinstance(pipe.scheduler, PiflowScheduler) or pipe.scheduler.config.n_grid != 10:
        raise ValueError('Expected the Lite distilled PiFlow scheduler with n_grid=10')
    pipe.vae.enable_tiling()
    if has_image:
        # One reference-frame encode next to the first DiT block; DiT blocks keep their device map.
        pipe.vae.to('cuda:0' if placement == 'balanced' else placement)
    # Do not call pipe.to(...) or enable_model_cpu_offload() over a sharded transformer.
    return pipe, device_map

# %% Joint generation and a reusable latent checkpoint
@torch.inference_mode()
def sample_joint(cfg, prompt, negative, conditioning_path, image, latent_path, trace_path=None):
    if latent_path.exists():
        with safe_open(str(latent_path), framework='pt') as f:
            if f.metadata().get('job_key') != cfg['job_key']:
                raise ValueError('Cached latent belongs to a different configuration/reference')
        print('Reusing joint latents; no Qwen/DiT rerun')
        return {'cache_reused':True}
    conditioning = load_file(str(conditioning_path))
    with safe_open(str(conditioning_path), framework='pt') as f:
        if f.metadata().get('key') != text_key(prompt,cfg):
            raise ValueError('Stale text conditioning')
    pipe, device_map = load_sampling_pipeline(cfg, image is not None)
    calls = [0]
    def count_call(module, inputs):
        calls[0] += 1
    def progress(pipeline, step, timestep, tensors):
        finite(tensors['latents'], f'video step {step}')
        print(f'DiT {step+1}/10, timestep={float(timestep):.2f}', flush=True)
        return tensors
    hook = pipe.transformer.register_forward_pre_hook(count_call)
    # The trace covers only the denoising call, not weight loading or latent saving.
    trace = NvidiaSmiTrace(trace_path) if trace_path else None
    gpus = sorted(mapped_gpu_indices(device_map))
    try:
        with trace or contextlib.nullcontext():
            for i in gpus:
                torch.cuda.synchronize(i)
            start = time.perf_counter()
            output = pipe(image=image, negative_prompt=negative,
                height=cfg['height'], width=cfg['width'], num_frames=cfg['num_frames'], frame_rate=24,
                num_inference_steps=10, guidance_scale=1.0, sample_audio=True, expand_prompts=False,
                generator=torch.Generator(device='cpu').manual_seed(cfg['seed']),
                output_type='latent', callback_on_step_end=progress, **conditioning)
            for i in gpus:
                torch.cuda.synchronize(i)
            denoise_seconds = time.perf_counter()-start
        finite(output.frames, 'sampled video'); finite(output.audio, 'sampled audio')
        if calls[0] != 10:
            raise AssertionError(f'Expected 10 measured DiT calls, got {calls[0]}')
        save_file({'video':output.frames.detach().cpu().contiguous(),
                   'audio':output.audio.detach().cpu().contiguous()}, str(latent_path),
                  metadata={'job_key':cfg['job_key']})
        result = dict(dit_nfe=calls[0], dit_placement=cfg['dit_placement'], device_map=device_map,
                      denoise_seconds=denoise_seconds, cache_reused=False,
                      video_shape=list(output.frames.shape), audio_shape=list(output.audio.shape))
        if trace is not None:
            result['nvidia_smi'] = trace.summary
        del output
        return result
    finally:
        hook.remove()
        del pipe
        release_memory()


def compare_placements(results_dir):
    # One row per run directory whose sampling actually ran (not a reused latent cache).
    rows = []
    for sample_path in sorted(Path(results_dir).glob('*/sample.json')):
        sample = json.loads(sample_path.read_text())
        if sample.get('cache_reused') or 'denoise_seconds' not in sample:
            continue
        config = json.loads((sample_path.parent/'config.json').read_text())
        trace = sample.get('nvidia_smi', {})
        stage = sample.get('stage', {})
        rows.append(dict(run=sample_path.parent.name, placement=sample['dit_placement'],
            size=f"{config['width']}x{config['height']}x{config['num_frames']}",
            denoise_s=round(sample['denoise_seconds'], 2),
            peak_alloc_gib=[round(g['allocated_peak_gib'], 2) for g in stage.get('gpu', [])],
            all_busy=trace.get('all_busy_fraction'), one_busy=trace.get('one_busy_fraction')))
    return rows

# %% Decode the saved video and audio latents
@torch.inference_mode()
def decode_video(cfg, latent_path):
    source, kwargs = model_kwargs(cfg, cfg['media_cache'])
    vae = AutoencoderKLHunyuanVideo.from_pretrained(source, subfolder='vae', torch_dtype=torch.float32, **kwargs).to('cuda:1')
    vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256,
        tile_sample_min_num_frames=16, tile_sample_stride_height=192,
        tile_sample_stride_width=192, tile_sample_stride_num_frames=12)
    try:
        z = load_file(str(latent_path))['video'].to('cuda:1',dtype=vae.dtype)
        decoded = vae.decode(z/vae.config.scaling_factor, return_dict=False)[0]
        finite(decoded, 'decoded video')
        frames = VideoProcessor(vae_scale_factor=8).postprocess_video(decoded, output_type='np')[0]
        if tuple(frames.shape) != (cfg['num_frames'], cfg['height'], cfg['width'],3):
            raise ValueError(f'Unexpected decoded frame shape: {frames.shape}')
        return frames[:cfg['export_frames']].copy()
    finally:
        del vae
        release_memory()


@torch.inference_mode()
def decode_audio(cfg, latent_path):
    source, kwargs = model_kwargs(cfg, cfg['media_cache'])
    vae = MMAudioVAE.from_pretrained(source, subfolder='audio_vae', torch_dtype=torch.float32, **kwargs).to('cuda:0')
    vocoder = MMAudioVocoder.from_pretrained(source, subfolder='vocoder', torch_dtype=torch.float32, **kwargs).to('cuda:0')
    try:
        z = load_file(str(latent_path))['audio'].to('cuda:0',dtype=vae.dtype)
        mel = vae.decode(z/vae.config.scaling_factor, return_dict=False)[0]
        waveform = vocoder(mel.to(vocoder.dtype), return_dict=False)[0][0,0].float().cpu()[None]
        finite(waveform, 'waveform')
        rate = vae.config.sample_rate
        samples = round(cfg['export_frames']/24*rate)
        if waveform.shape[-1] < samples:
            raise ValueError('Decoded audio is shorter than the requested export')
        return waveform[:,:samples].contiguous(), rate
    finally:
        del vae, vocoder
        release_memory()

# %% Export and verify the actual MP4 streams

def export_clip(frames, waveform, rate, cfg, path):
    import av
    if path.exists():
        raise FileExistsError(f'Choose a new output name to preserve the previous clip: {path}')
    encode_video(frames, fps=24, output_path=str(path), audio=waveform, audio_sample_rate=rate)
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        if len(container.streams.audio) != 1:
            raise ValueError('Expected one audio stream')
        if (video.width,video.height)!=(cfg['width'],cfg['height']):
            raise ValueError('Wrong MP4 geometry')
        fps=float(video.average_rate)
        count=sum(1 for _ in container.decode(video=0))
    with av.open(str(path)) as container:
        audio_samples=sum(f.samples for f in container.decode(audio=0))
    if count!=cfg['export_frames'] or fps!=24 or abs(audio_samples/rate-count/fps)>0.15:
        raise ValueError('MP4 frame/FPS/audio-duration verification failed')
    return dict(mp4=str(path), exported_frames=count, fps=fps, seconds=count/fps,
                audio_sample_rate=rate, decoded_audio_samples=audio_samples)

# %% Optional soundtrack supplied as a local audio file

def soundtrack_audio(generated, rate, music_path=None, offset_seconds=0.0, mode='generated', gain=0.65):
    if mode == 'generated':
        return generated, {'source':'generated'}
    if mode not in ('music','mix') or not music_path or offset_seconds < 0 or not 0 <= gain <= 1:
        raise ValueError('Set a local MUSIC_PATH, nonnegative offset, mode music/mix and gain in [0,1]')
    import av
    wanted = generated.shape[-1]
    skip = round(offset_seconds*rate)
    chunks, remaining = [], wanted
    with av.open(str(music_path)) as container:
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=rate)
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                values = torch.from_numpy(converted.to_ndarray().copy()).float()
                dropped = min(skip,values.shape[-1]);skip-=dropped
                values=values[:,dropped:]
                if values.shape[-1]:
                    part=values[:,:remaining];chunks.append(part);remaining-=part.shape[-1]
                if remaining == 0:break
            if remaining == 0:break
    if remaining:
        raise ValueError('Music file is shorter than offset plus clip duration')
    music=torch.cat(chunks,dim=-1)*gain
    mixed=music if mode=='music' else generated*0.65+music
    finite(mixed,'soundtrack')
    peak=float(mixed.abs().max())
    if peak>0.98:mixed=mixed*(0.98/peak)
    # Short fades avoid clicks without changing the requested number of samples.
    n=min(round(rate*0.01),wanted//2)
    mixed[:,:n]*=torch.linspace(0,1,n)
    mixed[:,-n:]*=torch.linspace(1,0,n)
    return mixed.contiguous(),{'source':mode,'filename':Path(music_path).name,
                              'offset_seconds':offset_seconds,'gain':gain,'mono':True}
