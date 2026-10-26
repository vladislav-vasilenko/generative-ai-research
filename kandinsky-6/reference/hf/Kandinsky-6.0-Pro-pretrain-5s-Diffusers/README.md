---
license: mit
library_name: diffusers
tags:
- video
- video-generation
- text-to-video
- image-to-video
- audio-generation
- kandinsky
- diffusers
---

<div align="center">
  <img src="https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-pretrain-5s-Diffusers/resolve/main/assets/promo.webp" width="100%">
</div>

<br>

<div align="center">
<img src="https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-pretrain-5s-Diffusers/resolve/main/assets/Kandinsky_LOGO_6_Horizontal_black.png" width="80%">
</div>


<div align="center">

<a href="https://kandinskylab.ai/models/video/"><img alt="KandinskyLab" src="https://img.shields.io/badge/KandinskyLab-76E0B7?style=for-the-badge"></a>
<a href="https://arxiv.org/pdf/2610.05608"><img alt="Report" src="https://img.shields.io/badge/Report-9C2731?style=for-the-badge"></a>
<a href="#"><img alt="PyPI" src="https://img.shields.io/badge/PyPI-3171B2?style=for-the-badge"></a>
<a href="https://huggingface.co/spaces/kandinskylab/Kandinsky-6.0-Pro-distill-5s"><img alt="HF Demo" src="https://img.shields.io/badge/HF%20Demo-D9622B?style=for-the-badge"></a>
<a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky6"><img alt="Diffusers" src="https://img.shields.io/badge/Diffusers-F8D44E?style=for-the-badge"></a>
<a href="https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/models/kandinsky6/"><img alt="vLLM" src="https://img.shields.io/badge/vLLM-54A0F8?style=for-the-badge"></a>
<a href="https://docs.sglang.io/cookbook/diffusion/Kandinsky/Kandinsky6"><img alt="SGLang" src="https://img.shields.io/badge/SGLang-C6602D?style=for-the-badge"></a>
<a href="https://haoailab.com/FastVideo/inference/kandinsky6/"><img alt="FastVideo" src="https://img.shields.io/badge/FastVideo-436BF6?style=for-the-badge"></a>
<a href="https://registry.comfy.org/nodes/kandinsky6"><img alt="ComfyUI" src="https://img.shields.io/badge/ComfyUI-EBF764?style=for-the-badge"></a>


</div>

<h1>Kandinsky 6.0: A family of diffusion models for Video + Audio generation</h1>

We present Kandinsky 6.0 Video, a family of foundation diffusion models for synchronized text-to-audio-video generation, comprising Kandinsky 6.0 Video Lite (3B parameters) and Kandinsky 6.0 Video Pro (29B parameters). Both models generate 5-second video clips with synchronized 44 kHz audio, including lip-sync, in text-to-audio-video (T2AV) and image-to-audio-video (TI2AV) modes; a plug-in super-resolution model raises the output resolution to Full-HD (1920×1080).

## Usage guide

This repository contains **Kandinsky 6.0 Pro, pretrained**: the pretrained Pro checkpoint. It uses the same architecture and sampler settings as the main Pro model.

### Text-to-audio-video (T2AV)

```python
import torch
from diffusers import Kandinsky6TI2VAPipeline
from diffusers.utils import encode_video

DEFAULT_PROMPT = (
    "cinematic shot: a giant stone samurai on a stormy cliff above a neon city opens glowing golden eyes and "
    "raises a katana. Blue lightning strikes the blade, creating a massive shockwave through the clouds. The "
    "camera rapidly pulls back from a low angle. Photorealistic, epic scale, dark blue and gold lighting, rain, "
    "sparks, volumetric lightning, blockbuster quality. Audio: heavy rain, deep thunder, metallic sword hum, "
    "rising brass and choir, electrical crackle, perfectly synchronized lightning impact, sub-bass shockwave. "
    "No dialogue, text, or logos."
)

pipe = Kandinsky6TI2VAPipeline.from_pretrained(
    "kandinskylab/Kandinsky-6.0-Pro-pretrain-5s-Diffusers", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

output = pipe(
    prompt=DEFAULT_PROMPT,
    height=480,
    width=864,
    num_frames=121,
    num_inference_steps=50,
    guidance_scale=5.0,
)
encode_video(
    output.frames[0],
    fps=24,
    output_path="output.mp4",
    audio=output.audio[0][None],
    audio_sample_rate=pipe.audio_sample_rate,
)
```

### Image-to-audio-video (TI2AV)

Pass a reference image with `image=`. It conditions the first frame, and the pipeline resizes and center-crops it to `height` × `width`. The prompt should describe what happens next in that image, so the example uses its own prompt rather than `DEFAULT_PROMPT`. It reuses `pipe` from the T2AV example.

```python
from diffusers.utils import load_image

IMAGE_PROMPT = (
    "cinematic shot: a red dragon wakes on a mound of gold coins in a sunlit cave, slowly unfolds its wings "
    "and lowers its head toward the glowing stream below. Coins slide and clink down the pile as the dragon "
    "shifts its weight. The camera slowly pushes in toward the dragon's eyes. Photorealistic, fantasy, warm "
    "golden light, drifting dust motes, lush cave flowers. Audio: deep rumbling breath, leathery wing flaps, "
    "clinking coins, trickling water, distant echoing roar. No dialogue, text, or logos."
)

image = load_image(
    "https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-pretrain-5s-Diffusers/resolve/main/assets/i2va_input.jpg"
)

output = pipe(
    prompt=IMAGE_PROMPT,
    image=image,
    height=480,
    width=864,
    num_frames=121,
    num_inference_steps=50,
    guidance_scale=5.0,
)
encode_video(
    output.frames[0],
    fps=24,
    output_path="output_ti2av.mp4",
    audio=output.audio[0][None],
    audio_sample_rate=pipe.audio_sample_rate,
)
```

### Options

Pass `sample_audio=False` to generate video only, and `expand_prompts=True` to let the Qwen2.5-VL text encoder rewrite a short prompt into a detailed one before encoding. `expand_prompts` adds latency but no extra model weights.


## Super Res guide

Kandinsky 6.0 video super-resolution is a separate pipeline, `Kandinsky6SRPipeline`. It takes the frames produced above and upscales them by ×2, ×2.25 or ×4 with tiled diffusion in K-VAE latent space, blending overlapping tiles back with Hann windows. The example uses the flow-matching checkpoint, which runs in 4 steps per tile.

```python
import torch
from diffusers import Kandinsky6SRPipeline
from diffusers.utils import encode_video

torch._inductor.config.max_autotune = True  # required: lets inductor pick flex-attention tiles that fit the SR block mask

sr_pipe = Kandinsky6SRPipeline.from_pretrained(
    "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers", torch_dtype=torch.bfloat16
)
sr_pipe.transformer.set_attention_backend("flex")
sr_pipe.enable_model_cpu_offload()
sr_pipe.transformer.compile_repeated_blocks(fullgraph=True)

upscaled = sr_pipe(
    video=output.frames[0],
    resolution_scale=2.25,  # 2, 2.25 or 4
    num_inference_steps=4,  # 2 with Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers
).frames[0]
encode_video(
    upscaled,
    fps=24,
    output_path="output_sr.mp4",
    audio=output.audio[0][None],
    audio_sample_rate=pipe.audio_sample_rate,
)
```

The example continues from the Usage guide: `pipe` and `output` come from it. The SR pipeline works on frames only, so the audio is passed to `encode_video` from the generation step.

- `resolution_scale=2.25` (the default, and the route used after Kandinsky 6 generation) is a ×1.125 bilinear pre-upscale followed by the ×2 path. `2` and `4` are the other options.
- `min_overlap` (default `0.2`) sets the minimum tile overlap. `tiles_batch_size` (default `1`) sets how many tiles go through the model per call, trading VRAM for speed.

## Components

| Folder | Class | Role |
|---|---|---|
| `transformer/` | `Kandinsky6Transformer3DModel` | Multimodal video+audio diffusion transformer (Pro) |
| `vae/` | `AutoencoderKLHunyuanVideo` | Video VAE: encodes the reference image, decodes the generated video |
| `text_encoder/` + `tokenizer/` | `Qwen2_5_VLForConditionalGeneration` + `Qwen2_5_VLProcessor` | Token-level text embeddings; also used for `expand_prompts=True` |
| `text_encoder_2/` + `tokenizer_2/` | `CLIPTextModel` + `CLIPTokenizer` | Pooled text embedding |
| `audio_vae/` | `MMAudioVAE` | Audio latents ↔ mel spectrogram |
| `vocoder/` | `MMAudioVocoder` (BigVGAN-style) | Mel spectrogram → waveform |
| `scheduler/` | `FlowMatchEulerDiscreteScheduler` | Flow matching, `shift = 5.0` |

`model_index.json` names the pipeline class `Kandinsky6TI2VAPipeline`.

### Kandinsky 6.0 family

| Checkpoint | Repository | Use |
|---|---|---|
| Pro | [kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers) | generation |
| Pro, distilled, 10 steps | [kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers) | generation |
| Pro, pretrained (this repository) | [kandinskylab/Kandinsky-6.0-Pro-pretrain-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-pretrain-5s-Diffusers) | generation |
| Lite | [kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers) | generation |
| Lite, distilled, 10 steps | [kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers) | generation |
| Lite, pretrained | [kandinskylab/Kandinsky-6.0-Lite-pretrain-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-pretrain-5s-Diffusers) | generation |
| VSR, flow matching, 4 steps/tile | [kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers) | upscale |
| VSR, distilled, 2 steps/tile | [kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers) | upscale |

# Citation

```bibtex
@misc{kandinsky2026kandinsky60videofoundation,
      title={Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation}, 
      author={Team Kandinsky and Julia Agafonova and Bulat Akhmatov and Mikhail Aksyutin and Grigorii Alekseenko and Anastasia Aliaskina and Olga Androsova and Vladimir Arkhipkin and Anna Averchenkova and Alexander Belykh and Serafima Bocharova and Sofiya Bogakovskaya and Anton Bukashkin and Mark Bulygin and Kirill Buzygin and Irina Cheremnykh and Kirill Chernyshev and Mikhail Chernyshov and Vladimir Chernyy and David Chikovani and Georgy Daniltsev and Denis Dimitrov and Anna Dmitrienko and Vladimir Dokholyan and Sergey Emelyanov and Dmitry Ermilov and Georgii Fedorov and Polina Gavrilova and Nikolai Gerasimenko and Aleksandr Gordeev and Andrey Inozemtsev and Andrei Ivaniuta and Alexander Ivanov and Mikhail Karaev and Anastasiia Kargapoltseva and Ivan Kirillov and Nikita Kiselev and Valeria Kobenko and Yury Kolabushin and Denis Koposov and Anatoly Korobov and Vladimir Korviakov and Kirill Kozlov and Denis Krzhivokolskiy and Konstantin Kuklev and Alexander Kunitsyn and Sergey Kuzin and Vladislav Lakhtionov and Alexey Letunovskiy and Maxim Litvinov and Alexander Lyulkov and Georgy Makarov and Kirill Malakhov and Egor Malykh and Mikhail Mamaev and Dmitrii Mikhailov and Polina Mikhailova and Ivan Mikheev and Elizaveta Muromtseva and Nikolai Nazarkin and Tatiana Nikulina and Lev Novitskiy and Stanislav Onuchin and Nikita Osterov and Denis Parkhomenko and Anatoliy Parpara and Vladimir Polovnikov and Konstantin Reznikov and Azat Saginbaev and Nikita Samsonov and Alexander Sentsov and Nikita Shaimov and Artem Sherstyuk and Andrey Shutkin and Egor Silvestrov and Bulat Suleimanov and Matvey Suprunov and Sergey Taranov and Irina Tolstykh and Tatiana Trofimuk and Ilya Trushkin and Aleksandra Tsybina and Olga Varlashina and Viacheslav Vasilev and Ilya Vasiliev and Eugeny Vilisov and Sergey Yakubson and Konstantin Zakharov},
      year={2026},
      eprint={2610.05608},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.05608}, 
}
```