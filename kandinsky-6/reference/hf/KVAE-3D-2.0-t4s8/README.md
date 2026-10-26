---
library_name: kvae
license: apache-2.0
tags:
  - pytorch
  - model_hub_mixin
  - pytorch_model_hub_mixin
  - video-tokenizer
  - causal-video-tokenizer
  - variational-autoencoder
  - video-reconstruction
  - diffusion-models
---

<div align="center">

  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="https://huggingface.co/kandinskylab/KVAE-3D-2.0-t4s8/resolve/main/assets/kvae-white-2.0.png"
    >
    <source
      media="(prefers-color-scheme: light)"
      srcset="https://huggingface.co/kandinskylab/KVAE-3D-2.0-t4s8/resolve/main/assets/kvae-black-2.0.png"
    >
    <img
      src="https://huggingface.co/kandinskylab/KVAE-3D-2.0-t4s8/resolve/main/assets/kvae-black-2.0.png"
      alt="KVAE 2.0 logo"
      width="600"
    >
  </picture>

  <a href="https://github.com/kandinskylab/kvae">GitHub</a> |
  <a href="https://huggingface.co/papers/2608.05798">Technical report</a> |
  <a href="https://habr.com/ru/companies/sberbank/articles/1016814/">Habr article</a> |
  <a href="https://huggingface.co/kandinskylab/KVAE-3D-2.0-t4s16">KVAE-Video 2.0 t4s16</a>

</div>

# KVAE 2.0: Video tokenizer (t4s8)

KVAE-Video 2.0 t4s8 is the 4 x 8 x 8 variant of KVAE 2.0, a family of causal video tokenizers designed as latent representations for diffusion models. It compresses videos into continuous 16-channel latents and reconstructs them with high fidelity. Its fully convolutional architecture supports long videos through temporal block caching.

## Model zoo

| Model | Modality | Compression | Latent channels |
| --- | :---: | :---: | :---: |
| [KVAE-Image 2.0](https://huggingface.co/kandinskylab/KVAE-2D-2.0) | Image | 8 x 8 | 32 |
| [KVAE-Video 2.0 t4s8](https://huggingface.co/kandinskylab/KVAE-3D-2.0-t4s8) | Video | 4 x 8 x 8 | 16 |
| [KVAE-Video 2.0 t4s16](https://huggingface.co/kandinskylab/KVAE-3D-2.0-t4s16) | Video | 4 x 16 x 16 | 64 |

## Inference

Run from the [KVAE source repository](https://github.com/kandinskylab/kvae) root. The reference environment uses Python 3.11, PyTorch 2.8.0, and CUDA 12.8.

```bash
pip install -r requirements.txt
pip install --editable .
```

```python
import torch

from data import VideoReader
from kvae import KVAEVideo

device = torch.device("cuda:0")
dtype = torch.bfloat16

model = (
    KVAEVideo.from_pretrained("kandinskylab/KVAE-3D-2.0-t4s8").eval().to(device=device, dtype=dtype)
)
reader = VideoReader(stream_pattern="*.png", input_norm="m11")
video = reader.read_video("path/to/video_frames")["frames"].unsqueeze(0)
video = video.to(device=device, dtype=dtype)

with torch.no_grad():
    latent = model.encode(video, seg_len=16).latent_dist.mode()
    reconstruction = model.decode(latent, seg_len=16).clip(-1, 1)
```

Temporal segments are processed through internal block caches. Do not interleave independent videos on the same model instance; use one `KVAEVideo` instance per concurrent stream.

## Evaluation

Reconstruction was evaluated on MCL-JCV at 720p and BVI-DVC. The compact table below reports MCL-JCV; all compared models use 4 x 8 x 8 compression with 16 latent channels.

| Model | PSNR↑ | SSIM↑ | LPIPS↓ |
| --- | ---: | ---: | ---: |
| HunyuanVideo 1.0 | 34.3 | 0.90 | 0.047 |
| Wan 2.1 | 34.3 | 0.89 | **0.044** |
| **KVAE-Video 2.0 t4s8** | **36.0** | **0.92** | 0.047 |

<details>
<summary><b>Show reconstruction figures</b></summary>

### Qualitative comparison

Columns from left to right: original video, KVAE-Video 2.0 t4s8, HunyuanVideo 1.0, and Wan 2.1.

<img src="assets/kvae3d-20-comparison-s8-artifacts-example.png" alt="Video reconstruction comparison at 4 x 8 x 8 compression" />

</details>

## Citation

```bibtex
@misc{kvae20,
  title         = {KVAE 2.0: video tokenizers for Image & Video generation models},
  author        = {Andrey Shutkin, Denis Parkhomenko, Kirill Chernyshev,
                  Ivan Kirillov, Denis Dimitrov,
                  Valeriya Kobenko, Kirill Malakhov},
  year          = {2026},
  eprint        = {2608.05798},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.05798}
}
```