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
      srcset="https://huggingface.co/kandinskylab/KVAE-3D-2.0-t4s16/resolve/main/assets/kvae-white-2.0.png"
    >
    <source
      media="(prefers-color-scheme: light)"
      srcset="https://huggingface.co/kandinskylab/KVAE-3D-2.0-t4s16/resolve/main/assets/kvae-black-2.0.png"
    >
    <img
      src="https://huggingface.co/kandinskylab/KVAE-3D-2.0-t4s16/resolve/main/assets/kvae-black-2.0.png"
      alt="KVAE 2.0 logo"
      width="600"
    >
  </picture>

  <a href="https://github.com/kandinskylab/kvae">GitHub</a> |
  <a href="https://huggingface.co/papers/2608.05798">Technical report</a> |
  <a href="https://habr.com/ru/companies/sberbank/articles/1016814/">Habr article</a> |
  <a href="https://huggingface.co/kandinskylab/KVAE-3D-2.0-t4s8">KVAE-Video 2.0 t4s8</a>

</div>

# KVAE 2.0: Video tokenizer (t4s16)

KVAE-Video 2.0 t4s16 is the higher-compression 4 x 16 x 16 video tokenizer in the KVAE 2.0 family. It is designed to provide the latent space for diffusion models, encoding videos into continuous 64-channel representations and reconstructing them with high fidelity. Its causal, fully convolutional architecture supports long videos through segment-wise processing with temporal caching.

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
    KVAEVideo.from_pretrained("kandinskylab/KVAE-3D-2.0-t4s16").eval().to(device=device, dtype=dtype)
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

Reconstruction was evaluated on MCL-JCV at 720p and BVI-DVC. The compact table below reports MCL-JCV; all compared models use 4 x 16 x 16 compression. HunyuanVideo 1.5 uses tiling with its default parameters because of its full attention block.

| Model | PSNR↑ | SSIM↑ | LPIPS↓ |
| --- | ---: | ---: | ---: |
| HunyuanVideo 1.5 | 34.4 | 0.89 | 0.073 |
| Wan 2.2 | 34.2 | 0.89 | **0.037** |
| **KVAE-Video 2.0 t4s16** | **35.4** | **0.91** | 0.048 |

The latent spaces of KVAE-Video 2.0 t4s16 and HunyuanVideo 1.5 were also compared through side-by-side human evaluation under a fixed generative-model setup. Against HunyuanVideo 1.5, KVAE-Video 2.0 t4s16 achieved reported win rates of 56% for prompt adherence, 54% for visual quality, and 55% for semantic quality.


<details>
<summary><b>Show reconstruction and human-evaluation figures</b></summary>

### Qualitative comparison

Columns from left to right: original video, KVAE-Video 2.0 t4s16, HunyuanVideo 1.5, and Wan 2.2.

<img src="assets/kvae3d-20-comparison-s16-artifacts-example.png" alt="Video reconstruction comparison at 4 x 16 x 16 compression" />

### Latent-space quality for generation

<img src="assets/kvae3d-20-latent-space-qualities-bars.png" alt="Human side-by-side evaluation of KVAE-Video 2.0 and HunyuanVideo 1.5" />

</details>

## Citation

```bibtex
@misc{kvae2026,
  title         = {KVAE: Family of Tokenizers for Multimodal Generative Models},
  author        = {Andrey Shutkin, Denis Parkhomenko, Ivan Kirillov,
                   Kirill Chernyshev, Kirill Malakhov, Ilia Vasiliev,
                   Ilia Trushkin, Valeriya Kobenko, David Chikovani,
                   Alexander Ivanov, Azat Saginbaev, Egor Silvestrov,
                   Ivan Mikheev, Konstantin Zakharov},
  year          = {2026},
  eprint        = {2608.05798},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.05798}
}
```