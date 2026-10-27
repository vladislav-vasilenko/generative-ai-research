---
library_name: kvae
license: apache-2.0
tags:
  - pytorch
  - model_hub_mixin
  - pytorch_model_hub_mixin
  - image-tokenizer
  - variational-autoencoder
  - diffusion-models
---

<div align="center">

  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="https://huggingface.co/kandinskylab/KVAE-2D-2.0/resolve/main/assets/kvae-white-2.0.png"
    >
    <source
      media="(prefers-color-scheme: light)"
      srcset="https://huggingface.co/kandinskylab/KVAE-2D-2.0/resolve/main/assets/kvae-black-2.0.png"
    >
    <img
      src="https://huggingface.co/kandinskylab/KVAE-2D-2.0/resolve/main/assets/kvae-black-2.0.png"
      alt="KVAE 2.0 logo"
      width="600"
    >
  </picture>

  <a href="https://github.com/kandinskylab/kvae">GitHub</a> |
  <a href="https://huggingface.co/papers/2608.05798">Technical report</a> 
</div>

# KVAE 2.0: Image tokenizer

KVAE-Image 2.0 is the image tokenizer from KVAE 2.0, a family of image and video tokenizers designed to provide latent representations for diffusion models. It compresses RGB images by 8 x 8 into continuous 32-channel latents and reconstructs them with high fidelity.

## Model zoo

| Model | Modality | Compression | Latent channels |
| --- | :---: | :---: | :---: |
| [KVAE-Image 2.0](https://huggingface.co/kandinskylab/KVAE-2D-2.0) | Image | 8 x 8 | 32 |
| [KVAE-Video 2.0 t4s8](https://huggingface.co/kandinskylab/KVAE-3D-2.0-t4s8) | Video | 4 x 8 x 8 | 16 |
| [KVAE-Video 2.0 t4s16](https://huggingface.co/kandinskylab/KVAE-3D-2.0-t4s16) | Video | 4 x 16 x 16 | 64 |

## Versions

A new revision `v2.1` is available and `main` points to it (`from_pretrained(..., revision="v2.0")` for the initial release); reconstruction quality on COCO val2017 (5000 images, native resolution, center crop to a multiple of 32, bf16):

| Model | PSNR | SSIM | LPIPS |
| --- | :---: | :---: | :---: |
| KVAE-Image 2.1 | 32.15 | 0.9057 | 0.0264 |
| KVAE-Image 2.0 | 32.28 | 0.9121 | 0.0318 |
| FLUX.2 | 31.37 | 0.8924 | 0.0167 |
| FLUX.1 | 29.89 | 0.8662 | 0.0234 |

In a side-by-side human evaluation of text-to-image generations, KVAE-Image 2.1 is preferred over 2.0 with an overall win rate of 54/46, driven by prompt following (56/44) and semantic quality (56/44), with visual quality on par (51/49).

Note: `v2.0` and `v2.1` have different encoders, so their latent spaces are not compatible.

## Inference

Run from the [KVAE source repository](https://github.com/kandinskylab/kvae) root. The reference environment uses Python 3.11, PyTorch 2.8.0, and CUDA 12.8.

```bash
pip install -r requirements.txt
pip install --editable .
```

```python
import torch

from data import read_image
from kvae import KVAEImage

device = torch.device("cuda:0")
dtype = torch.bfloat16

model = KVAEImage.from_pretrained("kandinskylab/KVAE-2D-2.0").eval().to(device=device, dtype=dtype)

image = read_image("path/to/image.png").unsqueeze(0)
image = image.to(device=device, dtype=dtype)

with torch.no_grad():
    latent = model.encode(image).latent_dist.mode()
    reconstruction = model.decode(latent).clip(-1, 1)
```

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
