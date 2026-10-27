---
library_name: kvae
license: mit
tags:
  - pytorch
  - model_hub_mixin
  - pytorch_model_hub_mixin
  - image-tokenizer
  - variational-autoencoder
  - image-reconstruction
  - diffusion-models
---

<div align="center">

  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="https://huggingface.co/kandinskylab/KVAE-2D-1.0/resolve/main/assets/kvae-white.png"
    >
    <source
      media="(prefers-color-scheme: light)"
      srcset="https://huggingface.co/kandinskylab/KVAE-2D-1.0/resolve/main/assets/kvae-black.png"
    >
    <img
      src="https://huggingface.co/kandinskylab/KVAE-2D-1.0/resolve/main/assets/kvae-black.png"
      alt="KVAE 1.0 logo"
      width="600"
    >
  </picture>

  <a href="https://github.com/kandinskylab/kvae">GitHub</a> |
  <a href="https://habr.com/ru/companies/sberbank/articles/966450/">Habr article</a> |
  <a href="https://huggingface.co/kandinskylab/KVAE-3D-1.0">KVAE-Video 1.0</a>

</div>

# KVAE 1.0: Image tokenizer

KVAE-Image 1.0 is the image tokenizer from KVAE 1.0, a family of image and video tokenizers designed for latent diffusion models. It compresses RGB images into compact continuous latent representations and reconstructs them with high fidelity.

## Model zoo

| Model | Modality | Compression | Latent channels |
| --- | :---: | :---: | :---: |
| [KVAE-Image 1.0](https://huggingface.co/kandinskylab/KVAE-2D-1.0) | Image | 8 x 8 | 16 |
| [KVAE-Video 1.0](https://huggingface.co/kandinskylab/KVAE-3D-1.0) | Video | 4 x 8 x 8 | 16 |

## Inference

Run from the [KVAE source repository](https://github.com/kandinskylab/kvae) root. The reference environment uses Python 3.11, PyTorch 2.8.0, and CUDA 12.8.

```bash
pip install -r requirements.txt
```

```python
import torch

from data import read_image
from kvae import KVAEImage

device = torch.device("cuda:0")
dtype = torch.bfloat16

model = KVAEImage.from_pretrained("kandinskylab/KVAE-2D-1.0").eval().to(device=device, dtype=dtype)
image = read_image("assets/image_test/0002.png").unsqueeze(0)
image = image.to(device=device, dtype=dtype)

with torch.no_grad():
    latent = model.encode(image).latent_dist.mode()
    reconstruction = model.decode(latent).clip(-1, 1)
```

## Evaluation

Reconstruction was evaluated on the validation splits of [ImageNet-256](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256) and [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/). All compared models use 8 x 8 spatial compression and 16 latent channels.

| Dataset | Model | PSNR↑ | SSIM↑ | LPIPS↓ | rFID↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| ImageNet-256 (val) | Wan 2.1 | 29.03 | 0.85 | 0.069 | 0.62 |
| ImageNet-256 (val) | FLUX | 31.11 | **0.91** | **0.041** | **0.11** |
| ImageNet-256 (val) | **KVAE-Image 1.0** | **31.71** | **0.91** | 0.054 | 0.46 |
| DIV2K | Wan 2.1 | 31.87 | 0.89 | 0.069 | — |
| DIV2K | FLUX | 32.64 | 0.91 | 0.061 | — |
| DIV2K | **KVAE-Image 1.0** | **33.67** | **0.92** | **0.060** | — |

<details>
<summary><b>Show reconstruction and DiT training figures</b></summary>

### Reconstruction comparison

Columns from left to right: original image, KVAE-Image 1.0 reconstruction, and FLUX VAE reconstruction.

<img src="assets/kvae2d-comparison.png" alt="Image reconstruction comparison of the original, KVAE-Image 1.0, and FLUX VAE" />

### DiT training metrics

CLIP score and FID during DiT training: blue is `DiT + FLUX`; green and red are two versions of `DiT + KVAE-Image 1.0`.

<img src="assets/kvae2d-charts.png" alt="CLIP score and FID during DiT training with FLUX and KVAE-Image 1.0" />

</details>

## Citation

```bibtex
@misc{kvae_1_2025,
  author       = {Kirill Chernyshev, Andrey Shutkin, Ilia Vasiliev,
                  Denis Parkhomenko, Ivan Kirillov,
                  Dmitrii Mikhailov, Denis Dimitrov},
  title        = {KVAE 1.0: Image and Video Tokenizers for Image and Video Generation Models},
  howpublished = {\url{https://github.com/kandinskylab/kvae}},
  year         = {2025}
}
```