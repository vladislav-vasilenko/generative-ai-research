---
library_name: kvae
license: mit
tags:
  - pytorch
  - model_hub_mixin
  - pytorch_model_hub_mixin
  - video-tokenizer
  - variational-autoencoder
  - video-reconstruction
  - diffusion-models
---

<div align="center">

  <picture>
    <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="https://huggingface.co/kandinskylab/KVAE-3D-1.0/resolve/main/assets/kvae-white.png"
    >
    <source
      media="(prefers-color-scheme: light)"
      srcset="https://huggingface.co/kandinskylab/KVAE-3D-1.0/resolve/main/assets/kvae-black.png"
    >
    <img
      src="https://huggingface.co/kandinskylab/KVAE-3D-1.0/resolve/main/assets/kvae-black.png"
      alt="KVAE 1.0 logo"
      width="600"
    >
  </picture>
  </picture>

  <a href="https://github.com/kandinskylab/kvae">GitHub</a> |
  <a href="https://habr.com/ru/companies/sberbank/articles/966450/">Habr article</a> |
  <a href="https://huggingface.co/kandinskylab/KVAE-2D-1.0">KVAE-Image 1.0</a>

</div>

# KVAE 1.0: Video tokenizer

KVAE-Video 1.0 is the video tokenizer from KVAE 1.0, a family of image and video tokenizers designed for latent diffusion models. Its causal, fully convolutional architecture encodes videos into compact continuous latent representations and reconstructs them with high fidelity.

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

from data import VideoReader
from kvae import KVAEVideo

device = torch.device("cuda:0")
dtype = torch.bfloat16

model = (
    KVAEVideo.from_pretrained("kandinskylab/KVAE-3D-1.0")
    .eval()
    .to(device=device, dtype=dtype)
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

Reconstruction was evaluated on [MCL-JCV](https://mcl.usc.edu/mcl-jcv-dataset/) downsampled to 540p because of the model's limitations at high resolutions. All compared models use 4 x 8 x 8 compression with 16 latent channels.

| Model | PSNR↑ | SSIM↑ | LPIPS↓ |
| --- | ---: | ---: | ---: |
| Wan 2.1 | 33.75 | 0.90 | 0.089 |
| HunyuanVideo 1.0 | 33.91 | 0.91 | 0.103 |
| **KVAE-Video 1.0** | **35.63** | **0.92** | **0.088** |

<details>
<summary><b>Show reconstruction figures</b></summary>


### Qualitative comparison

Columns from left to right: original video, KVAE-Video 1.0 reconstruction, and HunyuanVideo 1.0 reconstruction.

<img src="assets/kvae3d-comparison.png" alt="Video reconstruction comparison of the original, KVAE-Video 1.0, and HunyuanVideo 1.0" />

</details>

## Citation

```bibtex
@misc{kvae_1_2025,
  author       = {Kirill Chernyshev and Andrey Shutkin and Ilia Vasiliev and Denis Parkhomenko and Ivan Kirillov and Dmitrii Mikhailov and Denis Dimitrov},
  title        = {KVAE 1.0: Image and Video Tokenizers for Image and Video Generation Models},
  howpublished = {\url{https://github.com/kandinskylab/kvae}},
  year         = {2025}
}
```

## License

[MIT](LICENSE.txt)
