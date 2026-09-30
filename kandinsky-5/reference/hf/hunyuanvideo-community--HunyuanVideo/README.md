---
base_model:
- tencent/HunyuanVideo
library_name: diffusers
---

Unofficial community fork for Diffusers-format weights on [`tencent/HunyuanVideo`](https://huggingface.co/tencent/HunyuanVideo).

### Using Diffusers

HunyuanVideo can be used directly from Diffusers. Install the latest version of Diffusers.

```python
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)

# Enable memory savings
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
).frames[0]
export_to_video(output, "output.mp4", fps=15)
```

Refer to the [documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/hunyuan_video) for more information.

