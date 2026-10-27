"""Execute pinned generation functions with explicit, isolated test dependencies.

Original function bodies/decorators are preserved. CPU mode substitutes only
CUDA generator creation, CUDA autocast and empty_cache through a local facade.
Fake model components test orchestration, not pretrained image/video quality.
"""
import ast
from contextlib import nullcontext
import torch
import torchvision.transforms.functional as vision_F
from .common import SOURCE

GENERATION_FILE=SOURCE/'kandinsky/generation_utils.py'

class _CPUCache:
    def __init__(self,log): self.log=log
    def empty_cache(self): self.log.append(('cuda.empty_cache','CPU no-op'))
    def __getattr__(self,name): return getattr(torch.cuda,name)

class _CPUTorch:
    def __init__(self,log):
        self.log=log; self.cuda=_CPUCache(log)
    def __getattr__(self,name): return getattr(torch,name)
    def Generator(self,device='cpu'):
        self.log.append(('Generator',str(device),'cpu'))
        if str(device).startswith('cuda'): device='cpu'
        return torch.Generator(device=device)
    def autocast(self,device_type,**kwargs):
        if device_type=='cuda':
            self.log.append(('autocast','cuda','CPU nullcontext'))
            return nullcontext()
        return torch.autocast(device_type=device_type,**kwargs)

def load_generation_functions(cpu_cuda=False,quiet=True,adapter_log=None):
    """Return a new namespace; never patch torch or the installed Kandinsky package."""
    adapter_log=[] if adapter_log is None else adapter_log
    namespace={'torch':_CPUTorch(adapter_log) if cpu_cuda else torch,
               'F':vision_F,'Tensor':torch.Tensor,
               'all_gather':torch.distributed.all_gather}
    if quiet: namespace['tqdm']=lambda items: items
    else:
        from tqdm import tqdm
        namespace['tqdm']=tqdm
    utils_path=SOURCE/'kandinsky/models/utils.py'
    utils_ast=ast.parse(utils_path.read_text())
    mask=next(n for n in utils_ast.body if isinstance(n,ast.FunctionDef) and n.name=='fast_sta_nabla')
    exec(compile(ast.Module(body=[mask],type_ignores=[]),str(utils_path),'exec'),namespace)
    source_ast=ast.parse(GENERATION_FILE.read_text())
    nodes=[n for n in source_ast.body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(GENERATION_FILE),'exec'),namespace)
    namespace['_function_names']=[n.name for n in nodes]
    namespace['_adapter_log']=adapter_log
    namespace['_cpu_cuda']=cpu_cuda
    return namespace


"""Small instrumented components for exact generation_utils orchestration tests.

These components test tensor contracts and call order; they contain no learned
representation and do not approximate quality or latency of pretrained models.
"""
from types import SimpleNamespace

import torch
import torch.nn.functional as torch_f


def make_generation_conf(patch_size=(1, 1, 1)):
    return SimpleNamespace(
        model=SimpleNamespace(
            dit_params=SimpleNamespace(patch_size=patch_size),
            attention=SimpleNamespace(type="flash", wT=3, wH=3, wW=3, P=0.9, add_sta=True, method="topcdf"),
        ),
        metrics=SimpleNamespace(scale_factor=(1, 1, 1)),
    )


class FakeDiT:
    """Predict constant velocity; preserve snapshots of each DiT invocation."""

    def __init__(self, log=None, channels=2, visual_cond=False,
                 instruct_type=None, velocity=0.25):
        self.log = [] if log is None else log
        self.channels = channels
        self.visual_cond = visual_cond
        self.instruct_type = instruct_type
        self.velocity = velocity
        self.device = "cpu"
        self.calls = []

    def to(self, device, **kwargs):
        self.device = str(device)
        self.log.append(("dit.to", self.device))
        return self

    def __call__(self, x, text_embeds, pooled_embed, time,
                 visual_rope_pos, text_rope_pos, **kwargs):
        self.log.append(("dit.forward", tuple(x.shape)))
        self.calls.append(dict(
            x=x.detach().clone(), time=time.detach().clone(),
            text_embeds=text_embeds.detach().clone(),
            pooled_embed=pooled_embed.detach().clone(),
            visual_rope_pos=[p.detach().clone() for p in visual_rope_pos],
            text_rope_pos=text_rope_pos.detach().clone(), kwargs=kwargs,
        ))
        return torch.full_like(x[..., :self.channels], self.velocity)


class FakeTextEmbedder:
    """Emit deterministic fixed-width tensors while recording text/image routing."""

    def __init__(self, log=None):
        self.log = [] if log is None else log
        self.calls = []
        self.device = "cpu"
        self.embedder = SimpleNamespace(mode="t2v")

    def to(self, device, **kwargs):
        self.device = str(device)
        self.log.append(("text.to", self.device))
        return self

    def encode(self, captions, type_of_content="video", images=None):
        text = captions[0]
        length = 4 if text else 2
        self.log.append(("text.encode", type_of_content, text))
        self.calls.append(dict(
            captions=list(captions), type_of_content=type_of_content,
            images=None if images is None else [im.detach().clone() for im in images],
            mode=self.embedder.mode,
        ))
        value = 1.0 if text else 0.0
        embeds = {
            "text_embeds": torch.full((1, length, 4), value),
            "pooled_embed": torch.full((1, 4), value),
        }
        return embeds, torch.tensor([0, length]), torch.ones(1, length, dtype=torch.bool)


class FakeVAE:
    """Spatial factor 8 and video factor 4, implemented without learned weights.

    Encoding keeps the first two RGB channels after average pooling; decoding
    adds their mean as a third channel. A video decode expands T to 4(T-1)+1.
    The image variant exposes the image encode API (latent_dist.sample).
    """

    def __init__(self, log=None, image_vae=False, channels=2, scaling_factor=0.5):
        if channels != 2:
            raise ValueError("This teaching VAE is defined for two latent channels.")
        self.log = [] if log is None else log
        self.image_vae = image_vae
        self.config = SimpleNamespace(scaling_factor=scaling_factor)
        self.device = "cpu"
        self.encode_calls = []
        self.decode_calls = []
        self.encodings = []

    def to(self, device, **kwargs):
        self.device = str(device)
        self.log.append(("vae.to", self.device))
        return self

    def encode(self, x):
        self.log.append(("vae.encode", tuple(x.shape)))
        self.encode_calls.append(x.detach().clone())
        if x.ndim == 4:
            z = torch_f.avg_pool2d(x.float()[:, :2], 8).to(x.dtype)
        elif x.ndim == 5:
            b, c, t, h, w = x.shape
            flat = x[:, :2, ::4].permute(0, 2, 1, 3, 4)
            latent_t = flat.shape[1]
            z = torch_f.avg_pool2d(flat.reshape(b * latent_t, 2, h, w).float(), 8)
            z = z.reshape(b, latent_t, 2, h // 8, w // 8).permute(0, 2, 1, 3, 4).to(x.dtype)
        else:
            raise ValueError("encode expects BCHW or BCTHW")
        self.encodings.append(z.detach().clone())
        if self.image_vae:
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: z))
        return (z,)

    def decode(self, z):
        self.log.append(("vae.decode", tuple(z.shape)))
        self.decode_calls.append(z.detach().clone())
        if z.ndim == 4:
            rgb = torch.cat([z, z.mean(1, keepdim=True)], dim=1)
            rgb = torch_f.interpolate(rgb.float(), scale_factor=8, mode="nearest").to(z.dtype)
        elif z.ndim == 5:
            b, c, t, h, w = z.shape
            rgb = torch.cat([z, z.mean(1, keepdim=True)], dim=1)
            rgb = rgb.repeat_interleave(4, dim=2)[:, :, :4 * (t - 1) + 1]
            out_t = rgb.shape[2]
            rgb = torch_f.interpolate(
                rgb.permute(0, 2, 1, 3, 4).reshape(b * out_t, 3, h, w).float(),
                scale_factor=8, mode="nearest",
            ).reshape(b, out_t, 3, h * 8, w * 8).permute(0, 2, 1, 3, 4).to(z.dtype)
        else:
            raise ValueError("decode expects BCHW or BCTHW")
        return SimpleNamespace(sample=rgb)
