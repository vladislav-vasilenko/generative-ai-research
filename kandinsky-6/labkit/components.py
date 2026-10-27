"""Small CPU teaching utilities and explicit access to pinned K6 source.

No pretrained weights, network, CUDA application imports or global torch patches.
Native DiT uses the existing local dense-SDPA adapter; algorithm bodies are unmodified.
"""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import sys
import types
from datetime import datetime, timezone

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reference/kandinsky-6/kandinsky'
HF = ROOT / 'reference/hf/Kandinsky-6.0-Lite-distill-5s-Diffusers'
ARTIFACTS = ROOT / 'artifacts/component_labs'


def config(component='transformer'):
    return json.loads((HF / component / 'config.json').read_text())


def native(relative='core.components.dit'):
    """Load only the tensor-only modules used in these labs, in an isolated package."""
    from .torch_reference import load_class
    load_class()
    namespace = '_k6_eager_reference'
    allowed = {'core.components.dit', 'core.components.rope', 'core.components.piflow_dit',
               'core.tensors', 'core.types'}
    if relative not in allowed and not relative.startswith('core.algo.'):
        raise ValueError('Only pinned core tensor components/algorithms are supported')
    for suffix in ('core', 'core.components', 'core.algo'):
        name = namespace + '.' + suffix
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        sys.modules[name].__path__ = [str(SOURCE.joinpath(*suffix.split('.')))]
    return importlib.import_module(namespace + '.' + relative)


def show_source(relative, symbol=None):
    """Print the actual definition; no imports or heavyweight initialization."""
    path = SOURCE / relative
    source = path.read_text()
    if symbol:
        tree = ast.parse(source)
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == symbol)
        print('\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]))
    else:
        print(source)


def native_causal_conv():
    """Execute only the pinned causal-conv class, avoiding the Diffusers VAE imports.

    This is the original class definition, not a full loaded Hunyuan VAE.
    """
    path = SOURCE / 'core/components/hunyuan_vae.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'HunyuanVideoCausalConv3d')
    code = 'from __future__ import annotations\n' + '\n'.join(path.read_text().splitlines()[node.lineno-1:node.end_lineno])
    scope = {'torch': torch, 'nn': nn, 'F': F, '__name__': '_k6_causal_conv_definition'}
    exec(compile(code, str(path), 'exec'), scope)
    return scope[node.name]


def tiny_model(grid=None):
    """Random native 2-block K6, not a released Lite checkpoint."""
    from .torch_reference import tiny_config
    kwargs = tiny_config()
    kwargs.update(in_audio_dim=4, out_audio_dim=4,
                  visual_token_type_num_embeddings=2)
    if grid:
        cls = native('core.components.piflow_dit').PiFlowDiffusionTransformer3D
        return cls(n_grid=grid, **kwargs)
    return native().DiffusionTransformer3D(**kwargs)


def tiny_inputs(model, batch=1):
    # Physical state: video [B,2,4,4,4], audio [B,6,4]. Input carries 4+4+1 channels.
    video = torch.randn(batch, 2, 4, 4, 4)
    audio = torch.randn(batch, 6, 4)
    kwargs = dict(x_video=torch.cat((video, torch.zeros_like(video), torch.zeros_like(video[..., :1])), -1),
                  x_audio=audio, text_embed=torch.randn(batch, 5, 12),
                  pooled_text_embed=torch.randn(batch, 8), time=torch.full((batch,), 500.),
                  visual_rope=model.visual_rope((2, 2, 2), [torch.arange(2)] * 3),
                  audio_rope=model.audio_rope(torch.arange(6)),
                  text_rope=[model.video_text_rope(torch.arange(5)), model.audio_text_rope(torch.arange(5))],
                  attention_mask=torch.ones(batch, 5, dtype=torch.bool))
    return video, audio, kwargs


def activate_gates(model, std=0.08):
    """Nonzero random gates for sensitivity probes; deliberately not pretrained weights."""
    cls = native().Modulation
    for module in model.modules():
        if isinstance(module, cls):
            nn.init.normal_(module.out_layer.weight, std=std)
            nn.init.normal_(module.out_layer.bias, std=std)


def fresh_forward(model, kwargs):
    # Projection cache is an inference optimization. Training must not reuse its autograd graph.
    model.clear_text_proj_cache()
    return model(**kwargs)


def write_metrics(name, metrics):
    folder = ARTIFACTS / name
    folder.mkdir(parents=True, exist_ok=True)
    data = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'kind': 'CPU educational / random weights or analytic controls',
            'pretrained_inference': False, 'metrics': metrics}
    path = folder / 'metrics.json'
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    return path


def figure_path(name, filename):
    folder = ARTIFACTS / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (filename + '.png')
