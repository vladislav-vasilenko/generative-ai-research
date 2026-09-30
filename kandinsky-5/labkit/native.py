"""Изолированный eager-порт официального DiT для учебных CPU/MPS экспериментов.

Без изменения оригинала и без monkey patch глобального torch.
Архитектура/имена параметров сохранены; CUDA autocast/compile удалены через AST.
Данный путь НЕ является официально поддержанным MPS-пайплайном.
"""
import ast
import sys
import types
from .common import SOURCE

class EagerTransform(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list = [d for d in node.decorator_list
                               if not ast.unparse(d).startswith(('torch.compile(', 'torch.autocast('))]
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if ast.unparse(node) == 'torch.bfloat16':
            return ast.copy_location(ast.Name(id='KD_DTYPE', ctx=ast.Load()), node)
        return self.generic_visit(node)

def load_dit_class(dtype=None, query_chunk=128):
    import torch
    import torch.nn.functional as F
    dtype = dtype or torch.float32
    # Разные настройки получают отдельные модули; ранее созданные модели не меняются.
    name = f'_k5_lab_{str(dtype).split(".")[-1]}_{query_chunk}'
    if name+'.dit' in sys.modules: return sys.modules[name+'.dit'].DiffusionTransformer3D
    package = types.ModuleType(name); package.__path__ = []
    sys.modules[name] = package

    def sdpa(q, k, v, attn_mask=None):
        q, k, v = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
        # Делим только запросы; каждый запрос по-прежнему видит ВСЕ ключи.
        # Маска padding [B,L] транслируется на heads и query.
        if attn_mask is not None and attn_mask.ndim == 2:
            attn_mask = attn_mask[:, None, None, :]
        chunks = []
        for start in range(0, q.shape[2], query_chunk):
            mask = attn_mask
            if mask is not None and mask.shape[-2] > 1:
                mask = mask[..., start:start+query_chunk, :]
            chunks.append(F.scaled_dot_product_attention(q[:,:,start:start+query_chunk], k, v, attn_mask=mask))
        return torch.cat(chunks, dim=2).transpose(1, 2).contiguous()

    class SelfAttentionEngine:
        def __init__(self, engine='sdpa'):
            if engine not in ('sdpa', 'auto'): raise ValueError('Учебный порт поддерживает только dense SDPA')
        def get_attention(self): return sdpa

    attention = types.ModuleType(name+'.attention')
    attention.SelfAttentionEngine = SelfAttentionEngine
    sys.modules[attention.__name__] = attention
    for filename in ('utils', 'nn', 'dit'):
        path = SOURCE / 'kandinsky' / 'models' / (filename+'.py')
        tree = EagerTransform().visit(ast.parse(path.read_text()))
        ast.fix_missing_locations(tree)
        module = types.ModuleType(name+'.'+filename)
        module.__package__ = name
        module.KD_DTYPE = dtype
        sys.modules[module.__name__] = module
        exec(compile(tree, str(path), 'exec'), module.__dict__)
    return sys.modules[name+'.dit'].DiffusionTransformer3D

def tiny_dit(device='cpu'):
    import torch
    cls = load_dit_class(torch.float32)
    return cls(in_visual_dim=16, out_visual_dim=16, in_text_dim=32, in_text_dim2=16,
               model_dim=48, time_dim=32, ff_dim=96, axes_dims=(4,4,4),
               num_text_blocks=1, num_visual_blocks=2, patch_size=(1,2,2),
               visual_cond=True, attention_engine='sdpa', text_token_padding=True).to(device).eval()

def real_dit(weights, device='cpu', dtype=None, conf_path=None):
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    dtype = dtype or torch.bfloat16
    conf = OmegaConf.load(conf_path or SOURCE / 'configs' / 'k5_lite_t2v_5s_distil_sd.yaml')
    cls = load_dit_class(dtype)
    kwargs = OmegaConf.to_container(conf.model.dit_params)
    kwargs.update(attention_engine='sdpa', text_token_padding=True)
    with torch.device('meta'): model = cls(**kwargs)
    state = load_file(str(weights), device='cpu')
    model.load_state_dict(state, strict=True, assign=True)
    del state
    # Nonpersistent RoPE/frequency buffers aren't stored in checkpoints.
    # Recreate only those small buffers from a meta model structure.
    from .common import release
    for module in model.modules():
        if hasattr(module, 'max_period') and hasattr(module, 'model_dim') and hasattr(module, 'freqs'):
            module.freqs = torch.exp(-__import__('math').log(module.max_period)*torch.arange(module.model_dim//2,dtype=torch.float32)/(module.model_dim//2))
        if hasattr(module, 'max_pos') and hasattr(module, 'axes_dims'):
            for i, (dim, limit) in enumerate(zip(module.axes_dims, module.max_pos)):
                freqs=torch.exp(-__import__('math').log(module.max_period)*torch.arange(dim//2,dtype=torch.float32)/(dim//2))
                setattr(module, f'args_{i}', torch.outer(torch.arange(limit,dtype=torch.float32),freqs))
        elif hasattr(module, 'args') and module.args.device.type == 'meta':
            # RoPE1D: max_pos and dimension recoverable from original buffer shape.
            limit, half_dim = module.args.shape
            freqs=torch.exp(-__import__('math').log(10000.)*torch.arange(half_dim,dtype=torch.float32)/half_dim)
            module.args=torch.outer(torch.arange(limit,dtype=torch.float32),freqs)
    release()
    # Keep positional/frequency buffers FP32 as upstream does.
    buffers = [(module, key, value.float()) for module in model.modules() for key, value in module._buffers.items() if value is not None and value.is_floating_point()]
    model = model.to(device=device, dtype=dtype).eval()
    for module, key, value in buffers: module._buffers[key] = value.to(device)
    return model, conf
