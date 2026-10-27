"""Load unmodified K6 component code with a local dense SDPA backend.

The CUDA-oriented application and its package-level imports are not imported.
No global torch functions are patched. This is an eager reference for parity,
not the released end-to-end deployment pipeline.
"""
from pathlib import Path
import importlib.util
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reference/kandinsky-6/kandinsky'

def load_class():
    import torch.nn.functional as F
    name='_k6_eager_reference'
    if name+'.core.components.dit' in sys.modules:
        return sys.modules[name+'.core.components.dit'].DiffusionTransformer3D
    for suffix in ['', '.core', '.core.components', '.core.components.attention']:
        module=types.ModuleType(name+suffix);module.__path__=[]
        sys.modules[module.__name__]=module
    def sdpa(q,k,v,attn_mask=None):
        result=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=attn_mask)
        return result.transpose(1,2)
    class Engine:
        def __init__(self,engine='sdpa'):
            if engine not in ('auto','sdpa'):raise ValueError('Reference supports dense SDPA only')
        def get_attention(self):return sdpa
    dispatch=types.ModuleType(name+'.core.components.attention.dispatch')
    dispatch._sdpa=sdpa;dispatch.SelfAttentionEngine=Engine
    sys.modules[dispatch.__name__]=dispatch
    for suffix,file in [('.core.tensors','core/tensors.py'),('.core.components.rope','core/components/rope.py'),('.core.components.dit','core/components/dit.py')]:
        spec=importlib.util.spec_from_file_location(name+suffix,SOURCE/file)
        module=importlib.util.module_from_spec(spec);sys.modules[module.__name__]=module
        spec.loader.exec_module(module)
    return sys.modules[name+'.core.components.dit'].DiffusionTransformer3D

def audio_key(key):return key.startswith('audio_') or '.audio_dec_block.' in key

def tiny_config():
    return dict(in_visual_dim=4,out_visual_dim=4,in_text_dim=12,in_text_dim2=8,
                time_dim=16,patch_size=(1,2,2),model_dim=48,ff_dim=96,
                num_text_blocks=1,num_visual_blocks=2,axes_dims=(2,2,4),
                visual_cond=True,is_multimodal=True,in_audio_dim=4,out_audio_dim=40,
                model_dim_a=32,time_dim_a=16,ff_dim_a=64,axes_dims_a=(2,2,4),
                audio_freqs_scaling=0.144,text_token_padding=True,
                attention_engine='sdpa',ca_rope=True,cross_gates=True,fix_modulation=True)

def load_pretrained_audio(weights,config,dtype=None):
    import torch
    from safetensors import safe_open
    from .check_weights import assert_complete
    assert_complete(weights)
    cls=load_class();config=dict(config);config.pop('scale_factor',None)
    config['attention_engine']='sdpa'
    with torch.device('meta'):model=cls(**config)
    # Remove only modules that the native x_video=None branch never reads.
    for name in list(model._modules):
        if name!='visual_transformer_blocks' and not name.startswith('audio_'):delattr(model,name)
    for block in model.visual_transformer_blocks:
        for name in list(block._modules):
            if name!='audio_dec_block':delattr(block,name)
    # Map/read only audio tensors; do not create a second checkpoint on SSD.
    with safe_open(str(weights),framework='pt',device='cpu') as handle:
        state={k:handle.get_tensor(k) for k in handle.keys() if audio_key(k)}
    assert all(audio_key(k) for k in state)
    model.load_state_dict(state,strict=True,assign=True)
    # Nonpersistent position/frequency buffers are deliberately not in weights.
    get_freqs=sys.modules['_k6_eager_reference.core.tensors'].get_freqs
    for module in model.modules():
        if hasattr(module,'freqs'):
            module.freqs=get_freqs(module.freqs.shape[0])
        if hasattr(module,'args') and hasattr(module,'max_pos'):
            freq=get_freqs(module.dim//2,module.max_period)*module.freqs_scaling
            module.args=torch.outer(torch.arange(module.max_pos,dtype=torch.float32),freq)
    return model.to(dtype=dtype or torch.float32).eval().requires_grad_(False)

def forward_audio(model,x,text,pooled,time,mask=None):
    import torch
    model.clear_text_proj_cache()
    return model(x_video=None,x_audio=x,text_embed=text,pooled_text_embed=pooled,time=time,
                 visual_rope=None,audio_rope=model.audio_rope(torch.arange(x.shape[1],device=x.device)),
                 text_rope=model.audio_text_rope(torch.arange(text.shape[1],device=x.device)),
                 attention_mask=mask)
