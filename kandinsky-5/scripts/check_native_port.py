"""Meaningful component contract checks, no pretrained downloads."""
from pathlib import Path
import sys,tempfile,json
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
from safetensors.torch import save_file
from omegaconf import OmegaConf
from labkit.native import tiny_dit,real_dit,load_dit_class
from labkit.common import device_info
checks=[]
def inputs(device,dtype):
    return (torch.randn(1,4,4,33,device=device,dtype=dtype),torch.randn(1,6,32,device=device,dtype=dtype),
            torch.randn(1,16,device=device,dtype=dtype),torch.tensor([500.],device=device),
            [torch.arange(1,device=device),torch.arange(2,device=device),torch.arange(2,device=device)],torch.arange(6,device=device))
model=tiny_dit(); args=inputs('cpu',torch.float32)
out=model(*args); assert out.shape==(1,4,4,16) and torch.isfinite(out).all()
out.square().mean().backward(); assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
checks.append('CPU tiny DiT forward/backward')
# Same checkpoint keys, with a deliberately small configuration for strict round-trip.
cfg={'model':{'dit_params':{'in_visual_dim':16,'out_visual_dim':16,'in_text_dim':32,'in_text_dim2':16,
     'model_dim':48,'time_dim':32,'ff_dim':96,'axes_dims':[4,4,4],'num_text_blocks':1,'num_visual_blocks':2,
     'patch_size':[1,2,2],'visual_cond':True},'attention':{'type':'flash'}},'metrics':{'scale_factor':[1,1,1]}}
with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
    p=Path(temp); save_file(model.state_dict(),p/'tiny.safetensors'); OmegaConf.save(OmegaConf.create(cfg),p/'tiny.yaml')
    restored,_=real_dit(p/'tiny.safetensors',dtype=torch.float32,conf_path=p/'tiny.yaml')
    with torch.no_grad(): assert torch.allclose(model(*args),restored(*args),atol=1e-6)
    assert not any(b.is_meta for b in restored.buffers())
    checks.append('strict meta checkpoint load + regenerated nonpersistent buffers')
    bf16,_=real_dit(p/'tiny.safetensors',dtype=torch.bfloat16,conf_path=p/'tiny.yaml')
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        result=bf16(*inputs('cpu',torch.bfloat16))
    assert torch.isfinite(result).all(); assert all(b.dtype==torch.float32 for b in bf16.buffers())
    checks.append('BF16 eager forward with FP32 positional buffers')
# Query chunking equivalence under mask, using the same trained parameter values.
cls=load_dit_class(torch.float32,query_chunk=2)
chunked=cls(**cfg['model']['dit_params'],attention_engine='sdpa',text_token_padding=True)
chunked.load_state_dict(model.state_dict()); mask=torch.tensor([[True,True,True,False,False,False]])
with torch.no_grad(): assert torch.allclose(model(*args,attention_mask=mask),chunked(*args,attention_mask=mask),atol=1e-5)
checks.append('query chunk equivalence, padded bool mask')
if torch.backends.mps.is_available():
    for dtype in [torch.float32,torch.bfloat16]:
        m=load_dit_class(dtype)(**cfg['model']['dit_params'],attention_engine='sdpa',text_token_padding=True).to('mps',dtype)
        with torch.no_grad(), (nullcontext() if dtype==torch.float32 else torch.autocast('mps',dtype=dtype)): y=m(*inputs('mps',dtype))
        torch.mps.synchronize(); assert torch.isfinite(y).all()
        checks.append('MPS tiny DiT forward '+str(dtype))
report={'environment':device_info(),'passed':checks,'pretrained_weights_tested':False,'full_5s_inference_tested':False}
(ROOT/'reports/native-port-validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))
