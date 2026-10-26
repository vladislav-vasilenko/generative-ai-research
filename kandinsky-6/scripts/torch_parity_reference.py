"""Generate a PyTorch numerical reference; never label random fixtures pretrained."""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from safetensors.torch import save_file
from labkit.torch_reference import load_class,tiny_config,audio_key,load_pretrained_audio,forward_audio

parser=argparse.ArgumentParser();parser.add_argument('--pretrained',action='store_true')
parser.add_argument('--weights',type=Path);args=parser.parse_args()
torch.set_num_threads(4);torch.manual_seed(1234)
out=ROOT/'artifacts/parity';out.mkdir(exist_ok=True,parents=True)
if args.pretrained:
 config=json.loads((ROOT/'reference/hf/Kandinsky-6.0-Lite-distill-5s-Diffusers/transformer/config.json').read_text())
 weights=args.weights or ROOT/'artifacts/weights/k6-lite-distill-audio.safetensors'
 model=load_pretrained_audio(weights,config)
 tag='pretrained'
else:
 config=tiny_config();model=load_class()(**config).eval()
 # A zero-gated random initialization would hide attention-port mistakes.
 for name,param in model.named_parameters():
  if 'modulation.out_layer' in name:
   with torch.no_grad():param.normal_(0,0.04)
 weights=out/'fixture.safetensors'
 save_file({k:v.contiguous() for k,v in model.state_dict().items() if audio_key(k)},str(weights))
 tag='random_fixture'

x=torch.randn(1,12,config['in_audio_dim']);text=torch.randn(1,7,config['in_text_dim'])
pooled=torch.randn(1,config['in_text_dim2']);time_t=torch.tensor([532.],dtype=torch.float32)
mask=torch.tensor([[True,True,True,True,True,False,False]])
start=time.perf_counter()
with torch.no_grad():
 expected=forward_audio(model,x,text,pooled,time_t,mask)
elapsed=time.perf_counter()-start
assert torch.isfinite(expected).all()
np.savez(out/f'{tag}.npz',x=x.numpy(),text=text.numpy(),pooled=pooled.numpy(),time=time_t.numpy(),mask=mask.numpy(),expected=expected.numpy())
(out/f'{tag}.json').write_text(json.dumps({'config':config,'weights':str(weights.resolve()),
 'pretrained_weights':args.pretrained,'parameter_count':sum(p.numel() for n,p in model.named_parameters() if audio_key(n)),
 'reference_backend':'PyTorch CPU FP32 + dense SDPA','reference_seconds':elapsed,
 'output_shape':list(expected.shape),'torch':torch.__version__},indent=2)+'\n')
print('Reference:',tag,'shape:',tuple(expected.shape),'seconds:',round(elapsed,3),'params:',sum(p.numel() for n,p in model.named_parameters() if audio_key(n)),flush=True)
