"""CPU experiments with pinned method bodies; explicit educational dependency adapters.

No downloads, checkpoint loading, global torch patches or CUDA allocations.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass
import hashlib
import json
from math import ceil, sqrt
from pathlib import Path
from types import SimpleNamespace
import time
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

COURSE=Path(__file__).resolve().parents[1]
ROOT=COURSE.parent
SOURCE=ROOT/'kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py'
SOURCE_SHA='f5e8dd402aba625f23aef86ab7d5324dc6a4e8585b72917a3bf27bd46745bbaf'
ARTIFACTS=COURSE/'artifacts'

def source_hash():
    value=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert value==SOURCE_SHA, 'Pinned source changed: review provenance before executing'
    return value

def show_source(name):
    source_hash();text=SOURCE.read_text();tree=ast.parse(text)
    n=next(n for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
    print(f'{SOURCE.relative_to(ROOT)}:{n.lineno}–{n.end_lineno}')
    print('\n'.join(text.splitlines()[n.lineno-1:n.end_lineno]))

class DiagonalPosterior:
    """Educational dependency adapter, NOT Diffusers' distribution implementation."""
    def __init__(self,moments):
        self.mean,self.logvar=moments.chunk(2,dim=1)
        self.logvar=self.logvar.clamp(-30,20)
        self.std=torch.exp(.5*self.logvar)
    def mode(self):return self.mean
    def sample(self,generator=None):
        return self.mean+self.std*torch.randn(self.mean.shape,device=self.mean.device,dtype=self.mean.dtype,generator=generator)
    def kl(self):
        return .5*(self.mean.square()+self.logvar.exp()-1-self.logvar).flatten(1).sum(1)

@dataclass
class EncoderOutput:
    latent_dist: DiagonalPosterior
    def __getitem__(self,i):
        if i!=0:raise IndexError(i)
        return self.latent_dist
@dataclass
class DecoderOutput:
    sample: torch.Tensor
    def __getitem__(self,i):
        if i!=0:raise IndexError(i)
        return self.sample

def activation(name):
    if name not in ('silu','swish'):raise ValueError('Only source-used SiLU supported')
    return nn.SiLU()

def unavailable_attention(*args,**kwargs):
    raise RuntimeError('Tiny native codec disables optional mid attention; use Diffusers for that path')

def native_scope(torch_backend=torch):
    """Execute verbatim AST definitions; remove only import-dependent hook decorators.

    Pure component bodies are unchanged. The VAE facade below uses educational
    Gaussian/output adapters and no mid Attention, not a complete Diffusers runtime.
    A local backend may simulate mem_get_info only; it never patches torch globally.
    """
    source_hash();tree=ast.parse(SOURCE.read_text())
    pure=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
          and getattr(n,'name','')!='AutoencoderKLHunyuanVideo']
    vae=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AutoencoderKLHunyuanVideo')
    methods=[n for n in vae.body if isinstance(n,ast.FunctionDef) and n.name!='__init__']
    for n in methods:n.decorator_list=[]
    facade=ast.ClassDef(name='NativeMethods',bases=[],keywords=[],body=methods,decorator_list=[])
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*pure,facade],type_ignores=[])
    ast.fix_missing_locations(module)
    scope=dict(torch=torch_backend,nn=nn,F=F,np=np,ceil=ceil,sqrt=sqrt,
               get_activation=activation,Attention=unavailable_attention,
               DiagonalGaussianDistribution=DiagonalPosterior,
               AutoencoderKLOutput=EncoderOutput,DecoderOutput=DecoderOutput,
               __name__='_hunyuan_vae_educational_scope')
    exec(compile(module,str(SOURCE),'exec'),scope)
    return SimpleNamespace(**scope)

NATIVE=native_scope()

class TinyCodec(NATIVE.NativeMethods):
    """Random original encoder/decoder classes, reduced widths, no mid attention.

    Retains four stages, temporal factor 4 and spatial factor 8. Never pretrained.
    """
    def __init__(self,seed=0):
        torch.manual_seed(seed)
        opts=dict(block_out_channels=(4,8,8,8),layers_per_block=1,norm_num_groups=2,
                  mid_block_add_attention=False)
        self.encoder=NATIVE.HunyuanVideoEncoder3D(in_channels=3,out_channels=2,**opts).eval()
        self.decoder=NATIVE.HunyuanVideoDecoder3D(in_channels=2,out_channels=3,**opts).eval()
        self.quant_conv=nn.Conv3d(4,4,1).eval()
        self.post_quant_conv=nn.Conv3d(2,2,1).eval()
        self.spatial_compression_ratio=8;self.temporal_compression_ratio=4
        self.use_tiling=False;self.use_framewise_encoding=False;self.use_framewise_decoding=False
        self.tile_sample_min_height=32;self.tile_sample_min_width=32
        self.tile_sample_stride_height=16;self.tile_sample_stride_width=16
        self.tile_sample_min_num_frames=16;self.tile_sample_stride_num_frames=8
        self.tile_size=None;self.config=SimpleNamespace(scaling_factor=.476986)
    def modules(self):return (self.encoder,self.decoder,self.quant_conv,self.post_quant_conv)
    def parameters(self):
        for m in self.modules():yield from m.parameters()
    def set_tiles(self,tile=(17,32,32),stride=(8,16,16),spatial=True,temporal=True):
        self.apply_tiling((1,*tile),stride)
        self.use_tiling=spatial;self.use_framewise_decoding=temporal
        return self

class GridEncoder(nn.Module):
    """Exact coordinate oracle for stitching, not a learned codec."""
    def forward(self,x):return x[:,:,::4,::8,::8]
class GridDecoder(nn.Module):
    """First-frame preserving nearest-neighbor oracle, no learned reconstruction."""
    def forward(self,z):
        z=torch.cat((z[:,:,:1],z[:,:,1:].repeat_interleave(4,2)),2)
        return z.repeat_interleave(8,3).repeat_interleave(8,4)

def oracle_codec():
    obj=TinyCodec.__new__(TinyCodec)
    obj.encoder=GridEncoder();obj.decoder=GridDecoder()
    obj.quant_conv=nn.Identity();obj.post_quant_conv=nn.Identity()
    obj.spatial_compression_ratio=8;obj.temporal_compression_ratio=4
    obj.use_tiling=True;obj.use_framewise_decoding=True;obj.tile_size=None
    obj.apply_tiling((1,17,32,32),(8,16,16))
    return obj

class TorchBudgetFacade:
    """Local namespace adapter for ORIGINAL planner, synthetic CUDA budget only."""
    def __init__(self,free_bytes):
        self.cuda=SimpleNamespace(mem_get_info=lambda device=None:(free_bytes,free_bytes))
    def __getattr__(self,name):return getattr(torch,name)

def plan(shape,free_gib=None,device='cpu'):
    backend=torch if free_gib is None else TorchBudgetFacade(free_gib*2**30)
    cls=native_scope(backend).NativeMethods
    # Supply device explicitly: no parameter or actual CUDA device is needed here.
    return cls().get_enc_optimal_tiling(shape,device=device)

def axis_coverage(length,tile,stride):
    if tile<=0 or stride<=0:return dict(starts=[],covered=0,missing=length,valid=False)
    starts=list(range(0,length-tile+1,stride))
    covered=starts[-1]+tile if starts else 0
    return dict(starts=starts,covered=covered,missing=length-covered,valid=bool(starts) and covered==length)

def anchored_starts(length,tile,stride,align=8):
    """Explicit teaching alternative: anchor last full tile, keep grid alignment."""
    if length%align or tile%align or stride%align:
        raise ValueError('Pad to aligned dimensions first; do not silently shift latent phase')
    if not 0<stride<=tile<=length:raise ValueError('Require 0 < stride <= tile <= length')
    result=list(range(0,length-tile+1,stride))
    if result[-1]+tile<length:result.append(length-tile)
    return result

def synthetic_video(frames=17,height=48,width=48,seed=0):
    g=torch.Generator().manual_seed(seed)
    t=torch.linspace(0,1,frames)[None,None,:,None,None]
    y=torch.linspace(-1,1,height)[None,None,None,:,None]
    x=torch.linspace(-1,1,width)[None,None,None,None,:]
    wave=torch.sin(8*(x+t))*torch.cos(6*(y-t))
    return (wave.expand(1,3,frames,height,width)+.02*torch.randn(1,3,frames,height,width,generator=g)).clamp(-1,1)

def errors(a,b,data_range=2.):
    if a.shape!=b.shape:raise ValueError(f'Shape mismatch: {a.shape} vs {b.shape}')
    e=(a.float()-b.float()).square();mse=float(e.mean())
    return dict(mse=mse,max_abs=float((a-b).abs().max()),
                psnr_db=float(10*np.log10(data_range**2/max(mse,1e-30))))

def measure(fn,repeats=3):
    with torch.inference_mode():
        fn() # warmup excluded
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter();value=fn();samples.append(time.perf_counter()-start)
    return value,dict(seconds_median=float(np.median(samples)),seconds_samples=samples,
                      device='cpu',memory_metric='not measured; CPU time is not VRAM')

def save_metrics(lab,metrics):
    out=ARTIFACTS/lab;out.mkdir(parents=True,exist_ok=True)
    result=dict(lab=lab,source_sha256=source_hash(),torch_version=torch.__version__,
                evidence='CPU; synthetic inputs; random tiny weights or exact coordinate oracle',**metrics)
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result

def figure_path(lab,name):
    path=ARTIFACTS/lab;path.mkdir(parents=True,exist_ok=True)
    return path/(name+'.png')
