"""Pinned K5/K6 CPU teaching tools, explicit SDPA/precision adapters, no pretrained weights."""
from __future__ import annotations
import ast,copy,hashlib,importlib,importlib.util,json,sys,types
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

COURSE=Path(__file__).resolve().parents[1]
ROOT=COURSE.parent
K6=ROOT/'kandinsky-6/reference/kandinsky-6/kandinsky'
K5=ROOT/'kandinsky-5/reference/kandinsky-5/kandinsky/models'
ARTIFACTS=COURSE/'artifacts'

def check_sources():
    manifest=json.loads((COURSE/'sources.json').read_text())
    for item in manifest['files']:
        p=ROOT/item['repository_path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],f'Changed pinned source: {p}'
    return len(manifest['files'])

def _support():
    check_sources();name='_dit_course_k6_support'
    if name not in sys.modules:
        mod=types.ModuleType(name);mod.__path__=[str(ROOT/'kandinsky-6/labkit')];sys.modules[name]=mod
    return importlib.import_module(name+'.components')

def native(relative='core.components.dit'):return _support().native(relative)
def tiny_model(grid=None):return _support().tiny_model(grid)
def tiny_inputs(model,batch=1):return _support().tiny_inputs(model,batch)
def activate_gates(model,std=.08):return _support().activate_gates(model,std)
def fresh_forward(model,kwargs):return _support().fresh_forward(model,kwargs)

def show_source(relative,symbol=None,family='k6'):
    path=(K6 if family=='k6' else K5)/relative
    src=path.read_text();lines=src.splitlines()
    if symbol:
        node=next(n for n in ast.walk(ast.parse(src)) if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==symbol)
        print(f'{path.relative_to(ROOT)}:{node.lineno}–{node.end_lineno}');print('\n'.join(lines[node.lineno-1:node.end_lineno]))
    else:print(src)

def load_k5_block():
    """Original K5 class bodies; decorators omitted, local FP32 tensor helpers/tuple SDPA.

    Scope: dense tiny block only, no sparse/NABLA/CUDA/distributed parity claim.
    """
    check_sources()
    def tuple_sdpa(q,k,v,attn_mask=None):
        was_unbatched=q.ndim==3
        if was_unbatched:q,k,v=[x.unsqueeze(0) for x in (q,k,v)]
        if attn_mask is not None and attn_mask.ndim==2:attn_mask=attn_mask[:,None,None,:]
        out=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=attn_mask).transpose(1,2)
        return (out.squeeze(0) if was_unbatched else out,)
    class Engine:
        def __init__(self,*args):pass
        def get_attention(self):return tuple_sdpa
    class StripDecorators(ast.NodeTransformer):
        def visit_FunctionDef(self,n):n.decorator_list=[];return self.generic_visit(n)
    names={'Modulation','MultiheadSelfAttentionDec','MultiheadCrossAttention','FeedForward'}
    nodes=[n for n in ast.parse((K5/'nn.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name in names]
    nodes+=[n for n in ast.parse((K5/'dit.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='TransformerDecoderBlock']
    tree=StripDecorators().visit(ast.Module(body=nodes,type_ignores=[]));ast.fix_missing_locations(tree)
    tensors=native('core.tensors')
    scope=dict(torch=torch,nn=nn,F=F,SelfAttentionEngine=Engine,
               apply_scale_shift_norm=tensors.apply_scale_shift_norm,
               apply_gate_sum=tensors.apply_gate_sum,apply_rotary=tensors.apply_rotary)
    exec(compile(tree,'<pinned_K5_dense_block>','exec'),scope)
    return types.SimpleNamespace(**scope)

def manual_single(block,x,text,time,rope,mask=None):
    t=native('core.tensors');M,G=t.apply_scale_shift_norm,t.apply_gate_sum
    sa,ca,ff=block.visual_modulation(time).chunk(3,-1)
    b,s,g=sa.chunk(3,-1);x=G(x,block.self_attention(M(block.self_attention_norm,x,s,b),rope),g)
    b,s,g=ca.chunk(3,-1);x=G(x,block.cross_attention(M(block.cross_attention_norm,x,s,b),text,mask),g)
    b,s,g=ff.chunk(3,-1);return G(x,block.feed_forward(M(block.feed_forward_norm,x,s,b)),g)

def manual_fused(block,v,a,tv,ta,cv,ca,rv,ra,mask=None,video_gate_scale=1.,audio_gate_scale=1.,kv_after_text=False):
    """Independent stepwise derivation for active two-stream path, verified against native forward."""
    tensors=native('core.tensors');M,G=tensors.apply_scale_shift_norm,tensors.apply_gate_sum
    vb,ab=block.video_dec_block,block.audio_dec_block
    sa,ctx,ffv=vb.visual_modulation(cv).chunk(3,-1)
    b,s,g=sa.chunk(3,-1);v=G(v,vb.self_attention(M(vb.self_attention_norm,v,s,b),rv),g)
    b,s,gv=ctx.chunk(3,-1);vpre=M(vb.cross_attention_norm,v,s,b);vt=vb.cross_attention(vpre,tv,mask)
    sa,ctx,ffa=ab.visual_modulation(ca).chunk(3,-1)
    b,s,g=sa.chunk(3,-1);a=G(a,ab.self_attention(M(ab.self_attention_norm,a,s,b),ra),g)
    b,s,ga=ctx.chunk(3,-1);apre=M(ab.cross_attention_norm,a,s,b);a=G(a,ab.cross_attention(apre,ta,mask),ga)
    pv=block.va_modulation(cv if block.fix_modulation else ca)
    pa=block.av_modulation(ca if block.fix_modulation else cv)
    if block.cross_gates:
        bv,sv,gate_a=pv.split([block.model_dim,block.model_dim,block.model_dim_a],-1)
        ba,sa,gate_v=pa.split([block.model_dim_a,block.model_dim_a,block.model_dim],-1)
    else:
        bv,sv,gate_v=pv.chunk(3,-1);ba,sa,gate_a=pa.chunk(3,-1)
    v=G(v,vt,gv);qv=M(block.va_normalization,v,sv,bv);qa=M(block.av_normalization,a,sa,ba)
    # Deliberately wrong option for a falsifiable ordering counterexample.
    kv_a,kv_v=(a,v) if kv_after_text else (apre,vpre)
    rvq,raq=(rv,ra) if block.ca_rope else (None,None)
    dv=block.va_cross_attention(qv,kv_a,rope_q=rvq,rope_kv=raq)
    da=block.av_cross_attention(qa,kv_v,rope_q=raq,rope_kv=rvq)
    v=G(v,dv,gate_v*video_gate_scale);a=G(a,da,gate_a*audio_gate_scale)
    b,s,g=ffv.chunk(3,-1);v=G(v,vb.feed_forward(M(vb.feed_forward_norm,v,s,b)),g)
    b,s,g=ffa.chunk(3,-1);a=G(a,ab.feed_forward(M(ab.feed_forward_norm,a,s,b)),g)
    return v,a

class CountingLinearField(nn.Module):
    """Analytic two-state ODE oracle, explicitly NOT a DiT checkpoint."""
    visual_cond=False;is_multimodal=True
    def __init__(self,matrix=None,text_bias=0.):
        super().__init__();self.matrix=torch.tensor([[.7,.2],[.3,.4]]) if matrix is None else matrix
        self.text_bias=text_bias;self.calls=0;self.time_log=[]
    def forward(self,x_video,x_audio,text_embed,time,**kwargs):
        self.calls+=1;self.time_log.append([t.detach().cpu().tolist() for t in time] if isinstance(time,list) else time.detach().cpu().tolist())
        vmean=x_video.mean().to(x_audio);amean=x_audio.mean().to(x_video)
        m=self.matrix.to(x_video);bias=self.text_bias*text_embed.mean()
        v=m[0,0]*x_video+m[0,1]*amean+bias
        a=m[1,0]*vmean+m[1,1]*x_audio+bias
        return v,a

def tiny_bundle(batch=1,video_frames=2,audio_frames=6,channels=4):
    Bundle=native('core.types').LatentBundle
    return Bundle(torch.randn(batch,video_frames,4,4,channels),torch.randn(batch,audio_frames,channels),
                  torch.arange(batch+1,dtype=torch.int32)*video_frames,
                  torch.arange(batch+1,dtype=torch.int32)*audio_frames)

def clone_bundle(b):
    return native('core.types').LatentBundle(**{k:(v.clone() if v is not None else None) for k,v in vars(b).items()})

def scalar_bundle(v=1.,a=-.5):
    B=native('core.types').LatentBundle
    return B(torch.tensor([[[[[v]]]]]),torch.tensor([[[a]]]),torch.tensor([0,1],dtype=torch.int32),torch.tensor([0,1],dtype=torch.int32))

def vector_from_bundle(b):return torch.stack((b.video.flatten()[0],b.audio.flatten()[0]))
def dummy_text(batch=1,positive=1.):return {'text_embeds':torch.full((batch,3,12),positive),'pooled_embed':torch.full((batch,8),positive)}

def save_metrics(lab,metrics):
    folder=ARTIFACTS/lab;folder.mkdir(parents=True,exist_ok=True)
    data={'lab':lab,'torch_version':torch.__version__,'checked_source_files':check_sources(),
          'evidence':'CPU; random native tiny / analytic oracle / labelled dependency doubles',
          'pretrained_inference':False,'metrics':metrics}
    (folder/'metrics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return data

def figure_path(lab,name):
    folder=ARTIFACTS/lab;folder.mkdir(parents=True,exist_ok=True);return folder/(name+'.png')
