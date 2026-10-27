"""Dry-run by default; manually benchmark a LOCAL Diffusers Hunyuan VAE, no downloads."""
from __future__ import annotations
import argparse,hashlib,json,os,statistics,time
from pathlib import Path

C=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*2**20),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,help='Local Diffusers-format VAE folder (config.json + weights)')
    p.add_argument('--input',type=Path,help='Local .npy BCTHW RGB floats in [-1,1], F=4k+1,H/W multiple of 8')
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float32')
    p.add_argument('--policy',choices=['full','spatial','temporal','combined'],default='full')
    p.add_argument('--tile',type=int,nargs=3,default=[17,256,256],metavar=('F','H','W'))
    p.add_argument('--stride',type=int,nargs=3,default=[12,192,192],metavar=('F','H','W'))
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,default=C/'artifacts/pretrained/run.json')
    p.add_argument('--execute',action='store_true',help='Explicitly load local weights and perform reconstruction')
    a=p.parse_args()
    record={'execute':a.execute,'implementation':'Diffusers AutoencoderKLHunyuanVideo; NOT attached adaptive implementation',
            'device':a.device,'dtype':a.dtype,'policy':a.policy,'tile':a.tile,'stride':a.stride,
            'local_model':str(a.model) if a.model else None,'local_input':str(a.input) if a.input else None,
            'repeats':a.repeats,'posterior':'mode','network_allowed':False,'status':'dry-run'}
    if not a.execute:
        print(json.dumps(record,indent=2));return
    if not a.model or not a.input:p.error('--execute requires --model and --input')
    model=a.model.expanduser().resolve();inp=a.input.expanduser().resolve()
    if not model.is_dir() or not (model/'config.json').is_file():p.error('model must be a local folder with config.json')
    if not inp.is_file() or inp.suffix!='.npy':p.error('input must be a local .npy file')
    if a.repeats<1:p.error('--repeats must be positive')
    if a.device=='cpu' and a.dtype!='float32':p.error('CPU smoke uses float32')
    ft,ht,wt=a.tile;fs,hs,ws=a.stride
    if min(fs,hs,ws)<=0 or ft<5 or (ft-1)%4 or fs%4 or fs>ft-1 or ht%8 or wt%8 or hs%8 or ws%8 or hs>ht or ws>wt:
        p.error('Require temporal tile F=4k+1, stride multiple of 4; spatial tile/stride multiples of 8, 0<stride<=tile')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    import diffusers
    from diffusers import AutoencoderKLHunyuanVideo
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        arr=np.load(inp,allow_pickle=False)
        if arr.ndim!=5 or arr.shape[1]!=3 or not np.isfinite(arr).all() or arr.min()<-1 or arr.max()>1:
            raise ValueError('Require finite B,3,F,H,W in [-1,1]')
        if (arr.shape[2]-1)%4 or arr.shape[3]%8 or arr.shape[4]%8:
            raise ValueError('Pad/alignment and valid-frame mask must be prepared explicitly')
        record.update(torch_version=torch.__version__,diffusers_version=diffusers.__version__,
                      input_sha256=digest(inp),input_shape=list(arr.shape),
                      model_files_sha256={str(f.relative_to(model)):digest(f) for f in sorted(model.rglob('*')) if f.is_file()})
        dtype=getattr(torch,a.dtype)
        vae=AutoencoderKLHunyuanVideo.from_pretrained(str(model),torch_dtype=dtype,local_files_only=True).to(a.device).eval()
        vae.requires_grad_(False)
        record['vae_config']=dict(vae.config)
        use_spatial=a.policy in ('spatial','combined');use_temporal=a.policy in ('temporal','combined')
        flags={'use_tiling':use_spatial,'use_framewise_encoding':use_temporal,'use_framewise_decoding':use_temporal}
        for name,value in flags.items():
            if not hasattr(vae,name):raise RuntimeError(f'Unsupported Diffusers API: missing {name}; no fallback')
            setattr(vae,name,value)
        attrs={'tile_sample_min_num_frames':ft-1,'tile_sample_min_height':ht,'tile_sample_min_width':wt,
               'tile_sample_stride_num_frames':fs,'tile_sample_stride_height':hs,'tile_sample_stride_width':ws}
        for name,value in attrs.items():
            if not hasattr(vae,name):raise RuntimeError(f'Unsupported Diffusers tile API: missing {name}')
            setattr(vae,name,value)
        record['effective_flags']=flags;record['effective_tiles']=attrs
        x=torch.from_numpy(arr.copy()).to(device=a.device,dtype=dtype)
        def sync():
            if a.device=='cuda':torch.cuda.synchronize()
            if a.device=='mps':torch.mps.synchronize()
        def counters():
            if a.device=='cuda':return {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
            if a.device=='mps':return {'current_allocated':torch.mps.current_allocated_memory(),'driver_allocated':torch.mps.driver_allocated_memory(),'kind':'point counters, NOT peak'}
            return {'kind':'CPU memory not measured'}
        def timed(fn):
            out=fn();sync();del out # warmup excluded from peak window
            before=counters()
            if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
            times=[]
            for _ in range(a.repeats):
                sync();start=time.perf_counter();out=fn();sync();times.append(time.perf_counter()-start)
                del out # do not overlap previous output allocation with the next repetition
            after=counters();measurement={'seconds_median':statistics.median(times),'seconds_samples':times,
                                         'before':before,'after':after}
            if a.device=='cuda':
                measurement.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                   incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-before['allocated'])
            return measurement
        with torch.inference_mode():
            # No scaling here: encoder and decoder share VAE units; no DiT involved.
            z=vae.encode(x).latent_dist.mode();sync()
            record['encode']=timed(lambda:vae.encode(x).latent_dist.mode())
            record['decode']=timed(lambda:vae.decode(z).sample)
            out=vae.decode(z).sample;sync()
            if out.shape!=x.shape:raise ValueError(f'Output shape mismatch {tuple(out.shape)} vs {tuple(x.shape)}')
            if not torch.isfinite(out).all():raise ValueError('Non-finite reconstruction')
            out_np=out.float().cpu().numpy();mse=float(np.mean((arr.astype(np.float64)-out_np)**2))
            record.update(latent_shape=list(z.shape),output_shape=list(out.shape),
                          mse_input=mse,psnr_input_db=float(10*np.log10(4/max(mse,1e-30))),status='passed')
            np.save(a.output.with_suffix('.reconstruction.npy'),out_np)
        record['scope']='Local pretrained codec reconstruction; not generation. Compare paired .npy outputs to full baseline separately.'
    except Exception as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        a.output.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str)+'\n')
        print('FAILED; evidence saved:',a.output);raise
    a.output.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str)+'\n')
    print('Saved',a.output)
if __name__=='__main__':main()
