"""Validate single-video tile dispatch/IPC against native ComfyUI on CPU, without downloads."""
from __future__ import annotations
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

CHECK = r'''
import json, os, sys, subprocess, tempfile, time
from pathlib import Path
runtime, comfy, output = map(Path, sys.argv[1:])
sys.path[:0] = [str(runtime), str(comfy)]
sys.argv = ['tile-parity', '--cpu', '--fp32-vae', '--disable-dynamic-vram']
import comfy.options
comfy.options.enable_args_parsing()
import torch
import torch.nn.functional as F
import comfy.utils
import comfy.sd
import nodes
from kaggle_parallel_vae_tiles import iter_tiles, tile_parameters, dispatch_vae, DiskTileWorker
torch.set_num_threads(1)

params = dict(tile_t=4, tile_x=32, tile_y=32, overlap=(1,8,8))
plan_counts = []
for shape in [(1,16,9,32,54),(1,16,31,60,108),(2,16,2,3,3),(1,16,5,6,7)]:
    source = torch.arange(__import__('math').prod(shape), dtype=torch.float32).reshape(shape)
    seen = []
    def observe(tile):
        seen.append(tile.clone())
        return tile[:, :1]
    comfy.utils.tiled_scale_multidim(source, observe, tile=(4,32,32), overlap=(1,8,8), upscale_amount=1, out_channels=1)
    planned = list(iter_tiles(source, params))
    assert len(planned)==len(seen)
    assert all(index==i and torch.equal(tile,seen[i]) for i,(index,tile) in enumerate(planned))
    plan_counts.append(len(planned))
assert plan_counts[:2] == [6,150]

class Recorder:
    def temporal_compression_decode(self): return 4
    def spacial_compression_decode(self): return 8
    def decode_tiled(self, latent, **kwargs):
        self.kwargs=kwargs
        return torch.empty(0,0,0,3)
for values in [(256,64,16,4),(64,64,8,8),(128,16,32,4)]:
    cfg=dict(zip(['tile_size','tile_overlap','temporal_size','temporal_overlap'],values))
    recorder=Recorder()
    nodes.VAEDecodeTiled().decode(recorder,{'samples':torch.zeros(1,16,3,8,8)},*values)
    actual=recorder.kwargs
    expected=tile_parameters(recorder,cfg)
    assert expected==dict(tile_t=actual['tile_t'],tile_x=actual['tile_x'],tile_y=actual['tile_y'],
                          overlap=(actual['overlap_t'],actual['overlap'],actual['overlap']))

with tempfile.TemporaryDirectory(prefix='k6-tile-ipc-check-') as folder:
    folder=Path(folder)
    fixture=folder/'fixture_decoder.py'
    fixture.write_text("import torch\nimport torch.nn.functional as F\n"
       "def decode(x):\n"
       "    weight=torch.arange(3*x.shape[1]*27,dtype=x.dtype).reshape(3,x.shape[1],3,3,3)\n"
       "    weight=(weight.remainder(17)-8)/(17*x.shape[1]*27)\n"
       "    y=F.conv3d(F.pad(x,(1,1,1,1,2,0)),weight)\n"
       "    return F.interpolate(y,size=(4*x.shape[2]-3,2*x.shape[3],2*x.shape[4]),mode='trilinear',align_corners=False)\n")
    sys.path.insert(0,str(folder))
    from fixture_decoder import decode
    source=torch.linspace(-1,1,16*5*6*7).reshape(1,16,5,6,7)
    import numpy as np
    np.save(folder/'input.npy',source.numpy())
    p=dict(tile_t=3,tile_x=4,tile_y=4,overlap=(1,2,2))
    (folder/'params.json').write_text(json.dumps(p))
    child=folder/'producer.py'
    child.write_text("import sys,json,os\nfrom pathlib import Path\n"
        "sys.path.insert(0,"+repr(str(runtime))+")\n"
        "import torch,numpy as np\nfrom fixture_decoder import decode\n"
        "from kaggle_parallel_vae_tiles import produce_odd_tiles,watch_parent\n"
        "watch_parent(int(sys.argv[2]))\ntorch.set_num_threads(1)\nroot=Path(sys.argv[1])\n"
        "result=produce_odd_tiles(torch.from_numpy(np.load(root/'input.npy')),json.loads((root/'params.json').read_text()),decode,root)\n"
        "(root/'done.json').write_text(json.dumps(result))\n")
    vae=comfy.sd.VAE.__new__(comfy.sd.VAE)
    vae.upscale_ratio=(lambda t:max(0,4*t-3),2,2)
    vae.upscale_index_formula=(4,2,2)
    vae.output_channels=3
    vae.output_device=torch.device('cpu')
    vae.vae_dtype=torch.float32
    vae.device=torch.device('cpu')
    vae.process_output=lambda x: x.add_(1).div_(2).clamp_(0,1)
    class Model:
        def decode(self,x): return decode(x)
    vae.first_stage_model=Model()
    vae.vae_output_dtype=lambda:torch.float32
    with torch.inference_mode():
        reference=vae.decode_tiled_3d(source,**p)
        remote=DiskTileWorker([sys.executable,str(child),str(folder),str(os.getpid())],os.environ.copy(),folder,timeout=30)
        count=[0,0]
        sequence=[0]
        def route(tile):
            index=sequence[0];sequence[0]+=1
            count[index%2]+=1
            return remote.receive(index,tile.shape) if index%2 else decode(tile)
        try:
            parallel=dispatch_vae(vae,route).decode_tiled_3d(source,**p)
            remote.finish()
        finally:
            remote.close()
    assert count==[6,6]
    assert json.loads((folder/'done.json').read_text())['tiles']==6
    assert torch.equal(reference,parallel),float((reference-parallel).abs().max())
    assert not list(folder.glob('tile-*.npy')) and not list(folder.glob('tile-*.json'))
    errors=[]
    for command,timeout,exception in [([sys.executable,'-c','raise SystemExit(7)'],5,RuntimeError),
                                       ([sys.executable,'-c','import time;time.sleep(20)'],0.2,TimeoutError)]:
        worker=DiskTileWorker(command,os.environ.copy(),folder,timeout=timeout)
        try:
            try: worker.receive(999,(1,16,1,1,1))
            except exception: errors.append(True)
            else: raise AssertionError('Missing worker error')
        finally: worker.close()
        assert worker.process.poll() is not None
    assert len(errors)==2
    report=dict(native_plan_counts=plan_counts,parameter_conversion_matches_native=True,
                processes=2,device='cpu',fixture='causal Conv3d plus interpolation; no pretrained weights',
                fixture_shape=list(parallel.shape),tiles_per_process=count,max_abs_error=0.0,
                native_blending_order_preserved=True,failed_worker_detected=True,
                timeout_detected=True,child_processes_cleaned=True,queue_files_consumed=True,
                cuda_two_gpu_execution=False,pretrained_models_loaded=False)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('TILE_PARITY_OK',json.dumps(report))
'''


PRETRAINED_CHECK = r'''
import sys,json,tempfile,time
from pathlib import Path
runtime,comfy,weight,output=map(Path,sys.argv[1:])
sys.path.insert(0,str(runtime))
from k6_common import bootstrap
from kaggle_parallel_vae_tiles import decode_two_gpus
with tempfile.TemporaryDirectory(prefix='k6-pretrained-vae-cpu-') as d:
    model_root=Path(d)/'models'
    (model_root/'vae').mkdir(parents=True)
    (model_root/'vae/hunyuan_video_vae_bf16.safetensors').symlink_to(weight.resolve())
    cfg=dict(comfy_root=str(comfy),
             model_root=str(model_root),output_root=d,run_root=d,
             cpu_threads=1,video_tile_gpus=[0,1],tile_size=64,tile_overlap=16,
             temporal_size=8,temporal_overlap=4,video_tile_timeout_seconds=300)
    torch,nodes,mm=bootstrap(cfg,'video_cpu_validation',cpu=True)
    from safetensors.torch import save_file
    latent=torch.linspace(-0.1,0.1,16*3*8*8).reshape(1,16,3,8,8)
    save_file({'video':latent},str(Path(d)/'joint-latents.safetensors'))
    vae,=nodes.VAELoader().load_vae('hunyuan_video_vae_bf16.safetensors')
    started=time.perf_counter()
    with torch.inference_mode():
        reference,=nodes.VAEDecodeTiled().decode(vae,{'samples':latent},64,16,8,4)
        ref_seconds=time.perf_counter()-started
        report={}
        started=time.perf_counter()
        parallel=decode_two_gpus(cfg,vae,latent,report,cpu_validation=True)
        parallel_seconds=time.perf_counter()-started
    error=(parallel-reference).abs()
    assert torch.isfinite(parallel).all()
    assert torch.equal(reference,parallel),float(error.max())
    import hashlib,subprocess
    result=dict(vae_sha256=hashlib.file_digest(weight.open('rb'),'sha256').hexdigest(),
        comfy_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=comfy,text=True).strip(),
        pretrained_models_loaded=True,weights='hunyuan_video_vae_bf16.safetensors',
        latent_shape=list(latent.shape),output_shape=list(parallel.shape),device='cpu',
        processes=2,tiles_per_process=report['tile_parallelism']['tiles_per_worker'],
        max_abs_error=float(error.max()),bitwise_equal=True,
        native_seconds=ref_seconds,parallel_cpu_seconds=parallel_seconds,
        cuda_two_gpu_execution=False,synthetic_latent=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('PRETRAINED_CPU_TILE_PARITY_OK',json.dumps(result))
'''

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--comfy-root',type=Path,required=True)
    parser.add_argument('--vae-path',type=Path,help='Optional existing Hunyuan VAE; no download or model generation')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    runtime=args.root/'kandinsky-6/kaggle_runtime'
    output=args.root/'kandinsky-6/reports/parallel-vae-tiles-validation.json'
    with tempfile.TemporaryDirectory(prefix='k6-parallel-validator-') as folder:
        script=Path(folder)/'check.py'
        script.write_text(CHECK)
        result=subprocess.run([str(args.comfy_root/'.venv/bin/python'),str(script),str(runtime),str(args.comfy_root),str(output)],
                              cwd=args.comfy_root,text=True,capture_output=True,timeout=120)
        if result.returncode:
            print(result.stdout,result.stderr)
            raise RuntimeError('Parallel VAE CPU checks failed')
        print(next(line for line in result.stdout.splitlines() if line.startswith('TILE_PARITY_OK ')))
        if args.vae_path:
            if not args.vae_path.is_file():
                raise FileNotFoundError(args.vae_path)
            script.write_text(PRETRAINED_CHECK)
            pretrained_report=args.root/'kandinsky-6/reports/parallel-vae-pretrained-cpu-validation.json'
            result=subprocess.run([str(args.comfy_root/'.venv/bin/python'),str(script),str(runtime),
                str(args.comfy_root),str(args.vae_path),str(pretrained_report)],
                cwd=args.comfy_root,text=True,capture_output=True,timeout=300)
            if result.returncode:
                print(result.stdout,result.stderr)
                raise RuntimeError('Pretrained VAE CPU tile integration check failed')
            print(next(line for line in result.stdout.splitlines() if line.startswith('PRETRAINED_CPU_TILE_PARITY_OK ')))


if __name__=='__main__':
    main()
