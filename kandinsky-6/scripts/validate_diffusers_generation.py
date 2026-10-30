"""Validate classic Diffusers generation cells, presets and real tiny pipeline; no pretrained downloads."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import tempfile
import time
from types import SimpleNamespace

import torch
from PIL import Image, __version__ as pillow_version
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def validate_bootstrap(nb):
    import socket, sys
    from importlib.metadata import version, PackageNotFoundError
    from urllib.request import Request
    from urllib.error import HTTPError, URLError
    from unittest.mock import MagicMock
    from packaging.requirements import Requirement
    source=next(c['source'] for c in nb['cells'] if c['cell_type']=='code' and 'def ensure_runtime_dependencies(' in c['source'])
    names={'pending_dependencies','check_pypi_access','check_loaded_dependency_versions','ensure_runtime_dependencies'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==4
    opener,run=MagicMock(),MagicMock()
    namespace=dict(Requirement=Requirement,version=version,PackageNotFoundError=PackageNotFoundError,
        socket=socket,Request=Request,HTTPError=HTTPError,URLError=URLError,urlopen=opener,
        sys=SimpleNamespace(executable=sys.executable,modules={}),subprocess=SimpleNamespace(run=run))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'bootstrap-unit-check','exec'),namespace)
    required=['diffusers==0.42.1','pillow>=10.4']
    assert required[1] in source and 'pillow>=10.4,<13' not in source
    installed={'diffusers':'0.42.1','pillow':'13.0.0'}
    # Kaggle preloads PIL from its image; Pillow 13 must not trigger a downgrade and a kernel restart.
    assert namespace['pending_dependencies'](required,installed.__getitem__)==[]
    assert namespace['pending_dependencies'](required,lambda n:'12.3.0' if n=='pillow' else installed[n])==[]
    assert namespace['pending_dependencies'](required,lambda n:'0.41.0' if n=='diffusers' else installed[n])==['diffusers==0.42.1']
    pending=MagicMock(return_value=[])
    namespace['pending_dependencies']=pending
    assert namespace['ensure_runtime_dependencies'](required,Path('constraints.txt')) is False
    opener.assert_not_called();run.assert_not_called()
    pending.return_value=['diffusers==0.42.1']
    for error,expected in [(URLError(socket.gaierror(-3,'Temporary failure in name resolution')),'DNS'),
                           (HTTPError('https://pypi.org',403,'Forbidden',{},None),'HTTP 403'),
                           (TimeoutError('fixture timeout'),'timed out')]:
        opener.side_effect=error
        try:namespace['ensure_runtime_dependencies'](required,Path('constraints.txt'))
        except RuntimeError as caught:assert expected in str(caught)
        else:raise AssertionError('Network error did not stop the installer')
        run.assert_not_called()
    opener.side_effect=None
    assert namespace['ensure_runtime_dependencies'](required,Path('constraints.txt')) is True
    command=run.call_args.args[0]
    assert command[command.index('--retries')+1]=='1'
    assert command[command.index('--timeout')+1]=='15'
    assert '-c' in command and 'diffusers==0.42.1' in command and '--trusted-host' not in command
    request=opener.call_args.args[0]
    assert request.full_url.startswith('https://pypi.org/') and opener.call_args.kwargs['timeout']==10
    namespace['version']=installed.__getitem__
    pending.return_value=[]
    namespace['sys'].modules={'PIL':SimpleNamespace(__version__='13.0.0')}
    assert namespace['ensure_runtime_dependencies'](required,Path('constraints.txt')) is False
    namespace['sys'].modules={'diffusers':SimpleNamespace(__version__='0.41.0')}
    try:namespace['ensure_runtime_dependencies'](required,Path('constraints.txt'))
    except RuntimeError as error:assert 'Restart the Python kernel' in str(error)
    else:raise AssertionError('Stale imported diffusers was accepted')
    return dict(preloaded_pillow_13_kept_without_restart=True,compatible_dependencies_skip_network_and_pip=True,
        dns_http_timeout_stop_before_pip=True,limited_pip_retries_and_timeout=True,
        tls_verification_preserved=True,stale_imported_module_requires_kernel_restart=True,
        network_requests_during_tests=0,real_pip_installs_during_tests=0)


def check_supplied_soundtrack(runner,path):
    import av
    with av.open(str(path)) as container:
        stream=container.streams.audio[0]
        codec=stream.codec_context.name
        source_rate=stream.codec_context.sample_rate
    clips=[]
    for seconds in [1,5]:
        mixed,info=runner.soundtrack_audio(torch.zeros(1,seconds*44100),44100,path,0.0,'mix',0.65)
        assert mixed.shape==(1,seconds*44100) and torch.isfinite(mixed).all()
        assert 0<float(mixed.abs().max())<=0.98
        clips.append(dict(seconds=seconds,shape=list(mixed.shape),peak=float(mixed.abs().max()),rms=float(mixed.square().mean().sqrt())))
    return dict(provenance='User-provided AAC; decoded locally, not uploaded to Kaggle or GitHub',
        filename=path.name,bytes=path.stat().st_size,codec=codec,source_sample_rate=source_rate,
        output_sample_rate=44100,verified_excerpts=clips,audio_file_published=False)


def check_nvidia_smi_trace(runner,d):
    import os
    import matplotlib
    matplotlib.use('Agg')
    # Three polling ticks: GPU 0 busy, GPU 1 busy, then both busy.
    csv=d/'trace.csv'
    csv.write_text(''.join(f'2026/10/30 12:00:0{t}.000, {i}, {u}, {m}, [N/A]\n'
        for t,ticks in enumerate([(97,3),(2,96),(80,70)]) for i,(u,m) in enumerate(zip(ticks,(9000,8000)))))
    summary=runner.summarize_nvidia_smi(csv,200)
    assert summary['samples']==3 and [g['busy_fraction'] for g in summary['gpu']]==[2/3,2/3]
    assert summary['all_busy_fraction']==1/3 and summary['one_busy_fraction']==2/3 and summary['none_busy_fraction']==0
    runner.plot_nvidia_smi(csv,d/'trace.png');assert (d/'trace.png').stat().st_size>0
    # A fake nvidia-smi exercises the real Popen/terminate path on a machine without NVIDIA GPUs.
    fake=d/'bin'/'nvidia-smi';fake.parent.mkdir()
    fake.write_text('#!/bin/sh\nwhile true; do\n echo "2026/10/30 12:00:00.000, 0, 99, 9000, 60.0"\n'
                    ' echo "2026/10/30 12:00:00.000, 1, 1, 8000, [N/A]"\n /bin/sleep 0.05\ndone\n')
    fake.chmod(0o755)
    saved=os.environ['PATH']
    try:
        os.environ['PATH']=str(fake.parent)+os.pathsep+saved
        with runner.NvidiaSmiTrace(d/'live.csv') as trace:time.sleep(1.5)
        assert trace.summary['samples']>=2 and trace.summary['one_busy_fraction']==1 and trace.process.poll() is not None
        os.environ['PATH']=str(d/'empty')
        with runner.NvidiaSmiTrace(d/'missing.csv') as trace:pass
        assert trace.summary==dict(available=False)
    finally:
        os.environ['PATH']=saved
    return True

def check_parallel_clips(runner,d,presets):
    import threading
    # Decoder defaults keep the single-clip layout: video on cuda:1, audio on cuda:0.
    assert inspect.signature(runner.decode_video).parameters['device'].default=='cuda:1'
    assert inspect.signature(runner.decode_audio).parameters['device'].default=='cuda:0'
    assert isinstance(runner._LOAD_LOCK,type(threading.Lock()))
    base=dict(model_id='m',revision='r',packages={},seed=42,results=str(d/'parallel'),local_model_dir=None,
              media_cache=str(d/'cache'),profile='smoke',job_key='x',dit_placement='cuda:0')
    (d/'parallel').mkdir()
    runs=[]
    for i,name in enumerate(['orbit_woman','strange_tram','mechanical_bloom']):
        run=runner.prepare_run(base,presets[name],name,'smoke',f'cuda:{i%2}')
        run['key']=name;runs.append(run)
    assert len({r['cfg']['job_key'] for r in runs})==3 and runs[1]['cfg']['dit_placement']=='cuda:1'
    spans,lock={},threading.Lock()
    def fake_sample(cfg,prompt,negative,conditioning_path,image,latent_path,trace_path=None):
        if cfg['run_name'].startswith('mechanical_bloom'):
            raise torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 1.00 GiB')
        start=time.perf_counter();time.sleep(0.5)
        with lock:spans[cfg['run_name']]=(start,time.perf_counter())
        return dict(denoise_seconds=0.3,dit_placement=cfg['dit_placement'])
    for run in runs:run['cfg']['run_name']=run['run_dir'].name
    patched=dict(sample_joint=fake_sample,decode_video=lambda cfg,latents,device:np.zeros((24,8,8,3),np.float32),
        decode_audio=lambda cfg,latents,device:(torch.zeros(1,44100),44100),
        export_clip=lambda frames,waveform,rate,cfg,path:dict(mp4=str(path),exported_frames=len(frames)))
    cuda=dict(set_device=lambda gpu:None,reset_peak_memory_stats=lambda gpu:None,max_memory_allocated=lambda gpu:2**30)
    saved={k:getattr(runner,k) for k in patched};saved_cuda={k:getattr(torch.cuda,k) for k in cuda}
    try:
        for k,v in patched.items():setattr(runner,k,v)
        for k,v in cuda.items():setattr(torch.cuda,k,v)
        summary=runner.run_parallel_clips(runs,{r['key']:d/'unused' for r in runs},d/'parallel-trace.csv')
    finally:
        for k,v in saved.items():setattr(runner,k,v)
        for k,v in saved_cuda.items():setattr(torch.cuda,k,v)
    results=summary['results']
    assert [r['gpu'] for r in results]==[0,0,1] and summary['clips']==2
    assert [bool(r.get('oom')) for r in results]==[False,True,False]
    (a0,a1),(b0,b1)=spans[runs[0]['run_dir'].name],spans[runs[1]['run_dir'].name]
    assert max(a0,b0)<min(a1,b1), 'clips on the two GPUs did not overlap'
    assert summary['clips_per_hour']>0 and summary['sum_of_clip_seconds']>=1.0
    assert all((r['run_dir']/'parallel.json').is_file() for r in runs)
    return True

def check_weight_snapshots(runner,d):
    revision='4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81'
    def part(root,name,files,rev=revision):
        root.mkdir(parents=True)
        for f in files:(root/f).parent.mkdir(parents=True,exist_ok=True);(root/f).write_text('{}')
        (root/'weights-manifest.json').write_text(json.dumps(dict(part=name,revision=rev)))
    inputs=d/'input'
    part(inputs/'k6-lite-diffusers-weights-media'/'k6-lite-diffusers-media','media',
         ['model_index.json','transformer/config.json','text_encoder_2/config.json'])
    assert runner.assemble_local_snapshot(inputs,d/'snap-missing',revision) is None
    part(inputs/'k6-lite-diffusers-weights-text'/'k6-lite-diffusers-text','text',['text_encoder/config.json'])
    assert runner.assemble_local_snapshot(inputs,d/'snap-other',revision.replace('4','5')) is None
    target=Path(runner.assemble_local_snapshot(inputs,d/'snap',revision))
    assert sorted(p.name for p in target.iterdir())==['model_index.json','text_encoder','text_encoder_2','transformer']
    assert (target/'transformer/config.json').read_text()=='{}' and (target/'text_encoder').is_symlink()
    assert runner.assemble_local_snapshot(inputs,d/'snap',revision)==str(target)  # idempotent
    # Weight notebooks: private, CPU-only, and the two parts partition the snapshot.
    for name,number in [('media',17),('text',18)]:
        meta=json.loads((ROOT/f'kandinsky-6/generation/weights/{name}/kernel-metadata.json').read_text())
        assert meta['is_private'] and not meta['enable_gpu'] and meta['enable_internet']
        nb=json.loads((ROOT/f'kandinsky-6/notebooks/{number}_kaggle_k6_weights_{name}.ipynb').read_text())
        cells=[c['source'] for c in nb['cells'] if c['cell_type']=='code']
        for cell in cells:ast.parse(cell)
        assert f"PART={name!r}" in cells[0]
        source=next(c for c in cells if 'PARTS = {' in c)
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and n.targets[0].id=='PARTS')
        parts=eval(compile(ast.Expression(node.value),'parts','eval'))
    for path,expected in [('text_encoder/model-00001-of-00004.safetensors','text'),
                          ('text_encoder_2/model.safetensors','media'),('transformer/config.json','media'),
                          ('model_index.json','media')]:
        assert [k for k,f in parts.items() if f(path)]==[expected],path
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--soundtrack',type=Path);args=p.parse_args()
    path=ROOT/'kandinsky-6/generation/diffusers_runner.py'
    spec=importlib.util.spec_from_file_location('k6_diffusers_generation',path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    notebook=ROOT/'kandinsky-6/notebooks/16_kaggle_diffusers_lite_generation.ipynb'
    nb=json.loads(notebook.read_text())
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            ast.parse(cell['source']);assert not cell['outputs'] and cell['execution_count'] is None
            assert not cell['source'].startswith('%%')
    bootstrap=validate_bootstrap(nb)
    presets=json.loads((ROOT/'kandinsky-6/generation/prompt_presets.json').read_text())
    prompts=[runner.compile_prompt(preset,profile)[0] for preset in presets.values() for profile in runner.PROFILES]
    assert len(prompts)==14 and all('<AUDCAP>' in x and '<ENDAUDCAP>' in x for x in prompts)
    for profile in runner.PROFILES.values():
        assert (profile['num_frames']-1)%4==0
        assert profile['export_frames']/24==profile['seconds']
    mask=torch.tensor([[True,False,True,False]])
    tokens=torch.arange(16).reshape(1,4,4).float()
    compact=runner.compact_conditioning((tokens,torch.ones(1,3),mask))
    assert torch.equal(compact['prompt_embeds'],tokens[:,[0,2]])
    assert runner.mapped_gpu_indices({'blocks.0':0,'blocks.1':'cuda:1'})=={0,1}
    assert runner.dit_load_options('balanced')==(dict(device_map='balanced',max_memory={0:'10GiB',1:'10GiB'}),{0,1})
    assert runner.dit_load_options('cuda:1')==(dict(device_map='cuda:1'),{1})
    assert runner.mapped_gpu_indices({'':str(torch.device('cuda:1'))})=={1}
    try:runner.dit_load_options('auto')
    except ValueError:pass
    else:raise AssertionError('Unsupported DiT placement accepted')
    try:runner.compact_conditioning((torch.full((1,1,4),float('nan')),torch.ones(1,3),None))
    except FloatingPointError:pass
    else:raise AssertionError('NaN conditioning accepted')
    # Public API accepts the runner's cached embeddings and latent output.
    required={'image','prompt_embeds','pooled_prompt_embeds','output_type','frame_rate','sample_audio','callback_on_step_end'}
    assert required <= set(inspect.signature(runner.Kandinsky6TI2VAPipeline.__call__).parameters)
    torch.set_num_threads(1);torch.manual_seed(42)
    transformer=runner.Kandinsky6Transformer3DModel(in_visual_dim=16,out_visual_dim=160,
        in_audio_dim=40,out_audio_dim=400,in_text_dim=8,in_text_dim2=4,
        model_dim=32,model_dim_a=16,time_dim=16,time_dim_a=16,ff_dim=64,ff_dim_a=32,
        num_text_blocks=1,num_visual_blocks=1,axes_dims=(2,2,4),axes_dims_a=(2,2,4),
        visual_cond=True,ca_rope=True,cross_gates=True,fix_modulation=True,visual_token_type_num_embeddings=2)
    vae=runner.AutoencoderKLHunyuanVideo(block_out_channels=(8,8,8,8),layers_per_block=1,norm_num_groups=4)
    class AudioConfigFixture(torch.nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(sample_rate=44100);self.latent_hop_length=1024
            self.device=torch.device('cpu');self.dtype=torch.float32
    pipe=runner.Kandinsky6TI2VAPipeline(transformer=transformer,vae=vae,
        text_encoder=None,text_encoder_2=None,tokenizer=None,tokenizer_2=None,
        audio_vae=AudioConfigFixture(),vocoder=AudioConfigFixture(),
        scheduler=runner.PiflowScheduler(n_grid=10))
    params=dict(prompt_embeds=torch.randn(1,6,8),pooled_prompt_embeds=torch.randn(1,4),
        height=64,width=64,num_frames=5,frame_rate=24,num_inference_steps=10,
        guidance_scale=1.0,sample_audio=True,output_type='latent')
    native=[]
    for mode,image in [('t2v',None),('i2v',Image.new('RGB',(64,64),(80,160,40)))]:
        calls=[0]
        hook=transformer.register_forward_pre_hook(lambda m,a:calls.__setitem__(0,calls[0]+1))
        start=time.perf_counter()
        with torch.inference_mode():
            result=pipe(image=image,generator=torch.Generator('cpu').manual_seed(42),**params)
        hook.remove()
        assert calls[0]==10
        assert result.frames.shape==(1,16,2,8,8)
        assert result.audio.shape==(1,40,math.ceil(5/24*44100/1024))
        runner.finite(result.frames,mode);runner.finite(result.audio,mode+' audio')
        native.append(dict(mode=mode,dit_nfe=calls[0],video_latent_shape=list(result.frames.shape),
                           audio_latent_shape=list(result.audio.shape),seconds=time.perf_counter()-start))
    with tempfile.TemporaryDirectory(prefix='k6-diffusers-validation-') as d:
        d=Path(d)
        portraits=[]
        for key,size in [('mario_poster',(415,739)),('mario_pipe_illustration',(640,640))]:
            ref=d/(key+'.png');Image.new('RGB',size,(90,130,170)).save(ref)
            for profile in runner.PROFILES:
                im,sha,w,h=runner.reference_and_size(presets[key],profile,ref)
                assert im.size==(w,h) and len(sha)==64 and w%16==h%16==0
                portraits.append([key,profile,w,h])
        frames=np.zeros((24,32,32,3),dtype=np.float32)
        wave=(torch.sin(torch.arange(44100)*440*2*math.pi/44100)*0.02)[None]
        media=runner.export_clip(frames,wave,44100,dict(width=32,height=32,export_frames=24),d/'test.mp4')
        media.pop('mp4')
        # Reuse a generated audio file as the local soundtrack; no song download.
        mixed,soundtrack=runner.soundtrack_audio(torch.zeros(1,22050),44100,d/'test.mp4',0.25,'mix',0.4)
        assert mixed.shape==(1,22050) and torch.isfinite(mixed).all()
        assert 0<float(mixed.abs().max())<=0.98 and soundtrack['source']=='mix'
        try:runner.soundtrack_audio(torch.zeros(1,22050),44100,d/'test.mp4',1.0,'music',0.4)
        except ValueError:pass
        else:raise AssertionError('Short soundtrack was padded silently')
        checkpoint=d/'joint-latents.safetensors'
        runner.save_file({'video':torch.zeros(1,16,2,8,8),'audio':torch.zeros(1,40,9)},str(checkpoint),metadata={'job_key':'fixture'})
        assert runner.sample_joint({'job_key':'fixture'},'', '',d/'unused.safetensors',None,checkpoint)['cache_reused']
        try:runner.sample_joint({'job_key':'stale'},'', '',d/'unused.safetensors',None,checkpoint)
        except ValueError:pass
        else:raise AssertionError('Stale joint latent accepted')
        nvidia_smi_trace=check_nvidia_smi_trace(runner,d)
        parallel_clips=check_parallel_clips(runner,d,presets)
        weight_snapshots=check_weight_snapshots(runner,d)
        # Placement comparison reads only runs whose denoising actually ran.
        for name,sample in [('a',dict(cache_reused=False,dit_placement='balanced',denoise_seconds=13.4,
                                      stage=dict(gpu=[dict(allocated_peak_gib=3.0),dict(allocated_peak_gib=3.1)]),
                                      nvidia_smi=dict(all_busy_fraction=0.03,one_busy_fraction=0.9))),
                            ('b',dict(cache_reused=True))]:
            (d/'runs'/name).mkdir(parents=True)
            (d/'runs'/name/'sample.json').write_text(json.dumps(sample))
            (d/'runs'/name/'config.json').write_text(json.dumps(dict(width=432,height=256,num_frames=25)))
        rows=runner.compare_placements(d/'runs')
        assert rows==[dict(run='a',placement='balanced',size='432x256x25',denoise_s=13.4,
                           peak_alloc_gib=[3.0,3.1],all_busy=0.03,one_busy=0.9)]
        # Existing cache keys must not change when the key is built by job_key_for.
        base=dict(model_id='m',revision='r',packages={'diffusers':'0.42.1'},seed=42,width=432,height=256,
                  num_frames=25,dit_placement='balanced')
        assert runner.job_key_for(base,'p',None)==runner.digest(dict(model='m',revision='r',
            packages={'diffusers':'0.42.1'},prompt='p',seed=42,width=432,height=256,num_frames=25,image=None,
            steps=10,cfg=1.0,dtype='float16',placement='balanced-two-gpu'))
        assert runner.job_key_for(dict(base,dit_placement='cuda:0'),'p',None)!=runner.job_key_for(base,'p',None)
        # Measurement mode: one pair succeeds, the other runs out of memory and is recorded, not raised.
        saved_sample_joint=runner.sample_joint
        def fake_sample_joint(cfg,prompt,negative,conditioning_path,image,latent_path,trace_path=None):
            if cfg['profile']=='baseline':
                raise torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.00 GiB')
            return dict(dit_placement=cfg['dit_placement'],denoise_seconds=1.5,cache_reused=False,
                        nvidia_smi=dict(all_busy_fraction=0.0,one_busy_fraction=1.0))
        try:
            runner.sample_joint=fake_sample_joint
            (d/'measure').mkdir()
            measured_rows=runner.measure_placements(dict(base,results=str(d/'measure'),profile='smoke',job_key='x'),
                presets['orbit_woman'],'orbit_woman',[('smoke','cuda:0'),('baseline','cuda:0')],
                {'smoke':d/'unused.safetensors','baseline':d/'unused.safetensors'})
        finally:
            runner.sample_joint=saved_sample_joint
        assert [r.get('oom',False) for r in measured_rows]==[False,True]
        table=runner.compare_placements(d/'measure')
        assert sorted((r['size'],r.get('oom',False)) for r in table)==[('432x256x25',False),('864x480x121',True)]
    import diffusers,transformers,accelerate
    report=dict(date='2026-10-30',notebook=str(notebook.relative_to(ROOT)),
        notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
        runtime_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bootstrap_checks=bootstrap,
        code_cells=sum(c['cell_type']=='code' for c in nb['cells']),classic_python_cells=True,
        prompt_scenarios=7,prompt_timelines=14,profile_frame_contracts=True,
        cached_conditioning_padding_removed=True,nan_rejected=True,
        local_soundtrack_mix_and_offset=True,short_soundtrack_rejected=True,
        joint_latent_cache_reused_without_models=True,stale_latent_rejected=True,
        nvidia_smi_trace_summary_and_fallback=nvidia_smi_trace,
        dit_placement_options_and_comparison=True,
        placement_measurement_mode_records_oom=True,
        parallel_clips_two_gpu_queues_overlap_and_oom=parallel_clips,
        attached_weight_snapshots_assembled=weight_snapshots,
        native_tiny_pipeline=native,reference_shapes=portraits,synthetic_mp4=media,
        versions=dict(torch=torch.__version__,diffusers=diffusers.__version__,
                      transformers=transformers.__version__,accelerate=accelerate.__version__,pillow=pillow_version),
        pretrained_weights_downloaded=False,pretrained_models_loaded=False,
        two_gpu_cuda_tested=False,kaggle_inference_tested=False,
        boundary='Real random-weight Diffusers DiT and Hunyuan VAE; audio codecs are config-only fixtures for latent output; no pretrained quality/performance claim')
    if args.soundtrack:
        report['supplied_soundtrack']=check_supplied_soundtrack(runner,args.soundtrack)
    output=args.output or ROOT/'kandinsky-6/reports/diffusers-generation-validation.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
