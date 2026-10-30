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
from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args()
    path=ROOT/'kandinsky-6/generation/diffusers_runner.py'
    spec=importlib.util.spec_from_file_location('k6_diffusers_generation',path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    notebook=ROOT/'kandinsky-6/notebooks/16_kaggle_diffusers_lite_generation.ipynb'
    nb=json.loads(notebook.read_text())
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            ast.parse(cell['source']);assert not cell['outputs'] and cell['execution_count'] is None
            assert not cell['source'].startswith('%%')
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
    import diffusers,transformers,accelerate
    report=dict(date='2026-10-30',notebook=str(notebook.relative_to(ROOT)),
        notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
        runtime_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        code_cells=sum(c['cell_type']=='code' for c in nb['cells']),classic_python_cells=True,
        prompt_scenarios=7,prompt_timelines=14,profile_frame_contracts=True,
        cached_conditioning_padding_removed=True,nan_rejected=True,
        local_soundtrack_mix_and_offset=True,short_soundtrack_rejected=True,
        joint_latent_cache_reused_without_models=True,stale_latent_rejected=True,
        native_tiny_pipeline=native,reference_shapes=portraits,synthetic_mp4=media,
        versions=dict(torch=torch.__version__,diffusers=diffusers.__version__,
                      transformers=transformers.__version__,accelerate=accelerate.__version__),
        pretrained_weights_downloaded=False,pretrained_models_loaded=False,
        two_gpu_cuda_tested=False,kaggle_inference_tested=False,
        boundary='Real random-weight Diffusers DiT and Hunyuan VAE; audio codecs are config-only fixtures for latent output; no pretrained quality/performance claim')
    output=args.output or ROOT/'kandinsky-6/reports/diffusers-generation-validation.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
