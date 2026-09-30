"""Загрузка компонентов отдельно и последовательный T2V-инференс."""
from pathlib import Path
import ast
import json
import time
from .common import ROOT, SOURCE, TARGET, release, synchronize, shifted_schedule, save_json

REVISIONS = json.loads((ROOT/'reference/hf-revisions.json').read_text()) if (ROOT/'reference/hf-revisions.json').exists() else {}
REPOS = {'dit': TARGET, 'qwen': 'Qwen/Qwen2.5-VL-7B-Instruct',
         'clip': 'openai/clip-vit-large-patch14', 'vae': 'hunyuanvideo-community/HunyuanVideo'}

def download_component(name, weights=ROOT/'weights'):
    from huggingface_hub import snapshot_download
    repo=REPOS[name]
    if repo not in REVISIONS: raise RuntimeError('Нет зафиксированной версии '+repo)
    patterns={'dit':['model/*'], 'vae':['vae/*'], 'qwen':['*.json','*.safetensors','*.txt','*.model','*.jinja'],
              'clip':['*.json','*.safetensors','*.txt','*.bin']}[name]
    return Path(snapshot_download(repo_id=repo, revision=REVISIONS[repo], allow_patterns=patterns,
                                 local_dir=Path(weights)/name))

def prompt_template(kind='video'):
    tree=ast.parse((SOURCE/'kandinsky/models/text_embedders.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen2_5_VLTextEmbedder')
    assignment=next(n for n in cls.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPT_TEMPLATE' for t in n.targets))
    data=ast.literal_eval(assignment.value)
    return '\n'.join(data['template'][kind]),data['crop_start'][kind]

def encode_conditioning(prompt, weights=ROOT/'weights', device='cpu', dtype=None, max_length=256):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, CLIPTokenizer, CLIPTextModel
    dtype=dtype or torch.bfloat16
    qwen_path=Path(weights)/'qwen'; clip_path=Path(weights)/'clip'
    template,crop=prompt_template('video')
    processor=AutoProcessor.from_pretrained(qwen_path,local_files_only=True,use_fast=True)
    inputs=processor(text=[template.format(prompt)], images=None, videos=None, max_length=max_length+crop,
                     truncation=True, padding='max_length', return_tensors='pt').to(device)
    # Eager + SDPA; no compile, no bitsandbytes, no automatic device map.
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_path,local_files_only=True,
             dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval().requires_grad_(False).to(device)
    with torch.inference_mode():
        # Exactly as upstream: no attention_mask is passed into Qwen here.
        out=model(input_ids=inputs['input_ids'],return_dict=True,output_hidden_states=True)
        embeddings=out.hidden_states[-1][:,crop:].float().cpu().contiguous()
    mask=inputs['attention_mask'][:,crop:].bool().cpu()
    del model, out, inputs, processor; release()
    tokenizer=CLIPTokenizer.from_pretrained(clip_path,local_files_only=True)
    inputs=tokenizer([prompt],max_length=77,truncation=True,padding='max_length',return_tensors='pt').to(device)
    model=CLIPTextModel.from_pretrained(clip_path,local_files_only=True).eval().requires_grad_(False).to(device)
    with torch.inference_mode(): pooled=model(**inputs).pooler_output.cpu().float()
    del model,inputs,tokenizer; release()
    return {'text_embeds':embeddings,'pooled_embed':pooled,'attention_mask':mask}

def load_video_vae(weights=ROOT/'weights', device='cpu'):
    import torch
    from diffusers import AutoencoderKLHunyuanVideo
    # Reference baseline uses the same Hunyuan weights, without the CUDA-aware tiling planner.
    return AutoencoderKLHunyuanVideo.from_pretrained(Path(weights)/'vae',subfolder='vae',
               torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)

def sample_latents(dit, conditioning, conf, height=512, width=768, seconds=5, steps=16, scale=5., seed=42,
                   device='mps', dtype=None, on_step=None):
    import torch
    dtype=dtype or torch.bfloat16
    if height%16 or width%16 or seconds<0: raise ValueError('H/W кратны 16, seconds >= 0')
    if conf.model.attention.type != 'flash': raise ValueError('Этот порт только для dense 5s конфигурации')
    frames=seconds*24//4+1
    if max(frames,height//16,width//16)>128: raise ValueError('Превышены пределы RoPE')
    # CPU seed is portable across MPS/CUDA, but won't be bit-identical to native CUDA RNG.
    x=torch.randn(frames,height//8,width//8,16,generator=torch.Generator().manual_seed(seed)).to(device,dtype)
    text=conditioning['text_embeds'].to(device,dtype); pooled=conditioning['pooled_embed'].to(device,dtype)
    mask=conditioning['attention_mask'].to(device)
    positions=[torch.arange(frames,device=device),torch.arange(height//16,device=device),torch.arange(width//16,device=device)]
    text_positions=torch.arange(text.shape[1],device=device)
    schedule=shifted_schedule(steps,scale).to(device)
    with torch.inference_mode(), torch.autocast(device_type=str(device).split(':')[0],dtype=dtype):
        for i in range(steps):
            # T2V visual_cond=true: extra channels and mask stay zero (33 channels total).
            model_input=torch.cat([x,torch.zeros_like(x),torch.zeros_like(x[...,:1])],-1)
            velocity=dit(model_input,text,pooled,(schedule[i]*1000).reshape(1),positions,text_positions,
                         scale_factor=conf.metrics.scale_factor,sparse_params=None,attention_mask=mask)
            if not torch.isfinite(velocity).all(): raise FloatingPointError('DiT returned NaN/Inf')
            x=x+(schedule[i+1]-schedule[i])*velocity
            if on_step: on_step(i,x)
    return x.float().cpu()

def decode_latents(latents, weights=ROOT/'weights', device='cpu', tiling=True):
    import torch
    vae=load_video_vae(weights,device)
    if tiling: vae.enable_tiling()
    z=latents.permute(3,0,1,2).unsqueeze(0).to(device,torch.float32)/vae.config.scaling_factor
    with torch.inference_mode(): video=vae.decode(z).sample.cpu()
    del vae,z; release()
    if not torch.isfinite(video).all(): raise FloatingPointError('VAE returned NaN/Inf')
    return ((video[0].clamp(-1,1)+1)*127.5).to(torch.uint8).permute(1,2,3,0).numpy()

def sequential_t2v(prompt, output=ROOT/'artifacts/mac_distill.mp4', height=512,width=768,seconds=5,
                   steps=16,scale=5.,seed=42,device='mps',weights=ROOT/'weights'):
    import torch
    import imageio.v2 as imageio
    from .native import real_dit
    if device=='cpu': raise RuntimeError('Полный DiT запускать на MPS/CUDA; учебные эксперименты работают на CPU')
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    timings={}; start=time.perf_counter()
    cond=encode_conditioning(prompt,weights,device,torch.bfloat16)
    synchronize(device); timings['conditioning_seconds']=time.perf_counter()-start
    torch.save(cond,output.with_suffix('.conditioning.pt'))
    start=time.perf_counter()
    checkpoint=Path(weights)/'dit/model/kandinsky5lite_t2v_distilled16steps_5s.safetensors'
    dit,conf=real_dit(checkpoint,device,torch.bfloat16)
    latents=sample_latents(dit,cond,conf,height,width,seconds,steps,scale,seed,device,torch.bfloat16,
                          on_step=lambda i,x: print(f'DiT {i+1}/{steps}'))
    synchronize(device); timings['dit_load_and_sample_seconds']=time.perf_counter()-start
    del dit,cond; release()
    torch.save(latents,output.with_suffix('.latents.pt'))
    start=time.perf_counter()
    # CPU decoder is a correctness baseline on Mac; potentially very slow for 121 frames.
    frames=decode_latents(latents,weights,'cpu',tiling=True)
    timings['vae_load_and_decode_seconds']=time.perf_counter()-start
    imageio.mimwrite(output,frames,fps=24,codec='libx264',quality=8)
    save_json(output.with_suffix('.json'),dict(prompt=prompt,height=height,width=width,seconds=seconds,
          decoded_frames=len(frames),steps=steps,NFE=steps,guidance=1.,scheduler_scale=scale,seed=seed,
          device=device,dtype='bfloat16',vae_device='cpu',revisions=REVISIONS,timings=timings))
    return output
