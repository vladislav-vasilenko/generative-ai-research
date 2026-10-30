"""Check standalone SR graphs against a running ComfyUI, without submitting a prompt or loading weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NATIVE_CHECK = r'''
import json,sys,types
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root),str(root/'custom_nodes/kandinsky6-sr')]
sys.argv=['sr-contract-check','--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch
from kandinsky6_vsr.nodes import trim_comfy_audio
from kandinsky6_vsr.runtime.pipeline.upscale_utils import pre_upscale_video,resolve_scale_request
from kandinsky6_vsr.runtime.pipeline.output_resize import resolve_target_hw
from kandinsky6_vsr.runtime.pipeline.video_io import resample_to_target_fps,clip_to_aligned_frames
from kandinsky6_vsr.sampling import steps_per_tile
video=torch.zeros((121,3,16,16),dtype=torch.uint8)
resampled,fps=resample_to_target_fps(video,24.0)
assert fps==24 and clip_to_aligned_frames(resampled).shape[0]==121
audio={'waveform':torch.arange(230000).reshape(1,1,-1).float(),'sample_rate':44100}
trimmed=trim_comfy_audio(audio,121,24)
assert trimmed['waveform'].shape[-1]==222338
assert torch.equal(trimmed['waveform'],audio['waveform'][...,:222338])
geometry=[]
for scale in [2.0,2.25]:
 integer,pre=resolve_scale_request(scale)
 source=torch.empty((121,3,480,864),device='meta',dtype=torch.uint8)
 sized=pre_upscale_video(source,pre,16) if pre!=1 else source
 h,w=[side*integer for side in sized.shape[-2:]]
 fit=resolve_target_hw('fullhd',(h,w))
 geometry.append({'scale':scale,'raw_width':w,'raw_height':h,'fullhd_fit_hw':list(fit) if fit else None})
assert geometry[0]['raw_width']==1728 and geometry[0]['raw_height']==960
assert geometry[1]['raw_width']==1952 and geometry[1]['raw_height']==1088
assert geometry[1]['fullhd_fit_hw']==[1070,1920]
holder=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=types.SimpleNamespace(n_grid=10)))
assert steps_per_tile(holder,2)==2 and steps_per_tile(holder,5)==2
print('SR_CONTRACTS_OK',json.dumps({'geometry':geometry,'source_frames_preserved':121,
 'source_fps_preserved':24,'audio_trim_samples':222338,'audio_prefix_unchanged':True,
 'torch_version':torch.__version__,'device':'cpu',
 'distilled_nfe_per_tile_batch':2,'pretrained_models_loaded':False,'sr_inference_run':False}))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default='http://127.0.0.1:8188')
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--models-root', type=Path, action='append', required=True)
    options = parser.parse_args()
    with urllib.request.urlopen(options.server + '/object_info', timeout=15) as response:
        schemas = json.load(response)
    report = {'comfyui_jobs_submitted': 0, 'weights_downloaded': False, 'sr_inference_run': False,
              'mps_sr_inference_verified': False,
              'checked_date_utc': datetime.now(timezone.utc).date().isoformat(),
              'workflows': []}
    for profile in ['2x', '2_25x']:
        path = ROOT / f'workflows/k6_sr_standalone_{profile}.json'
        graph = json.loads(path.read_text())
        nodes = {n['id']: n for n in graph['nodes']}
        assert len(nodes) == 9 and len(graph['links']) == 11
        assert not any(n['type'] in ['Kandinsky6Sampler', 'DualCLIPLoader', 'Kandinsky6AudioVAELoader'] for n in nodes.values())
        for node in nodes.values():
            # MarkdownNote is an official frontend-only node, absent from /object_info.
            if node['type'] != 'MarkdownNote':
                assert node['type'] in schemas, f"Node not installed: {node['type']}"
        for number, source, out_index, target, in_index, kind in graph['links']:
            assert nodes[source]['outputs'][out_index]['type'] == kind
            pin = nodes[target]['inputs'][in_index]
            assert pin['type'] == kind and pin['link'] == number
            schema = schemas[nodes[target]['type']]['input']
            all_inputs = {**schema.get('required', {}), **schema.get('optional', {})}
            assert pin['name'] in all_inputs and all_inputs[pin['name']][0] == kind
            assert schemas[nodes[source]['type']]['output'][out_index] == kind
        assert nodes[2]['outputs'][1]['links'] == [3]  # source audio -> SR
        assert nodes[4]['outputs'][2]['links'] == [10]  # trimmed source audio -> CreateVideo
        settings = nodes[4]['widgets_values_named']
        assert settings['steps'] == 2 and settings['tiles_batch_size'] == 1 and not settings['use_nabla']
        assert settings['target_resolution'] == 'none'
        assert nodes[8]['widgets_values'][1] == '2x'
        input_file = options.input_root / nodes[1]['widgets_values'][0]
        assert input_file.is_file(), f'Missing source video: {input_file}'
        assert nodes[1]['widgets_values'][0] in schemas['LoadVideo']['input']['required']['file'][1]['options']
        report['workflows'].append({'profile': profile, 'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'nodes': 9, 'typed_links': 11,
            'registered_execution_nodes': 8, 'frontend_notes': 1,
            'source_video_exists': True, 'source_audio_connected': True})
    manifest = json.loads((ROOT / 'workflows/sr-model-manifest.json').read_text())
    assert len(manifest['files']) == 8
    report['bundle_revision'] = manifest['revision']
    report['bundle_bytes'] = sum(f['size'] for f in manifest['files'])
    report['missing_files'] = [f['filename'] for f in manifest['files'] if not any(
        (root / 'Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers' / f['filename']).is_file()
        for root in options.models_root)]
    report['all_component_files_present'] = not report['missing_files']
    with tempfile.TemporaryDirectory(prefix='k6-sr-contracts-') as folder:
        check = Path(folder) / 'check.py'
        check.write_text(NATIVE_CHECK)
        result = subprocess.run([str(options.comfy_root / '.venv/bin/python'), str(check), str(options.comfy_root)],
                                 cwd=options.comfy_root, text=True, capture_output=True, timeout=90)
        if result.returncode:
            print(result.stdout, result.stderr)
            raise RuntimeError('Native SR contract checks failed')
        line = next(line for line in result.stdout.splitlines() if line.startswith('SR_CONTRACTS_OK '))
        report['native_cpu_contracts'] = json.loads(line.removeprefix('SR_CONTRACTS_OK '))
    destination = ROOT / 'reports/sr-workflows-validation.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
