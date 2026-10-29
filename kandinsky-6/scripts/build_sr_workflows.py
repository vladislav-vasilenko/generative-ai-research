"""Derive standalone x2/x2.25 Kandinsky SR workflows from an installed official template."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = 'Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers'
VIDEO = 'k6_baseline_864x480_121f.mp4'
PROFILES = {'2x': '2x', '2_25x': '2.25x'}
TEMPLATE_SHA256 = 'da30efcaab1c4c0dcb0065a4abd4a1ad886d0b205f7c7b6eceaf33e422dfca4d'


def build(template: Path, output_dir: Path, input_video: str = VIDEO):
    if Path(input_video).name != input_video or input_video in ('', '.', '..'):
        raise ValueError('Use a video filename from the ComfyUI input folder')
    template_bytes = template.read_bytes()
    if hashlib.sha256(template_bytes).hexdigest() != TEMPLATE_SHA256:
        raise ValueError('Official template changed; review it and update source provenance before rebuilding')
    source = json.loads(template_bytes)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = {1: 'LoadVideo', 2: 'GetVideoComponents', 3: 'UNETLoader',
                4: 'Kandinsky6VSRUpscale', 5: 'CreateVideo', 6: 'SaveVideo',
                7: 'Kandinsky6SRVAELoader', 8: 'Kandinsky6LatentUpscalerLoader', 9: 'MarkdownNote'}
    assert {n['id']: n['type'] for n in source['nodes']} == expected, 'Review a changed official template'
    for profile, scale in PROFILES.items():
        graph = copy.deepcopy(source)
        nodes = {n['id']: n for n in graph['nodes']}
        nodes[1]['widgets_values'] = [input_video]
        nodes[1]['widgets_values_named'] = {'file': input_video}
        nodes[1]['title'] = 'Готовое видео — повторный Sampler не нужен'
        nodes[3]['widgets_values'] = [f'{BUNDLE}/transformer/diffusion_pytorch_model.safetensors', 'default']
        nodes[4]['widgets_values'] = [42, 'fixed', 2, 1, scale, 0.2, 24.0, 'none', False]
        nodes[4]['widgets_values_named'] = {'seed': 42, 'control_after_generate': 'fixed',
            'steps': 2, 'tiles_batch_size': 1, 'scale': scale, 'overlap': 0.2,
            'frame_rate': 24.0, 'target_resolution': 'none', 'use_nabla': False}
        nodes[4]['title'] = f'Kandinsky SR {scale} — distilled, 2 NFE per tile'
        nodes[5]['widgets_values'] = [24.0, 'auto', 'sRGB', 'none']
        nodes[5]['widgets_values_named'] = {'fps': 24.0, 'bit_depth': 'auto', 'color_space': 'sRGB', 'codec': 'none'}
        prefix = f'kandinsky6_research/sr_{profile}/baseline'
        nodes[6]['widgets_values'] = [prefix, 'mp4', 'h264', 'auto', 'auto']
        nodes[6]['widgets_values_named'] = {'filename_prefix': prefix, 'format': 'mp4',
                                           'format.codec': 'h264', 'format.codec.encoding': 'auto', 'codec': 'auto'}
        nodes[8]['widgets_values'] = [f'{BUNDLE}/latent_upscaler/diffusion_pytorch_model.safetensors', '2x']
        nodes[9]['size'] = [720, 560]
        nodes[9]['widgets_values'] = [f'''## Отдельный Kandinsky 6 SR {scale}

Вход — уже готовый MP4 со звуком: `{input_video}`. Здесь нет Qwen, CLIP, Lite generator, audio VAE или повторного T2AV sampling.

Цепочка: LoadVideo → frames/audio/fps → SR KVAE + latent-upscaler bank + distilled SR DiT → CreateVideo → SaveVideo.

Настройки: `scale={scale}`, latent-upscaler `2x`, batch=1, overlap=0.2, seed=42 fixed, `use_nabla=false`. Distilled checkpoint использует 2 вызова SR DiT на tile; это не два вызова на весь ролик. `target_resolution=none` сохраняет исходный SR размер.

Для источника 864×480 / 121 кадр / 24 FPS: 2x → 1728×960; 2.25x → 1952×1088 после округления промежуточного grid. При `target_resolution=fullhd` 2.25x результат вписывается в 1920×1080 с сохранением аспекта (примерно 1920×1070).

Нужны 8 файлов (3 safetensors + 5 JSON), всего 17.45 GB, из `{BUNDLE}`. JSON должны лежать рядом с соответствующими весами и в корне bundle. Кнопка Download models относится только к SR bundle. Проверьте свободный диск до загрузки.

Новые веса не загружаются при открытии графа. Плагин документирует CUDA-установку; выполнение SR на MPS в этом стенде пока не проверено. Текущие задания и cache генератора этот отдельный граф не меняет.

Результат: `output/{prefix}_*.mp4`. Аудио и FPS в CreateVideo берутся из выходов SR node, чтобы учесть его выравнивание длительности.''']
        nodes[9]['title'] = f'Настройка SR {scale} / модели / вход'
        nodes[9]['properties']['kandinsky6_download_package'] = 'kandinsky6-sr'
        for link_id, a, slot, b, index, kind in graph['links']:
            assert nodes[a]['outputs'][slot]['type'] == kind
            assert nodes[b]['inputs'][index]['type'] == kind
            assert nodes[b]['inputs'][index]['link'] == link_id
            assert link_id in nodes[a]['outputs'][slot]['links']
        assert graph['links'] == source['links']
        assert all(n['type'] not in {'Kandinsky6Sampler', 'DualCLIPLoader', 'Kandinsky6TextEncode',
                                    'Kandinsky6AudioVAELoader'} for n in graph['nodes'])
        graph['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 'k6-study-standalone-sr-' + profile))
        graph['revision'] = 0
        graph['extra'] = {'name': f'Kandinsky 6 SR - {scale}',
                          'description': 'Standalone SR of an existing MP4; source audio preserved.',
                          'source': 'https://github.com/kandinskylab/kandinsky-6-sr/tree/b15e2aac526519c51f59dfec2b848355bb8902de/comfyui',
                          'attribution': 'Derived from the Kandinsky Lab SR example, Apache-2.0; see workflows/licenses/kandinsky-6-sr-LICENSE.',
                          'modifications': '2026-10-30: Baseline input, fixed seed, distilled steps, batch=1, x2/x2.25 profiles, native output dimensions, audio wiring retained, MP4/H.264 output and study notes.',
                          'template_sha256': hashlib.sha256(template.read_bytes()).hexdigest()}
        path = output_dir / f'k6_sr_standalone_{profile}.json'
        path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + '\n')
        print(path, '9 nodes, 11 verified links, audio preserved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=Path, required=True, help='Installed official Video Super Resolution workflow JSON')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'workflows')
    parser.add_argument('--input-video', default=VIDEO)
    args = parser.parse_args()
    build(args.template, args.output_dir, args.input_video)
