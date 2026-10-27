"""Publish a concise evidence report and selected figures from successful local executions."""
from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'reports/component-validation.json').read_text())
assert report['passed'] and all(r['executed'] for r in report['notebooks']), 'Run full --execute validation first'
rows={r['notebook'][:2]:r for r in report['notebooks']}
def metric(n,key):return rows[n]['measurements'][key]
count=sum(r['cells'] for r in rows.values());code=sum(r['code_cells'] for r in rows.values())
text=f'''# K6 component labs: проверка 27 октября 2026

[Курс](../README.md) · [Полное содержание](../../curriculum/CONTENTS.md) · [Машиночитаемый execution report](component-validation.json).

**14 новых notebooks выполнены в отдельных свежих CPU kernels: {count} ячеек / {code} code cells.** Все assertions прошли, локальные ссылки проверены, {report['pinned_source_hashes_checked']} pinned source hashes совпали. Курс работает через основное зафиксированное окружение `uv run --locked --offline`; модельные веса/датасеты не загружались, ComfyUI jobs не отправлялись, MLX не исполнялся. Colab bootstrap подготовлен, но удалённый Colab runtime не проверялся.

## Что именно исполнялось

- Оригинальные native DiT / fused blocks / PiFlow wrapper на уменьшенных размерах, случайные веса; dense SDPA adapter явно описан в labkit.
- Оригинальные tensor-only schedule/CFG/Euler/PiFlow/conditioning/codec-wrapper functions.
- Изолированное оригинальное определение Hunyuan causal Conv3d, без загрузки полного VAE.
- Учебный MLP-VAE, synthetic joint FM training, аналитические ODE/grid controls.
- Там, где нужны pretrained encoders/decoder/vocoder, используются явно названные test doubles; они проверяют API/layout, не медиакачество.

## Измерения

| Опыт | Получено | Область вывода |
|---|---|---|
| Full Lite Distill constructor / meta | {metric('01','total_parameters'):,} parameters | Точная структура выбранного native config, без чтения 3B весов |
| Text padding perturbation | max error {metric('02','masked_value_perturbation_max_error'):.1g} | Mask закрывает padded K/V |
| Projection cache | {metric('02','projection_calls_two_reads')} call на 2 reads | Time-independent projection reuse |
| Toy beta-VAE objective | {metric('03','toy_objective_initial'):.6f} → {metric('03','toy_objective_final'):.6f} | Synthetic moving-blob data, 100 steps |
| Native causal Conv3d | past-output max error {metric('03','native_causal_conv_past_error'):.1g} | Изменено только будущее, kernel=3/stride=1 |
| STFT inverse | max error {metric('04','stft_inverse_max_error'):.3g} | Сохранённая complex phase, не mel vocoder |
| Visual patch manual/native | max error {metric('05','visual_patch_max_error'):.1g} | Порядок patch features |
| Zero residual gate | identity max error {metric('05','zero_gate_identity_error'):.1g} | При этом modulation gradient ненулевой |
| Manual attention / SDPA | max error {metric('06','manual_sdpa_max_error'):.3g} | FP64, small unmasked case |
| Text perturbation after text-CA cut | max change {metric('07','token_condition_cut_delta'):.1g} | Изолирован token-level condition path |
| AV directional cuts | оба cut deltas = 0; active deltas ненулевые | Проверка wiring одного native fused block |
| Audio-output loss → video input | gradient norm {metric('08','audio_loss_gradient_to_video_on'):.6g} active / {metric('08','audio_loss_gradient_to_video_off'):.1g} cut | Межмодальный gradient path |
| Tiny joint FM held-out video MSE | {metric('09','heldout_initial_video_audio')[0]:.6f} → {metric('09','heldout_final_video_audio')[0]:.6f} | {metric('09','trainable_parameters'):,} parameters, synthetic conditioning |
| Tiny joint FM held-out audio MSE | {metric('09','heldout_initial_video_audio')[1]:.6f} → {metric('09','heldout_final_video_audio')[1]:.6f} | 100 training steps, один fixed validation batch |
| Oracle constant-grid policy | endpoint error {metric('10','oracle_constant_grid_endpoint_error'):.3g} | Аналитический контроль, student не обучался |
| Native tiny PiFlow sampler | {metric('10','native_tiny_sampler_model_NFE')} model NFE, finite outputs обеих модальностей | Grid-head/rollout contract, random weights |
| Native Euler CFG accounting | 4 calls при w=1, 8 при w=0/3 | 4 steps, analytic constant predictor double |
| Reference append/restore | max restore error 0 | Temporal reference + mask/TRE layout |
| ComfyUI Smoke | 12 nodes / 14 links | Только JSON/static inspection; runtime не запускался |

Исторические округления статьи по потокам не заменяются нашими числами: в `meta`-подсчёте fusion включает также cross-modal modulation. Логический размер attention score matrix — арифметическая оценка, а не измеренная peak memory SDPA/FlashAttention. CPU microtiming из notebook 13 относится только к tiny CPU сети и не экстраполируется на M3 GPU или pretrained Lite.

## Графики и их смысл

![Dual-stream architecture](figures/components/dual_stream_architecture.png)

Два потока имеют разные widths; стрелки определяют источник K/V и получателя Q. Детали snapshot states объясняются в notebook 08.

![Directional cross-modal ablation](figures/components/cross_modal_ablation.png)

При отключении нужного cross gate исчезает соответствующая input/output sensitivity. Это контроль структуры вычислений, не lip-sync benchmark.

![Joint Flow Matching learning](figures/components/joint_fm_training.png)

Снижение loss относится к синтетическим latent distributions и небольшой native сети. Для validation используются другой seed/batch; статистика по нескольким обучающим seeds здесь ещё не проведена.

![Toy VAE](figures/components/vae_training.png)

MLP beta-VAE показывает reconstruction/KL objective; это не тренировка Hunyuan/MMAudio codec.

![Euler integration error](figures/components/euler_error.png)

У поля u=x известен точный endpoint, поэтому наблюдаем discretization error отдельно от ошибки neural predictor.

## Как повторить

```bash
uv run --locked --offline python kandinsky-6/scripts/validate_component_notebooks.py --execute
uv run --locked --offline python kandinsky-6/scripts/summarize_components.py
```

Execution outputs остаются в ignored `reports/executed/`, метрики и локальные изображения — `artifacts/component_labs/`. Здесь опубликованы summary JSON и выбранные фигуры. Notebook sources публикуются без outputs. Генераторы и hashes позволяют проверить, какие именно версии исполнялись.

## Что остаётся отдельным этапом

Pretrained AV generation, реальная реконструкция кодеками в ComfyUI, MPS/MLX parity, полный joint MLX pipeline, обучение voice-FiLM на VCTK latents, waveform sampling с адаптером и SECS/WER/FAD/no-reference regression. Эти этапы не выводятся из выполненных CPU контролей; прежнее решение пользователя об SSD 50 GB+ сохраняется. [Статус исследования](../STATUS.md).
'''
(ROOT/'reports/COMPONENT_VALIDATION.md').write_text(text)
figures={'08_dual_stream_audio_video_crossdit':['dual_stream_architecture','cross_modal_ablation'],
         '09_flow_matching_training':['joint_fm_training'],'03_video_vae':['vae_training'],
         '11_sampling_cfg_and_nfe':['euler_error']}
dest=ROOT/'reports/figures/components';dest.mkdir(parents=True,exist_ok=True)
for stem,names in figures.items():
    for name in names:shutil.copyfile(ROOT/f'artifacts/component_labs/{stem}/{name}.png',dest/f'{name}.png')
print('Published report:',count,'cells,',code,'code cells,',sum(map(len,figures.values())),'figures')
