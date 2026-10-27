"""Publish measured DiT/CrossDiT/π-Flow results, compact CSVs and selected plots."""
from pathlib import Path
import csv,hashlib,json,shutil
C=Path(__file__).resolve().parents[1];ROOT=C.parent
r=json.loads((C/'reports/execution.json').read_text());math=json.loads((C/'reports/math-rendering.json').read_text())
assert r['passed'] and len(r['notebooks'])==11 and all(x['executed'] for x in r['notebooks'])
assert math['passed']
for row in r['notebooks']:assert hashlib.sha256((C/'notebooks'/row['notebook']).read_bytes()).hexdigest()==row['source_sha256']
for path,h in r['helper_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
for row in math['files']:assert hashlib.sha256((C/'notebooks'/row['notebook']).read_bytes()).hexdigest()==row['source_sha256']
m={x['notebook'][:2]:x['measurements'] for x in r['notebooks']}
def csv_write(name,rows,fields):
 with (C/'reports'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows({k:row[k] for k in fields} for row in rows)
csv_write('euler-convergence.csv',m['06']['native_euler_convergence'],['steps','NFE','error'])
csv_write('policy-rollout-error.csv',m['07']['policy_rollout_error'],['grid','density','error'])
csv_write('policy-budget.csv',m['08']['policy_schedule_counts'],['network_NFE','grid_points_example','policy_density','policy_evaluations_per_modality'])
csv_write('heldout-trajectory.csv',m['09']['heldout_trajectory_probe'],['fraction_of_segment','state_MSE','network_calls_per_source_segment','policy_density'])
selected={'01_dit_primitives_and_tensor_geometry':['attention'],
 '02_single_stream_crossdit_k5':['conditioning_routes'],
 '03_dual_stream_crossdit_k6':['fusion_controls'],
 '05_visual_conditioning_references_and_rope':['reference_positions'],
 '06_denoise_loop_euler_cfg_and_partial_sampling':['euler_convergence'],
 '07_piflow_math_dx_policy':['time_warp','grid_error','endpoint_sensitivity','rollout_error'],
 '08_piflow_grid_head_and_native_sampler':['substep_accounting'],
 '09_miniature_policy_imitation_distillation':['imitation_training']}
for lab,names in selected.items():
 for name in names:shutil.copyfile(C/'artifacts'/lab/(name+'.png'),C/'reports/figures'/(lab[:2]+'_'+name+'.png'))
toy=m['09']['toy_pi_ID']
text=f'''# DiT / CrossDiT / π-Flow: измеренные CPU результаты

[Маршрут](../README.md) · [MIT questions](../MIT_BRIDGE.md) · [Границы source/adapter](../SOURCE_MAP.md) · [Полный execution JSON](execution.json) · [Формулы](math-rendering.json).

**11/11 notebooks, {sum(x['code_cells'] for x in r['notebooks'])} code cells, {sum(x['cells'] for x in r['notebooks'])} cells всего исполнены в fresh CPU kernels.** Дата UTC: {r['created_utc']}. Python {r['python'].split()[0]}. Проверено {r['source_files_checked']} selected source/config files и {r['MIT_originals_checked']} canonical MIT originals. Hash каждого notebook и adapter/helper сохранён. Все **{math['total_formulas']} display formulas** прошли strict KaTeX parse/render; это не GitHub/Colab renderer parity.

Материалы проверяют analytic fields, random reduced K5/K6 blocks и explicit dependency doubles. Pretrained weights, CUDA/MPS execution, production latency/VRAM, ComfyUI runtime и MLX parity **не выполнялись**.

## Гипотезы и controls

1. **Matched dense block:** K5 visual+text block и K6 single-stream counterpart при одинаковых весах/FP32 adapter дают одинаковую функцию. Проверяется narrow block, не whole-model parity.
2. **Fusion order/dependence:** original K6 block равен stepwise manual assembly; неправильный pre/post-text K/V порядок меняет output. Отдельный directional cut устраняет только выбранную зависимость в одном блоке.
3. **Solver:** native Euler сходится к exact matrix-exponential solution известного linear field; CFG меняет branch count, а не число интегрируемых trajectories.
4. **Policy:** grid approximation и ODE integration errors различаются; substep density не равна NFE×density.
5. **Imitation:** detached student rollout / frozen teacher сохраняют gradient routing; обучаемый MLP уменьшает fixed held-out velocity error на 2D field. Это single-seed toy experiment, а не статистический benchmark K6.

## Формулы и native блоки

| Проверка | Измерение | Область вывода |
|---|---|---|
| Affine velocity finite difference | max error {m['00']['affine_velocity_fd_max_error']:.3g} | Time convention/derivative |
| Correlated Gaussian velocity→score | max error {m['00']['correlated_Gaussian_score_bridge_max_error']:.3g} | Conditional expectation / covariance algebra |
| Manual attention vs native | max error {m['01']['attention_manual_error']:.3g} | Dense tiny FP32 attention |
| adaLN-Zero inner gradient | {m['01']['adaLN_zero']['inner_weight_gradient']:.3g} | Closed residual branch, first backward |
| adaLN modulation gradient | {m['01']['adaLN_zero']['modulation_gradient']:.6f} | Zero init не запрещает обучение gates |
| K5↔K6 matched block | max error {m['02']['k5_k6_dense_block_parity_error']:.3g} | Explicit FFN key/precision/backend mapping |
| Fused manual assembly | max error {m['03']['manual_block_max_error']:.3g} | Both streams, all four flags checked |
| Wrong K/V ordering | max difference {m['03']['wrong_KV_order_max_error']:.6f} | State ordering действительно меняет функцию |
| Audio→video Jacobian | norm {m['03']['cross_jacobian_norm']:.6f} | Local dependence присутствует |
| Audio→video cut Jacobian | norm {m['03']['cut_jacobian_norm']:.3g} | Chosen edge removed, one block |
| Two asymmetric cross-attentions | {m['03']['cross_attention_parameters']['measured']} parameters | Совпало с аналитическим count, excludes outside gates/norm |
| Masked text perturbation | max error {m['04']['bool_mask_error']:.3g} | Boolean keep-mask contract |
| RoPE precomputed/rebuilt | max error {m['06']['native_tiny_rope_AB']['max_error']:.3g} | Same coordinates, random native tiny |

![Fusion controls](figures/03_fusion_controls.png)

Conditioning helpers исполнялись на codec doubles; input validation, RGB normalization, scale, layout и no_grad проверены, semantic encoder/codec quality не измерялась. Visual reference tests покрывают batched/packed B=2,R=2, role masks, in-place restoration, projector idempotence и semantic t0/tlast RoPE.

![Reference positions](figures/05_reference_positions.png)

## Euler convergence и CFG accounting

[CSV](euler-convergence.csv). Exact reference: exp(−A)x₁ для A=[[0.7,0.2],[0.3,0.4]]. Source denoise loop использует negative shifted time increments и network time 1000σ.

| Steps / NFE | Endpoint L2 error |
|---|---|
'''
for row in m['06']['native_euler_convergence']:text+=f"| {row['NFE']} | {row['error']:.8f} |\n"
text+='''
![Euler convergence](figures/06_euler_convergence.png)

Для 8 steps CFG=1 дал 8 network calls; CFG=0/3 дали 16. Constant-bias field endpoints совпали с algebra (float32 tolerance). При frozen audio tensor не изменился и audio conditioning time был 0. Полный fused DiT при этом всё ещё может вычислять оба streams.

## π-Flow grid, время и реальная стоимость substeps

[Grid/solver error CSV](policy-rollout-error.csv), [budget CSV](policy-budget.csv). Warp inverse/Jacobian и interpolation curvature bound проверены. Polynomial branch протестирована как coefficient basis, не как grid samples. copy/detach не объявляются deep clone; MultimodalDXPolicy покрыта отдельно.

![Grid interpolation error](figures/07_grid_error.png)

![Policy rollout error](figures/07_rollout_error.png)

При fixed grid рост density уменьшает discretization error до representation floor на аналитическом curved-anchor примере. Near-endpoint velocity error усиливается как 1/max(σ,eps). Это не универсальная гарантия quality при росте substeps у trained модели.

| Network NFE | Grid G (example) | Configured density | Actual policy evaluations per modality |
|---|---|---|---|
'''
for row in m['08']['policy_schedule_counts']:text+=f"| {row['network_NFE']} | {row['grid_points_example']} | {row['policy_density']} | {row['policy_evaluations_per_modality']} |\n"
text+=f'''
Для S=10,r=.5,K=128 фактически {m['08']['policy_schedule_counts'][-1]['policy_evaluations_per_modality']} evaluations: 9×13+7. Native random grid-DiT при S=2 вызван ровно 2 раза при densities 8/32/128; outputs сохранили physical shapes и finite values. В batch с двумя durations source делает 12 vectorized policy calls, active updates на rows — 12 и 5; не 17 отдельных batch calls.

![Substep accounting](figures/08_substep_accounting.png)

## Маленький π-ID эксперимент

Teacher — frozen 2D linear field, student — MLP с G={toy['grid_points']} anchors, {toy['steps']} Adam steps, seed23. Training: fresh sources, detached on-policy rollout, 500 teacher batch calls / {toy['teacher_training_states']} queried training states. Fixed held-out probe содержит 512 sources и exact teacher states, построенные до training; target не используется optimizer.

| Метрика | До | После |
|---|---|---|
| Fixed held-out velocity MSE | {toy['heldout_before']:.8f} | {toy['heldout_after']:.8f} |
| Средний train loss, первые/последние 50 | {toy['first_50_train_mean']:.8f} | {toy['last_50_train_mean']:.8f} |

Held-out loss уменьшился в {toy['improvement_factor']:.2f} раза относительно этой инициализации. Teacher weights неизменны, teacher.grad отсутствует, rollout states не требуют gradient. Train и fixed probe distributions различаются, поэтому их loss не надо напрямую приравнивать.

![Toy policy imitation](figures/09_imitation_training.png)

[Held-out trajectory CSV](heldout-trajectory.csv):

| Fraction of segment | State MSE to exact teacher | Student source calls |
|---|---|---|
'''
for row in m['09']['heldout_trajectory_probe']:text+=f"| {row['fraction_of_segment']} | {row['state_MSE']:.8g} | {row['network_calls_per_source_segment']} |\n"
text+='''
Это поддерживает гипотезу о работоспособности данного toy imitation mechanism. Не подтверждает K6/AV perceptual quality, production distillation speed или Grönwall uniform assumptions по одному среднему loss. Реальная K6 distillation/Sim-LADD, teacher CFG и paired AV data не воспроизводились.

## Воспроизведение

```bash
uv run --locked --offline python dit/scripts/validate_labs.py --execute
uv run --locked --offline python dit/scripts/summarize_labs.py
```

Authored notebooks чистые; full executed copies и local HTML previews — ignored `artifacts/`. CSV, компактные JSON и 11 selected figures публикуются. [Lab10](../notebooks/10_comfyui_and_manual_research_protocol.ipynb) содержит ручной protocol после SSD gate; graph изучен статически, ни один ComfyUI job не отправлен.
'''
(C/'reports/VALIDATION.md').write_text(text)
print('Published 11-lab report, four CSVs and',sum(len(v) for v in selected.values()),'figures')
