"""Publish compact measured CPU results, CSV and selected plots after successful execution."""
from pathlib import Path
import csv,hashlib,json,shutil,sys
C=Path(__file__).resolve().parents[1]
r=json.loads((C/'reports/execution.json').read_text())
assert r['passed'] and len(r['notebooks'])==9 and all(n['executed'] for n in r['notebooks'])
for row in r['notebooks']:
 assert hashlib.sha256((C/'notebooks'/row['notebook']).read_bytes()).hexdigest()==row['source_sha256'],'Re-execute after changing a notebook'
assert hashlib.sha256((C/'labkit/vae_lab.py').read_bytes()).hexdigest()==r['helper_sha256']
lookup={n['notebook'][:2]:n['measurements'] for n in r['notebooks']}
bench=lookup['07']['runs']
with (C/'reports/tiny-benchmark.csv').open('w') as f:
 cols=['seed','policy','seconds_median','mse_to_full','psnr_to_input_db','seam_MSE','interior_MSE']
 w=csv.DictWriter(f,fieldnames=cols,lineterminator='\n');w.writeheader()
 for x in bench:w.writerow({k:x[k] for k in ['seed','policy','seconds_median','seam_MSE','interior_MSE']}|{'mse_to_full':x['difference_to_full']['mse'],'psnr_to_input_db':x['reconstruction_vs_input']['psnr_db']})
selected={'01_elbo_and_probabilistic_codec':['training'],
 '02_causal_convolution_and_geometry':['causal_mask'],
 '04_spatial_tiling_and_blending':['spatial_error'],
 '05_temporal_tiling_and_first_frame':['length_audit','temporal_error'],
 '06_free_vram_planner':['planner_sweep'],
 '07_controlled_benchmark':['benchmark','benchmark_error_maps']}
for lab,names in selected.items():
 for name in names:shutil.copyfile(C/'artifacts'/lab/(name+'.png'),C/'reports/figures'/(lab[:2]+'_'+name+'.png'))
text=f'''# HunyuanVideo VAE: проверка и измеренные результаты

[Маршрут](../README.md) · [Источник и область выводов](../SOURCE_MAP.md) · [Полный JSON](execution.json) · [CSV benchmark](tiny-benchmark.csv).

**{len(r['notebooks'])}/9 notebooks исполнены в свежих CPU kernels**, {sum(n['code_cells'] for n in r['notebooks'])} code cells, {sum(n['cells'] for n in r['notebooks'])} cells всего. Дата UTC: {r['created_utc']}. Python: {sys.version.split()[0]}, PyTorch: {lookup['00']['torch_version']}. Pinned source SHA: `{r['pinned_source_sha256']}`. Report содержит hashes каждого notebook и helper, чтобы не приписывать старый результат новому коду.

Pretrained weights не загружались. GPU peak memory, Diffusers runtime, MPS execution и ComfyUI jobs **не проверялись**. Timings ниже — локальный CPU, synthetic RGB и random reduced native encoder/decoder; это не latency/quality Hunyuan checkpoint.

Все 32 display formulas прошли строгий KaTeX parse/render: [отчёт](math-rendering.json). Два локальных HTML previews и графики осмотрены; это не проверка GitHub/Colab browser parity. Ссылки на MIT Q4.1–Q5.1 и исходные cell indices сверены по canonical notebook.

## Теория и локальные операторы

| Проверка | Измерение | Допустимый вывод |
|---|---|---|
| ELBO MC | {lookup['01']['elbo']['MC']:.6f}; exact log p(x) {lookup['01']['elbo']['exact_log_marginal']:.6f}; MC SE {lookup['01']['elbo']['standard_error']:.6f} | Lower bound согласован с известной scalar marginal |
| Closed-form KL vs torch | max error {lookup['01']['kl_max_error']:.3g} | Gaussian algebra и reductions согласованы |
| MIT Q4.7 Gaussian reconstruction NLL vs torch | max error {lookup['01']['MIT_Q4_7_NLL_max_error']:.3g} | Learned-variance likelihood и sum reduction согласованы |
| Reparameterization gradient | max finite-difference error {lookup['01']['gradient_max_error']:.3g} | Pathwise derivative проверена при фиксированном ε |
| Conv future perturbation | prefix error {lookup['02']['causal_conv_prefix_error']:.3g} | Default isotropic native conv causal |
| GroupNorm future perturbation | prefix error {lookup['02']['groupnorm_future_prefix_error']:.6f} | Whole-tensor GN не strict online causal |
| Manual attention vs SDPA | max error {lookup['03']['attention_formula_sdpa_max_error']:.3g} | Attention formula/mask согласованы на малом тензоре; не full Diffusers parity |
| Local convolution with halo | max error {lookup['04']['halo_local_conv']['with_halo_max_error']:.3g} | Halo+crop exact для этого локального оператора |

![Linear VAE learning curve](figures/01_training.png)

## Stitching и coverage

Aligned coordinate oracle дал **точное совпадение** spatial и temporal outputs с full. Это проверка indices без context. Random native codec дал ненулевое отличие, несмотря на одинаковые input/weights.

Spatial counterexample: L=88,Q=32,S=24 оставляет {lookup['04']['tail_counterexample']['coverage']['missing']} pixels; фактическая latent width={lookup['04']['tail_counterexample']['actual']}, ожидается 11. Этот вызов относится к приложенной адаптации, не ко всем upstream versions. Temporal sweep проверяет настоящий `_encode` dispatch; короткий input не отправлялся в неподходящий tile path.

| Input F | Expected Fz | Actual Fz | Shape |
|---|---|---|---|
'''
for x in lookup['05']['temporal_encode_lengths']:text+=f"| {x['frames']} | {x['expected_Fz']} | {x['actual_Fz']} | {'OK' if x['same_shape'] else 'tail missing'} |\n"
text+='''
![Temporal coverage audit](figures/05_length_audit.png)

Original planner на CPU/MPS использует бесконечный budget. Synthetic CUDA sweep выявляет, где grid coverage требует дополнительной проверки; very low budget может давать invalid stride. Это не измеренная VRAM и не подтверждение OOM safety.

![Synthetic budget sweep](figures/06_planner_sweep.png)

## Paired CPU tiny benchmark

Input B=1,C=3,F=33,H=W=48. Native widths 4/8/8/8, latent=2, layers=1, mid attention=False. Mode posterior, three seeds; у каждой policy warmup и 3 timed repeats. «MSE к full» означает deviation от той же случайной модели, а «PSNR к input» — reconstruction случайного codec на RGB range=2.

| Seed | Policy | Median CPU seconds | MSE to full | PSNR to input dB | Seam MSE | Interior MSE |
|---|---|---|---|---|---|---|
'''
for x in bench:text+=f"| {x['seed']} | {x['policy']} | {x['seconds_median']:.5f} | {x['difference_to_full']['mse']:.6f} | {x['reconstruction_vs_input']['psnr_db']:.3f} | {x['seam_MSE']:.6f} | {x['interior_MSE']:.6f} |\n"
text+='''
![CPU paired comparison](figures/07_benchmark.png)

![Spatial difference maps](figures/07_benchmark_error_maps.png)

На выбранной grid все policies сохранили BCTHW. Spatial/temporal/combined outputs отличались от full у всех трёх seeds, включая interior. Это поддерживает гипотезу о неэквивалентности с context/normalization. Full baseline совпал сам с собой. Нулевой oracle error и ненулевой native error показывают разницу index correctness и full-model equivalence.

Tiling повторяет overlapping вычисления, поэтому CPU time может вырасти. Отсюда нельзя заключить, сколько VRAM экономится или какой tile быстрее на CUDA/MPS. Header source утверждает lower peak относительно Diffusers, но это требует отдельного pretrained paired benchmark.

## Воспроизведение и следующий шаг

```bash
uv run --locked --offline python vae/scripts/validate_labs.py --execute
uv run --locked --offline python vae/scripts/summarize_labs.py
```

Clean authored notebooks не содержат outputs; полные исполненные копии — в ignored `artifacts/executed/`. CSV/JSON и выбранные PNG публикуются как компактное evidence. [Lab 08](../notebooks/08_pretrained_and_comfyui_protocol.ipynb) содержит ручной no-download pretrained/ComfyUI протокол после SSD gate. Для реального вывода нужны local checkpoint SHA, clips, paired float reconstructions, synchronized latency и настоящий memory measurement.
'''
(C/'reports/VALIDATION.md').write_text(text)
print('Published measured report and',len(bench),'benchmark rows')
