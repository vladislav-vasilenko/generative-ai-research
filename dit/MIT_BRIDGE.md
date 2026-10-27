# MIT 6.S184 → DiT/CrossDiT и π-Flow: точное соответствие

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../curriculum/README.md) · [Полное содержание](../curriculum/CONTENTS.md).
<!-- study-navigation:end -->

[Маршрут](README.md) · [Курс MIT](https://diffusion.csail.mit.edu/2026/index.html) · [Notes](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Lecture guides](../curriculum/CONTENTS.md).

Сначала пройдите исходные понятия, затем подробный архитектурный case. PDF pages — one-based; notebook cells — **zero-based**. Проверены headings/количество cells закреплённых originals; они не переписываются и не запускаются как готовые решения.

| MIT material / original | Что освоить | Наши labs / измерение |
|---|---|---|
| Lecture 1, notes §2.1 pp.7–9; [Lab 1](../lecture_01/references/lab_one_original.ipynb) Q1.1, cells 9–13 | Euler как quadrature; truncation vs global error | [00](notebooks/00_probability_path_and_learning_route.ipynb), [06](notebooks/06_denoise_loop_euler_cfg_and_partial_sampling.ipynb), [07](notebooks/07_piflow_math_dx_policy.ipynb): exact field и convergence |
| Lecture 2, notes §3.1–3.2 pp.14–18; [Lab 2](../lecture_02/references/lab_two_original.ipynb) Q2.1, cells 14–20; Q2.2, 21–24; Q2.3, 25–30 | Gaussian/affine paths, target, time convention | [00](notebooks/00_probability_path_and_learning_route.ipynb): derivative/sign; [07](notebooks/07_piflow_math_dx_policy.ipynb): u→x0 и warp |
| Lecture 2, notes §3.3 pp.19–24; Lab 2 Q3.1, cells 38–49 | Conditional squared regression → marginal field | [00](notebooks/00_probability_path_and_learning_route.ipynb), [09](notebooks/09_miniature_policy_imitation_distillation.ipynb): отличие data CFM и teacher imitation |
| Lecture 3A, notes §4 pp.25–33; Lab 2 Q2.4 cells 31–36, Q3.3 59–63 | Score по x, conditional/marginal, velocity conversion | [00](notebooks/00_probability_path_and_learning_route.ipynb): Gaussian score/velocity bridge и endpoints |
| Lecture 3B, notes §5 pp.34–40; [Lab 3](../lecture_04/references/lab_three_original.ipynb) Q2.2 cells 27–31, Q2.3 32–33, sanity 34–36 | Class dropout, null branch, CFG | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb): условия; [06](notebooks/06_denoise_loop_euler_cfg_and_partial_sampling.ipynb): exact NFE и CFG limits |
| Lecture 4, notes §6.1 pp.41–46; Lab 3 Q3.1 cells 38–39 | Fourier time encoder | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb): sinusoid→MLP→modulation |
| Lab 3 Q3.2 cells 40–42 | Patchifier, divisibility, token count | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb): exact index roundtrip и native input projection |
| Lab 3 Q3.3 cells 43–44 | Attention, modulation, residual blocks | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb), [02](notebooks/02_single_stream_crossdit_k5.ipynb), [03](notebooks/03_dual_stream_crossdit_k6.ipynb) |
| Lab 3 Q3.4 cells 45–46, Q3.5 cells 47–48 | Depatchifier и whole model contracts | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb), [08](notebooks/08_piflow_grid_head_and_native_sampler.ipynb): standard/grid head |
| Lab 3 pixel training cells 49–54 | Held-out field error отдельно от sampling | [06](notebooks/06_denoise_loop_euler_cfg_and_partial_sampling.ipynb), [09](notebooks/09_miniature_policy_imitation_distillation.ipynb) |
| Lecture 4, notes §6.2 pp.46–52; Lab 3 Parts 4–5 cells 55–78, Q5.1 76–77 | Frozen codec при latent DiT training, encode/decode units | [VAE route](../vae/MIT_LAB3_BRIDGE.md), [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb), [05](notebooks/05_visual_conditioning_references_and_rope.ipynb), [10](notebooks/10_comfyui_and_manual_research_protocol.ipynb) |

## Что мы добавляем сверх MIT

- Text token cross-attention K5: SA→text CA→FFN; original class comparison, token vs pooled routes.
- Dual AV CrossDiT K6: asymmetric widths, bidirectional residual CA, pre/post-text states, flags/gates и directional Jacobian controls.
- Clean modality encoding/reference tails: batched/packed indices, semantic RoPE и constraint projector.
- Production loop wiring: cache, distinct null conditions, partial sampling, actual transformer calls.
- π-Flow DX: policy parameterization, raw/shifted time, interpolation/polynomial, native cheap substeps и grid output.
- Toy policy imitation: [π-Flow paper §4/Alg.1](https://arxiv.org/html/2510.14974v1#S4), а не отсутствующее MIT упражнение. K6-specific release behaviour читается по source и [K6 §4.5](https://arxiv.org/html/2610.05608v1#S4.SS5).

MIT MNIST class-conditioning не содержит Qwen/CLIP, audio/video CrossDiT или готовой π-Flow distillation. Это перенос базовых навыков на новые архитектуры; приписывать extensions исходному курсу было бы неверно.

## Три контрольные точки

1. **После MIT базы / lab00–01:** вывести знак velocity, сделать patch roundtrip, собрать attention и объяснить gradient gate. Не переходить только по факту Run All.
2. **После lab02–06:** нарисовать Q/K/V для каждого направления; отдельно объяснить conditioning tensor, mask, RoPE и time; показать parity и one-factor ablation.
3. **После lab07–10:** отличить G/NFE/policy evaluations, вывести warp и DX, получить held-out imitation evidence, составить real measurement protocol с ограничениями.

На M3 Max CPU опыты автономны. Canonical training cells с CUDA и большими budgets требуют отдельной рабочей копии, device adaptation и проверки памяти; этот курс не заявляет, что исходный MIT full MNIST training был выполнен заново. Big-weight ComfyUI/MLX этап остаётся отложенным.
