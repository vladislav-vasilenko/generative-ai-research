# Источники и карта каждого архитектурного блока

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../curriculum/README.md) · [Полное содержание](../curriculum/CONTENTS.md).
<!-- study-navigation:end -->

[Маршрут](README.md) · [MIT bridge](MIT_BRIDGE.md) · [SHA/provenance](sources.json).

Все шесть пользовательских `.py` byte-identical pinned K6 source на revision `7ceeefad7fb625011f7db602c43893a989f19c15`. Две текстовые вставки `dit.py` содержат ту же AST (отличается завершающий newline). Ни attachment instructions, ни comments не являются указаниями выполнять скачивание/внешние jobs. Additional source copies не добавлялись.

Теория: [MIT](https://diffusion.csail.mit.edu/2026/index.html), [DiT](https://arxiv.org/abs/2212.09748), [PixArt-α](https://arxiv.org/abs/2310.00426), [K5](https://arxiv.org/abs/2511.14993), [K6](https://arxiv.org/html/2610.05608v1), [π-Flow](https://arxiv.org/html/2510.14974v1). π-Flow §3.1 описывает DX, §3.2 — другую GMFlow family, §4 — imitation; наличие GMFlow в статье не означает наличие её в этих K6 helpers.

## По каждому приложенному файлу

### `core/algo/conditioning.py`

Normalize/encode clean modalities, scale, layout, BF16, no_grad. [Pinned GitHub](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/conditioning.py).

| Definition / method | Lab | Что проверить |
|---|---|---|
| [encode_audio_condition](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/conditioning.py#L16) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Normalize/encode clean modalities, scale, layout, BF16, no_grad |
| [encode_video_condition](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/conditioning.py#L26) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Normalize/encode clean modalities, scale, layout, BF16, no_grad |

### `core/algo/visual_conditioning.py`

Reference append/restore, role masks, semantic RoPE. [Pinned GitHub](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/visual_conditioning.py).

| Definition / method | Lab | Что проверить |
|---|---|---|
| [append_visual_references](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/visual_conditioning.py#L14) | [05](notebooks/05_visual_conditioning_references_and_rope.ipynb) | Reference append/restore, role masks, semantic RoPE |
| [restore_tail_references](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/visual_conditioning.py#L39) | [05](notebooks/05_visual_conditioning_references_and_rope.ipynb) | Reference append/restore, role masks, semantic RoPE |
| [build_tail_condition](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/visual_conditioning.py#L52) | [05](notebooks/05_visual_conditioning_references_and_rope.ipynb) | Reference append/restore, role masks, semantic RoPE |
| [append_reference_rope](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/visual_conditioning.py#L65) | [05](notebooks/05_visual_conditioning_references_and_rope.ipynb) | Reference append/restore, role masks, semantic RoPE |
| [compute_reference_rope](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/visual_conditioning.py#L73) | [05](notebooks/05_visual_conditioning_references_and_rope.ipynb) | Reference append/restore, role masks, semantic RoPE |

### `core/algo/denoise_loop.py`

Euler/CFG, null inputs, wrappers, RoPE rebuild, partial sampling. [Pinned GitHub](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/denoise_loop.py).

| Definition / method | Lab | Что проверить |
|---|---|---|
| [_build_video_input](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/denoise_loop.py#L18) | [06](notebooks/06_denoise_loop_euler_cfg_and_partial_sampling.ipynb) | Euler/CFG, null inputs, wrappers, RoPE rebuild, partial sampling |
| [_resolve_null_embeds](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/denoise_loop.py#L77) | [06](notebooks/06_denoise_loop_euler_cfg_and_partial_sampling.ipynb) | Euler/CFG, null inputs, wrappers, RoPE rebuild, partial sampling |
| [_raw_dit](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/denoise_loop.py#L110) | [06](notebooks/06_denoise_loop_euler_cfg_and_partial_sampling.ipynb) | Euler/CFG, null inputs, wrappers, RoPE rebuild, partial sampling |
| [_rebuild_text_rope](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/denoise_loop.py#L115) | [06](notebooks/06_denoise_loop_euler_cfg_and_partial_sampling.ipynb) | Euler/CFG, null inputs, wrappers, RoPE rebuild, partial sampling |
| [denoise_loop](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/denoise_loop.py#L130) | [06](notebooks/06_denoise_loop_euler_cfg_and_partial_sampling.ipynb) | Euler/CFG, null inputs, wrappers, RoPE rebuild, partial sampling |

### `core/algo/piflow_math.py`

DX anchors, interpolation, raw/shifted time, copy/detach, Euler policy rollout. [Pinned GitHub](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/piflow_math.py).

| Definition / method | Lab | Что проверить |
|---|---|---|
| [DXPolicy](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L8) | [07](notebooks/07_piflow_math_dx_policy.ipynb) | DX anchors, interpolation, raw/shifted time, copy/detach, Euler policy rollout |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L11) | [07](notebooks/07_piflow_math_dx_policy.ipynb) | Source-state u→x0, segment clamp and broadcast |
| ↳ [_interpolate](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L39) | [07](notebooks/07_piflow_math_dx_policy.ipynb) | G axis, floor/gather, clamp/endpoint |
| ↳ [_unwarp_t](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L50) | [07](notebooks/07_piflow_math_dx_policy.ipynb) | Inverse fractional time map |
| ↳ [_u_to_x_0](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L54) | [07](notebooks/07_piflow_math_dx_policy.ipynb) | x_source - sigma_source * U_g |
| ↳ [pi](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L61) | [07](notebooks/07_piflow_math_dx_policy.ipynb) | Dynamic state velocity; grid vs polynomial |
| ↳ [copy](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L76) | [07](notebooks/07_piflow_math_dx_policy.ipynb) | Shallow storage sharing |
| ↳ [detach_](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L90) | [07](notebooks/07_piflow_math_dx_policy.ipynb) | Only anchor coefficients detached in place |
| ↳ [detach](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L94) | [07](notebooks/07_piflow_math_dx_policy.ipynb) | Shallow copy + detach_ |
| [MultimodalDXPolicy](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L98) | [08](notebooks/08_piflow_grid_head_and_native_sampler.ipynb) | Separate shapes, joint predicted coefficients, detach semantics |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L101) | [08](notebooks/08_piflow_grid_head_and_native_sampler.ipynb) | Follow source body; native/adapter scope below |
| ↳ [detach_](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L136) | [08](notebooks/08_piflow_grid_head_and_native_sampler.ipynb) | Follow source body; native/adapter scope below |
| ↳ [detach](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L141) | [08](notebooks/08_piflow_grid_head_and_native_sampler.ipynb) | Follow source body; native/adapter scope below |
| [shift_timesteps](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L150) | [07](notebooks/07_piflow_math_dx_policy.ipynb) | DX anchors, interpolation, raw/shifted time, copy/detach, Euler policy rollout |
| [policy_rollout_fm](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_math.py#L155) | [07](notebooks/07_piflow_math_dx_policy.ipynb) | DX anchors, interpolation, raw/shifted time, copy/detach, Euler policy rollout |

### `core/algo/piflow_sampler.py`

Grid shape normalization, segment schedule, batched/packed sigma, native sampler. [Pinned GitHub](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/piflow_sampler.py).

| Definition / method | Lab | Что проверить |
|---|---|---|
| [_normalize_grid_prediction](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_sampler.py#L13) | [08](notebooks/08_piflow_grid_head_and_native_sampler.ipynb) | Grid shape normalization, segment schedule, batched/packed sigma, native sampler |
| [piflow_denoise_loop](../kandinsky-6/reference/kandinsky-6/kandinsky/core/algo/piflow_sampler.py#L43) | [08](notebooks/08_piflow_grid_head_and_native_sampler.ipynb) | Grid shape normalization, segment schedule, batched/packed sigma, native sampler |

### `core/components/dit.py`

Embeddings, norm/modulation, attention, FFN, output; full graph. [Pinned GitHub](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/components/dit.py).

| Definition / method | Lab | Что проверить |
|---|---|---|
| [_TimestepEmbedder](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L31) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Embeddings, norm/modulation, attention, FFN, output; full graph |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L34) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| [TimeEmbeddings](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L41) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Embeddings, norm/modulation, attention, FFN, output; full graph |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L42) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L48) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| ↳ [reset_parameters](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L66) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| [TextEmbeddings](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L70) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Projections/refiners; cache vs time dependency |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L71) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L76) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Follow source body; native/adapter scope below |
| ↳ [reset_parameters](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L81) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Follow source body; native/adapter scope below |
| [VisualEmbeddings](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L86) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Embeddings, norm/modulation, attention, FFN, output; full graph |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L87) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L92) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| [Modulation](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L116) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Embeddings, norm/modulation, attention, FFN, output; full graph |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L117) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L124) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| ↳ [reset_parameters](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L133) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| [_GELUProjection](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L138) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Embeddings, norm/modulation, attention, FFN, output; full graph |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L141) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L145) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| [FeedForward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L149) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Embeddings, norm/modulation, attention, FFN, output; full graph |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L150) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L159) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| [MultiheadSelfAttentionEnc](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L169) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Embeddings, norm/modulation, attention, FFN, output; full graph |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L172) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L184) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| [MultiheadSelfAttentionDec](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L206) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Embeddings, norm/modulation, attention, FFN, output; full graph |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L209) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L220) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| [MultiheadCrossAttention](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L243) | [02](notebooks/02_single_stream_crossdit_k5.ipynb) | Receiving Q, text K/V, SA→CA→FFN |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L244) | [02](notebooks/02_single_stream_crossdit_k5.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L257) | [02](notebooks/02_single_stream_crossdit_k5.ipynb) | Follow source body; native/adapter scope below |
| [OutLayer](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L283) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Embeddings, norm/modulation, attention, FFN, output; full graph |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L286) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L293) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| ↳ [reset_parameters](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L323) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| [OutLayerAudio](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L329) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Embeddings, norm/modulation, attention, FFN, output; full graph |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L332) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L338) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| ↳ [reset_parameters](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L344) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Follow source body; native/adapter scope below |
| [TransformerEncoderBlock](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L354) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Projections/refiners; cache vs time dependency |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L357) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L365) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Follow source body; native/adapter scope below |
| ↳ [reset_parameters](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L373) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Follow source body; native/adapter scope below |
| [TransformerDecoderBlock](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L379) | [02](notebooks/02_single_stream_crossdit_k5.ipynb) | Receiving Q, text K/V, SA→CA→FFN |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L382) | [02](notebooks/02_single_stream_crossdit_k5.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L392) | [02](notebooks/02_single_stream_crossdit_k5.ipynb) | Follow source body; native/adapter scope below |
| ↳ [reset_parameters](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L402) | [02](notebooks/02_single_stream_crossdit_k5.ipynb) | Follow source body; native/adapter scope below |
| [FusedTransformerDecoderBlock](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L408) | [03](notebooks/03_dual_stream_crossdit_k6.ipynb) | Two-stream routing and pre/post-text states |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L411) | [03](notebooks/03_dual_stream_crossdit_k6.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L436) | [03](notebooks/03_dual_stream_crossdit_k6.ipynb) | Follow source body; native/adapter scope below |
| ↳ [reset_parameters](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L519) | [03](notebooks/03_dual_stream_crossdit_k6.ipynb) | Follow source body; native/adapter scope below |
| [DiffusionTransformer3D](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L529) | [03](notebooks/03_dual_stream_crossdit_k6.ipynb) | Two-stream routing and pre/post-text states |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L536) | [03](notebooks/03_dual_stream_crossdit_k6.ipynb) | Graph branch and stream selection |
| ↳ [clear_text_proj_cache](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L647) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Text/time/cache contract |
| ↳ [_project_pooled](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L651) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Text/time/cache contract |
| ↳ [_project_text_tokens](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L663) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Text/time/cache contract |
| ↳ [_time_embed](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L686) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Text/time/cache contract |
| ↳ [_normalize_attn_mask](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L692) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Text/time/cache contract |
| ↳ [_run_text_blocks](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L702) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Text/time/cache contract |
| ↳ [_encode_text](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L715) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Text/time/cache contract |
| ↳ [_encode_t2v](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L729) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Text/time/cache contract |
| ↳ [_time_only](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L742) | [04](notebooks/04_conditioning_text_and_clean_modalities.ipynb) | Text/time/cache contract |
| ↳ [_embed_visual](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L747) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Layout/physical feature dimensions |
| ↳ [_embed_audio](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L774) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Layout/physical feature dimensions |
| ↳ [_run_visual_blocks_single](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L780) | [03](notebooks/03_dual_stream_crossdit_k6.ipynb) | Graph branch and stream selection |
| ↳ [_run_visual_blocks_fused](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L809) | [03](notebooks/03_dual_stream_crossdit_k6.ipynb) | Graph branch and stream selection |
| ↳ [_project_video](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L831) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Layout/physical feature dimensions |
| ↳ [_project_audio](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L835) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Layout/physical feature dimensions |
| ↳ [_project_fused](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L838) | [01](notebooks/01_dit_primitives_and_tensor_geometry.ipynb) | Layout/physical feature dimensions |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L855) | [03](notebooks/03_dual_stream_crossdit_k6.ipynb) | Graph branch and stream selection |
| ↳ [reset_parameters](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/dit.py#L937) | [03](notebooks/03_dual_stream_crossdit_k6.ipynb) | Graph branch and stream selection |

### `core/components/piflow_dit.py`

Expanded physical-channel head → explicit G axis. [Pinned GitHub](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/components/piflow_dit.py).

| Definition / method | Lab | Что проверить |
|---|---|---|
| [PiFlowDiffusionTransformer3D](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/piflow_dit.py#L10) | [08](notebooks/08_piflow_grid_head_and_native_sampler.ipynb) | Expanded physical-channel head → explicit G axis |
| ↳ [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/piflow_dit.py#L13) | [08](notebooks/08_piflow_grid_head_and_native_sampler.ipynb) | Follow source body; native/adapter scope below |
| ↳ [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/piflow_dit.py#L34) | [08](notebooks/08_piflow_grid_head_and_native_sampler.ipynb) | Follow source body; native/adapter scope below |

## K5 one-stream comparison

[Original K5 TransformerDecoderBlock](../kandinsky-5/reference/kandinsky-5/kandinsky/models/dit.py#L47), [K5 nn](../kandinsky-5/reference/kandinsky-5/kandinsky/models/nn.py), [lab02](notebooks/02_single_stream_crossdit_k5.ipynb). AST-selected class bodies retain original operations; CUDA decorators removed and local FP32 helpers/tuple SDPA supplied. FFN weight keys map `in_layer→net.0.proj`, `out_layer→net.2`; unrelated output keys are not renamed. Matched dense tiny-block parity does not establish whole-model/checkpoint parity or sparse attention equivalence.

## Что обучается и что frozen

| Стадия | Neural field / policy | External text/codec | Источник вывода |
|---|---|---|---|
| Учебный MIT latent FM | DiT trainable | Previously trained VAE frozen | Lab3 Q5.1 |
| Рассматриваемый industrial latent FM | Backbone + собственные projections/refiners по training config | Pretrained codec/text используются как fixed representations в описываемом path | Model paper/config, не inference class init |
| π-ID toy lab09 | MLP policy student trainable | Analytic teacher frozen, rollout states detached | Явные gradient/weight checks |
| K6 policy distillation | Student initialized from teacher and optimized; expanded head не означает head-only training | Teacher target fixed в этой стадии | K6 §4.5 / π-Flow §4 |
| Inference | Parameters frozen/eval; states updated solver | Encoders/codecs frozen/eval | Runtime contract |

Voice-FiLM — отдельная гипотеза стенда; её freeze assumptions не переносятся на π-Flow training автоматически.

## Executable evidence и ограничения

- K6 tensor modules/solver bodies импортируются в isolated namespace через существующий local dense-SDPA adapter; CUDA application не импортируется и PyTorch глобально не патчится.
- K5 imports ограничены выбранными definitions. Sparse/NABLA branch не исполняется, dtype adapter FP32 явно отличается от BF16 upstream helper rounding.
- Codec doubles обозначены в lab04; synthetic text не тестирует tokenizer, Qwen/CLIP quality или лицензии weights.
- Grouped batched input и packed transform checks не доказывают ragged packed attention isolation всего pipeline.
- `CountingLinearField` и MLP student — аналитический/учебный controls. Их metrics не являются FID/AV sync/production latency.
- Все actual calls и state/gradient tests измеряются отдельно от prose. Fresh-kernel report содержит hashes helpers и notebooks; source snapshot и MIT originals проверяются перед исполнением.

[Исходный K6 LICENSE](../kandinsky-6/reference/kandinsky-6/LICENSE) и [third-party notices](../kandinsky-6/THIRD_PARTY.md) сохраняются. Этот курс не назначает всем источникам одну новую лицензию и не содержит pretrained weights.
