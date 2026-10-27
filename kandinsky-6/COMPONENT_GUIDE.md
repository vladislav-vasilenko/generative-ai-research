# K6: карта архитектурных компонентов и оригинального кода

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../curriculum/README.md) · [Полное содержание](../curriculum/CONTENTS.md).
<!-- study-navigation:end -->

[Курс / последовательность](README.md) · [Полное содержание проекта](../curriculum/CONTENTS.md) · [K6 paper](https://huggingface.co/papers/2610.05608).

## Как читать один компонент

1. Зафиксировать вход/выход, layout, units и смысл timestep.
2. Прочитать соответствующий раздел отчёта, затем конкретный class/function из закреплённого source.
3. Воспроизвести маленький случай и предсказать результат до исполнения.
4. Изменить один фактор, измерить форму/ошибку/градиент; записать ограниченный вывод.
5. Найти тот же контракт в ComfyUI node или готовом pipeline.

## Сопоставление K5 → K6 → практика

| Тема | Предпосылка K5 | K6 компонент | Следующий опыт |
|---|---|---|---|
| Текст | text_embedders, lab 01 | [02: Qwen/CLIP → separate refiners](notebooks/02_text_conditioning.ipynb) | masks и cache |
| Visual codec | vae, lab 02/11 | [03: HunyuanVideo](notebooks/03_video_vae.ipynb) | ComfyUI VAE reconstruction C4 |
| Audio codec | Новый канал относительно K5 T2V | [04: MMAudio + vocoder](notebooks/04_audio_vae_and_vocoder.ipynb) | mel/latent/waveform и durations |
| Строительные блоки | nn, attention, utils | [05](notebooks/05_nn_patches_and_modulation.ipynb), [06](notebooks/06_attention_and_rope.ipynb) | numeric identity/parity |
| DiT / CrossDiT | K5 lab 06 | [07: DiT и текстовая CA](notebooks/07_dit_and_text_crossdit.ipynb) | shapes from config, adaLN-Zero invariants, token vs pooled path, velocity vs grid head |
| AV fusion | Новый joint backbone | [08: Dual-stream CrossDiT](notebooks/08_dual_stream_audio_video_crossdit.ipynb) | manual block = native, flag routing, directional cut, unimodal = joint with cut |
| Обучение | K5 FM lab 06 | [09: joint FM](notebooks/09_flow_matching_training.ipynb) | synthetic held-out loss |
| Distillation | K5 sampling 07 | [10: π-Flow](notebooks/10_piflow_distillation.ipynb) | grid policy / NFE |
| Solver | K5 generation_utils, sampling | [11](notebooks/11_sampling_cfg_and_nfe.ipynb), [12](notebooks/12_latents_conditioning_and_output.ipynb) | Euler/CFG/layout contracts |
| Deployment | K5 MPS/CUDA/parallelize | [13: runtime/MLX](notebooks/13_runtime_memory_and_mlx.ipynb) | budget, parity gates |
| UI → research | K5 inference labs | [14: ComfyUI](notebooks/14_comfyui_component_experiments.ipynb) | [отдельный voice protocol](COMFYUI_AND_EXPERIMENT_RU.md) |

## По одному fused блоку K6

Внутри каждого потока: self-attention → text cross-attention → cross-modal exchange → FFN, с отдельными residual gates. Video/audio widths различаются. Video queries читают audio K/V и обновляют video; audio queries читают video K/V и обновляют audio. Номер позиции в каждой модальности выражен собственным RoPE. Q каждой cross-modal ветви берётся из потока-получателя **после** его text update, K/V — из сохранённого pre-text-CA state (`vis_pre_ca`/`aud_pre_ca`) другого потока. Оба cross outputs вычисляются до применения любого из них. Полное тело класса и ablations исполняются в notebook 08.

`cross_gates=True` даёт video-sized gate из av-modulation и audio-sized gate из va-modulation. `fix_modulation=True` подаёт в va/av-modulation время потока-получателя. В released Lite/Pro (оба флага True) shift/scale query задаёт время получателя, а gate его обновления — время источника; таблица всех четырёх комбинаций проверяется в notebook 08. Zero-init выключает residual-влияние на шаге 0; параметры при этом могут обучаться. `OutLayerAudio` содержит второй norm; его нельзя убрать при переносе без изменения функции.

## Границы исходников и воспроизведения

`labkit/components.py` загружает tensor-only native modules в изолированный namespace, используя существующий dense SDPA adapter из `torch_reference.py`. Тела DiT/solver функций не переписываются; CUDA application imports обходятся явно. `native_causal_conv()` выполняет только оригинальное определение causal Conv3d; это не полный Hunyuan VAE. Текстовые encoders и codecs в CPU-лабораторных не загружаются; synthetic embeddings и test doubles помечены перед кодом. Все original files остаются неизменными и проверяются по manifest.

## Полный индекс выбранных class/function definitions

Список ниже автоматически извлекается через AST из файлов, на которые ссылаются лабораторные. Это навигация по **реальным** именам и методам, а не список отдельных pretrained моделей. Переход «Код» закреплён на revision; «Лабы» открывает объяснение/эксперименты.

### `core/algo/denoise_loop.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/denoise_loop.py) · Лабы: [08](notebooks/08_dual_stream_audio_video_crossdit.ipynb) · [09](notebooks/09_flow_matching_training.ipynb) · [11](notebooks/11_sampling_cfg_and_nfe.ipynb).

- `_build_video_input`.
- `_resolve_null_embeds`.
- `_raw_dit`.
- `_rebuild_text_rope`.
- `denoise_loop`.

### `core/algo/flow_matching.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/flow_matching.py) · Лабы: [09](notebooks/09_flow_matching_training.ipynb) · [11](notebooks/11_sampling_cfg_and_nfe.ipynb).

- `flow_match_timesteps`.

### `core/algo/guidance.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/guidance.py) · Лабы: [11](notebooks/11_sampling_cfg_and_nfe.ipynb).

- `apply_cfg`.

### `core/algo/mux.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/mux.py) · Лабы: [12](notebooks/12_latents_conditioning_and_output.ipynb).

- `_require_av`.
- `_video_np`.
- `_audio_np`.
- `extract_audio_from_video`.
- `mux_video_audio`.

### `core/algo/piflow_math.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/piflow_math.py) · Лабы: [10](notebooks/10_piflow_distillation.ipynb).

- **`DXPolicy`**: `__init__`, `_interpolate`, `_unwarp_t`, `_u_to_x_0`, `pi`, `copy`, `detach_`, `detach`.
- **`MultimodalDXPolicy`**: `__init__`, `detach_`, `detach`.
- `shift_timesteps`.
- `policy_rollout_fm`.

### `core/algo/piflow_sampler.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/piflow_sampler.py) · Лабы: [08](notebooks/08_dual_stream_audio_video_crossdit.ipynb) · [10](notebooks/10_piflow_distillation.ipynb) · [14](notebooks/14_comfyui_component_experiments.ipynb).

- `_normalize_grid_prediction`.
- `piflow_denoise_loop`.

### `core/algo/postprocess_audio.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/postprocess_audio.py) · Лабы: [04](notebooks/04_audio_vae_and_vocoder.ipynb) · [12](notebooks/12_latents_conditioning_and_output.ipynb) · [14](notebooks/14_comfyui_component_experiments.ipynb).

- `postprocess_audio`.

### `core/algo/postprocess_video.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/postprocess_video.py) · Лабы: [03](notebooks/03_video_vae.ipynb) · [12](notebooks/12_latents_conditioning_and_output.ipynb) · [14](notebooks/14_comfyui_component_experiments.ipynb).

- `postprocess_video`.

### `core/algo/prepare_latents.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/prepare_latents.py) · Лабы: [12](notebooks/12_latents_conditioning_and_output.ipynb).

- `prepare_video_latents`.
- `audio_latent_duration`.
- `prepare_audio_latents`.
- `prepare_visual_rope_pos`.
- `prepare_audio_rope_pos`.
- `prepare_text_rope_pos`.
- `encode_video`.
- `resize_image`.
- `_load_pil_rgb`.
- `encode_i2va_first_frame`.
- `_append_i2va_tail_condition_batch`.
- `append_i2va_tail_condition`.

### `core/algo/visual_conditioning.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/visual_conditioning.py) · Лабы: [12](notebooks/12_latents_conditioning_and_output.ipynb).

- `append_visual_references`.
- `restore_tail_references`.
- `build_tail_condition`.
- `append_reference_rope`.
- `compute_reference_rope`.

### `core/components/dit.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/components/dit.py) · Лабы: [01](notebooks/01_system_and_training_map.ipynb) · [02](notebooks/02_text_conditioning.ipynb) · [05](notebooks/05_nn_patches_and_modulation.ipynb) · [06](notebooks/06_attention_and_rope.ipynb) · [07](notebooks/07_dit_and_text_crossdit.ipynb) · [08](notebooks/08_dual_stream_audio_video_crossdit.ipynb) · [09](notebooks/09_flow_matching_training.ipynb) · [14](notebooks/14_comfyui_component_experiments.ipynb).

- **`_TimestepEmbedder`**: `__init__`.
- **`TimeEmbeddings`**: `__init__`, `forward`, `reset_parameters`.
- **`TextEmbeddings`**: `__init__`, `forward`, `reset_parameters`.
- **`VisualEmbeddings`**: `__init__`, `forward`.
- **`Modulation`**: `__init__`, `forward`, `reset_parameters`.
- **`_GELUProjection`**: `__init__`, `forward`.
- **`FeedForward`**: `__init__`, `forward`.
- **`MultiheadSelfAttentionEnc`**: `__init__`, `forward`.
- **`MultiheadSelfAttentionDec`**: `__init__`, `forward`.
- **`MultiheadCrossAttention`**: `__init__`, `forward`.
- **`OutLayer`**: `__init__`, `forward`, `reset_parameters`.
- **`OutLayerAudio`**: `__init__`, `forward`, `reset_parameters`.
- **`TransformerEncoderBlock`**: `__init__`, `forward`, `reset_parameters`.
- **`TransformerDecoderBlock`**: `__init__`, `forward`, `reset_parameters`.
- **`FusedTransformerDecoderBlock`**: `__init__`, `forward`, `reset_parameters`.
- **`DiffusionTransformer3D`**: `__init__`, `clear_text_proj_cache`, `_project_pooled`, `_project_text_tokens`, `_time_embed`, `_normalize_attn_mask`, `_run_text_blocks`, `_encode_text`, `_encode_t2v`, `_time_only`, `_embed_visual`, `_embed_audio`, `_run_visual_blocks_single`, `_run_visual_blocks_fused`, `_project_video`, `_project_audio`, `_project_fused`, `forward`, `reset_parameters`.

### `core/components/hunyuan_vae.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/components/hunyuan_vae.py) · Лабы: [03](notebooks/03_video_vae.ipynb).

- `prepare_causal_attention_mask`.
- **`HunyuanVideoCausalConv3d`**: `__init__`, `forward`.
- **`HunyuanVideoUpsampleCausal3D`**: `__init__`, `forward`.
- **`HunyuanVideoDownsampleCausal3D`**: `__init__`, `forward`.
- **`HunyuanVideoResnetBlockCausal3D`**: `__init__`, `forward`.
- **`HunyuanVideoMidBlock3D`**: `__init__`, `forward`.
- **`HunyuanVideoDownBlock3D`**: `__init__`, `forward`.
- **`HunyuanVideoUpBlock3D`**: `__init__`, `forward`.
- **`HunyuanVideoEncoder3D`**: `__init__`, `forward`.
- **`HunyuanVideoDecoder3D`**: `__init__`, `forward`.
- **`AutoencoderKLHunyuanVideo`**: `__init__`, `_encode`, `encode`, `_decode`, `decode`, `blend_v`, `blend_h`, `blend_t`, `tiled_encode`, `tiled_decode`, `_temporal_tiled_encode`, `_temporal_tiled_decode`, `forward`, `apply_tiling`, `get_enc_optimal_tiling`, `get_dec_optimal_tiling`.

### `core/components/mmaudio/features_utils.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/components/mmaudio/features_utils.py) · Лабы: [04](notebooks/04_audio_vae_and_vocoder.ipynb).

- **`FeaturesUtils`**: `__init__`, `compile`, `train`, `encode_audio`, `decode`, `device`, `dtype`, `wrapped_decode`, `wrapped_encode`.

### `core/components/piflow_dit.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/components/piflow_dit.py) · Лабы: [01](notebooks/01_system_and_training_map.ipynb) · [07](notebooks/07_dit_and_text_crossdit.ipynb) · [08](notebooks/08_dual_stream_audio_video_crossdit.ipynb) · [10](notebooks/10_piflow_distillation.ipynb).

- **`PiFlowDiffusionTransformer3D`**: `__init__`, `forward`.

### `core/components/rope.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/components/rope.py) · Лабы: [06](notebooks/06_attention_and_rope.ipynb).

- **`RoPE1D`**: `__init__`, `forward`, `reset_parameters`.
- **`RoPE3D`**: `__init__`, `forward`, `reset_parameters`.

### `core/components/text_embedder.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/components/text_embedder.py) · Лабы: [02](notebooks/02_text_conditioning.ipynb).

- `clip_text_state`.
- `clip_tokenizer_dir`.
- `qwen_processor_dir`.
- `_load_clip_text_model`.
- **`Kandinsky6TextEmbedder`**: `__init__`, `_load_models`, `_weights_ready`, `encode`, `encode_qwen`, `encode_clip`, `release`, `ensure_loaded`, `to`.

### `core/components/vae_audio.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/components/vae_audio.py) · Лабы: [04](notebooks/04_audio_vae_and_vocoder.ipynb).

- `build_audio_vae`.
- `build_vocoder`.

### `core/components/vae_video.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/components/vae_video.py) · Лабы: [03](notebooks/03_video_vae.ipynb).

- `build_vae`.

### `core/tensors.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/tensors.py) · Лабы: [05](notebooks/05_nn_patches_and_modulation.ipynb) · [06](notebooks/06_attention_and_rope.ipynb) · [07](notebooks/07_dit_and_text_crossdit.ipynb).

- `get_freqs`.
- `apply_scale_shift_norm`.
- `apply_gate_sum`.
- `apply_rotary`.

### `pipeline/config.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/pipeline/config.py) · Лабы: [07](notebooks/07_dit_and_text_crossdit.ipynb).

- **`BeautifierConfig`**: typed data structure.
- **`PathsConfig`**: typed data structure.
- **`DiTConfig`**: `_as_tuple`.
- **`GenerationConfig`**: `_as_tuple`.
- **`TextEmbedderConfig`**: typed data structure.
- **`AudioVAEConfig`**: typed data structure.
- **`VocoderConfig`**: typed data structure.
- **`PipelineConfig`**: `_audio_components_together`.
- `_read_mapping`.
- `_deep_merge`.
- `_checkpoint_file`.
- `_module_file`.
- `_per_grid_overrides`.
- `_expand_dit`.
- `_compose`.
- `load_config`.

### `runtime/cache/magcache.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/runtime/cache/magcache.py) · Лабы: [13](notebooks/13_runtime_memory_and_mlx.ipynb).

- **`MagCacheConfig`**: typed data structure.
- `select_mag_ratios`.
- `nearest_interp`.
- `prepare_mag_ratios`.
- **`MagCache`**: `_reset_mag_state`, `_mag_should_skip`, `_mag_advance`, `_magcache_forward`, `_mag_forward_fused`, `_mag_forward_video`.

### `runtime/kernels/sdpa.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/runtime/kernels/sdpa.py) · Лабы: [06](notebooks/06_attention_and_rope.ipynb).

- `sdpa`.

### `runtime/offload/block.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/runtime/offload/block.py) · Лабы: [13](notebooks/13_runtime_memory_and_mlx.ipynb).

- **`_Binding`**: `bind`.
- **`_TensorGroup`**: `__init__`, `add_modules`, `restore`, `bind`.
- **`_Slot`**: `__init__`, `load`, `clear`.
- **`BlockOffload`**: `__init__`, `release`, `releases`, `register`, `_register_dit`, `_decorate`, `_prefetch`, `_use_block`, `use`.

### `runtime/offload/module.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/runtime/offload/module.py) · Лабы: [13](notebooks/13_runtime_memory_and_mlx.ipynb).

- **`ModuleOffload`**: `__init__`, `register`, `releases`, `prefetch`, `_wait_ready`, `_ensure`, `release`, `empty_cache`, `use`.

### `runtime/sampler.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/runtime/sampler.py) · Лабы: [01](notebooks/01_system_and_training_map.ipynb) · [10](notebooks/10_piflow_distillation.ipynb) · [13](notebooks/13_runtime_memory_and_mlx.ipynb).

- **`PiFlowConfig`**: typed data structure.
- `dit_class`.
- `reject_incompatible_cache`.

### `runtime/weights.py`

[Код](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/runtime/weights.py) · Лабы: [13](notebooks/13_runtime_memory_and_mlx.ipynb).

- **`Checkpoint`**: `weight_gb`.
- `get_checkpoint`.
- `_cache_root`.
- `_cached_bytes`.
- `_disk_free`.
- `_require_download_space`.
- `_text_encoder_ready`.
- `_snapshot_ready`.
- `_local_snapshot`.
- `_ensure_snapshot`.
- `ensure_checkpoint`.
- `materialize_meta_buffers`.
- `load_bf16_checkpoint`.

## Самостоятельные архитектурные продолжения

SR-DiT/Latent Upscaler используют KVAE и описаны в K6 §3.3/4.6; они не требуются для основного Lite генератора. Сначала пройдите [самостоятельный visual codec опыт K5 11](../kandinsky-5/notebooks/11_kvae_separate.ipynb), затем сравните latent contracts по [pinned KVAE](reference/kvae/README.md) и [KVAE-Audio](reference/kvae-audio/README.md). Новые pretrained кодеки нельзя подставлять в уже обученный DiT только по совпадению shapes.

Для сравнений HunyuanVideo/Wan/LTX-Video/Mochi/Open-Sora используйте [общую industrial map](../curriculum/INDUSTRY_BRIDGE.md), а не переносите K6-specific значения в другие семейства.
