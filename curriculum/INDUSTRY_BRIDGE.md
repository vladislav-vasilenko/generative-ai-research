# Как читать промышленный pipeline после Lab 1–3

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](README.md) · [Полное содержание](CONTENTS.md).
<!-- study-navigation:end -->

Цель — переносить понятия и проверяемые контракты. Сначала проследите один prompt/noise до decoder, затем выберите один компонент. Одновременный полный запуск всех семейств не нужен.

## Общая схема

`text → tokenizer → text embeddings`, параллельно `noise/conditioning latents → time-dependent DiT → predicted field/policy → sampler update → decoder → media`.

По каждому звену выпишите shape, dtype, device, time convention, scale/shift, trainable/frozen и источник весов. Tokenizer rules, text encoder, VAE и DiT — разные уровни. Distilled policy head не обязан предсказывать velocity.

| Семейство / вариант | После какой темы | Что читать и воспроизводить маленьким кодом | Сравнение в ComfyUI |
|---|---|---|---|
| HunyuanVideo, оригинальный repo | P6–P7 | Causal 3D VAE; переход dual-stream → single-stream Transformer; text conditioning; flow scheduler | Сначала отдельный VAE reconstruction C4; затем совместимый официальный pipeline |
| Wan2.1 | P3, P5–P7 | Wan-VAE, T5 text embeddings, time modulation/cross-attention, flow solver | Один T2V profile, CFG и NFE; фиксировать T5/codec и initial noise |
| Wan2.2 A14B | После Wan2.1 | High-noise/low-noise experts и switching по timestep/config boundary | Логировать выбранный expert по шагам; active и stored parameters различаются |
| Wan2.2 TI2V-5B | После codec/shape maps | Это dense вариант, не A14B two-expert architecture; иной high-compression codec | Отдельный graph/model+codec, не подмена VAE в Wan2.1 |
| Kandinsky 5 | P3–P7 + [14 labs](../kandinsky-5/README.md) | Qwen/CLIP, Hunyuan VAE, DiT, LTF/cross-attention и generation utilities | Сравнить non-distilled vs distilled по разным profile; не смешивать objectives |
| Kandinsky 6 Lite | После K5 или P6–P7 | [14 компонентных labs](../kandinsky-6/README.md): [DiT](../kandinsky-6/notebooks/07_dit_and_text_crossdit.ipynb), [dual AV fusion](../kandinsky-6/notebooks/08_dual_stream_audio_video_crossdit.ipynb), MMAudio/Hunyuan, PiFlow | [Подготовленные Lite graphs](../kandinsky-6/README.md), затем измерения AV |
| LTX-Video | После P7 | Собственные codec/Transformer/RF scheduler contracts; выбрать конкретную revision | Сопоставить shape map с её VAE, не подменять его Hunyuan/Wan codec |
| Mochi 1 | После P6–P7 | Asymmetric visual/text streams, VAE и pipeline | Сначала интерфейсы блоков, затем отдельный совместимый workflow |
| Open-Sora, HPC-AI Tech | После P7 | Конкретный release/config: MMDiT, codec и sampling path | Фиксировать release и graph; это не Open-Sora-Plan |

Это каталог кейсов после MIT-основы, не рейтинг и не обязательная последовательность всех моделей. Архитектурные особенности сверены с официальными исходниками; конкретные Mac throughput/peak memory здесь не измерены. В Wan2.2 не все варианты MoE. В K6 audio-only path пропускает cross-modal interactions и не эквивалентен совместному sampler.

## Что воспроизводить по ступеням

1. **Контракт:** random tensor нужной формы проходит один patch/time/attention/output block.
2. **Математика:** ручная формула против framework operation, включая masks, normalization epsilon, RoPE и dtype.
3. **Pretrained parity:** один и тот же block с одинаковыми weights/inputs в двух реализациях; ошибки по слоям.
4. **Trajectory parity:** один solver update, затем несколько шагов; drift error может накапливаться.
5. **Media и стоимость:** codec output, качество и runtime, cold/warm, memory/swap, NFE.
6. **Изменение модели:** adapter/новое условие только после baseline; controls, validation/test и regression.

Название `FlowMatchDiscreteScheduler` означает дискретизацию времени scheduler и само по себе не делает генерацию CTMC из Lecture 5. Discrete token IDs Qwen/T5/CLIP могут быть только text conditioning.

## MLX и MPS

Сначала PyTorch CPU reference, потом MPS/MLX: передавать сохранённые arrays, сохранять double normalization или необычный eps, сравнивать output и gradients. MLX-порт архитектуры/весов обычно не требует полного переобучения, но unsupported operations, quantization и изменение sampler требуют отдельных checks. Causal cache layout и fused kernels не переносятся автоматически.

Для benchmark принудительно завершать lazy/asynchronous вычисления и прогревать один profile. В отчёте отдельно указывать weights resident memory, activations и optimizer state; trainable parameter count не равен полной стоимости backward.

## Закреплённые точки входа

Следующие ссылки ведут на конкретные revisions, проверенные 27 октября 2026. Исходники индексировались как текст и не исполнялись. [Полный manifest](references/industry_manifest.json).

### Tencent-Hunyuan/HunyuanVideo

Revision: `4eda9cb44c3be82a6b9c4261897cb6bd3cfb084e`.

- [hyvideo/modules/models.py](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/4eda9cb44c3be82a6b9c4261897cb6bd3cfb084e/hyvideo/modules/models.py) — MMDoubleStreamBlock, MMSingleStreamBlock, HYVideoDiffusionTransformer.
- [hyvideo/vae/__init__.py](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/4eda9cb44c3be82a6b9c4261897cb6bd3cfb084e/hyvideo/vae/__init__.py) — load_vae.
- [hyvideo/diffusion/schedulers/scheduling_flow_match_discrete.py](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/4eda9cb44c3be82a6b9c4261897cb6bd3cfb084e/hyvideo/diffusion/schedulers/scheduling_flow_match_discrete.py) — FlowMatchDiscreteSchedulerOutput, FlowMatchDiscreteScheduler.

### Wan-Video/Wan2.1

Revision: `9737cba9c1c3c4d04b33fcad41c111989865d315`.

- [wan/modules/model.py](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py) — sinusoidal_embedding_1d, rope_params, rope_apply.
- [wan/modules/vae.py](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py) — CausalConv3d, RMS_norm, Upsample.
- [wan/modules/t5.py](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/t5.py) — fp16_clamp, init_weights, GELU.
- [wan/text2video.py](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py) — WanT2V.
- [wan/utils/fm_solvers.py](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers.py) — get_sampling_sigmas, retrieve_timesteps, FlowDPMSolverMultistepScheduler.

### Wan-Video/Wan2.2

Revision: `1ea34ff48f87168174e12956e200b1d908b1c5ff`.

- [wan/modules/model.py](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/model.py) — sinusoidal_embedding_1d, rope_params, rope_apply.
- [wan/modules/vae2_2.py](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/vae2_2.py) — CausalConv3d, RMS_norm, Upsample.
- [wan/text2video.py](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/text2video.py) — WanT2V.
- [wan/configs/wan_ti2v_5B.py](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/configs/wan_ti2v_5B.py).
- [wan/configs/wan_t2v_A14B.py](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/configs/wan_t2v_A14B.py).

### kandinskylab/kandinsky-5

Revision: `3a74261b957df0446c1969c5cedf8bbb07969ac2`.

- [kandinsky/models/dit.py](https://github.com/kandinskylab/kandinsky-5/blob/3a74261b957df0446c1969c5cedf8bbb07969ac2/kandinsky/models/dit.py) — TransformerEncoderBlock, TransformerDecoderBlock, DiffusionTransformer3D.
- [kandinsky/models/vae.py](https://github.com/kandinskylab/kandinsky-5/blob/3a74261b957df0446c1969c5cedf8bbb07969ac2/kandinsky/models/vae.py) — prepare_causal_attention_mask, HunyuanVideoCausalConv3d, HunyuanVideoUpsampleCausal3D.
- [kandinsky/models/text_embedders.py](https://github.com/kandinskylab/kandinsky-5/blob/3a74261b957df0446c1969c5cedf8bbb07969ac2/kandinsky/models/text_embedders.py) — ClipTextEmbedder, Qwen2_5_VLTextEmbedder, Kandinsky5TextEmbedder.
- [kandinsky/generation_utils.py](https://github.com/kandinskylab/kandinsky-5/blob/3a74261b957df0446c1969c5cedf8bbb07969ac2/kandinsky/generation_utils.py) — get_sparse_params, adaptive_mean_std_normalization, normalize_first_frame.

### kandinskylab/kandinsky-6

Revision: `7ceeefad7fb625011f7db602c43893a989f19c15`.

- [kandinsky/core/components/dit.py](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/components/dit.py) — _TimestepEmbedder, TimeEmbeddings, TextEmbeddings.
- [kandinsky/core/components/piflow_dit.py](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/components/piflow_dit.py) — PiFlowDiffusionTransformer3D.
- [kandinsky/core/algo/piflow_sampler.py](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/piflow_sampler.py) — _normalize_grid_prediction, piflow_denoise_loop.
- [kandinsky/core/algo/flow_matching.py](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/kandinsky/core/algo/flow_matching.py) — flow_match_timesteps.

### Lightricks/LTX-Video

Revision: `4b2d053057623ddd4d0a1d3e9cd28890e9ef487f`.

- [ltx_video/models/autoencoders/causal_video_autoencoder.py](https://github.com/Lightricks/LTX-Video/blob/4b2d053057623ddd4d0a1d3e9cd28890e9ef487f/ltx_video/models/autoencoders/causal_video_autoencoder.py).
- [ltx_video/models/transformers/transformer3d.py](https://github.com/Lightricks/LTX-Video/blob/4b2d053057623ddd4d0a1d3e9cd28890e9ef487f/ltx_video/models/transformers/transformer3d.py).
- [ltx_video/schedulers/rf.py](https://github.com/Lightricks/LTX-Video/blob/4b2d053057623ddd4d0a1d3e9cd28890e9ef487f/ltx_video/schedulers/rf.py).

### genmoai/mochi

Revision: `15c9bad662df1fa3e7b53136423695b286484c08`.

- [src/genmo/mochi_preview/dit/joint_model/asymm_models_joint.py](https://github.com/genmoai/mochi/blob/15c9bad662df1fa3e7b53136423695b286484c08/src/genmo/mochi_preview/dit/joint_model/asymm_models_joint.py).
- [src/genmo/mochi_preview/vae/models.py](https://github.com/genmoai/mochi/blob/15c9bad662df1fa3e7b53136423695b286484c08/src/genmo/mochi_preview/vae/models.py).
- [src/genmo/mochi_preview/pipelines.py](https://github.com/genmoai/mochi/blob/15c9bad662df1fa3e7b53136423695b286484c08/src/genmo/mochi_preview/pipelines.py).

### hpcaitech/Open-Sora

Revision: `7ad6a96a135feb81f755c84fb391818718f6beb2`.

- [opensora/models/mmdit/model.py](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/model.py).
- [opensora/models/hunyuan_vae/autoencoder_kl_causal_3d.py](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/hunyuan_vae/autoencoder_kl_causal_3d.py).
- [opensora/utils/sampling.py](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/sampling.py).

Mochi 1: прежний URL `genmoai/models` перенаправляется на `genmoai/mochi`. Для LTX выбран LTX-Video, а не отдельное семейство LTX-2. Open-Sora здесь означает проект HPC-AI Tech. Ни одно из новых семейств в этом стенде не запускалось с pretrained weights.

## Исполняемая ветвь HunyuanVideo VAE

[Девять подробных labs](../vae/README.md) дополняют source map: вывод ELBO/KL, native tiny encoder/decoder, causal normalization caveat, coverage spatial/temporal loops и synthetic free-VRAM planner. [Source provenance](../vae/SOURCE_MAP.md) различает Tencent paper v6, приложенную Kandinsky адаптацию и Diffusers runtime.

## Глубокий исполняемый CrossDiT / π-Flow case

[Общий DiT курс](../dit/README.md) разворачивает K5 single generated visual stream + text context и K6 dual Audio/Video streams в formulas, Q/K/V dimensions и original code. Native block parity, directional Jacobian cuts, visual references, Euler convergence и toy policy imitation имеют [отдельные измерения](../dit/reports/VALIDATION.md). [MIT bridge](../dit/MIT_BRIDGE.md) отделяет исходный курс от AV fusion/π-Flow extensions.
