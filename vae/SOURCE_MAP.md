# HunyuanVideo VAE: карта источников, компонентов и границ выводов

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../curriculum/README.md) · [Полное содержание](../curriculum/CONTENTS.md).
<!-- study-navigation:end -->

[Курс](README.md) · [Маршрут](../curriculum/CONTENTS.md) · [Машиночитаемая provenance](sources.json).

## Как читать источники

1. [HunyuanVideo v6](https://arxiv.org/pdf/2412.03603v6), стр. 5–6 / §4.1: архитектура, Fig. 6; §4.1.1 / Eq. 1: L1 + perceptual + adversarial + KL; §4.1.2: overlapping tiling и дополнительное обучение с tiling. PDF v6 предоставлен пользователем; 35 страниц, SHA в manifest; полный PDF в Git не включён.
2. [Официальная Tencent реализация](https://github.com/Tencent-Hunyuan/HunyuanVideo/tree/4eda9cb44c3be82a6b9c4261897cb6bd3cfb084e/hyvideo/vae) — первичный код семейства. Для исходной теории VAE: [Kingma & Welling](https://arxiv.org/abs/1312.6114).
3. **Исследуемый файл пользователя** — адаптация из Kandinsky; побайтово совпал с [pinned source](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py) на revision `7ceeefad7fb625011f7db602c43893a989f19c15`. SHA-256 `f5e8dd402aba625f23aef86ab7d5324dc6a4e8585b72917a3bf27bd46745bbaf`. Это не доказательство идентичности всех Tencent/Diffusers versions.
4. [Diffusers Hunyuan VAE](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py) — отдельный runtime для будущего ручного reference benchmark. Он не подменяет исследуемый pinned algorithm.

## Все классы и функция

| Source symbol | Назначение | Основная lab |
|---|---|---|
| [prepare_causal_attention_mask](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L24) | Frame-causal dense attention mask; FHW token layout | [02](notebooks/02_causal_convolution_and_geometry.ipynb) |
| [HunyuanVideoCausalConv3d](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L39) | Replicate padding → Conv3d; isotropic/dilation assumptions | [02](notebooks/02_causal_convolution_and_geometry.ipynb) |
| [HunyuanVideoUpsampleCausal3D](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L80) | First-frame spatial only; rest temporal/spatial nearest | [02](notebooks/02_causal_convolution_and_geometry.ipynb) |
| [HunyuanVideoDownsampleCausal3D](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L125) | Strided causal conv | [03](notebooks/03_architecture_posterior_and_api.ipynb) |
| [HunyuanVideoResnetBlockCausal3D](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L147) | GN → SiLU → causal conv twice; shortcut | [02](notebooks/02_causal_convolution_and_geometry.ipynb) |
| [HunyuanVideoMidBlock3D](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L196) | Resnet; optional flattened attention + causal mask; resnet | [03](notebooks/03_architecture_posterior_and_api.ipynb) |
| [HunyuanVideoDownBlock3D](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L283) | Residual stack + optional downsample | [03](notebooks/03_architecture_posterior_and_api.ipynb) |
| [HunyuanVideoUpBlock3D](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L340) | Residual stack + optional upsample | [03](notebooks/03_architecture_posterior_and_api.ipynb) |
| [HunyuanVideoEncoder3D](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L396) | Four stages → posterior moments | [03](notebooks/03_architecture_posterior_and_api.ipynb) |
| [HunyuanVideoDecoder3D](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L506) | Latents → reversed stages → RGB | [03](notebooks/03_architecture_posterior_and_api.ipynb) |
| [AutoencoderKLHunyuanVideo](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L617) | Posterior API, quant/postquant projections, stateful tile dispatch | [03](notebooks/03_architecture_posterior_and_api.ipynb) |

## Каждый метод AutoencoderKLHunyuanVideo

| Method | Что проследить | Lab |
|---|---|---|
| [__init__](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L629) | projections/config/default flags | [03](notebooks/03_architecture_posterior_and_api.ipynb) |
| [_encode](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L713) | temporal → spatial → full dispatch; decoding flag | [03](notebooks/03_architecture_posterior_and_api.ipynb) |
| [encode](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L731) | planner override, posterior creation, tuple return | [03](notebooks/03_architecture_posterior_and_api.ipynb) |
| [_decode](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L765) | latent thresholds, temporal → spatial → full | [03](notebooks/03_architecture_posterior_and_api.ipynb) |
| [decode](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L798) | recompute planner; ignores additional kwargs | [03](notebooks/03_architecture_posterior_and_api.ipynb) |
| [blend_v](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L826) | vertical in-place ramp | [04](notebooks/04_spatial_tiling_and_blending.ipynb) |
| [blend_h](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L836) | horizontal in-place ramp | [04](notebooks/04_spatial_tiling_and_blending.ipynb) |
| [blend_t](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L846) | temporal in-place ramp | [05](notebooks/05_temporal_tiling_and_first_frame.ipynb) |
| [tiled_encode](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L856) | pixel loop → latent blending → crop; edge coverage | [04](notebooks/04_spatial_tiling_and_blending.ipynb) |
| [tiled_decode](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L930) | latent loop → RGB blending → crop | [04](notebooks/04_spatial_tiling_and_blending.ipynb) |
| [_temporal_tiled_encode](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L1014) | Q+1 slice, drop first latent, append limits | [05](notebooks/05_temporal_tiling_and_first_frame.ipynb) |
| [_temporal_tiled_decode](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L1062) | Qz+1 slice, drop first frame, append limits | [05](notebooks/05_temporal_tiling_and_first_frame.ipynb) |
| [forward](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L1124) | posterior.mode/sample then decode; no automatic DiT scaling | [03](notebooks/03_architecture_posterior_and_api.ipynb) |
| [apply_tiling](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L1148) | ft−1, stateful thresholds/stride | [06](notebooks/06_free_vram_planner.ipynb) |
| [get_enc_optimal_tiling](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L1163) | memory heuristic, sqrt partition, rounding | [06](notebooks/06_free_vram_planner.ipynb) |
| [get_dec_optimal_tiling](../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py#L1209) | latent → sample shape; same encode heuristic | [06](notebooks/06_free_vram_planner.ipynb) |

## Исполняемый уровень доказательств

- AST loader исполняет исходные class definitions и method bodies. Hook decorators снимаются, потому что full Diffusers application не импортируется.
- `get_activation` сопоставляет только source-used SiLU/Swish. Optional Diffusers Attention **отключена и не эмулируется** в tiny codec; mask отдельно проверяется оригинальной функцией.
- `DiagonalPosterior`, `EncoderOutput`, `DecoderOutput` — учебные dependency adapters, не оригинальные Diffusers classes. Gaussian formulas сверяются с torch.distributions; source return contract читается напрямую.
- Native tiny использует random weights и меньшие widths/layers/latent count. Index oracle намеренно ещё проще: выбор grid coordinates и nearest repetition.
- Budget facade меняет только локальную `torch.cuda.mem_get_info` зависимость внутри namespace planner; глобальный PyTorch не патчится. CPU/MPS вызовы исходного planner используют настоящий branch budget=∞.
- Runtime Diffusers runner по умолчанию dry-run; pretrained benchmark и peak VRAM ещё не выполнялись. Адаптивная vs fixed policy latency/peak сравниваются только после двух реальных одинаковых runtime runs.

## Ограничения, которые проверяются на конкретных входах

| Наблюдение | Подтверждение | Чего из него не следует |
|---|---|---|
| GroupNorm меняет прошлое при изменении future frames | Lab 02 perturbation | Ошибка всей архитектуры или отсутствие пользы causal conv |
| General kernel/dilation padding требует аудита | Lab 02 constructor examples | Поломка default isotropic k=3,d=1 |
| Some spatial dimensions оставляют tail | Lab 04, L=88,Q=32,S=24 | Ошибка всех входов и всех upstream implementations |
| Some temporal lengths теряются после stitching | Lab 05 real dispatch sweep | Невозможность tiling; valid grid можно проверить/дополнить |
| Non-CUDA planner не меряет free memory | Lab 06 CPU/MPS branch | MPS автоматически OOM или CPU/MPS невозможны |
| Very small budget даёт invalid stride | Lab 06 synthetic budget | Численная модель memory является rigorous upper bound |
| Pinned encode выдаёт posterior, wrapper ожидает tensor | Lab 03 source contracts | Любой runtime использует именно эту несовместимую пару без adapter |

## Атрибуция и лицензии

Новые материалы ссылаются на существующий snapshot; дополнительные копии третьестороннего кода не добавлялись. [Kandinsky source LICENSE](../kandinsky-6/reference/kandinsky-6/LICENSE) сохранён. Tencent, Diffusers и checkpoints имеют собственные условия: название Hunyuan не означает, что все источники/веса покрыты одной лицензией. Этот курс не назначает им новую лицензию и не утверждает blanket-MIT для репозитория. Свои API adapters и комментарии не скрывают происхождение исполняемых native definitions.

## Дополнительные первичные ссылки

[Conv3d shape rule](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html), [GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html), [CUDA peak allocated](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html), [MPS allocation counter](https://docs.pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory.html). Конкретные версии runtime сохраняются в reports.
