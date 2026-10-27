# MIT Lab 3 → VAE → промышленный видеокодек

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../curriculum/README.md) · [Полное содержание](../curriculum/CONTENTS.md).
<!-- study-navigation:end -->

[Общий VAE маршрут](README.md) · [Lecture 4](../lecture_04/README.md) · [Исходный Lab 3](../lecture_04/references/lab_three_original.ipynb) · [MIT course](https://diffusion.csail.mit.edu/2026/index.html).

Базовый скелет — MIT 6.S184 (2026), **Lab 3 Part 4: Training a Variational Autoencoder**, затем **Part 5: Training a Latent Diffusion Model**. Наши notebooks дополняют originals: задания остаются незаполненными и неизменёнными. Нумерация cells ниже — **zero-based**, как в Python `nb['cells'][i]`; номера страниц PDF — one-based. Исходный Lab 3 содержит 80 cells.

## Что прочитать и в каком порядке работать

1. [Lecture 4 slides 3–19](https://diffusion.csail.mit.edu/2026/docs/20260128_Lecture_04_edited.pdf), [notes §6.2, pp.46–52](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf#page=46). Это overview VAE и latent generation; Appendix D, pp.77–82 — дополнительные derivations.
2. Если идёте по всему MIT Lab 3, сохраняйте его порядок: Parts 1–2 conditioning → Part 3 DiT → Part 4 VAE → Part 5 latent model. Для отдельной VAE ветви достаточно изучить нужные tensor/attention основы, а готовый DiT понадобится позже в Part 5.
3. Пройдите наши 00–01, затем реализуйте Q4.1–4.7 в своей рабочей копии MIT Lab 3. Не переписывайте canonical original в `lecture_04/references/`.
4. Используйте наши 02–03 как сравнение image-VAE с промышленным 3D codec. Вернитесь к MIT training/interpolation и Q5.1 для замороженного codec.
5. Только после full reconstruction переходите к 04–07: tiling и memory policy. Завершите ручным ComfyUI/pretrained опытом 08 после resource gate.

## Соответствие каждого задания

| MIT Lab 3 | Cells | Что реализовать / проверить в MIT | Наше продолжение | Критерий понимания |
|---|---|---|---|---|
| Part 4 overview | 55 | Разделить encoder, posterior, decoder и objective | [00 карта](notebooks/00_route_and_source_contracts.ipynb) | μ/logvar не перепутаны с sampled z |
| Q4.1 ResidualBlock | 56–57 | Residual path, norm, activation, convolution | [02 convolution](notebooks/02_causal_convolution_and_geometry.ipynb) | Shapes residual суммируются; объяснить GroupNorm axes |
| Q4.2 AttentionBlock | 58–59 | Q/K/V, reshape, softmax по key axis | [03 attention/architecture](notebooks/03_architecture_posterior_and_api.ipynb) | N=HW vs FHW; mask и memory N² |
| Q4.3 EncoderBlock | 60–61 | Residual/attention/downsampling | [02 geometry](notebooks/02_causal_convolution_and_geometry.ipynb), [03 stages](notebooks/03_architecture_posterior_and_api.ipynb) | Предсказать каждый output shape до forward |
| Q4.4 Encoder | 62–63 | Разделить mean и log variance | [01 posterior/KL](notebooks/01_elbo_and_probabilistic_codec.ipynb), [03 API](notebooks/03_architecture_posterior_and_api.ipynb) | std=exp(logvar/2), correct channel split |
| Q4.5 DecoderBlock | 64–65 | Upsample и residual blocks | [02 first-frame rule](notebooks/02_causal_convolution_and_geometry.ipynb) | Image upsample vs causal video 1+r(F−1) |
| Q4.6 Decoder | 66–67 | Conditional reconstruction parameters | [01 NLL](notebooks/01_elbo_and_probabilistic_codec.ipynb), [03 decoder](notebooks/03_architecture_posterior_and_api.ipynb) | Learned likelihood logvar в MIT не равна posterior logvar |
| Q4.7 VAE | 68–69 | Reparameterization, compute_loss, KL и reconstruction NLL | [01 ELBO](notebooks/01_elbo_and_probabilistic_codec.ipynb) | NLL проверена torch.distributions; reductions согласованы |
| Training / interpolation | 70–74 | MNIST training, reconstruction, interpolation | [01 tiny training](notebooks/01_elbo_and_probabilistic_codec.ipynb), [07 controlled evaluation](notebooks/07_controlled_benchmark.ipynb) | Held-out reconstruction, KL, active dimensions; не один train loss |
| Q5.1 LatentCFGTrainer | 76–77 | Frozen VAE, latent path, DiT optimizer | [03 scaling/contracts](notebooks/03_architecture_posterior_and_api.ipynb), [K6 FM](../kandinsky-6/notebooks/09_flow_matching_training.ipynb) | Градиент у DiT, codec не обновляется |
| Part 5 training | 78 | Latent sampling → decode | [08 deployment](notebooks/08_pretrained_and_comfyui_protocol.ipynb) | Отделить codec reconstruction от генерации ODE |

MIT code `VAE.forward` возвращает z_mean,z_logvar,x_mean,x_logvar. В trainer встречаются имена `z_std/x_std`, но семантика значений задаётся forward; не подменяйте log variance стандартным отклонением из-за имени переменной. В Hunyuan codec posterior — distribution object, а RGB decoder с L1/perceptual/adversarial loss не обязан выдавать learned x_logvar как учебный Gaussian decoder MIT. Это различие моделей, а не повод насильно выравнивать API.

## Где заканчивается исходное задание MIT

| Тема | Есть в исходном Lab 3 | Наше промышленное расширение |
|---|---|---|
| Reparameterization / KL / Gaussian likelihood | Part 4 | Выводы, finite differences, source contracts |
| 2D residual, attention, encoder/decoder | Q4.1–4.6 | 3D causal conv, frame lattice, 16 latent channels |
| VAE freeze при latent DiT training | Q5.1 | Scale/layout/provenance в K5/K6 |
| Spatial/temporal overlapping tiling | Не является отдельным заданием Q4.1–4.7 | Labs 04–05, Hunyuan v6 §4.1.2 |
| Free-VRAM planner и MPS counters | Не является отдельным заданием | Lab 06, конкретная Kandinsky adaptation |
| ComfyUI, CUDA peak, device deployment | Не является отдельным заданием | Lab 08, реальные веса вручную |

## Три уровня итогового результата

1. **Я понял формулу:** получить Gaussian KL/NLL и проверить finite-difference gradient. Выполненная ячейка сама по себе этого не доказывает.
2. **Я собрал codec:** все формы, reconstruction baseline и posterior sampling понятны; VAE frozen там, где обучается DiT.
3. **Я провёл исследование:** сравнить policy при неизменных input/checkpoint, сохранить MSE/PSNR, lengths, latency и корректную memory metric. [Измеренный CPU отчёт](reports/VALIDATION.md) отделён от ещё не выполненного pretrained этапа.

На M3 Max 48 GB выполняйте CPU примеры 00–07. Исходный MIT training cell с `device='cuda'` не запустится на Mac без изменения **рабочей копии** и проверки поддерживаемых операций/budget. Полный MNIST training не объявлен выполненным этим курсом; маленький linear VAE проверяет только механизм objective. ComfyUI image reconstruction не заменяет Q4.7 или temporal video experiment.
