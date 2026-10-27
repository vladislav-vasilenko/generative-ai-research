# Lecture 4 → Lab 3: DiT, VAE и обучение в latent space

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../curriculum/README.md) · [Полное содержание](../curriculum/CONTENTS.md).

**Связанные лекции:** [Lecture 4](https://diffusion.csail.mit.edu/2026/docs/20260128_Lecture_04_edited.pdf).
<!-- study-navigation:end -->

[Общий маршрут](../curriculum/README.md), [слайды](https://diffusion.csail.mit.edu/2026/docs/20260128_Lecture_04_edited.pdf), [исходный Lab 3](references/lab_three_original.ipynb), 80 ячеек. Это исходник с заготовками, не полностью решённая лабораторная.

Предварительно выполнены Lab 3 Parts 1–2 из [Lecture 3B](../lecture_03/README.md). В slides VAE идёт раньше DiT. Прочитайте overview, но исполняйте notebook в его порядке: **Part 3 → Part 4 → Part 5**.

| Блок | Читать | Реализовать | Проверить |
|---|---|---|---|
| DiT | Slides pp.20–28; notes §6.1 pp.41–46 | Q3.1–3.5, cells 38–48: time embedding, patchifier, MHA, modulation, depatchifier | Shapes, softmax по keys, head scaling, patch roundtrip; output в размерности target |
| Pixel training | Notebook cells 49–54 | Обучение DiT на изображениях | Фиксированные held-out noise/time и samples, а не только train loss |
| VAE | Slides pp.3–19; notes §6.2 pp.46–52, Appendix D pp.77–82 | Q4.1–4.7, cells 56–69; затем training cells 70–74 | Residual/attention/downsampling; std=exp(logvar/2), reconstruction NLL, KL reduction |
| Latent model | Slides p.18; notes pp.51–52 | Q5.1, cells 75–78 | Encoder под no_grad; frozen VAE, отдельный optimizer DiT; согласованные latent shapes/scales |

## Что именно учится

| Стадия | Encoder / decoder VAE | DiT / class embedding | Text encoder |
|---|---|---|---|
| Lab 3: обучение VAE | Trainable | Не участвует в этой objective | В MNIST lab не нужен |
| Lab 3: pixel DiT | Не участвует | Trainable | Вместо текста — class label |
| Lab 3: latent DiT | Frozen после VAE stage | Trainable | Вместо текста — class label |
| Промышленное обучение latent DiT | Pretrained codec frozen в рассматриваемой стадии; проверить config | Trainable backbone либо только выбранный adapter | Отдельные pretrained Qwen/T5/CLIP; freeze policy задана training config |
| Промышленный inference baseline | Frozen / eval | Frozen / eval | Frozen / eval |

Tokenizer обычно означает фиксированные правила/vocabulary, а embedding matrix — обучаемые параметры конкретной сети. Не смешивайте это с VAE, который в статьях также могут называть visual tokenizer.

## Обязательные опыты

1. **VAE tradeoff:** sweep KL weight при неизменном preprocessing и budget; reconstruction MSE/NLL, KL, active latent dimensions. Само приближение aggregate posterior к prior не гарантирует идеальное reconstruction.
2. **Architecture:** один DiT, меняем patch size и считаем tokens, parameters, observed memory/time. Не обещайте 16× экономии всей памяти только из формы attention scores.
3. **Latent vs pixel:** одинаковая data split и оценочный protocol. Отдельно сравните стоимость codec и generative model. VAE reconstruction и генерация из latent prior — разные эксперименты.
4. **ComfyUI [C4](../curriculum/COMFYUI_LABS.md):** reconstruction через уже имеющийся HunyuanVideo VAE. Этот graph не вычисляет VAE training loss и не доказывает качество KL regularization.

Для подробного разложения используйте [K5 VAE lab](../kandinsky-5/notebooks/02_vae.ipynb), [attention](../kandinsky-5/notebooks/04_attention.ipynb), [DiT/FM](../kandinsky-5/notebooks/06_dit_and_flow_matching.ipynb). После P7 можно прослеживать [реальные архитектуры](../curriculum/INDUSTRY_BRIDGE.md).

**Готовность:** собрать shape diagram всего latent pipeline, показать gradient только у нужных parameters и объяснить, почему замороженный decoder всё равно нужен при оценке.

## Дополнительный разбор

После VAE/DiT: temporal metrics и causal Conv3d, опыт C. [Материалы, первичные источники и CPU-контроли](../curriculum/SUPPLEMENTARY_READING.md). Основной порядок лекций/labs не меняется.
