# Дополнительные материалы к основному курсу

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](README.md).
<!-- study-navigation:end -->

**Основной скелет — [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html).** Сначала соответствующая лекция и lab, затем один внешний разбор и небольшой контрольный опыт. Эти материалы не образуют параллельную обязательную программу.

| Когда | Дополнение | Зачем читать | Первичная проверка |
|---|---|---|---|
| После Lecture 2 / Lab 2 Part 4 | [lucidrains: rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch) | Сопоставить probability path, target, loss, sampler и teacher-generated coupling | [Rectified Flow paper](https://arxiv.org/abs/2209.03003) и закреплённый код |
| Вместе с Lecture 3 | [Lilian Weng: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | Дополнительное объяснение forward/reverse diffusion, noise prediction и guidance | [DDPM](https://arxiv.org/abs/2006.11239) |
| После Lecture 4 / Lab 3 | [Lilian Weng: Diffusion Models for Video Generation](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/) | Переход к временной структуре и чтению video architectures | [Video Diffusion Models](https://arxiv.org/abs/2204.03458) и official model code |

Обзор video diffusion опубликован в 2024: он даёт контекст, а архитектуры более поздних Wan/Kandinsky проверяются по отдельным официальным источникам. Обзоры используются как пояснения; математические identities и выводы опыта проверяются по первичным статьям/коду.

## Какие репозитории использовать

1. **[Meta flow_matching](https://github.com/facebookresearch/flow_matching)** — первая библиотечная сверка после ручной реализации MIT lab: отдельные probability paths, schedulers и solvers; есть continuous и discrete направления. Связан с [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264).
2. **[Оригинальный RectifiedFlow](https://github.com/gnobitab/RectifiedFlow)** — сверка с авторской реализацией метода и reflow.
3. **[lucidrains](https://github.com/lucidrains/rectified-flow-pytorch)** — дополнительная реализация с вариантами objectives/losses и follow-up research. Для учебного baseline сначала отключите необязательные изменения и проверьте normalization, time convention и solver.

Мы прочитали выбранные исходники, но эти пакеты здесь не устанавливались и не запускались. Учебные контрольные опыты ниже независимы от них. Revisions, URLs и SHA-256 — в [индексе](references/additional_sources.json).

В pinned lucidrains `RectifiedFlow.forward` для `predict='flow'` использует линейную интерполяцию noise→data и target `data-noise`; по умолчанию также есть image normalization. `Reflow` создаёт пары исходного noise и teacher sample. Не переносите image-wrapper settings на произвольные 2D vectors без проверки. `sample` использует ODE solver; число сохранённых time points не является универсальным NFE.

## Опыт A: два разных смысла velocity

В variance-preserving параметризации обозначим текущее состояние через `z_tau`:

$$
z_{\tau}=a_{\tau}x_{\mathrm{data}}+\sigma_{\tau}\epsilon,\qquad a_{\tau}^2+\sigma_{\tau}^2=1.
$$

Параметр `v` из [Salimans & Ho](https://arxiv.org/abs/2202.00512) задаётся так:

$$
v_{\mathrm{VP}}=a_{\tau}\epsilon-\sigma_{\tau}x_{\mathrm{data}}.
$$

Обратное преобразование:

$$
x_{\mathrm{data}}=a_{\tau}z_{\tau}-\sigma_{\tau}v_{\mathrm{VP}},\qquad
\epsilon=\sigma_{\tau}z_{\tau}+a_{\tau}v_{\mathrm{VP}}.
$$

Если `a = cos(phi)` и `sigma = sin(phi)`, производная по `tau` равна `dphi/dtau * v_VP`. Это не автоматически RF target `data-noise` и не любой output с названием velocity. В MIT Gaussian path beta — standard deviation; в DDPM beta_k — variance отдельного noising step.

**Проверить:** inverse errors на общих inputs, знак/масштаб при обращении времени, отличие VP v от RF velocity. Формулы не используются в singular endpoints без отдельного анализа.

## Опыт B: выпрямление и reflow

После Lab 2 Part 4 зафиксируйте teacher и формируйте paired endpoints:

1. Sample source `x0`.
2. Получить `x1 = teacher_sampler(x0)` без gradient через teacher.
3. Interpolate между **этими же** endpoints; target равен `x1 - x0`.
4. Обучать student; отдельно оценивать teacher cost, training budget и held-out sampling при одинаковых source samples.

Straight conditional targets сами по себе не гарантируют straight learned marginal trajectories или качественный one-step sampler. Reflow, progressive distillation и простое уменьшение steps — разные процедуры. Прямой coupling не обязан быть optimal transport в многомерном случае.

В контрольном опыте используется **аналитический teacher и известное выпрямленное поле**. Здесь endpoint map известен заранее, поэтому one-step equality — oracle control, не результат обучения student и не заявление об ускорении Kandinsky.

## Опыт C: temporal consistency не равна неподвижности

Синтетическая последовательность содержит движущийся объект. Сравните согласованное движение, повтор одного кадра и независимый шум:

- raw frame-to-frame MSE;
- MSE относительно известной reference sequence;
- residual после компенсации известного движения.

У замороженного кадра первая метрика равна нулю, хотя движение не выполнено. Это пример плохой единственной метрики, а не production video benchmark. В реальном видео motion estimation/occlusions добавляют собственную ошибку.

Второй тест изменяет future frames и проверяет past outputs causal Conv3d. Причинность video codec не означает, что весь diffusion Transformer обязан быть causal.

## Запуск контролей

Из корня репозитория:

```bash
uv run --locked python curriculum/scripts/supplementary_experiments.py
```

Только CPU, synthetic arrays и аналитические поля; нет weights/dataset downloads. [Результаты](reports/supplementary-experiments.json), plots сохраняются в ignored `curriculum/artifacts/supplementary/`. В ComfyUI C0 можно загрузить PNG-график/contact sheet как демонстрацию; математика выполняется Python-кодом, а не KSampler.

**Задание для отчёта:** сначала предсказать результат каждого контроля, затем записать ошибку, scope и следующий один опыт. После этого вернуться к основному MIT маршруту.
