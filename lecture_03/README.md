# Lecture 3A / 3B: score matching, stochastic sampling и CFG

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../curriculum/README.md) · [Полное содержание](../curriculum/CONTENTS.md).

**Связанные лекции:** [Lecture 3](https://diffusion.csail.mit.edu/2026/docs/20260123_Lecture_03.pdf).
<!-- study-navigation:end -->

[Общий маршрут](../curriculum/README.md), [слайды](https://diffusion.csail.mit.edu/2026/docs/20260123_Lecture_03.pdf). Это **два практических блока**, использующие разные notebooks.

## 3A: score и связь с Flow Matching

Читайте slides pp.6–23, notes §4 pp.25–33; затем [Lab 2](../lecture_02/references/lab_two_original.ipynb) Q2.4 (cells 31–36), Q3.2 (50–58), Q3.3 (59–63).

- Определите score как gradient **по x**, а не по t или weights.
- Для Gaussian выведите score и сравните с autograd; формула приведена ниже.
- Conditional score при известном clean endpoint — доступный target; marginal score требует усреднения с posterior weights.
- Для scalar diffusion `sigma(t)`, не зависящей от `x`, используйте drift из формулы ниже. Если diffusion зависит от `x`, нужны дополнительные члены; переносить формулу автоматически нельзя.
- Выведите Fokker–Planck и выделите continuity term. Различайте Gaussian denoising objective, stochastic sampler и reverse-time interpretation.

**Gaussian score:**

$$
\nabla_x\log p(x) = -\Sigma^{-1}(x-\mu).
$$

**SDE drift, сохраняющий заданные marginals:**

$$
b_t(x) = v_t(x) + \frac{\sigma_t^2}{2}\nabla_x\log p_t(x).
$$

Здесь `v_t` — marginal ODE field, `b_t` — SDE drift. Утверждение предполагает регулярность плотности/полей и spatially constant diffusion.

Опыт: одна target density, одинаковый time horizon и initial distribution, несколько diffusion coefficients. Сравнить moments/distribution distance и Monte Carlo uncertainty; равенство paths не требуется. Не называйте дополнительный шум автоматически улучшением качества. Прочитайте [уточнение Q3.3](../curriculum/SOURCE_NOTES.md).

## 3B: условная генерация и classifier-free guidance

Читайте slides pp.24–37, notes §5 pp.34–40; затем [Lab 3](../lecture_04/references/lab_three_original.ipynb) Parts 1–2 (cells 19–36).

Сначала class-conditioned MLP и label dropout, затем guidance:

$$
u_w(x,t,y) = u_{\varnothing}(x,t) + w\left(u_y(x,t)-u_{\varnothing}(x,t)\right).
$$

Проверьте w=0 и w=1. При w>1 это heuristic extrapolation, а не обещание точного sampling из исходного conditional data distribution. Unconditional branch появляется благодаря training dropout, а не благодаря пустому тексту в новом UI.

Опыт: фиксированные initial noise, class labels и solver; sweep w. Оцените class adherence, diversity, ошибки и количество branch evaluations. Negative prompt как baseline может отличаться от truly unconditional branch.

[ComfyUI C3](../curriculum/COMFYUI_LABS.md) проводить на поддерживающем обычный CFG non-distilled checkpoint. K6 Lite Distill по умолчанию CFG=1; такой checkpoint не подходит для универсального учебного guidance sweep.

**Готовность:** вывести Gaussian score; объяснить conditional vs marginal score; показать CFG identity w=1 и осознанно посчитать compute branches. После этого переходите к [Lecture 4](../lecture_04/README.md).

## Дополнительный разбор

Для сравнения noise/score/VP v/RF velocity: опыт A. [Материалы, первичные источники и CPU-контроли](../curriculum/SUPPLEMENTARY_READING.md). Основной порядок лекций/labs не меняется.
