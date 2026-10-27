# Lecture 3A / 3B: score matching, stochastic sampling и CFG

[Общий маршрут](../curriculum/README.md), [слайды](https://diffusion.csail.mit.edu/2026/docs/20260123_Lecture_03.pdf). Это **два практических блока**, использующие разные notebooks.

## 3A: score и связь с Flow Matching

Читайте slides pp.6–23, notes §4 pp.25–33; затем [Lab 2](../lecture_02/references/lab_two_original.ipynb) Q2.4 (cells 31–36), Q3.2 (50–58), Q3.3 (59–63).

- Определите score как gradient **по x**, а не по t или weights.
- Для Gaussian получите $\nabla_x\log p(x)=-\Sigma^{-1}(x-\mu)$ и сравните с autograd.
- Conditional score при известном clean endpoint — доступный target; marginal score требует усреднения с posterior weights.
- Для scalar diffusion sigma(t), не зависящей от x, drift $v_t+\sigma_t^2\nabla\log p_t/2$ сохраняет заданные marginals при выполнении регулярности. Если diffusion зависит от x, нужны дополнительные члены; эту формулу нельзя переносить автоматически.
- Выведите Fokker–Planck и выделите continuity term. Различайте Gaussian denoising objective, stochastic sampler и reverse-time interpretation.

Опыт: одна target density, одинаковый time horizon и initial distribution, несколько diffusion coefficients. Сравнить moments/distribution distance и Monte Carlo uncertainty; равенство paths не требуется. Не называйте дополнительный шум автоматически улучшением качества. Прочитайте [уточнение Q3.3](../curriculum/SOURCE_NOTES.md).

## 3B: условная генерация и classifier-free guidance

Читайте slides pp.24–37, notes §5 pp.34–40; затем [Lab 3](../lecture_04/references/lab_three_original.ipynb) Parts 1–2 (cells 19–36).

Сначала class-conditioned MLP и label dropout, затем guidance:

$$u_w(x,t,y)=u_\varnothing(x,t)+w\bigl(u_y(x,t)-u_\varnothing(x,t)\bigr).$$

Проверьте w=0 и w=1. При w>1 это heuristic extrapolation, а не обещание точного sampling из исходного conditional data distribution. Unconditional branch появляется благодаря training dropout, а не благодаря пустому тексту в новом UI.

Опыт: фиксированные initial noise, class labels и solver; sweep w. Оцените class adherence, diversity, ошибки и количество branch evaluations. Negative prompt как baseline может отличаться от truly unconditional branch.

[ComfyUI C3](../curriculum/COMFYUI_LABS.md) проводить на поддерживающем обычный CFG non-distilled checkpoint. K6 Lite Distill по умолчанию CFG=1; такой checkpoint не подходит для универсального учебного guidance sweep.

**Готовность:** вывести Gaussian score; объяснить conditional vs marginal score; показать CFG identity w=1 и осознанно посчитать compute branches. После этого переходите к [Lecture 4](../lecture_04/README.md).
