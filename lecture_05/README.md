# Lecture 5: CTMC и дискретный Flow Matching

В предоставленных трёх исходных labs нет отдельного задания на Lecture 5. Поэтому добавлен [самостоятельный CPU notebook](05_ctmc_and_discrete_flow.ipynb). Это новая учебная адаптация, не официальный Lab Four MIT.

Читать [slides](https://diffusion.csail.mit.edu/2026/docs/20260130_Lecture_05.pdf) pp.5–28,37–40; notes §7, PDF pp.54–66. Предпосылки: многомерные вероятности, условное ожидание, probability paths и CFM из P0/P3. Полный video inference для этой темы не требуется.

## Последовательность

1. Задать конечное state space и проверить rate matrix Q.
2. Различить transition **rates** Q и transition **probabilities** P(h). Rate может быть больше 1.
3. Получить exact law `p0 @ matrix_exp(t*Q)` и сравнить с Gillespie trajectories и finite-step approximation.
4. Проверить positivity, conservation of total probability и stationary law; исследовать ошибку шага и sampling error отдельно.
5. Построить factorized masking path для коротких categorical sequences.
6. Выполнить маленькое обучение denoising categorical posterior с cross-entropy; сравнить с аналитическим Bayes posterior игрушечного примера.
7. Объяснить, как posterior parametrizes off-diagonal rates; не объявлять one-step denoising полным CTMC sampler.

В notebook используем строки: `Q[i,j]` — rate из i в j, `p` — row vector, `dp/dt = p @ Q`. В notes запись $Q_t(y\mid x)$ фиксирует destination/source; при column-vector representation появляется transpose. Всегда сначала выпишите индексы.

## Связь с предыдущими темами

| Непрерывное состояние | Дискретное состояние |
|---|---|
| Velocity / score | Jump rates / denoising token posterior |
| Continuity / Fokker–Planck PDE | Kolmogorov forward equation |
| MSE на conditional velocity/noise target | Categorical cross-entropy для posterior |
| Euler / Euler–Maruyama | Gillespie или корректная finite-step transition approximation |

Вероятностный вектор CTMC удовлетворяет ODE, хотя сам categorical state не движется по обычной непрерывной trajectory. Token IDs Qwen/CLIP в видеопайплайне **не означают**, что video model обучен дискретной диффузии: они могут быть только conditioning input.

**ComfyUI:** [C6](../curriculum/COMFYUI_LABS.md) показывает сохранённый probability plot; стандартный KSampler не выполняет этот CTMC алгоритм. Реальный CTMC custom node — отдельное расширение после проверки Python реализации.

**Готовность:** корректная Q, exact/simulated probability comparison с uncertainty, явно отделённый toy denoising result и объяснение отличия от continuous latent video model.
