# Самопроверка студента

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](README.md) · [Полное содержание](CONTENTS.md).
<!-- study-navigation:end -->

Выполнение агентом smoke tests не означает освоение темы студентом. Заполняйте свой статус и ссылку на собственный отчёт. Рекомендуемый переход — только после объяснения без подсказки и воспроизводимого numerical evidence.

| Этап | Контрольный вопрос | Артефакт | Ваш статус |
|---|---|---|---|
| P0 | Почему batch samples могут быть независимы, а координаты одного вектора — нет? | Gaussian covariance и score check | не оценено |
| P1 | Как отличить ошибку сети от ошибки Euler? | convergence plot при фиксированном T | не оценено |
| P2 | Почему Brownian increment имеет std sqrt(h), а не h? | moments/uncertainty и OU transient | не оценено |
| P3 | Почему CFM видит clean endpoint, а inference не должен его получать? | held-out loss + marginal samples | не оценено |
| P4 | Почему одинаковые marginals допускают разные ODE/SDE paths? | score/conversion и moment comparison | не оценено |
| P5 | Откуда берётся unconditional branch CFG? | dropout, w=0/1 identities, adherence/diversity | не оценено |
| P6 | Как меняется число tokens при другой patch size? | shape diagram и measured time/memory | не оценено |
| P7 | Какие weights учатся на VAE stage и latent DiT stage? | optimizer/gradient map, reconstruction/KL | не оценено |
| P8 | Почему Q не является матрицей вероятностей? | exact/Monte Carlo CTMC, positivity | не оценено |
| R | Какой один factor изменён и что могло объяснить эффект иначе? | baseline, paired results, regression, conclusion | не оценено |

Чтение/реализация/измерение/объяснение — отдельные состояния. Если вы можете нажать Run All, но не можете предсказать знак или shape результата, вернитесь к меньшему примеру.
