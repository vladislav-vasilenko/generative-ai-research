# Материалы MIT и дополнения к лабораторным Kandinsky

Использована предоставленная пользователем версия курса MIT 6.S184 (2026), **An Introduction to Flow Matching and Diffusion Models**, Peter Holderrieth и Ezra Erives. [Сайт курса](https://diffusion.csail.mit.edu/). Сопровождающие notebooks: `lab_one.ipynb`, `lab_two.ipynb`, `lab_three.ipynb`.

Номера страниц ниже относятся к предоставленному PDF (84 страницы) и совпадают с напечатанными номерами. Номера ячеек исходных notebooks считаются с нуля и включают Markdown; они позволяют найти материал в точной версии из `source-manifest.json`. Исходные PDF/notebooks целиком в этот репозиторий не копируются. Новые пояснения и примеры синтезированы для курса Kandinsky, а не являются полным переводом исходных заданий.

| Источник | Концепция | Где применена |
|---|---|---|
| lecture_notes, с. 46–51; lab_three, ячейки 55–69 | Gaussian VAE, variance decoder, reconstruction NLL, latent space | 02: likelihood и единицы loss, отличие от Bernoulli и pretrained VAE |
| lecture_notes, с. 77–80 | Joint KL, marginalization, ELBO | 02: совместные распределения и aggregate posterior |
| lecture_notes, с. 50–51; lab_three, ячейки 74–78 | Диагностика латентов, интерполяция, frozen VAE → latent DiT | 02 и мост к 06 |
| lecture_notes, с. 15–24; lab_two, ячейки 17–29 | Gaussian conditional paths, conditional/marginal velocity, continuity, CFM | 06: общие пути и численные проверки |
| lecture_notes, с. 25–33; lab_two, ячейки 33–35; lab_one, ячейки 4, 7, 37–41 | Score, ODE/SDE, Euler–Maruyama | 06: связь score/velocity и отдельный учебный SDE |
| lecture_notes, с. 35–44; lab_three, ячейки 24–48 | Null-conditioning/CFG, DiT | 06: conditioning и соответствие настоящему Kandinsky |

## Исправления при адаптации

- В материалах время идёт от шума к данным. В Kandinsky лабораторных `t=0` соответствует данным, `t=1` — шуму; используем `s=1−t` и меняем знак velocity.
- В `lab_two`, ячейка 33, у Gaussian score записано равенство градиента log density и градиента density. Корректно: `grad log p = grad p / p`. Пример ниже использует градиент log density.
- Опечатка с нулевыми joint distributions на с. 80 PDF не переносится. Равенство normalized совместных распределений не означает их равенство нулю.
- `beta != 1` обычно меняет вариационную цель; термин negative ELBO применяем к согласованному likelihood и коэффициенту KL=1.
- Общий posterior/variational gap не называется автоматически amortization gap; приближённое семейство и параметризация encoder дают разные причины зазора.
- SDE пример использует возрастающее время и положительный шаг. Он не является заменой ODE sampler Kandinsky; прямое добавление шума к отрицательному Euler шагу некорректно.
- Учебный VAE и DiT MIT не идентичны HunyuanVideo VAE и CrossDiT Kandinsky. Представленные материалы не раскрывают полный training recipe внешних pretrained VAE.
