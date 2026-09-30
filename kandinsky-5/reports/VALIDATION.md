# Статус проверки KandinskyLab

Проверки выполнены 30 сентября — 1 октября 2026 года. Оборудование: Apple M3 Max / 48 ГБ RAM; Python 3.12.14, PyTorch 2.8.0. Полная среда сохранена в `environment-macos.txt`.

| Проверка | Результат | Что подтверждает |
|---|---|---|
| nbformat и Python syntax | 14/14 notebooks прошли | Корректные Jupyter-файлы и синтаксис ячеек |
| Последовательное выполнение в свежих kernels | 14/14 прошли | Учебные ячейки не требуют скрытых переменных других notebook |
| Уменьшенный официальный DiT CPU forward/backward | Прошёл | Контракт 33→16 channels, конечные значения, градиенты |
| DiT checkpoint round-trip через meta loading | Прошёл | Strict state_dict и восстановление nonpersistent RoPE/frequency buffers |
| BF16 eager DiT + FP32 position buffers | Прошёл | Рабочий dtype без разрушения позиционных частот |
| Query-chunked SDPA с padding mask | Прошёл | Численная эквивалентность на малых тензорах |
| MPS BF16 SDPA | Прошёл в notebook 09 | Metal операция на малом тензоре |
| Уменьшенный официальный DiT MPS FP32/BF16 | Прошёл | Проверены обе precision ветви на M3 Max |
| Mini-VAE training | Прошёл | Учебные reconstruction/KL loss и gradient update |
| Tiny DiT Flow Matching training | Прошёл | Velocity loss уменьшается на простой задаче |
| KVAEVideo: уменьшенные настоящие encoder/decoder | Прошёл | 9×32×32 RGB → 3×4×4 latent → исходные размеры |
| KVAE local checkpoint round-trip | Прошёл | Hub mixin загружает сохранённый локальный config/state_dict |
| Multi-GPU collectives | Не выполнялись | CPU shards показывают математику, не distributed performance |
| Настоящие Qwen / CLIP / Hunyuan / KVAE weights | Не загружались для execution tests | Download/forward ячейки opt-in; результата качества пока нет |
| Полный Mac MPS distill 5s | Не выполнялся | Experimental adapter подготовлен, full-scale memory/quality/latency не подтверждены |
| Полный Colab CUDA distill 5s | Не выполнялся | Official factory/config путь подготовлен, NVIDIA runtime отсутствует в этой сессии |

Учебный tiny DiT содержит 130 424 параметра; это та же структура классов с уменьшенными размерами. Tiny KVAE содержит 984 523 параметра со случайными весами. Эти модели не предназначены для качественной генерации.

На учебной fixed-latent задаче Flow Matching средний loss первых 10 шагов около 1.21, последних 10 около 0.74; validation на новых noise/time около 0.73. Это ограниченная проверка обучения, не метрика pretrained Kandinsky.

JSON отчёты: `notebook-validation.json`, `native-port-validation.json`. Исполненные notebooks с выводами/графиками находятся в `reports/executed/`; исходные notebooks оставлены чистыми для обучения.

Проверка MPS требует доступа к Metal. В sandbox PyTorch определяет устройство как CPU; при выполнении проверок вне sandbox MPS доступен. По этой причине CPU-only subprocess не является опровержением MPS совместимости оборудования.

Сетевые/весовые flags по умолчанию False. Факт прохождения notebook с такими flags означает прохождение учебного маршрута, а не пропущенных pretrained экспериментов.

## Углубление VAE и Flow Matching

Повторно выполнены расширенные notebooks 02 (44 ячейки) и 06 (38 ячеек). Прошли: exact ELBO/evidence/KL-gap identity; analytic Gaussian KL vs API; reparameterization gradients; Bernoulli NLL identity; β-VAE training; upstream Hunyuan encoder/decoder small forward и latent layout roundtrip; staged DiT forward == full forward; finite-difference velocity; conditional 2D flow training и reverse Euler sampling.

Уменьшенный Hunyuan из самого vae.py: RGB [1,3,9,32,32] → posterior parameters [1,8,3,4,4] → mode [1,4,3,4,4] → RGB исходной формы. DiT staged/direct maximum difference = 0.0 в проверке FP32. Для ToyVelocity generated means около [-1.93,-0.74] и [1.96,0.70], целевые centers [-2,-0.7] и [2,0.7]. Это учебные проверки, без pretrained weights.

## Разбор официального T2V примера

В notebook 10 добавлен разбор `examples/inference_examples_t2v.ipynb` из закреплённой ревизии исходников: выбор Distill 5s, неявные настройки pipeline и зависимости от CUDA. Notebook повторно выполнен успешно без загрузки больших весов. Полная генерация на NVIDIA и pretrained-инференс на M3 Max не проверены.

## GitHub / Colab entry points

После добавления Colab setup и кнопок все 12 notebooks повторно успешно выполнены в свежих локальных kernels. Локальная ветвь setup не меняет окружение. Установка пакетов и полный CUDA-инференс в настоящем Colab runtime не проверены.

## Расширения по приложенным материалам MIT

В лабораторную 02 добавлено 11 ячеек, в 06 — 15 (теория, CPU код, карта источников и упражнения). Обе выполнены целиком в свежих kernels. Новые Gaussian NLL, joint KL/ELBO и aggregate posterior / mutual information тождества проверены вычислениями; residual/attention encoder выдаёт правильные 2C moments. Диагностика и интерполяция используют уже обученные mini-VAE.

В FM опытах continuity residual около 4.34e-5; posterior average совпадает с аналитическим velocity до 1e-15; Euler endpoint error уменьшается с 0.54 при 4 шагах до 0.019 при 128 шагах. Corrected SDE variance около 0.358 при target 0.36; naive noise injection даёт около 0.816. Это небольшие синтетические задачи, а не оценка качества pretrained Kandinsky.

Графики новых блоков просмотрены. Полная пересборка курса воспроизводит содержание обоих расширенных notebooks; новые ячейки сохранены в генераторе. Источники/страницы/исправленные опечатки и SHA256 вложений указаны в `reference/course_materials/`. Файлы вложений целиком не включены в репозиторий.

## Лабораторная generation_utils.py

Notebook 12 (45 ячеек) целиком выполнен в свежем CPU kernel. Покрыты все 10 функций
закреплённого модуля: извлекаются их настоящие AST function bodies/decorators;
для wrapper-функций используется явный локальный CUDA→CPU facade, без глобальных
изменений PyTorch. Настоящие torchvision resize/crop и fast_sta_nabla выполнены;
уменьшенный официальный DiT подключён к настоящему generate (два forward).

Проверены CFG и число forward; направление/shift временной сетки Euler; inplace
обновление только velocity channels; visual/edit conditioning и маски; scaling/layout
codec; маршруты text/VAE; offload call order; воспроизводимость seed на CPU. Три
wrapper-функции выполняются с записывающими заглушками компонентов: они проверяют
контракты и управляющую логику, но не качество pretrained generation и не расход памяти.

Граничные случаи helper-функций воспроизведены и объяснены: normalize_first_frame с T=1
возвращает tuple, T=2…4 даёт пустую reference-группу; нулевая std требует отдельной
политики; generate с num_steps=0 не определяет pred; encode_video video-ветка
несовместима с posterior-returning Hunyuan API данной ревизии. Tensor parallel
разбиение показано на локальных shards; настоящие collectives не запускались.

Семь графиков просмотрены. Полная пересборка курса воспроизводит lab 12.
Полный pretrained-инференс и установка в настоящей Colab runtime остаются непроверенными.

## Отдельный аудит normalize_first_frame и encode_video

Лабораторная 13 (45 ячеек) выполнена целиком в свежем CPU kernel. GitHub API main на момент проверки совпал с SOURCE_REVISION; девять исходных файлов побайтово подтверждены по Git blob SHA-1/размеру, дополнительно записаны SHA-256. Manifest: reference/audits/normalization-main-2026-10-01.json.

Пройдены 160 matrix cases + 24 special cases; математические affine moment identities и decomposition глобальной reference variance; позитивные контроли T=5/9/31 и negative short/constant/singleton/layout cases; FP32/BF16 precision и constructed quantization collapse; I2V wrapper T=1/2/4/7/31 с записью decoder input. При T=1 получен точный AttributeError tuple/reshape, при T=7/31 decoder inputs конечны. Все wrapper-компоненты — записывающие заглушки; это не результат качества pretrained модели.

Настоящий уменьшенный upstream Hunyuan encode(opt_tiling=False) вернул AutoencoderKLOutput/DiagonalGaussianDistribution; исходный encode_video воспроизвёл TypeError на posterior multiplication, в том числе при return_dict=False. Явный mode adapter дал правильный scaled BTHWC Tensor. Настоящий малый diffusers AutoencoderKL подтвердил успешную image branch. CPU planner override явно указан; полноценные CUDA/tiling не проверены.

Важное уточнение: nFr=4 само по себе не объявляется ошибкой. T2V Distill не вызывает эти helpers; обычный I2V 5s T=31 не имеет empty reference; обычный I2V/I2I обходят несовместимую video encode branch. Source image-mode branch time_length=0 существует, но её полноценная pretrained поддержка не доказана. В notebook исследована отдельная skip/reject/FP32 policy с проверкой совпадения upstream на хорошо обусловленных FP32 входах. Она не подменяет оригинал.

Три графика просмотрены. Детальные результаты: reports/normalization-audit.json. Полный pretrained-инференс не запускался.
