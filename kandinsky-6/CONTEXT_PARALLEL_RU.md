# Kandinsky 6 и context parallelism: сравнение с LTX-2 и Wan в Diffusers 0.42.1

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../curriculum/README.md) · [Полное содержание](../curriculum/CONTENTS.md).
<!-- study-navigation:end -->

[K6](README.md) · [Diffusers runner](generation/README.md) · [Notebook 16](notebooks/16_kaggle_diffusers_lite_generation.ipynb) · [diffusers#15156](https://github.com/huggingface/diffusers/issues/15156) · [kandinskylab/kandinsky-6#22](https://github.com/kandinskylab/kandinsky-6/issues/22).

**Статус на 30 октября 2026:** это разбор исходного кода, а не реализация. Замеры текущего размещения без CP — в §6 и [отчёте Kaggle](reports/dit-placement-kaggle-measurements.json). Источник — установленный `diffusers==0.42.1` и `transformer/config.json` Lite-чекпойнта на ревизии `4c487e8b`. Ни один `_cp_plan` для Kandinsky 6 не запускался; выводы о корректности ниже — рассуждение по коду, которое нужно подтвердить численным тестом на двух GPU.

## 1. Зачем это нужно

Notebook 16 делит DiT по слоям (`device_map="balanced"`). Трасса `nvidia-smi` на 2×T4 (Smoke, 432×256, 25 кадров) показала: GPU работают по очереди, одновременно заняты обе только в 3 % отсчётов. Это ожидаемо для разбиения по слоям при batch 1. Чтобы обе GPU считали один ролик одновременно, нужен **context parallelism (CP)**: токены видео делятся между процессами, и каждый процесс проходит все слои над своей частью последовательности.

В Diffusers CP включается так:

```python
transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2))  # под torchrun, 2 процесса
```

Для этого у модели должен быть `_cp_plan`. У `Kandinsky6Transformer3DModel` его нет.

## 2. Как устроен `_cp_plan`

`_cp_plan` — словарь «имя модуля → что делить или собирать» ([`_modeling_parallel.py:243-343`](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/_modeling_parallel.py)). Хуки ставятся на модули, а не на строки внутри `forward`:

| Запись | Когда срабатывает | Что делает |
|---|---|---|
| `"": {"arg": ContextParallelInput(split_dim=…)}` | перед `forward` всей модели | делит входной тензор по последовательности |
| `"module": {"arg": ContextParallelInput(…)}` | перед `forward` модуля | делит его вход; имя аргумента сопоставляется по сигнатуре, позиционные аргументы тоже работают (`hooks/context_parallel.py:50-77`) |
| `"module": {0: ContextParallelInput(…, split_output=True)}` | после `forward` модуля | делит **выход** с индексом 0; так делят RoPE-таблицы |
| `"module": ContextParallelOutput(gather_dim=…)` | после `forward` модуля | собирает полную последовательность на каждом процессе |

`enable_parallelism` записывает конфиг во **все** attention processors модели (`modeling_utils.py:1813-1822`), после чего каждый вызов `dispatch_attention_fn(..., parallel_config=...)` идёт через Ulysses (all-to-all по головам) или Ring (обмен K/V с объединением по LSE).

## 3. Что уже есть у Kandinsky 6

Источник: [`transformer_kandinsky6.py`](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_kandinsky6.py).

- **Processor уже готов.** `Kandinsky6AttnProcessor` хранит `_parallel_config` и передаёт его в `dispatch_attention_fn` (строки 198-205). Менять attention не нужно.
- **Архитектура Lite** (`config.json`): `model_dim=1792`, `model_dim_a=896`, `axes_dims=(16, 24, 24)` → `head_dim=64`. Значит, **28 голов видео и 14 голов аудио**, `num_visual_blocks=32`, `num_text_blocks=2` на каждую ветку, `ca_rope=True`, `cross_gates=True`, `text_token_padding=True`.
- **Поток данных в `forward`** (строки 787-844):
  1. RoPE для текста из `arange(L_text)`; две текстовые ветки (video и audio) по 2 блока self-attention **с RoPE**.
  2. Видео: patchify `(B,T,H,W,C)` → `flatten(1,3)` → `(B, N_video, C)`; RoPE `(T,H,W,…)` → `flatten(0,2)`.
  3. Аудио: `(B, N_audio, C)` и 1D-RoPE.
  4. 32 fused-блока. В каждом: self-attention видео и аудио, cross-attention на текст, затем **va** (Q = видео, K/V = аудио) и **av** (Q = аудио, K/V = видео), с той же RoPE, что и self-attention (`ca_rope=True`, строки 551-564).
  5. `reshape(-1, T, H, W, C)` и `out_layer` с unpatchify внутри (строки 838-840, 341-349).

## 4. LTX-2 и Wan: что из них можно взять

**LTX-2** ([`transformer_ltx2.py:1125-1136`](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_ltx2.py)):

```python
_cp_plan = {
    "": {
        "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3),
        "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3),
        "encoder_attention_mask": ContextParallelInput(split_dim=1, expected_dims=2),
    },
    "rope": {0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
             1: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True)},
    "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
}
```

- Делятся только видео-токены, видео-текст с маской и RoPE видео self-attention.
- `audio_hidden_states`, `audio_encoder_hidden_states` и RoPE аудио в плане **не указаны**, то есть аудио на всех процессах полное.
- Отдельная RoPE для audio↔video cross-attention (`cross_attn_rope`, строка 1508) строится по всем видео-токенам и в плане тоже **не указана**, а Q видео в этом attention уже разделён. По коду не видно, как эти длины согласуются. Путь audio+video под CP у LTX-2 может быть непроверенным; без запуска я этого не утверждаю. Поэтому LTX-2 — образец для **аудио**, но не полный шаблон.

**Wan** ([`transformer_wan.py:552-569`](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_wan.py)): RoPE делится на выходе модуля `rope`, видео — на **входе `blocks.0`**, `encoder_hidden_states` **намеренно не делится** (комментарий в коде: 769 токенов не делятся на число устройств), сборка на `proj_out`. Это прецедент, что неразделённый текст под CP в Diffusers допустим.

| | LTX-2 | Wan | Kandinsky 6 (нужно) |
|---|---|---|---|
| Где делится видео | вход модели `""` | вход `blocks.0` | вход `visual_transformer_blocks.0` (patchify внутри `forward`) |
| RoPE видео | выход `rope`, уже плоский | выход `rope` | выход `visual_rope_embeddings`, но он **не плоский** — см. §5 |
| Текст | делится, RoPE в тексте нет | не делится | **не делить**: текстовые блоки используют RoPE от `arange(L)` |
| Аудио | не делится | — | не делить (короткое, длина нечётная) |
| Cross-modal RoPE | отдельная, не в плане | — | та же, что self-attention: делится вместе с видео |
| Сборка | `proj_out` (плоские токены) | `proj_out` | нет плоского модуля: unpatchify внутри `out_layer` — см. §5 |

## 5. Что придётся изменить в Kandinsky 6

Только декларативного плана недостаточно: нужны два небольших рефакторинга точек разбиения.

1. **Плоская RoPE видео.** `Kandinsky6RoPE3D.forward` возвращает `(T, H, W, 1, D, 2, 2)`, а `flatten(0, 2)` выполняется уже в `forward` модели. Делить выход по оси `T` нельзя: у Smoke 7 латентных кадров, а 3024 токена делятся пополам посреди кадра. Нужно, чтобы модуль возвращал `(T·H·W, 1, D, 2, 2)`; тогда `"visual_rope_embeddings": {0: ContextParallelInput(split_dim=0, split_output=True)}`. Одна разделённая таблица корректно обслуживает и self-attention, и va/av cross-attention.
2. **Сборка до unpatchify.** Сейчас `forward` делает `reshape(-1, T, H, W, C)` до `out_layer`, а на половине токенов это сломается. `out_layer` до unpatchify работает по токенам (modulation зависит только от времени), поэтому его можно разделить: плоская проекция → `ContextParallelOutput(gather_dim=1)` → unpatchify в `forward`. Так сделано в LTX-2 и Wan.

План после рефакторинга (набросок, не проверен):

```python
_cp_plan = {
    "visual_rope_embeddings": {0: ContextParallelInput(split_dim=0, split_output=True)},
    "visual_transformer_blocks.0": {"vis": ContextParallelInput(split_dim=1, expected_dims=3)},
    "out_layer.out_layer": ContextParallelOutput(gather_dim=1, expected_dims=3),  # после рефакторинга §5.2
}
```

**Почему неразделённые текст и аудио должны работать (гипотеза для теста).** Под Ulysses процесс со своей полной копией аудио отдаёт её в all-to-all, и в attention попадают K/V `[A, A]`. Softmax по дублированным ключам даёт те же веса, после обратного all-to-all каждый процесс получает свою копию `A`. Это верно, только если **маска отсутствует**: при `encoder_attention_mask` длины `L` и ключах длины `2L` форма не совпадёт. Runner notebook 16 вырезает padding и маску не передаёт, но pipeline при `text_token_padding=True` может её построить. Это главный кандидат на отдельный тест.

## 6. Ограничения для 2×T4

| Проверка | Значение | Вывод |
|---|---|---|
| Видео-токены, Smoke T2V | 7·16·27 = 3024 | делится на 2 и 4 |
| Видео-токены, Baseline T2V | 31·30·54 = 50 220 | делится на 2 и 4 |
| Видео-токены, I2V (+1 reference кадр) | 3456 / 51 840 | делится на 2 и 4 |
| Аудио-токены | Smoke 45, Baseline 218 | 45 нечётно → аудио не делить |
| Головы для Ulysses | видео 28, аудио 14 | `ulysses_degree=2` подходит; при 4 аудио не делится → нужен `ulysses_anything` |
| Backend attention | T4 = sm75, FlashAttention 2 требует sm80+ | остаётся `NATIVE` (SDPA), он помечен `supports_context_parallel=True` |
| Ring attention | требует LSE, а `NATIVE` его не возвращает (`attention_dispatch.py:876`) | на T4 практически только **Ulysses** |
| Связь между GPU | PCIe, без NVLink | all-to-all в каждом attention; на Smoke выигрыш может быть около нуля, смысл — Baseline |

Формы латентов подтверждены прогоном на Kaggle: Smoke `[1,16,7,32,54]` и аудио `[1,40,45]`, Baseline `[1,16,31,60,108]` и `[1,40,218]`.

**Замер без CP (Kaggle 2×T4, version 7, по одному прогону):**

| Профиль | `balanced` (2 GPU) | `cuda:0` (1 GPU) | Обе GPU заняты одновременно (`balanced`) |
|---|---|---|---|
| Smoke, 3024 видео-токена | 12,5 s, 3,1 + 3,2 GiB | 13,4 s, 6,1 GiB | 6,7 % |
| Baseline, 50 220 видео-токенов | 746 s, 5,6 + 5,6 GiB | 762 s, 8,6 GiB | 0 % |

Split не ускоряет ролик, и Baseline помещается на одну T4, так что CP для Lite нужен только ради latency, а не памяти. У Baseline в 16,6 раза больше токенов, но denoising дольше примерно в 60 раз: на полном размере доминирует attention. Именно здесь Ulysses мог бы окупить обмен по PCIe; оценка ускорения требует реализации и замера.

## 7. План работ и проверки

1. Рефакторинг §5 без `_cp_plan`. Критерий: выход одно-процессной модели **бит-в-бит** совпадает с текущим на tiny-конфиге (как в `scripts/validate_diffusers_generation.py`).
2. Добавить `_cp_plan` из §5. Критерий: под `torchrun --nproc_per_node=2` с `ulysses_degree=2` выход совпадает с одно-процессным в пределах fp16-допуска; отдельно проверить T2V, I2V (`visual_rope_pos` с повторным индексом 0) и случай с маской текста.
3. Замер на Kaggle 2×T4: Smoke и Baseline, `denoise_seconds` против `DIT_PLACEMENT='balanced'` и `'cuda:0'`, трасса `nvidia-smi`.
4. Если CP с маской текста не работает — либо делить текст после построения RoPE (выходы `*_text_embeddings` и `*_text_rope_embeddings`) с padding до кратной длины, либо требовать отсутствия маски.

Вопросы к авторам (#22): делят ли они последовательность во внутреннем инференсе (в `kandinsky/core/algo/prepare_latents.py` делимость зависит от `world_size` 2/4/8) и нужно ли что-то особое для va/av attention.
