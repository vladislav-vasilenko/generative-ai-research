"""Build the lightweight manual experiment notebook; no model downloads."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(dict(cell_type="markdown", metadata={}, source=dedent(source).strip() + "\n"))


def code(source):
    cells.append(dict(cell_type="code", metadata={}, source=dedent(source).strip() + "\n",
                      execution_count=None, outputs=[]))


md(r"""
# Kandinsky 6 Lite: от проверки ComfyUI до эксперимента с голосом

Это **пошаговый журнал исследования**, рассчитанный на M3 Max / 48 GB Unified Memory.
Сначала разберём систему на маленьких примерах, проверим файлы и подготовим измерения.
Затем вручную запустим исходную модель и только после этого перейдём к адаптеру голоса.

**Текущее решение: загрузка больших весов и генерация отложены до освобождения 50 GB+ на диске.**
Run All не скачивает модели, не отправляет задания в ComfyUI и не запускает обучение большой сети.
Для вычисляемых примеров нужны только NumPy и Matplotlib. MLX и pretrained-веса — отдельные этапы.

Запуск из корня репозитория: `uv run --locked jupyter lab kandinsky-6/notebooks/00_manual_experiment_comfy_mlx_film.ipynb`.
Окружение ComfyUI менять не нужно. [Полная инструкция](../COMFYUI_AND_EXPERIMENT_RU.md),
[состояние реализации](../STATUS.md), [Lite workflows](../workflows).

В конце получится отчёт с измеренными величинами и явно отмеченными незапущенными этапами.
Учебный пример FiLM не выдаётся за клонирование голоса.
""")
md(r"""
## 1. Что проверяем и что будет результатом

Основной вопрос: **может ли один speaker-conditioned FiLM, добавленный к замороженному
аудиотракту уже обученной CrossDiT, улучшить сходство голоса на новых текстах?**
Наша гипотеза H1 относится к adapter-only обучению. Это отдельная абляция:
[voice-cloning статья](https://arxiv.org/html/2608.15690v1) дообучает также audio backbone.
Обещания «10–20 записей, 5–15 минут» здесь не принимаются за измеренные результаты.

| Этап | Проверяем | Что сохраняем |
|---|---|---|
| A | Установка и место на SSD | пути, inventory, свободные байты |
| B | Исходный AV baseline в ComfyUI | MP4, workflow, логи, время, память |
| C | Корректность audio-forward в MLX | численная ошибка против PyTorch |
| D | Данные и обучение FiLM | manifest, split, loss, gradient, adapter |
| E | Польза голосового условия | SECS, WER/CER, естественность, регрессия |

Готовы workflow, проверки, подготовка VCTK и прототип audio-forward.
**Сквозной training/sampling pipeline для FiLM ещё не реализован и не проверен.**
Поэтому этапы D–E ниже имеют критерии и план реализации; команды несуществующего trainer не предлагаются.
""")
code("""
from pathlib import Path
import csv, json, os, shutil, subprocess, sys
from datetime import datetime, timezone
from collections import Counter

def find_course():
    for parent in (Path.cwd(), *Path.cwd().parents):
        for candidate in (parent, parent / 'kandinsky-6'):
            if (candidate / 'workflows/k6_lite_distill_smoke.json').is_file():
                return candidate.resolve()
    raise FileNotFoundError('Откройте notebook из репозитория generative-ai-research.')

ROOT = find_course()
ARTIFACTS = ROOT / 'artifacts/manual_experiment'
ARTIFACTS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(ARTIFACTS / 'mpl-cache'))
import numpy as np
import matplotlib.pyplot as plt

COMFY = Path('/Users/vladmac/ComfyUI-Installs/Vladislav/ComfyUI')
MODEL_ROOTS = [Path('/Users/vladmac/ComfyUI-Shared/models'), COMFY / 'models']
# При внешнем SSD добавьте его активный models-каталог в этот список.
MIN_FREE_BYTES = 50_000_000_000  # 50 десятичных GB; 46.57 GiB
RUN_TORCH_FIXTURE = False      # Только маленькая случайная сеть, НЕ генерация
report = {'created_utc': datetime.now(timezone.utc).isoformat(),
          'full_generation': 'not_run', 'voice_training': 'not_run',
          'voice_evaluation': 'not_run'}
print('Папка курса:', ROOT)
print('Все большие запуски выполняются вручную после проверки ресурсов.')
""")
md(r"""
## 2. Почему в исходном шаблоне было больше 100 GB

Исходный комплект включал Pro, Qwen3.5 для улучшения промпта и SR для повышения разрешения.
В **Lite Smoke** этих ветвей нет. Однако 3B transformer — только один компонент:
текстовые условия всё ещё вычисляет Qwen 2.5 VL 7B вместе с CLIP.

В вашем окне Missing models остались Qwen **15.45 GB** и Lite **5.94 GB** —
суммарно **21.39 GB по оценке интерфейса**. Это не размер RAM и не обещание пикового потребления.
Точные байты файлов смотрим отдельно: 1 GB = $10^9$ bytes, 1 GiB = $2^{30}$ bytes.
Архивы, временные файлы, результаты и swap тоже требуют места.

Вы выбрали порог 50 GB+. Следующая ячейка проверяет его **сейчас**, но ничего не загружает.
Если models будут на внешнем диске, учитывайте также свободное место системного диска для swap.
Не удаляйте существующие модели автоматически; настройка model roots выполняется отдельно в Comfy Desktop.
""")
code("""
disk_rows = []
for label, path in [('course/system volume', ROOT), *[(str(p), p) for p in MODEL_ROOTS if p.exists()]]:
    usage = shutil.disk_usage(path)
    row = {'path': label, 'free_bytes': usage.free,
           'threshold_passed': usage.free >= MIN_FREE_BYTES}
    disk_rows.append(row)
    print(f"{label}: {usage.free / 1e9:.2f} GB = {usage.free / 2**30:.2f} GiB; "
          f"50 GB gate: {row['threshold_passed']}")
disk_ready = all(r['threshold_passed'] for r in disk_rows)
report['disk'] = disk_rows
print('Решение:', 'место позволяет перейти к следующей проверке' if disk_ready
      else 'загрузку весов и генерацию пока откладываем')

fig, ax = plt.subplots(figsize=(8, 2.7))
ax.barh(['Свободно сейчас', 'Выбранный порог'],
        [disk_rows[0]['free_bytes'] / 1e9, MIN_FREE_BYTES / 1e9], color=['#3984b8', '#929292'])
ax.set_xlabel('GB, десятичные единицы'); ax.set_title('Проверка диска, не оценка RAM')
plt.tight_layout(); plt.show()
""")
md(r"""
## 3. Инвентаризация: что уже скачано

Проверим файловую систему, не загружая тензоры. Safetensors header хранит имена, dtype,
shape и byte offsets. Совпадение ожидаемой длины файла с фактической — структурная проверка.
Она не заменяет SHA-256 и может не распознать недописанный sparse-файл с заранее выставленным размером.
Файлы `.pt/.pth` здесь только перечисляются: unpickle не выполняется.

Список Missing models в ComfyUI полезен, но не является проверкой всех файлов runtime:
аудиокодек и vocoder также должны быть доступны. Конкретный loader может получать их отдельно.
""")
code("""
inventory_path = ARTIFACTS / 'comfy-model-inventory.json'
existing_roots = [p for p in MODEL_ROOTS if p.is_dir()]
inventory = {'model_files': [], 'weights_loaded': False}
if existing_roots:
    command = [sys.executable, str(ROOT / 'scripts/inspect_comfy_models.py')]
    for p in existing_roots:
        command += ['--models', str(p)]
    subprocess.run(command + ['--output', str(inventory_path)], check=True, capture_output=True, text=True)
    inventory = json.loads(inventory_path.read_text())
for row in inventory['model_files']:
    print(row['relative_path'], f"{row['bytes'] / 2**30:.3f} GiB",
          'complete=' + str(row.get('structurally_complete', 'header not inspected')))
print('Файлов:', len(inventory['model_files']), '; веса не загружались.')

required = [
 'diffusion_models/Kandinsky-6.0-Lite-distill-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors',
 'text_encoders/qwen_2.5_vl_7b.safetensors', 'text_encoders/clip_l.safetensors',
 'vae/hunyuan_video_vae_bf16.safetensors', 'audio_vae/v1-44.pth',
 'audio_vae/bigvgan_vocoder/bigvgan_generator.pt', 'audio_vae/bigvgan_vocoder/config.json']
missing = [name for name in required if not any((p / name).is_file() for p in MODEL_ROOTS)]
print('Не найдены по ожидаемым путям:')
print(chr(10).join(missing) if missing else 'Все ожидаемые пути найдены; далее нужны проверки целостности.')
report['missing_expected_paths'] = missing
""")
md(r"""
## 4. Из чего состоит pipeline и какие веса будем обучать

Текст → tokenizer → Qwen/CLIP embeddings → совместный video/audio CrossDiT → latents → VAE/vocoder.
CLIP даёт дополнительное текстовое условие; Qwen hidden states и pooled embeddings имеют разные роли.
Видео и аудио обрабатываются отдельными ветвями с cross-modal связями внутри общей модели.

| Компонент | Базовая модель | В нашем A1 |
|---|---|---|
| Tokenizers и Qwen2.5-VL-7B / CLIP | Внешние pretrained компоненты | Frozen |
| Video VAE | HunyuanVideo VAE | Frozen |
| Audio codec + vocoder | MMAudio 44.1 kHz, 40 latent channels | Frozen |
| CrossDiT, video/audio/cross-modal параметры | Pretrained Kandinsky 6 | Frozen |
| Speaker encoder | Отдельно выбранная pretrained сеть | Frozen, revision фиксируем |
| Новый FiLM | Нулевая инициализация | Trainable |

Отдельный **KVAE-Audio** имеет 48 kHz / 64 channels и не является прямой заменой MMAudio.
KVAE 2.0 тоже нельзя подставить в baseline только потому, что он новее.
Источник: [официальный K6 код и configs](../reference/kandinsky-6/README.md),
[закреплённые snapshots](../reference/source-manifest.json), [K6 report](https://arxiv.org/html/2610.05608v1).

**Упражнение:** почему frozen параметры экономят память optimizer, но не устраняют
активации и backward через слои после адаптера? Проследите цепочку производных от loss до FiLM.
""")
md(r"""
## 5. Проверяем Lite-шаблоны до запуска

В Desktop: **Templates → kandinsky6 → Kandinsky 6 Lite - Smoke**.
Вторая копия: **Workflows → K6 Research**. Если список ещё старый, обновите окно/перезапустите Desktop.
Можно также перетащить JSON из `workflows/` прямо на canvas.

`тип=kandinsky5` в Dual CLIP Loader соответствует исходному K6 workflow:
название этого загрузчика не выбирает поколение DiT. В UNETLoader должен быть именно **Lite Distill**.
Мы убрали Pro, SR, beautifier и MagCache из минимального графа.

Проверка ниже доказывает согласованность записанных links, но не исполнение nodes.
""")
code("""
workflow_reports = []
for profile in ['smoke', 'baseline']:
    path = ROOT / f'workflows/k6_lite_distill_{profile}.json'
    workflow = json.loads(path.read_text())
    nodes = {node['id']: node for node in workflow['nodes']}
    for link_id, source, out_slot, target, in_slot, kind in workflow['links']:
        assert nodes[source]['outputs'][out_slot]['type'] == kind
        assert nodes[target]['inputs'][in_slot]['type'] == kind
        assert nodes[target]['inputs'][in_slot]['link'] == link_id
        assert link_id in nodes[source]['outputs'][out_slot]['links']
    assert 'Kandinsky-6.0-Pro' not in json.dumps(workflow)
    assert nodes[1]['widgets_values'][0].startswith('Kandinsky-6.0-Lite-distill')
    row = {'profile': profile, 'nodes': len(nodes), 'links': len(workflow['links']),
           'geometry': nodes[7]['widgets_values'], 'sampler': nodes[8]['widgets_values']}
    workflow_reports.append(row)
    print(row)
report['workflow_static_checks'] = workflow_reports
""")
md(r"""
## 6. Ручной baseline — выполнить после освобождения места

1. Проверьте 50 GB gate, файлы и выбранные loader names. Закройте приложения, потребляющие много RAM.
2. Откройте Smoke: **432×240, 33 кадра, 24 fps, batch=1, steps=10, CFG=1, seed=42 fixed**.
3. Запустите Queue один раз. Сохраните workflow, полный console log и MP4.
4. Проверьте видео, аудиодорожку, длительность, sample rate, тишину/чёрные кадры и ошибки.
5. Только после успеха откройте Baseline: **864×480, 121 кадр**, прочие параметры те же.

Smoke ≈ 1.375 s по `frames/fps`; baseline ≈ 5.04 s. Проверяйте фактическую длительность выходного файла.
Малый режим — проверка системы, а не оценка качества штатной 5s генерации.
Для distill plugin использует **PiFlow**; видимые euler/simple относятся к общему интерфейсу sampler.

На Mac CPU и GPU разделяют 48 GB. CPU offload не добавляет физическую память.
Если получена ошибка, сохраните точный профиль и текст; изменение разрешения/precision — новый вариант опыта.
""")
md(r"""
## 7. Как измерить baseline без ложного ускорения от cache

Первый cold run включает загрузку весов. После него сделайте три warm run с **seed 43, 44, 45**.
Неизменный seed и graph могут дать cache hit без запуска sampler. Такое время не измеряет генерацию.
Повторяемость одного seed проверяйте отдельно в свежем процессе и отмечайте overhead загрузки.

Сохраняйте total prompt time и sampler time раздельно, если plugin их сообщает.
Запишите precision, checkpoint hash, версии, RAM pressure и Swap Used.
`torch allocated`, RSS и driver memory — разные показатели, их нельзя складывать как отдельную RAM.

Read-only мониторинг, вручную из `kandinsky-6` перед Queue:
```bash
python3 scripts/measure_comfy.py --url http://127.0.0.1:8188 --seconds 900 --output reports/baseline-memory.csv
```
Адрес замените на фактический адрес Desktop. Скрипт не отправляет prompts.
Максимум samples раз в секунду — наблюдаемый максимум, не гарантированный настоящий пик.
""")
code("""
# Создаём только пустую рабочую таблицу; измерения не придумываем.
run_csv = ARTIFACTS / 'baseline_runs.csv'
if not run_csv.exists():
    shutil.copyfile(ROOT / 'reports/templates/baseline_runs.csv', run_csv)
with run_csv.open(newline='') as stream:
    runs = list(csv.DictReader(stream))
groups = {}
for row in runs:
    if row.get('cold_or_warm') == 'warm' and row.get('total_seconds') and not row.get('error'):
        fields = ['profile', 'backend', 'checkpoint_sha256', 'width', 'height', 'frames',
                  'fps', 'steps', 'cfg', 'precision']
        key = tuple(row.get(field, '') for field in fields)
        groups.setdefault(key, []).append(float(row['total_seconds']))
print('Таблица для заполнения:', run_csv)
if groups:
    for key, warm in groups.items():
        print(key, 'Warm median:', np.median(warm), 's; min/max:', min(warm), max(warm))
else:
    print('Реальных baseline timings пока нет.')
""")
md(r"""
## 8. Что именно означает нулевая инициализация FiLM

Пусть $h$ — скрытые признаки target-аудио, $e$ — speaker embedding.
Один Linear вычисляет два вектора: $[\gamma,\beta]=We+b$.
FiLM изменяет признаки: $h'=h\odot(1+\gamma)+\beta$.
При $W=0,b=0$ имеем $h'=h$. Это простое тождество проверим численно.

Важно: сравниваются **одни и те же входы и attention context**.
Нулевой FiLM не отменяет изменения attention после добавления reference prefix.
Reference tokens не модулируются; они остаются фиксированными и при sampling.
""")
code("""
rng = np.random.default_rng(42)
B, L, D, E = 4, 8, 12, 6  # маленькие учебные размеры, не размеры K6
h = rng.normal(size=(B, L, D))
speaker = rng.normal(size=(B, E))
W = np.zeros((2 * D, E)); bias = np.zeros(2 * D)

def film(hidden, embedding, weight, bias, reference_tokens=0):
    gamma, beta = np.split(embedding @ weight.T + bias, 2, axis=-1)
    mask = (np.arange(hidden.shape[1]) >= reference_tokens)[None, :, None]
    return hidden * (1 + gamma[:, None, :] * mask) + beta[:, None, :] * mask

identity_error = float(np.max(np.abs(film(h, speaker, W, bias) - h)))
assert identity_error == 0
print('Zero-FiLM max error:', identity_error)
print('Trainable parameters в примере:', W.size + bias.size)
print('Для width=896 и speaker_dim=1024:', 2 * 896 * (1024 + 1))
report['educational_zero_film_error'] = identity_error
""")
md(r"""
### Почему reference prefix требует отдельного контроля

Attention — взвешенное среднее value-векторов. Добавив reference key/value, мы изменяем
набор элементов и знаменатель softmax. Выход может измениться даже без FiLM.
Следующий пример — обычная маленькая attention, без K6 весов.
""")
code("""
def attention(q, k, v):
    scores = q @ k.T / np.sqrt(q.shape[-1])
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities @ v

q, k, v = [rng.normal(size=(8, 12)) for _ in range(3)]
ref_k, ref_v = [rng.normal(size=(3, 12)) for _ in range(2)]
b0 = attention(q, k, v)
b2 = attention(q, np.concatenate([ref_k, k]), np.concatenate([ref_v, v]))
prefix_delta = float(np.max(np.abs(b2 - b0)))
print('Изменение только из-за reference context:', prefix_delta)
assert prefix_delta > 0
report['educational_reference_context_delta'] = prefix_delta
plt.figure(figsize=(7, 2.7)); plt.plot(b0[0], label='без prefix'); plt.plot(b2[0], label='с prefix')
plt.title('Один query: изменили context, а не FiLM'); plt.legend(); plt.tight_layout(); plt.show()
""")
md(r"""
### Маленькое обучение: может ли zero-init Linear получить gradient?

Да: нулевая инициализация одного адаптера не означает нулевой gradient.
Ниже target создаёт известный синтетический FiLM. По train примерам восстанавливаем его
матрицу простым gradient descent и проверяем на новых случайных признаках.
Это проверка математики и кода; здесь нет речи, pretrained backbone и оценки speaker identity.

Для MSE $\partial L/\partial h'=2(h'-y)/N$.
Производная по $\gamma$ суммирует $(\partial L/\partial h')\odot h$ по tokens,
а по $\beta$ — $\partial L/\partial h'$. Затем применяем правило цепочки к Linear.
""")
code("""
teacher_W = rng.normal(scale=0.07, size=W.shape)
teacher_b = rng.normal(scale=0.02, size=bias.shape)
train_h = rng.normal(size=(48, L, D)); train_e = rng.normal(size=(48, E))
val_h = rng.normal(size=(24, L, D)); val_e = rng.normal(size=(24, E))
train_y = film(train_h, train_e, teacher_W, teacher_b)
val_y = film(val_h, val_e, teacher_W, teacher_b)
fit_W = W.copy(); fit_b = bias.copy()
losses, val_losses = [], []
first_gradient_norm = None
for step in range(400):
    pred = film(train_h, train_e, fit_W, fit_b)
    error = pred - train_y
    grad_out = 2 * error / error.size
    grad_coeff = np.concatenate([(grad_out * train_h).sum(axis=1), grad_out.sum(axis=1)], axis=-1)
    grad_W = grad_coeff.T @ train_e; grad_b = grad_coeff.sum(axis=0)
    if step == 0:
        first_gradient_norm = float(np.linalg.norm(grad_W))
    losses.append(float(np.mean(error**2)))
    val_losses.append(float(np.mean((film(val_h, val_e, fit_W, fit_b) - val_y)**2)))
    fit_W -= 0.2 * grad_W; fit_b -= 0.2 * grad_b
assert first_gradient_norm > 0 and val_losses[-1] < val_losses[0] / 100
print('Gradient norm при zero-init:', first_gradient_norm)
print('Synthetic validation MSE:', val_losses[0], '→', val_losses[-1])
report['synthetic_film'] = {'scope': 'synthetic features only; not voice cloning',
    'initial_validation_mse': val_losses[0], 'final_validation_mse': val_losses[-1],
    'initial_weight_gradient_norm': first_gradient_norm}
plt.figure(figsize=(7, 2.7)); plt.semilogy(losses, label='train'); plt.semilogy(val_losses, label='validation')
plt.xlabel('Gradient step'); plt.ylabel('Synthetic MSE'); plt.legend(); plt.tight_layout(); plt.show()
""")
md(r"""
**Упражнения:** измените speaker embedding на константу и объясните потерю управляемости.
Проверьте finite-difference gradient для одного элемента W.
Вызовите `film(..., reference_tokens=3)` и убедитесь, что первые три tokens не меняются.
Почему хороший synthetic MSE ещё ничего не говорит о качестве речи?

## 9. Контролируемые варианты настоящего исследования

| Вариант | Reference prefix | FiLM | Назначение |
|---|---|---|---|
| B0 | нет | нет | исходная модель |
| B1 | нет | zero, frozen | проверка сохранения функции |
| B2 | есть | zero, frozen | отдельный эффект reference context |
| A1 | есть | обучается | проверка adapter-only гипотезы |
| A2, позже | есть | обучается + audio LoRA/backbone | отдельный дополнительный бюджет |

B0/B1 сравниваются с одинаковыми noise/time/text/context. Для всех вариантов фиксируются
число шагов, decoder, длина reference, геометрия и precision. Нельзя одновременно менять порт,
checkpoint, codec и адаптер, а затем приписывать разницу одному FiLM.

Для A1 сначала нужен короткий overfit test на 1–2 train utterances, затем три training seed,
выбор checkpoint по validation и единственная финальная test оценка.
""")
md(r"""
## 10. VCTK: данные и защита от утечки текста

Используем открытый pilot **p225, p226, p227, p228**, 120 клипов.
Это проверка пайплайна, не воспроизведение большого benchmark статьи.
Деление по utterance, а одинаковый нормализованный текст глобально закрепляется за одним split.
Проверяем не только filenames: у разных VCTK speakers бывают одинаковые предложения.
Задача этого pilot — новые тексты известных спикеров; unseen-speaker generalization им не проверяется.

Подготовка уже выполнялась; в Git хранятся результаты аудита, не аудиофайлы.
Для воспроизведения позднее, вручную: `python3 scripts/prepare_vctk.py --download`.
Скрипт забирает выбранные ZIP ranges, а не весь архив 6.28 GB; сеть всё равно нужна.
Run All эту команду не вызывает.
""")
code("""
dataset_summary = json.loads((ROOT / 'reports/vctk-pilot-preparation.json').read_text())
print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))
assert sum(dataset_summary['split_counts'].values()) == 120
assert all(v == 0 for v in dataset_summary['cross_split_text_overlap'].values())
manifest_path = ROOT / 'artifacts/vctk/manifest.json'
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    texts_by_split = {}
    for row in manifest['rows']:
        texts_by_split.setdefault(row['split'], set()).add(row['text_sha256'])
    for a in texts_by_split:
        for b in texts_by_split:
            if a < b:
                assert texts_by_split[a].isdisjoint(texts_by_split[b]), (a, b)
    print('Локальный manifest перепроверен:', len(manifest['rows']), 'клипов')
else:
    print('Локальных аудио/manifest нет; показан сохранённый результат подготовки, не новый аудит аудио.')
fig, ax = plt.subplots(figsize=(7, 2.7))
counts = dataset_summary['split_counts']
ax.bar(list(counts), list(counts.values())); ax.set_ylabel('Клипы'); ax.set_title('VCTK pilot split')
plt.tight_layout(); plt.show()
""")
md(r"""
Перед настоящим обучением зафиксируйте resampling/trim/normalization, reference duration,
codec revision, speaker encoder revision/dimension и cache hashes. Reference utterance не может
быть тем же фрагментом, на котором считается target loss. В тесте reference заранее фиксируется.

Qwen3-TTS embedding 1024 и Resemblyzer embedding 256 — разные представления.
Увеличить размерность padding до 1024 не означает воспроизвести Qwen3-TTS.

## 11. MLX: сначала равенство вычислений, потом обучение

Перенос архитектуры и pretrained-весов на MLX сам по себе не требует полного переобучения.
Но он требует корректного воспроизведения attention, norms, RoPE, dtype и sampler.
Наш прототип пока покрывает **audio-only forward** native CrossDiT, не весь AV pipeline.
Общий checkpoint ComfyUI содержит отдельно именованные audio/video/cross-modal параметры.
Выбор audio-параметров — абляция с отключёнными связями, а не эквивалентная совместная модель.

Начинаем с маленького настоящего класса K6 со случайными весами и ненулевыми gates.
Затем сравниваем pretrained audio-forward на одинаковых входах.
Проверки: max absolute error < $10^{-3}$, relative L2 < $10^{-4}$, zero-FiLM error = 0 для этого FP32 fixture.
Эти пороги не гарантируют равенство длинной генерации; нужна дальнейшая trajectory/audio проверка.
""")
code("""
# Флаг в первой ячейке по умолчанию False. Здесь нет pretrained-весов.
if RUN_TORCH_FIXTURE:
    comfy_python = COMFY / '.venv/bin/python'
    if not comfy_python.is_file():
        raise FileNotFoundError('Укажите Python с torch и safetensors; не меняйте окружение ComfyUI автоматически.')
    result = subprocess.run([str(comfy_python), str(ROOT / 'scripts/torch_parity_reference.py')],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    print(result.stdout)
else:
    print('CPU fixture пропущен. Инструкция и скрипт готовы, новый результат не заявляется.')

for name in ['mlx-random_fixture-parity.json', 'mlx-pretrained-parity.json']:
    path = ROOT / 'reports' / name
    if path.exists():
        parity = json.loads(path.read_text())
        print(name, 'passed=', parity['passed'], 'scope=', parity['scope'])
    else:
        print(name, 'NOT RUN — файла измерений нет')
""")
md(r"""
После освобождения места создайте отдельное окружение MLX по [инструкции](../COMFYUI_AND_EXPERIMENT_RU.md#7-проверка-mlx-порта-до-обучения).
Сначала выполните `scripts/torch_parity_reference.py`, затем `scripts/mlx_parity.py` в MLX окружении.

Для pretrained проверки **не нужна ещё одна копия весов на SSD**:
```bash
/Users/vladmac/ComfyUI-Installs/Vladislav/ComfyUI/.venv/bin/python scripts/torch_parity_reference.py --pretrained --weights /ПУТЬ/К/ПОЛНОМУ/LITE/diffusion_pytorch_model.safetensors
.venv-mlx/bin/python scripts/mlx_parity.py --pretrained
```
Это команды из папки `kandinsky-6`. Placeholder замените полным путём из inventory.
Проверка исходного pinned checkpoint включает SHA-256; файлы ComfyUI не изменяются.
PyTorch reader выбирает audio tensors. В MLX используются только выбранные параметры,
но режим чтения полного safetensors всё равно нужно проверять по фактической памяти runtime.

У native audio output head две normalization; RMSNorm eps зависит от dtype.
Не «упрощайте» их до прохождения parity.
""")
md(r"""
## 12. Почему Lite Distill не является обычной velocity-моделью

У non-distilled audio velocity выход имеет 40 channels.
У выбранного **PiFlow Distill** head выдаёт **400 = 10 policy-grid points × 40 channels**.
Это не waveform из 400 аудиоканалов и не основание применить обычный FM-MSE к этим 400 числам.
Официальный PiFlow denoise loop работает с обоими потоками.

Значит есть два разных маршрута:

1. Сохранить Distill для быстрого исходного AV baseline и исследовать корректный перенос его policy/sampler.
2. Для первого обычного FM adapter training явно выбрать **non-distilled Lite** и соответствующий codec/objective.

Второй маршрут потребует другого checkpoint и является другим опытом. Мы его автоматически не скачиваем.
Продолжение обучения PiFlow требует отдельно обоснованной distillation objective/teacher.
""")
code("""
configs = ROOT / 'reference/hf'
for name in ['Kandinsky-6.0-Lite-distill-5s-Diffusers', 'Kandinsky-6.0-Lite-5s-Diffusers']:
    config = json.loads((configs / name / 'transformer/config.json').read_text())
    print(name, {k: config.get(k) for k in ['model_dim_a', 'num_visual_blocks', 'in_audio_dim', 'out_audio_dim']})
""")
md(r"""
### Flow Matching для будущего non-distilled опыта

В выбранной convention $z$ — чистый latent при $t=0$, $\epsilon$ — Gaussian noise при $t=1$:
$$x_t=(1-t)z+t\epsilon,\qquad v^*=\epsilon-z.$$
Обучаем $v_\theta(x_t,t,c)$ приближать $v^*$; при sampling интегрируем от 1 к 0.
Loss считаем только на target tokens; reference prefix фиксирован.
Text и reference dropout 0.1 независимо дают вероятности .81/.09/.09/.01 для четырёх режимов.
Это важно для будущего split CFG, а не четыре равновероятных условия.

Ниже только проверка знака и формы пути на двух числах. Реальную модель эта ячейка не обучает.
""")
code("""
z = np.array([1.0, -0.5]); noise = np.array([-0.8, 1.2])
t = np.linspace(0, 1, 21)
path = (1 - t[:, None]) * z + t[:, None] * noise
velocity = noise - z
dt = -0.1
state = noise.copy()
for _ in range(10):
    state += dt * velocity
assert np.allclose(state, z)
print('Euler с точным постоянным velocity восстановил data:', state)
plt.figure(figsize=(5, 3)); plt.plot(path[:, 0], path[:, 1], 'o-')
plt.scatter(*z, label='data t=0'); plt.scatter(*noise, label='noise t=1')
plt.xlabel('latent coordinate 1'); plt.ylabel('latent coordinate 2'); plt.legend(); plt.tight_layout(); plt.show()
""")
md(r"""
## 13. Что измерять после реализации trainer и waveform sampler

Перед запуском сохраните имена optimizer parameters: в A1 там только W/bias FiLM.
Проверьте ненулевой gradient, frozen parameter delta = 0, masked reference loss и adapter reload equality.
Выбирайте checkpoint только по validation. После выбора не подстраивайте параметры по test.

| Величина | Как сравниваем |
|---|---|
| Train/validation loss | Фиксированный held-out noise/time; значение, steps, gradient norm |
| SECS | Один frozen speaker verifier; cosine vs reference и отдельно vs centroid |
| WER/CER | Одинаковый ASR/normalization; суммарные ошибки / суммарные слова или символы |
| Естественность | Слепое прослушивание; UTMOS только дополнительный proxy |
| CLAP | Соответствие тексту, не speaker identity |
| FAD | Dataset-level metric с указанным encoder, не оценка одного клипа |
| No-reference regression | B0 и trained adapter с отключённым reference на одинаковых условиях |
| Стоимость | step time, total time, observed peak memory, RAM/swap, число NFE |

Оценивайте парные разницы B2→A1 на одинаковых target/reference/seed.
Покажите результат каждого speaker и разброс по training seed. Bootstrap группируйте по speaker;
четыре speaker — слишком малый pilot для сильного заявления об обобщении.
P-value и проценты улучшения из статьи не являются результатами нашего опыта.
""")
code("""
voice_csv = ARTIFACTS / 'voice_evaluation.csv'
if not voice_csv.exists():
    shutil.copyfile(ROOT / 'reports/templates/voice_evaluation.csv', voice_csv)
print('Таблица голосовых измерений:', voice_csv)
print('Пока нет сгенерированного test audio — SECS/WER не вычисляем.')
""")
md(r"""
## 14. Вывод пишем по данным

- Нет свободного диска/файлов → вывод о readiness, а не о способности модели клонировать голос.
- Не пройден parity → сначала исследуем порт; training результат не интерпретируем.
- Падает train loss, но не улучшается held-out voice → overfit/неудачная гипотеза остаются возможными.
- Растёт SECS, портятся WER/естественность → фиксируем tradeoff.
- Стабильное улучшение на test при no-reference контроле → основание расширить pilot.

При переносе на Vast.AI сохраняем тот же experiment config, а не меняем сразу всё.
Даже frozen backbone требует память для gradient path. Покупать GPU только по размеру FiLM нельзя.
Для Pro 29B одни BF16 веса около 58 GB в десятичных единицах; RTX 4090 24 GB не хранит их целиком.

Отчёт ниже содержит только результаты выполненных учебных ячеек и read-only preflight.
Записи о настоящей генерации, training и voice evaluation остаются `not_run`.
""")
code("""
report_path = ARTIFACTS / 'manual-readiness-and-educational-results.json'
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + chr(10))
print('Сохранён отчёт:', report_path)
print('Генерация:', report['full_generation'])
print('Обучение голоса:', report['voice_training'])
print('Оценка голоса:', report['voice_evaluation'])
print('Следующий ручной шаг: освободить 50 GB+, повторить preflight, проверить обязательные модели.')
""")

notebook = dict(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"}}, nbformat=4, nbformat_minor=5)
for index, cell in enumerate(cells):
    cell['id'] = f'manual-{index:03d}'
out = ROOT / 'notebooks/00_manual_experiment_comfy_mlx_film.ipynb'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print(out, len(cells), 'cells')
