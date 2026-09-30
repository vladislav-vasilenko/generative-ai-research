"""Build a reproducible source, numerical and call-site audit, without weights."""
from pathlib import Path
import ast
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
base = json.loads((ROOT / 'notebooks/00_start_and_training_map.ipynb').read_text())


def md(source):
    return dict(cell_type='markdown', metadata={}, source=textwrap.dedent(source).strip() + '\n')


def code(source):
    source = textwrap.dedent(source).strip() + '\n'
    ast.parse(source)
    return dict(cell_type='code', metadata={}, source=source, execution_count=None, outputs=[])


cells = [md(r'''
# 13. Проверка утверждений: normalize_first_frame и encode_video

**Вопрос исследования:** действительно ли в основном Kandinsky есть ошибки, или мы неверно истолковали назначение функций и допустимые входы?

Краткий ответ, который дальше проверим: изменение первых **четырёх латентных кадров** — факт реализации, но само по себе не доказательство ошибки. Подтверждены непоследовательный return type, результаты NaN/exception на некоторых входах и несовместимость video encode с posterior API. Их достижимость зависит от конкретного caller. Обычный T2V Distill 5s не вызывает эти две функции; обычный I2V 5s имеет 31 латентный кадр и не попадает в случай пустого reference.

Это отдельный аудит с положительными и отрицательными контролями, а не повтор общей лабораторной 12. Сначала проверяем происхождение кода, затем математику, контрпримеры и пути вызова. В конце предложим явно обозначенную защитную политику; исходные функции сохраняем для сравнения.

Все опыты небольшие и CPU. Большие pretrained weights не загружаются. Для Colab используйте новую сессию и выполните setup: старый checkout может не содержать manifest нового аудита.
'''), *base['cells'][1:5]]

cells += [md(r'''
## 1. Какой именно main мы проверяем

1 октября 2026 года по московскому времени GitHub API вернул для `kandinskylab/kandinsky-5/main` commit **3a74261b957df0446c1969c5cedf8bbb07969ac2**. Он совпал с `SOURCE_REVISION` курса. Проверены Git blob SHA-1 и размеры девяти файлов: helpers, VAE, pipeline callers, ComfyUI и официальный I2V пример. Для дополнительного контроля сохранены SHA-256.

Результат проверки записан в `reference/audits/normalization-main-2026-10-01.json`. Это свидетельство состояния main **на момент проверки**, а не обещание, что ветка никогда не изменится. Notebook работает офлайн с проверенными файлами; при новом аудите нужно снова сверить main, а не молча заменить исходник.

Источники: [commit API](https://api.github.com/repos/kandinskylab/kandinsky-5/commits/main), [проверенный generation_utils.py](https://github.com/kandinskylab/kandinsky-5/blob/3a74261b957df0446c1969c5cedf8bbb07969ac2/kandinsky/generation_utils.py#L40), [VAE encode](https://github.com/kandinskylab/kandinsky-5/blob/3a74261b957df0446c1969c5cedf8bbb07969ac2/kandinsky/models/vae.py#L730).
'''), code(r'''
import ast, hashlib, inspect, json, warnings
from types import SimpleNamespace
from IPython.display import display, Markdown
from labkit.generation_lab import (
    load_generation_functions, FakeDiT, FakeVAE, FakeTextEmbedder, make_generation_conf,
)

manifest = json.loads((ROOT/'reference/audits/normalization-main-2026-10-01.json').read_text())
assert manifest['main_commit'] == REVISION
for entry in manifest['files']:
    raw = (SOURCE/entry['path']).read_bytes()
    git_blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    assert git_blob == entry['git_blob_sha1']
    assert hashlib.sha256(raw).hexdigest() == entry['sha256'] and len(raw) == entry['bytes']
print('Main verification UTC:', manifest['checked_at_utc'])
print('Exact upstream commit:', REVISION, '; files verified:', len(manifest['files']))

gu = load_generation_functions(cpu_cuda=False)
normalize = gu['normalize_first_frame']
adaptive = gu['adaptive_mean_std_normalization']
encode_video = gu['encode_video']
assert gu['_adapter_log'] == []  # helpers исполняются с настоящим torch, без CUDA facade
generation_ast = ast.parse((SOURCE/'kandinsky/generation_utils.py').read_text())
for name in ['adaptive_mean_std_normalization', 'normalize_first_frame', 'encode_video']:
    node = next(n for n in generation_ast.body if isinstance(n, ast.FunctionDef) and n.name == name)
    print('\n'+name, 'lines', node.lineno, '–', node.end_lineno)
    print(ast.get_source_segment((SOURCE/'kandinsky/generation_utils.py').read_text(), node))

def table(headers, rows):
    def clean(value): return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| '+' | '.join(map(clean, headers))+' |', '| '+' | '.join(['---']*len(headers))+' |']
    lines += ['| '+' | '.join(map(clean, row))+' |' for row in rows]
    display(Markdown('\n'.join(lines)))
'''), md(r'''
## 2. Сначала договоримся о значении слова «кадр»

Normalizer принимает **THWC**, то есть `[latent_time, latent_height, latent_width, latent_channels]`. Он не принимает RGB и не принимает VAE layout `[B,C,T,H,W]`. В текущем I2V caller batch равен 1 и уже объединён с временной осью sampler.

Для официальной подготовки I2V количество латентов равно `1`, если `time_length=0`, иначе `time_length*24//4+1`. При положительных **целых** секундах получаем 7, 13, 19…31 кадров. Hunyuan decoder для совместимой сетки даёт `4*(T-1)+1` RGB кадров. Поэтому «первые четыре» здесь означает четыре позиции **латентного** времени, а не четыре RGB картинки. Нельзя по одному числу определить все изменённые RGB пиксели: это зависит от decoder.

Прочитаем и исполним именно выражение из upstream, а не перепишем его по памяти. Аннотация `int` не является runtime validation. Дробное число секунд даёт float в вычислении shape и не является рабочим способом добраться до короткого целочисленного T.
'''), code(r'''
i2v_ast = ast.parse((SOURCE/'kandinsky/i2v_pipeline.py').read_text())
i2v_class = next(n for n in i2v_ast.body if isinstance(n, ast.ClassDef) and n.name == 'Kandinsky5I2VPipeline')
i2v_call = next(n for n in i2v_class.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
frames_expr = next(n.value for n in ast.walk(i2v_call) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'num_frames' for t in n.targets))
print('Actual expression:', ast.unparse(frames_expr))
frame_code = compile(ast.Expression(frames_expr), '<upstream-num-frames>', 'eval')
duration_rows = []
for seconds in [0, 1, 2, 3, 5, 10]:
    T = eval(frame_code, {'__builtins__': {}}, {'time_length': seconds})
    duration_rows.append((seconds, T, 4*(T-1)+1))
assert dict((s, T) for s, T, _ in duration_rows)[5] == 31
table(['seconds', 'latent T', 'decoder RGB frames'], duration_rows)
fractional_T = eval(frame_code, {'__builtins__': {}}, {'time_length': 0.5})
assert isinstance(fractional_T, float)
print('0.5 seconds produces shape value:', fractional_T, type(fractional_T).__name__)
'''), md(r'''
## 3. Положительный контроль: что функция делает при нормальном входе

В исходнике явно написано `nFr=4`. Функция клонирует Tensor, выделяет `samples[:4]` и reference `samples[4:4+min(reference_frames,T-1)]`, исправляет статистики первых четырёх и возвращает clone. Остальные кадры остаются равны исходным.

**Что не следует объявлять ошибкой:** число четыре, глобальные reference statistics, ограничение изменения статистик и изменение первого латента после I2V repin. Это предусмотренные кодом операции. Причина выбора четырёх не документирована; можно предполагать эвристику стабилизации, но намерение автора из кода не доказано. Требование «выходной latent обязан точно равняться image latent» также не задано.

Проверим на девяти непостоянных кадрах: reference содержит пять кадров, исходный Tensor не меняется, кадры с индексами 4…8 сохраняются, первые четыре изменяются. График показывает mean и sample standard deviation каждого кадра.
'''), code(r'''
rng = torch.Generator(device='cpu').manual_seed(2026)
frames = torch.randn(9, 3, 4, 2, generator=rng)
frames[:4] += torch.tensor([-1.0, -0.3, 0.8, 1.6])[:,None,None,None]
frames[4:] += 0.25
before = frames.clone()
after = normalize(frames)
assert isinstance(after, torch.Tensor) and after is not frames
assert torch.equal(frames, before) and torch.equal(after[4:], frames[4:])
assert all(not torch.equal(after[i], frames[i]) for i in range(4))
assert torch.isfinite(after).all()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, statistic, title in [(axes[0], 'mean', 'Mean по кадру'), (axes[1], 'std', 'Sample std по кадру')]:
    old = getattr(frames, statistic)(dim=(1,2,3))
    new = getattr(after, statistic)(dim=(1,2,3))
    ax.plot(range(9), old, 'o-', label='До'); ax.plot(range(9), new, 's--', label='После')
    ax.axvspan(-0.3, 3.3, color='orange', alpha=0.12, label='Изменяемые латенты 0…3')
    ax.set(xlabel='Индекс латентного кадра', title=title); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
'''), md(r'''
## 4. Математика adaptive_mean_std_normalization

Для каждого source-кадра считаются $m_i,s_i$ по всем HWC значениям вместе. У reference одна общая mean $m_R$ и одна общая std $s_R$, по всем кадрам, каналам и пикселям:

$$m_i^*=\mathrm{clip}(m_R,m_i-0.05,m_i+0.1),\qquad
s_i^*=\mathrm{clip}(s_R,s_i-0.1,s_i+0.25).$$

$$y_i=\frac{x_i-m_i}{s_i}s_i^*+m_i^*.$$

Это ограниченное приближение к reference statistics, а не полное выравнивание. Асимметричные границы — фактические constants в коде. При конечных статистиках, $s_i>0$ и без elementwise clipping получаем mean$(y_i)=m_i^*$ и sample std$(y_i)=s_i^*$. Проверим эти равенства, а не только `isfinite`.

[torch.std в PyTorch 2.8](https://docs.pytorch.org/docs/2.8/generated/torch.std.html) использует `correction=1`: делит sample variance на $N-1$. Для source нужны минимум два HWC значения **на кадр**; для reference — минимум два значения **суммарно**. Один reference frame с большим HWC достаточен.
'''), code(r'''
source, reference = frames[:4], frames[4:9]
mu = source.mean((1,2,3), keepdim=True)
sd = source.std((1,2,3), keepdim=True)
mu_target = reference.mean().clamp(mu-0.05, mu+0.1)
sd_target = reference.std().clamp(sd-0.1, sd+0.25)
manual = (source-mu)/sd*sd_target+mu_target
actual = adaptive(source, reference)
assert torch.allclose(actual, manual, atol=1e-7)
assert torch.allclose(actual.mean((1,2,3)), mu_target.flatten(), atol=1e-6)
assert torch.allclose(actual.std((1,2,3)), sd_target.flatten(), atol=1e-6)
assert torch.allclose(after[:4], actual, atol=1e-7)
table(['frame', 'source mean', 'target mean', 'source std', 'target std'],
      [(i, round(mu[i].item(),4), round(mu_target[i].item(),4),
        round(sd[i].item(),4), round(sd_target[i].item(),4)) for i in range(4)])
print('Reference mean/std:', reference.mean().item(), reference.std().item())
'''), md(r'''
### Глобальная std не равна среднему std отдельных кадров

Если разные reference-кадры имеют разные mean, общий разброс включает и вариацию внутри кадров, и различия между mean кадров. Это не ошибка вычисления; это выбранная область reduction.

Для M кадров с одинаковым числом N элементов sample variance всей reference равна:

$$s_R^2=\frac{(N-1)\sum_i s_i^2+N\sum_i(m_i-m_R)^2}{MN-1}.$$

Ниже сильно разнесём mean, сохранив небольшой внутренний шум, и проверим identity. После `clump_values=True` выполняется дополнительный clamp всех значений к min/max reference. Этот нелинейный шаг уже не сохраняет выведенные выше равенства mean/std.
'''), code(r'''
group_ref = torch.randn(5, 3, 4, 2, generator=rng)*0.1
group_ref += torch.tensor([-2., -1., 0., 1., 2.])[:,None,None,None]
M, N = group_ref.shape[0], group_ref[0].numel()
group_means = group_ref.mean((1,2,3)); group_stds = group_ref.std((1,2,3))
variance_identity = ((N-1)*group_stds.square().sum()+N*(group_means-group_ref.mean()).square().sum())/(M*N-1)
assert torch.allclose(group_ref.var(), variance_identity, atol=1e-6)
print('Mean of per-frame std:', group_stds.mean().item(), '; global std:', group_ref.std().item())
clamped = normalize(frames, clump_values=True)
assert clamped[:4].min() >= reference.min() and clamped[:4].max() <= reference.max()
assert torch.equal(clamped[4:], frames[4:])
print('Clipping can change mean/std beyond affine targets:',
      (clamped[:4].std((1,2,3))-sd_target.flatten()).abs().max().item())
'''), md(r'''
## 5. Матрица граничных случаев: длина, reference_frames, clipping, dtype

Не ограничимся одним случайным примером. Переберём 160 комбинаций: восемь длин T, два dtype, пять значений reference_frames, два режима clump. Возьмём простой непостоянный вход `arange/17`, чтобы нулевая source std не смешалась с ошибкой длины.

Записываем return type, NaN/Inf, exception, реальную длину slice, неизменность входа и warnings. `isfinite(empty).all()` формально True; пустой Tensor и ранний tuple-return не считаем успешной нормализацией. Таблица ниже показывает только defaults; полная matrix останется в `matrix` и итоговом JSON отчёте.
'''), code(r'''
def record_call(label, function, source, reference=None, **kwargs):
    before = source.clone()
    row = dict(label=label, shape=list(source.shape), dtype=str(source.dtype), kwargs=kwargs)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            raw = function(source, reference) if reference is not None else function(source, **kwargs)
            out = raw[0] if isinstance(raw, tuple) else raw
            row.update(return_type=type(raw).__name__, finite=bool(torch.isfinite(out).all()),
                       nan_count=int(out.isnan().sum()), inf_count=int(out.isinf().sum()),
                       same_object=out is source, source_unmodified=bool(torch.equal(source, before)))
        except Exception as error:
            row.update(exception_type=type(error).__name__, exception=str(error))
        row['warnings'] = [str(w.message) for w in notices]
    return row

matrix = []
for T in [0,1,2,3,4,5,9,31]:
    base_input = torch.arange(T*8, dtype=torch.float32).reshape(T,2,2,2)/17
    for dtype in [torch.float32, torch.bfloat16]:
        for ref_count in [0,1,5,-1,-5]:
            for clump in [False, True]:
                row = record_call(f'T{T}/ref{ref_count}/clump{clump}', normalize,
                                  base_input.to(dtype), reference_frames=ref_count, clump_values=clump)
                row.update(T=T, reference_frames=ref_count, clump=clump,
                           reference_T=len(range(T)[4:4+min(ref_count,T-1)]) if T>1 else None)
                matrix.append(row)
assert len(matrix) == 160 and all(r.get('source_unmodified', True) for r in matrix)
defaults = [r for r in matrix if r['reference_frames']==5 and not r['clump'] and r['dtype']=='torch.float32']
table(['T', 'reference T', 'return type', 'finite', 'NaN values', 'input object returned'],
      [(r['T'],r['reference_T'],r.get('return_type'),r.get('finite'),r.get('nan_count'),r.get('same_object'))
       for r in defaults])
for r in matrix:
    if r['T'] in [0,1]: assert r.get('return_type')=='tuple' and r['same_object']
    if r['T'] in [2,3,4] and r['reference_frames']==5:
        assert r.get('exception_type')=='RuntimeError' if r['clump'] else r['nan_count']==r['T']*8
    if r['T']>=5 and r['reference_frames'] in [1,5]: assert r['finite']
'''), md(r'''
### 5.1. T=2…4: почему возникает NaN

Все имеющиеся кадры попадают в `samples[:4]`. Reference начинается с индекса 4, поэтому он пуст. `mean(empty)` и `std(empty)` дают NaN; clamp не превращает NaN в корректную статистику. Без clump_values функция возвращает Tensor с NaN. С clump_values дополнительный `reference.min()` на пустом Tensor вызывает RuntimeError.

Это воспроизводимое нарушение численного контракта helper для таких входов. Оно **не доказывает**, что обычный I2V 5s не работает: его T=31. При положительных целых секундах Python pipeline не создаёт T=2…4. Низкоуровневый caller может передать их напрямую.
'''), code(r'''
short = torch.arange(3*8, dtype=torch.float32).reshape(3,2,2,2)/17
empty_ref = short[4:4+min(5,len(short)-1)]
assert empty_ref.numel()==0
with warnings.catch_warnings(record=True) as notices:
    warnings.simplefilter('always')
    bad = normalize(short)
    print('Reference shape:', tuple(empty_ref.shape), '; NaN values:', int(bad.isnan().sum()))
    print('Warnings:', [str(w.message) for w in notices])
assert bad.isnan().all()
try:
    normalize(short, clump_values=True)
except RuntimeError as error:
    print('Expected empty min/max failure:', str(error))
else:
    raise AssertionError('Ожидали подтверждённый upstream RuntimeError')
fig, ax = plt.subplots(figsize=(6,2.4))
ax.imshow(bad.isnan().reshape(3,-1), cmap='Reds', vmin=0, vmax=1, aspect='auto')
ax.set(xlabel='HWC элемент', ylabel='Латентный кадр', title='NaN mask: T=3, reference пуст')
ax.set_yticks(range(3)); plt.tight_layout(); plt.show()
'''), md(r'''
### 5.2. T≤1: return-type inconsistency

Ранняя ветка возвращает `(latents, message)`, обычная — Tensor. `generate_sample_i2v` не распаковывает tuple: сразу после normalization он вызывает `.reshape(...)`. Таким образом, для T=1 возникает **конкретный конфликт двух функций**, а не только стилистическое замечание.

`time_length=0` не запрещён в I2V `__call__`: комментарий параметра и отдельная image-result ветка предусматривают такой запрос. Но найденный официальный I2V notebook использует только 5s; наличие ветки не доказывает поддержку или успешность полного image-инференса. Ниже точный wrapper на CPU дойдёт до этого конфликта, если предыдущие компоненты выполнились.

Для wrapper используем знакомый **изолированный CUDA→CPU facade** и записывающие заглушки. Алгоритм, normalizer и reshape остаются исходными; качество, GPU behavior и pretrained-компоненты этим опытом не проверяются. Дополнительно наблюдаем decoder input: uint8-output сам по себе не позволяет обнаружить NaN в latent.
'''), code(r'''
gu_cpu = load_generation_functions(cpu_cuda=True)
wrapper_rows = []
first_latent = torch.linspace(-0.4,0.4,8).reshape(1,2,2,2).bfloat16()
for T in [1,2,4,7,31]:
    log = []; dit = FakeDiT(log, visual_cond=True); vae = FakeVAE(log); text = FakeTextEmbedder(log)
    row = dict(T=T)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            rgb = gu_cpu['generate_sample_i2v'](
                (1,T,2,2,2), 'Учебный куб', dit, vae, make_generation_conf(), text,
                images=first_latent, num_steps=1, guidance_weight=1., scheduler_scale=1.,
                seed=42, device='cpu', vae_device='cpu', progress=False, offload=False)
            row.update(rgb_shape=list(rgb.shape), rgb_dtype=str(rgb.dtype),
                       decoder_latent_finite=bool(torch.isfinite(vae.decode_calls[0]).all()))
            if row['decoder_latent_finite']:
                scaled=vae.decode_calls[0].permute(0,2,3,4,1)[0]*vae.config.scaling_factor
                row['first_latent_max_change']=float((scaled[:1].float()-first_latent.float()).abs().max())
        except Exception as error:
            row.update(exception_type=type(error).__name__, exception=str(error))
    wrapper_rows.append(row)
assert wrapper_rows[0]['exception_type']=='AttributeError' and 'reshape' in wrapper_rows[0]['exception']
assert all(not r['decoder_latent_finite'] for r in wrapper_rows if r['T'] in [2,4])
assert all(r['decoder_latent_finite'] for r in wrapper_rows if r['T'] in [7,31])
assert all(r['first_latent_max_change']>0 for r in wrapper_rows if r['T'] in [7,31])
table(['T', 'result / exception', 'decoder input finite', 'first latent change'],
      [(r['T'], r.get('exception',r.get('rgb_shape')), r.get('decoder_latent_finite'),
        r.get('first_latent_max_change')) for r in wrapper_rows])
'''), md(r'''
## 6. Где эти функции действительно вызываются

| Маршрут | normalize_first_frame | encode_video video branch | Что можно утверждать |
|---|---|---|---|
| T2V Lite Distill 5s | не вызывается | не вызывается | Найденные helper cases не объясняют отказ этого маршрута |
| Python I2V, положительные целые seconds | вызывается; T≥7 | не вызывается | Пустого reference из-за T нет; нужны конечные ненулевые source std |
| Python I2V, time_length=0 | вызывается; T=1 | не вызывается | После успешного sampling есть tuple/reshape conflict |
| ComfyUI I2V node | не вызывается | не в этом sampler path | У него отдельный generate → repin → return |
| Python I2I / T2I | не вызывается | image_vae=True вместо video branch | Проверяем image posterior API отдельно |
| Прямой TI2I helper, image + channel + image_vae=False | не вызывается | вызывается | Можно достичь проверяемой VAE type incompatibility |

Карту проверяем AST вызовами, явными keywords и официальным example. Это проверка маршрутов в данной ревизии, не доказательство отсутствия любых других ошибок. T2V может зависеть от других проблем независимо от этих helpers. ComfyUI поддерживает короткие length по собственному интерфейсу, но его результаты нельзя автоматически приписывать Python I2V wrapper.

Источники: [I2V prepare/first image](https://github.com/kandinskylab/kandinsky-5/blob/3a74261b957df0446c1969c5cedf8bbb07969ac2/kandinsky/i2v_pipeline.py#L37), [I2I image_vae=True](https://github.com/kandinskylab/kandinsky-5/blob/3a74261b957df0446c1969c5cedf8bbb07969ac2/kandinsky/i2i_pipeline.py#L195), [ComfyUI sampler](https://github.com/kandinskylab/kandinsky-5/blob/3a74261b957df0446c1969c5cedf8bbb07969ac2/comfyui/nodes_kandinsky.py#L364).
'''), code(r'''
def named_calls(node):
    return [(n.func.id if isinstance(n.func,ast.Name) else n.func.attr)
            for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))]

normalizer_callers = [f.name for f in generation_ast.body if isinstance(f,ast.FunctionDef)
                      and 'normalize_first_frame' in named_calls(f)]
encoder_callers = [f.name for f in generation_ast.body if isinstance(f,ast.FunctionDef)
                   and 'encode_video' in named_calls(f)]
assert normalizer_callers==['generate_sample_i2v'] and encoder_callers==['generate_sample_ti2i']
print('generation_utils callers:', normalizer_callers, encoder_callers)
i2i_ast = ast.parse((SOURCE/'kandinsky/i2i_pipeline.py').read_text())
image_calls = [n for n in ast.walk(i2i_ast) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Name) and n.func.id=='generate_sample_ti2i']
assert len(image_calls)==1 and any(k.arg=='image_vae' and isinstance(k.value,ast.Constant)
                                  and k.value.value is True for k in image_calls[0].keywords)
comfy_ast = ast.parse((SOURCE/'comfyui/nodes_kandinsky.py').read_text())
assert 'normalize_first_frame' not in named_calls(comfy_ast)
assert 'generate' in named_calls(comfy_ast)
example = json.loads((SOURCE/'examples/inference_examples_i2v.ipynb').read_text())
example_seconds = []
for cell in example['cells']:
    if cell['cell_type']!='code': continue
    src = ''.join(cell['source'])
    for node in ast.walk(ast.parse(src)):
        if isinstance(node,ast.keyword) and node.arg=='time_length' and isinstance(node.value,ast.Constant):
            example_seconds.append(node.value.value)
assert example_seconds and set(example_seconds)=={5}
print('Official I2V example seconds:', example_seconds)
source_excerpt('kandinsky/i2v_pipeline.py', 'lat_image = vae.encode', 3)
'''), md(r'''
## 7. reference_frames: подозрительный min не обязательно off-by-one bug

Для положительного r и T≥5 slice `[4:4+min(r,T-1)]` фактически содержит `min(r,T-4)` кадров: конец slice автоматически ограничивается длиной Tensor. Поэтому замена `T-1` на `T-4` в min сделает намерение яснее, но **не изменит выбранные кадры на этих допустимых входах**. Нельзя объявлять этот min самостоятельной доказанной ошибкой.

Другой вопрос — validation. `reference_frames=0` создаёт пустую группу, отрицательные значения интерпретируются Python slicing неожиданным образом. Например, r=−5 даёт конечный индекс −1 и может выбрать почти все оставшиеся кадры. Такие значения не означают корректное «количество reference frames»; функция их не отвергает.
'''), code(r'''
for T in [5,6,9,31]:
    for r in [1,5,100]:
        actual_indices = list(range(T))[4:4+min(r,T-1)]
        clearer_indices = list(range(T))[4:4+min(r,T-4)]
        assert actual_indices==clearer_indices and len(actual_indices)==min(r,T-4)
rows = []
for T in [9,31]:
    for r in [0,-1,-5,1,5]:
        indices = list(range(T))[4:4+min(r,T-1)]
        rows.append((T,r,len(indices),indices[:5]))
table(['T','reference_frames','actual count','first indices'],rows)
assert len(list(range(31))[4:4+min(-5,30)])==26
'''), md(r'''
## 8. Нулевая variance, singleton statistics и clipping

Непустого reference недостаточно. Постоянный source имеет std=0, а `(source-mean)/std` становится `0/0`. Нет epsilon или отдельной политики. Когда source содержит всего одно HWC значение, std с correction=1 также не определена. Одно **значение** reference — та же проблема; один **кадр** с множеством значений — другой, допустимый случай.

Постоянный reference сам по себе не обязан давать NaN: его std=0 определена при двух и более элементах. Ограничения могут сохранить часть source variance или свести output к константе. `clump_values=True` не лечит NaN source. Если reference constant, clamp к min=max делает первые кадры constant — ожидаемый результат такой политики, а не новая арифметическая ошибка.

Эти синтетические cases исследуют устойчивость; нет данных о том, как часто такие source statistics встречаются с pretrained-весами.
'''), code(r'''
special = []
varying = torch.linspace(-0.4,0.8,16).reshape(2,2,2,2)
for dtype in [torch.float32,torch.bfloat16]:
    ref = torch.linspace(-0.5,0.5,24).reshape(3,2,2,2).to(dtype)
    cases = [
        ('constant source', torch.ones(2,2,2,2).to(dtype),ref),
        ('constant reference',varying.to(dtype),torch.full_like(ref,0.12)),
        ('both constant',torch.ones(2,2,2,2).to(dtype),torch.ones_like(ref)),
        ('source singleton per frame',torch.tensor([0.,1.]).reshape(2,1,1,1).to(dtype),ref),
        ('reference singleton',varying.to(dtype),torch.ones(1,1,1,1).to(dtype)),
        ('small source std / constant ref',(varying*0.1).to(dtype),torch.zeros_like(ref)),
        ('BF16 quantization collapse',(64+varying*0.01).to(dtype),ref),
    ]
    for label,src,r in cases: special.append(record_call(label,adaptive,src,r))
    for batch in [1,2,5]:
        wrong = torch.arange(batch*2*9*2*2,dtype=torch.float32).reshape(batch,2,9,2,2).to(dtype)
        special.append(record_call(f'wrong BCTHW B={batch}',normalize,wrong))
    for clump in [False,True]:
        x = torch.arange(9*8,dtype=torch.float32).reshape(9,2,2,2)/17; x[4:]=0.123
        special.append(record_call('normalize constant reference',normalize,x.to(dtype),clump_values=clump))
assert len(special)==24
table(['case','dtype','finite','NaN','return type'],
      [(r['label'],r['dtype'],r.get('finite'),r.get('nan_count'),r.get('return_type'))
       for r in special if not r['label'].startswith('wrong')])
assert all(r['finite'] for r in special if r['label']=='constant reference')
assert all(r['nan_count']>0 for r in special if r['label']=='constant source')
constant_prefix = frames.clone(); constant_prefix[:4]=1.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for clump in [False,True]:
        bad = normalize(constant_prefix,clump_values=clump)
        assert bad[:4].isnan().all() and torch.equal(bad[4:],constant_prefix[4:])
print('clump_values does not repair division by zero.')
'''), md(r'''
## 9. BF16: ограниченная точность может превратить variance в ноль

Исходный helper не переводит статистики и affine arithmetic явно в FP32. Но dtype возвращённых statistics не доказывает dtype внутреннего accumulator kernel — такого утверждения не делаем.

Сначала сравним два обычных конечных результата. Затем специально возьмём большой offset и маленький spread: FP32 различает числа, а после BF16 cast они могут стать одинаковыми. Приведение уже округлённого input обратно в FP32 не вернёт потерянные различия. Epsilon предотвращает деление на ноль, но требует явного решения, что делать с таким кадром.
'''), code(r'''
source32 = torch.tensor([-0.02,0.01,0.05,0.09,0.14,0.21,0.28,0.33,
                          0.9,1.1,1.2,1.3,1.5,1.8,2.1,2.5]).reshape(2,2,2,2)
ref32 = torch.linspace(-1,1,24).reshape(3,2,2,2)
result32 = adaptive(source32,ref32)
resultbf = adaptive(source32.bfloat16(),ref32.bfloat16())
precision_difference = float((result32-resultbf.float()).abs().max())
assert torch.isfinite(result32).all() and torch.isfinite(resultbf).all()
collapsed32 = 64+varying*0.01
collapsedbf = collapsed32.bfloat16()
assert collapsed32.std()>0 and collapsedbf.unique().numel()==1
assert torch.isfinite(adaptive(collapsed32,ref32)).all()
assert adaptive(collapsedbf,ref32.bfloat16()).isnan().all()
print('Ordinary FP32/BF16 maximum difference:',precision_difference)
print('Constructed BF16 unique values:',collapsedbf.unique().float().tolist())
fig, ax = plt.subplots(figsize=(7,3))
ax.plot((collapsed32-64).flatten().numpy(),'o-',label='FP32 input − 64')
ax.plot((collapsedbf.float()-64).flatten().numpy(),'s--',label='BF16 input − 64')
ax.set(xlabel='Элемент source',ylabel='Малое отклонение',title='Сконструированный пример потери spread')
ax.legend(); plt.tight_layout(); plt.show()
'''), md(r'''
## 10. Layout misuse: finite не означает правильный результат

У helper нет rank/layout validation. При передаче BCTHW он принимает B за число кадров. Для B=1 возвращает tuple, для B=2 получает пустой reference, а при B=5 может вернуть finite output, нормализуя batch и оставляя одну ось не той, что требовалась.

Это недостаточная защита helper от неправильного входа. Настоящий исследованный caller передаёт THWC; факт отсутствия guard **не доказывает**, что caller ошибается с layout.
'''), code(r'''
wrong_rows = [r for r in special if r['label'].startswith('wrong')]
table(['shape','dtype','return type','finite'],
      [(r['shape'],r['dtype'],r.get('return_type'),r.get('finite')) for r in wrong_rows])
assert all(r['return_type']=='tuple' for r in wrong_rows if 'B=1' in r['label'])
assert all(not r['finite'] for r in wrong_rows if 'B=2' in r['label'])
assert all(r['finite'] for r in wrong_rows if 'B=5' in r['label'])
print('Actual caller core layout is THWC, not these 5D inputs.')
'''), md(r'''
## 11. encode_video: проверяем настоящий VAE, а не только похожую заглушку

Video branch делает `data=vae.encode(data)[0]`, затем `data*=scaling_factor`. Для такой операции `[0]` должен быть Tensor. В этой ревизии Hunyuan `encode` возвращает `AutoencoderKLOutput(latent_dist=posterior)`; индекс `[0]` — **DiagonalGaussianDistribution**. С `return_dict=False` возвращается `(posterior,)`, поэтому этот флаг сам по себе проблему не исправляет.

Создадим настоящий класс из upstream `vae.py`, уменьшая widths и число latent channels. Это его архитектура и публичный encode API со случайными весами. Вызов `opt_tiling=False` отключает автоматический CUDA memory planner для CPU; структура encoder и тип posterior остаются исходными. Таким способом мы проверяем API, а не качество или полный GPU inference.
'''), code(r'''
import importlib.util
vae_path = SOURCE/'kandinsky/models/vae.py'
spec = importlib.util.spec_from_file_location('_k5_normalization_audit_vae',vae_path)
vae_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(vae_module)
small_vae = vae_module.AutoencoderKLHunyuanVideo(
    latent_channels=4,block_out_channels=(8,16,16,16),layers_per_block=1,
    norm_num_groups=4,mid_block_add_attention=False).eval().requires_grad_(False)
video = torch.randn(1,3,9,16,16,generator=rng)
with torch.no_grad():
    encoded = small_vae.encode(video,opt_tiling=False)
    tuple_encoded = small_vae.encode(video,opt_tiling=False,return_dict=False)
assert isinstance(encoded[0],vae_module.DiagonalGaussianDistribution)
assert isinstance(tuple_encoded[0],vae_module.DiagonalGaussianDistribution)
assert not isinstance(encoded[0],torch.Tensor)
assert encoded.latent_dist.mode().shape==(1,4,3,2,2)
print('encode output:',type(encoded).__name__,'[0]:',type(encoded[0]).__name__)
print('return_dict=False:',type(tuple_encoded).__name__,'[0]:',type(tuple_encoded[0]).__name__)
print('Latent Tensor exists only after sample()/mode():',tuple(encoded.latent_dist.mode().shape))
'''), md(r'''
### 11.1. Точная ошибка video helper и два независимых препятствия CPU

Публичный default `opt_tiling=True` использует CUDA memory planner. Его отказ на CPU не является доказательством posterior mismatch. Поэтому разделим эти вопросы: proxy ниже меняет только аргумент planner (`opt_tiling=False`) и возвращает **настоящий** upstream encode result без конвертации. Исходный `encode_video` остаётся неизменным и после encode падает на умножении posterior.

На CUDA успешный planner не превратит posterior в Tensor; type mismatch остаётся. Однако мы не проверяли CUDA запуск. Реальные callers обходят этот helper в I2V или используют image_vae=True в I2I. Поэтому корректное заключение — несовместимость **video-ветки данного helper с данным codec API**, а не «весь Kandinsky VAE не работает».
'''), code(r'''
class NoAutomaticTiling:
    def __init__(self,vae,return_dict=True):
        self.vae=vae; self.config=vae.config; self.return_dict=return_dict
    def encode(self,pixels):
        return self.vae.encode(pixels,opt_tiling=False,return_dict=self.return_dict)

codec_errors=[]
with torch.no_grad():
    for return_dict in [True,False]:
        try:
            encode_video(video,NoAutomaticTiling(small_vae,return_dict),image_vae=False)
        except TypeError as error:
            codec_errors.append(dict(return_dict=return_dict,type=type(error).__name__,message=str(error)))
            print('Expected TypeError:',str(error))
        else:
            raise AssertionError('Настоящий posterior неожиданно стал Tensor')
assert len(codec_errors)==2 and all('DiagonalGaussianDistribution' in r['message'] for r in codec_errors)
'''), md(r'''
### 11.2. Явная posterior policy устраняет type mismatch

У posterior два разных решения: `mode()` возвращает mean, `sample()` добавляет случайность. Выбор — часть договора с downstream моделью; нельзя считать их взаимозаменяемыми по качеству conditioning. Истинный I2V first-image caller выбирает sample.

Ниже учебный adapter сначала получает **несмасштабированный Tensor**, клонирует mode (он может быть view posterior mean), а затем даёт исходному helper выполнить его собственные scaling/permute. Это локальная демонстрация интерфейса, не незаметная правка upstream и не предложение заменить production sampling на mode.
'''), code(r'''
class ExplicitPosteriorMode:
    def __init__(self,vae): self.vae=vae; self.config=vae.config
    def encode(self,pixels):
        posterior=self.vae.encode(pixels,opt_tiling=False).latent_dist
        return (posterior.mode().clone(),)

with torch.no_grad():
    fixed_latents = encode_video(video,ExplicitPosteriorMode(small_vae),image_vae=False)
    expected_latents = (small_vae.encode(video,opt_tiling=False).latent_dist.mode()*small_vae.config.scaling_factor).permute(0,2,3,4,1)
assert fixed_latents.shape==(1,3,2,2,4) and torch.isfinite(fixed_latents).all()
assert torch.allclose(fixed_latents,expected_latents,atol=1e-6)
print('Explicit mode contract: BCTHW RGB → BTHWC scaled latent:',tuple(fixed_latents.shape))
'''), md(r'''
### 11.3. Положительный контроль image branch

Image branch уже вызывает `.latent_dist.sample()` и удаляет/возвращает временную ось T=1. Проверим её на настоящем небольшом `diffusers.AutoencoderKL`, то есть том же классе API, который использует image factory. Это не FLUX pretrained weights и не точная FLUX конфигурация; spatial compression toy-модели здесь равна 1.

Убедимся, что проблема не в любом posterior и не в `permute` вообще. При корректном преобразовании posterior → Tensor image branch исполняется успешно. `sample()` стохастичен, поэтому сравним два запуска с одинаковым seed в локальном RNG контексте.
'''), code(r'''
from diffusers import AutoencoderKL
image_vae = AutoencoderKL(
    in_channels=3,out_channels=3,down_block_types=('DownEncoderBlock2D',),
    up_block_types=('UpDecoderBlock2D',),block_out_channels=(8,),latent_channels=4,
    norm_num_groups=4,layers_per_block=1,mid_block_add_attention=False).eval().requires_grad_(False)
with torch.random.fork_rng(devices=[]),torch.no_grad():
    torch.manual_seed(123)
    image_latent1=encode_video(video[:,:,:1],image_vae,image_vae=True)
    torch.manual_seed(123)
    image_latent2=encode_video(video[:,:,:1],image_vae,image_vae=True)
assert image_latent1.shape==(1,1,16,16,4) and torch.isfinite(image_latent1).all()
assert torch.equal(image_latent1,image_latent2)
print('Actual image API branch PASS:',tuple(image_latent1.shape))
del small_vae,image_vae,encoded,tuple_encoded; release()
'''), md(r'''
## 12. Учебная защитная версия: сначала выбираем политику

Не существует единственного «правильного» результата для отсутствующего reference или постоянного source. Один вариант — отвергать такие входы; другой — пропускать correction. Для сравнения ниже выбираем **консервативную учебную политику**:

1. При неверном rank, пустых измерениях, не-float, non-finite input или неположительном reference_frames — явный ValueError.
2. Всегда возвращать clone Tensor, включая T=1. При T≤4 пропускать correction, потому что будущего reference после первых четырёх нет.
3. При недостаточном sample count пропускать correction. У остальных кадров считать statistics в FP32 (FP64 оставляем FP64).
4. Кадры с std≤eps оставлять неизменными. Не пытаться восстановить потерянный BF16 spread. Остальные нормализовать как upstream; optional clipping применять только к ним.

Это **новое решение для неопределённых случаев**, а не реконструкция намерения авторов. Для хорошо обусловленных FP32/FP64 входов должны сохранить upstream формулу. В low precision промежуточная FP32 арифметика может изменить результат округления; это намеренное отличие. Full-scale quality/latency требует отдельной проверки и здесь не измеряется.
'''), code(r'''
def normalize_first_frame_safe(latents,reference_frames=5,clump_values=False,eps=1e-6):
    if not isinstance(latents,torch.Tensor) or latents.ndim!=4:
        raise ValueError('Expected a THWC Tensor')
    if not latents.is_floating_point() or any(s==0 for s in latents.shape):
        raise ValueError('Expected non-empty floating THWC dimensions')
    if type(reference_frames) is not int or reference_frames<1:
        raise ValueError('reference_frames must be a positive integer')
    if not np.isfinite(eps) or eps<=0 or not torch.isfinite(latents).all():
        raise ValueError('Expected positive finite eps and finite input')
    out=latents.clone()
    if len(out)<=4 or out[0].numel()<2:
        return out  # объявленная политика, не поведение upstream
    work_dtype=torch.float64 if out.dtype==torch.float64 else torch.float32
    source=out[:4].to(work_dtype)
    reference=out[4:4+min(reference_frames,len(out)-4)].to(work_dtype)
    mu=source.mean((1,2,3),keepdim=True); sd=source.std((1,2,3),keepdim=True)
    mu_target=reference.mean().clamp(mu-0.05,mu+0.1)
    sd_target=reference.std().clamp(sd-0.1,sd+0.25)
    valid=sd>eps
    normalized=(source-mu)/sd.clamp_min(eps)*sd_target+mu_target
    if clump_values:
        normalized=normalized.clamp(reference.min(),reference.max())
    out[:4]=torch.where(valid,normalized,source).to(out.dtype)
    return out
'''), md(r'''
### 12.1. Проверяем свойства новой политики

Для нормальных FP32 входов результаты должны совпадать с upstream; для коротких последовательностей и constants — оставаться finite и сохранять вход согласно нашей политике. Проверим отсутствие изменения исходника и alias-return. Затем неправильные входы должны дать понятный отказ.

Эта защита рассчитана на обычный численный диапазон. Она не доказывает невозможность overflow при любых конечных экстремальных числах или любых устройствах. Если production требует такой гарантии, необходимы дополнительные bounds/validation и согласование policy с обучением модели.
'''), code(r'''
safe_checks=[]
for dtype in [torch.float32,torch.float64]:
    for T in [1,2,3,4,5,9,31]:
        x=torch.randn(T,3,4,2,generator=rng,dtype=dtype); original=x.clone()
        for clump in [False,True]:
            fixed=normalize_first_frame_safe(x,clump_values=clump)
            assert fixed is not x and fixed.dtype==dtype and torch.equal(x,original) and torch.isfinite(fixed).all()
            if T<=4: assert torch.equal(fixed,x)
            else: assert torch.equal(fixed,normalize(x,clump_values=clump))
        safe_checks.append(dict(dtype=str(dtype),T=T,matched_upstream=T>=5,finite=True))
for dtype in [torch.float16,torch.bfloat16]:
    varying_input=torch.randn(9,2,2,2,generator=rng).to(dtype)
    finite_result=normalize_first_frame_safe(varying_input)
    assert finite_result.dtype==dtype and torch.isfinite(finite_result).all()
    assert torch.equal(finite_result[4:],varying_input[4:])
    x=torch.randn(9,2,2,2,generator=rng).to(dtype); x[:4]=64.
    fixed=normalize_first_frame_safe(x,clump_values=True)
    assert fixed.dtype==dtype and torch.isfinite(fixed).all() and torch.equal(fixed,x)
near_eps=torch.randn(9,2,2,2,generator=rng)
near_eps[:4]=torch.linspace(-1,1,8).reshape(1,2,2,2)*torch.tensor([1.,0.,1e-7,2e-6])[:,None,None,None]
near_result=normalize_first_frame_safe(near_eps)
assert torch.isfinite(near_result).all() and torch.equal(near_result[1:3],near_eps[1:3])
assert torch.equal(near_result[4:],near_eps[4:])
singleton=torch.arange(9,dtype=torch.float32).reshape(9,1,1,1)
assert torch.equal(normalize_first_frame_safe(singleton),singleton)
invalid_inputs=[(torch.empty(0,2,2,2),{}),(torch.zeros(1,2,9,2,2),{}),
                (torch.zeros(9,2,2,2),{'reference_frames':0}),
                (torch.full((9,2,2,2),float('nan')),{}),
                (torch.full((9,2,2,2),float('inf')),{}),(torch.ones(9,2,2,2,dtype=torch.int64),{})]
for x,kwargs in invalid_inputs:
    try: normalize_first_frame_safe(x,**kwargs)
    except ValueError: pass
    else: raise AssertionError('Invalid input was not rejected')
table(['dtype','T','finite','same as upstream where defined'],
      [(r['dtype'],r['T'],r['finite'],r['matched_upstream']) for r in safe_checks])
'''), md(r'''
## 13. Что подтверждено и что нужно исправить в прежней формулировке

| Утверждение | Вердикт и граница |
|---|---|
| «Normalizer меняет первые четыре латентных кадра» | Подтверждено; число четыре само по себе не доказанная ошибка |
| «Для коротких последовательностей результат некорректен» | Уточнить: T≤1 — tuple, T=2…4 — empty reference NaN/exception; T=5 с нормальными HWC statistics работает |
| «Обычный I2V 5s страдает от пустого reference» | Не подтверждено: T=31, reference непустой; full pretrained в этом аудите не запускался |
| «time_length=0 доходит до tuple/reshape conflict» | Подтверждена несовместимость downstream после успешных предыдущих стадий; не доказана поддержка полного image-инференса этой моделью |
| «После I2V repin первый latent изменяется» | Факт normalization; объявить ошибкой fidelity нельзя без требования точного равенства и quality comparison |
| «Video encode helper несовместим с Hunyuan output» | Подтверждено на настоящем encode API; return_dict=False не лечит тип |
| «Из-за этого обычный I2V/I2I не работает» | Такой вывод неверен: I2V получает sample напрямую, I2I выбирает image branch |
| «Непустого reference достаточно для finite result» | Неверно: нужны конечные statistics, достаточный sample count и ненулевая representable source std |

Ранее формулировка об «ошибках» была слишком общей. Контрпримеры helper-level остаются действительными; теперь явно отделяем их от обычных routes, неподтверждённого намерения и оценки качества модели.

**Задания:** предложите strict reject вместо skip и объясните tradeoff; исследуйте downstream image-mode contract; сравните preservation первого latent до/после normalization, не смешивая это с RGB identity; при будущем изменении upstream повторите main/blob checks и матрицу. Не отправляйте issue с утверждением «5s inference broken» на основании только короткого helper input.
'''), code(r'''
from labkit.common import save_json
report=dict(
    upstream_commit=REVISION, main_checked_at_utc=manifest['checked_at_utc'],
    verified_files=len(manifest['files']), torch_version=torch.__version__, device='cpu',
    matrix=matrix, special=special, wrapper_cases=wrapper_rows,
    actual_vae_video_type_errors=codec_errors, image_branch_passed=True,
    fp32_bf16_max_difference=precision_difference, safe_policy_checks=safe_checks,
    normalizer_callers=normalizer_callers, encode_video_callers=encoder_callers,
    pretrained_generation_tested=False, cuda_generation_tested=False,
)
save_json(ROOT/'reports/normalization-audit.json',report)
print('Audit complete:',len(matrix),'matrix cases +',len(special),'special cases.')
print('Actual VAE API and image positive control PASS; full pretrained generation not tested.')
''')]

for i, cell in enumerate(cells):
    cell['id'] = f'audit-{i:03}'
metadata = json.loads(json.dumps(base['metadata']))
metadata['colab']['name'] = '13_normalization_and_vae_contract_audit.ipynb'
nb = dict(nbformat=4, nbformat_minor=5, metadata=metadata, cells=cells)
path = ROOT / 'notebooks/13_normalization_and_vae_contract_audit.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
print('Built', path.name, len(cells), 'cells')
