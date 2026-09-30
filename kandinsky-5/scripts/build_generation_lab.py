"""Build a complete, output-free lab for every pinned generation utility."""
from pathlib import Path
import json,ast,textwrap
ROOT=Path(__file__).resolve().parents[1]
base=json.loads((ROOT/'notebooks/00_start_and_training_map.ipynb').read_text())
def md(s): return {'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip()+'\n'}
def code(s):
    s=textwrap.dedent(s).strip()+'\n'; ast.parse(s)
    return {'cell_type':'code','metadata':{},'source':s,'execution_count':None,'outputs':[]}
cells=[md(r'''# 12. generation_utils.py: от prompt до RGB по функциям

Эта лабораторная исследует **все 10 функций** закреплённого generation_utils.py. Сначала читаем контракт, затем предсказываем результат, выполняем небольшой опыт и проверяем исходник. Цель — понимать порядок операций, формы тензоров, CFG, sampling, conditioning и offload.

Содержание: 1) get_sparse_params; 2) adaptive_mean_std_normalization; 3) normalize_first_frame; 4) get_velocity; 5) generate; 6) resize_video; 7) encode_video; 8) generate_sample; 9) generate_sample_ti2i; 10) generate_sample_i2v.

Перед началом полезны лабораторные 01, 02, 04, 06 и 07. Большие веса не нужны: small tensor probes отделяют контракты от качества генерации.'''),*base['cells'][1:5],md(r'''## Как именно запускаем исходные функции

Loader извлекает AST FunctionDef из сохранённого файла и выполняет **неизменённые тела функций** в отдельном namespace. Реальны PyTorch, torchvision resize/crop и fast_sta_nabla. Вызовы DiT/VAE/text encoder заменяются маленькими **записывающими заглушками**, чтобы видеть параметры и порядок вызовов.

Чистые helpers и generate исполняются с настоящим torch на CPU. Для трёх orchestrators нужен `cpu_cuda=True`: отдельный facade переводит hardcoded CUDA Generator в CPU Generator, CUDA autocast в nullcontext и empty_cache в записываемый no-op. BF16 noise сохраняется. Это явный **учебный adapter**, не доказательство поддержки CPU/MPS исходным pipeline. Глобальный torch и сторонние пакеты не изменяются. Quiet tqdm отключает только отображение progress bar.

Заглушки не содержат pretrained weights, реальных tokenizer или codec. Их скорость и изображения не измеряют Kandinsky latency/quality. Реальный CUDA запуск остаётся в лабораторной 10.

В Colab начните с новой среды выполнения и выполните setup сверху: открытая ранее сессия может хранить старую копию labkit без generation_lab.py. После обновления файла notebook сам по себе уже загруженный код не обновляется.'''),code('''from labkit.generation_lab import load_generation_functions, FakeDiT, FakeVAE, FakeTextEmbedder, make_generation_conf
from types import SimpleNamespace
import ast,inspect,warnings
original_generator=torch.Generator
gu=load_generation_functions()
adapter_events=[]
gu_cpu=load_generation_functions(cpu_cuda=True,adapter_log=adapter_events)
assert torch.Generator is original_generator
function_nodes=[n for n in ast.parse((SOURCE/'kandinsky/generation_utils.py').read_text()).body if isinstance(n,ast.FunctionDef)]
assert set(gu['_function_names'])=={n.name for n in function_nodes} and len(function_nodes)==10
for n in function_nodes:
    print(f'{n.name:34} lines {n.lineno}–{n.end_lineno}  {inspect.signature(gu[n.name])}')
print('Pinned source:',REVISION)
'''),md(r'''## Карта потока и соглашение о формах

```mermaid
flowchart LR
    P[Caption + negative caption] --> TE[Text embedder]
    S[Seed] --> N[Gaussian noise THWC]
    TE --> G[get_velocity]
    N --> LOOP[generate: shifted grid + Euler]
    SP[get_sparse_params] --> LOOP
    LOOP --> G
    G --> LOOP
    LOOP --> D[Unscale + BCTHW + VAE decode]
    D --> U[clamp + uint8 RGB]
    I[Input RGB image] --> R[resize_video]
    R --> E[encode_video for TI2I]
    E --> LOOP
    IF[Scaled image latent for I2V] --> LOOP
    LOOP --> NF[normalize_first_frame for I2V]
    NF --> D
```

| Обозначение | Форма | Значение |
|---|---|---|
| Codec RGB input | B,C,T,H,W | Channel-first pixels |
| Wrapper shape | B,T,h,w,Cz | Задание размеров noise в latent space |
| Core sampler img | T,h,w,Cz | Batch=1 в настоящих callers; channels-last |
| DiT input с visual_cond | T,h,w,2Cz+1 | current state, zero visual channels, mask |
| Decode input | B,Cz,T,h,w | После деления на VAE scaling_factor |
| T2V/I2V return | B,3,Tpx,Hpx,Wpx uint8 | RGB tensor; MP4 сохраняет внешний pipeline |
| T2I/TI2I image return | B,3,Hpx,Wpx uint8 | При image_vae=True time axis удаляется |

`shape` допускает B синтаксически, но callers этой ревизии используют B=1: flatten B*T, RoPE длины T и один caption не образуют рабочий batch video pipeline. Не выводите поддержку batch>1 из одной сигнатуры.''')]
cells += [md(r'''## 1. get_sparse_params: подготовка NABLA mask

Контракт: `conf`, `batch_embeds['visual']` формы **T,h,w,C**, device → dict sparse_params или None. Название batch_embeds не означает наличие batch axis.

1. Проверяется temporal patch size=1, даже если attention не NABLA.
2. Размеры latent делятся на patch sizes: получается grid DiT tokens.
3. Для NABLA grid делится на spatial blocks 8×8 token positions; fast_sta_nabla строит boolean adjacency соседних blocks.
4. Dictionary сообщает attention engine mask, fractal layout, visual_shape, threshold P и параметры локального окна. Другие attention types дают None.

Mask имеет форму 1×1×Nblocks×Nblocks, не Ntokens². `P` — параметр последующего NABLA selection, не число шагов Euler. Для выбранного patch2 H/W latent должны делиться на 16 при block attention; floor division в этой функции не проверяет корректность дальнейшего fractal layout. Здесь проверяем только подготовку mask, не запускаем sparse GPU kernel.

Предскажите число blocks при T=3, h=w=32 и patch=(1,2,2).'''),code('''source_excerpt('kandinsky/generation_utils.py','def get_sparse_params',28)
mask_conf=make_generation_conf(); mask_conf.model.dit_params.patch_size=(1,2,2)
mask_input=torch.zeros(3,32,32,2)
assert gu['get_sparse_params'](mask_conf,{'visual':mask_input},'cpu') is None
mask_conf.model.attention.type='nabla'
mask_conf.model.attention.wT=3; mask_conf.model.attention.wH=1; mask_conf.model.attention.wW=1
sparse=gu['get_sparse_params'](mask_conf,{'visual':mask_input},'cpu')
assert sparse['visual_shape']==(3,16,16)
assert sparse['sta_mask'].shape==(1,1,12,12)
assert sparse['sta_mask'][0,0].diagonal().all()
print({k:v for k,v in sparse.items() if k!='sta_mask'})
plt.figure(figsize=(4,4)); plt.imshow(sparse['sta_mask'][0,0],cmap='gray',vmin=0,vmax=1)
plt.xlabel('key block'); plt.ylabel('query block'); plt.title('3 time × 2 × 2 spatial blocks'); plt.show()
'''),md(r'''## 2. adaptive_mean_std_normalization: ограниченная поправка статистик

Source имеет форму T,H,W,C. Mean и std считаются **отдельно для каждого source frame**, по осям H/W/C. Reference mean/std считаются **по всему reference tensor**, включая время. Стандартное torch.std использует correction=1; важно согласовать это при проверке результата.

Пусть source frame имеет mean m и std s. Целевые статистики ограничены:

$$m^*=\mathrm{clip}(m_{ref},m-0.05,m+0.1),\qquad s^*=\mathrm{clip}(s_{ref},s-0.1,s+0.25).$$
$$y=(x-m)s^*/s+m^*.$$

Получаем ограниченное изменение контраста/смещения латента. Эта операция не обязана полностью приблизить source к reference. Constants 0.05/0.1/0.25 — эвристика конкретной реализации, не VAE prior или Flow Matching loss. Input не меняется inplace. При s=0 отсутствует epsilon, поэтому constant source даёт NaN.'''),code('''source_excerpt('kandinsky/generation_utils.py','def adaptive_mean_std_normalization',18)
norm_rng=torch.Generator().manual_seed(1202)
norm_source=torch.randn(2,4,4,2,generator=norm_rng)*torch.tensor([0.7,1.2])[:,None,None,None]
norm_source+=torch.tensor([-0.5,0.5])[:,None,None,None]
norm_reference=torch.randn(5,4,4,2,generator=norm_rng)*0.5+1.5
norm_before=norm_source.clone()
norm_output=gu['adaptive_mean_std_normalization'](norm_source,norm_reference)
source_m=norm_source.mean((1,2,3)); source_s=norm_source.std((1,2,3))
expected_m=torch.clamp(norm_reference.mean(),source_m-0.05,source_m+0.1)
expected_s=torch.clamp(norm_reference.std(),source_s-0.1,source_s+0.25)
assert torch.allclose(norm_output.mean((1,2,3)),expected_m,atol=1e-6)
assert torch.allclose(norm_output.std((1,2,3)),expected_s,atol=1e-6)
assert torch.equal(norm_source,norm_before)
print('source means/std:',source_m.tolist(),source_s.tolist())
print('global reference mean/std:',norm_reference.mean().item(),norm_reference.std().item())
print('output means/std:',expected_m.tolist(),expected_s.tolist())
constant_output=gu['adaptive_mean_std_normalization'](torch.ones(2,4,4,2),norm_reference)
assert not torch.isfinite(constant_output).all()
print('Constant source produces nonfinite values:',not torch.isfinite(constant_output).all().item())
'''),md(r'''## 3. normalize_first_frame: первые четыре latent frames

Функция clone-ит input, выбирает `samples[:4]` и reference slice начиная с index4 (по умолчанию до index9). Применяет предыдущую нормализацию; `clump_values=True` дополнительно ограничивает результат global min/max reference. Кадры с index4 и далее сохраняются.

Это postprocessing latent перед decoder в I2V; четыре latent frames не равны четырём RGB кадрам из-за temporal compression. Название singular first_frame скрывает изменение четырёх кадров.

**Граничные случаи исходника:** T=1 возвращает tuple `(latents, message)`, хотя обычная ветка возвращает Tensor. При T=2…4 reference пуст и возникают NaNs; с clump_values=True min/max пустого tensor вызывает ошибку. Для обычной ветки нужны T≥5, непустой reference и ненулевые std. Показываем это как исследование контракта, не исправляем vendored source.'''),code('''source_excerpt('kandinsky/generation_utils.py','def normalize_first_frame',23)
frames=torch.randn(8,4,4,2,generator=norm_rng)
frames[:4]+=0.4
frames_before=frames.clone()
frames_normal=gu['normalize_first_frame'](frames)
frames_clamped=gu['normalize_first_frame'](frames,clump_values=True)
assert torch.equal(frames,frames_before)
assert torch.equal(frames_normal[4:],frames[4:])
assert frames_clamped[:4].min()>=frames[4:].min() and frames_clamped[:4].max()<=frames[4:].max()
plt.plot(frames.mean((1,2,3)), 'o-', label='input')
plt.plot(frames_normal.mean((1,2,3)), 'x--',label='first 4 normalized')
plt.axvline(3.5,color='gray',linestyle=':'); plt.xlabel('latent frame index'); plt.ylabel('mean'); plt.legend(); plt.show()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for length in (1,2,3,4,5,6):
        candidate=torch.randn(length,4,4,2,generator=norm_rng)
        result=gu['normalize_first_frame'](candidate)
        if length==1:
            assert isinstance(result,tuple) and result[0] is candidate
            print('T=1: tuple, original tensor returned')
        else:
            finite=bool(torch.isfinite(result).all()); assert finite==(length>=5)
            print('T=',length,'finite:',finite)
    try: gu['normalize_first_frame'](torch.randn(3,4,4,2),clump_values=True)
    except RuntimeError as error: print('Expected empty-reference clamp failure:',type(error).__name__)
'''),md(r'''## 4. get_velocity: один DiT forward или CFG из двух

Эта функция не двигает latent: она получает velocity нужного размера. `x` — текущий model input, `t` — обычно tensor shape[1] из диапазона [0,1]. В DiT поступает **t×1000**, text token states, pooled embedding, RoPE positions, mask и `conf.metrics.scale_factor`. Последний параметр масштабирует RoPE по трём осям: time, height, width. В выбранном config временной множитель равен 1. Это масштаб позиционных координат; scheduler_scale отдельно меняет временную сетку интегратора.

Сначала всегда считается conditional ветка. Если |guidance_weight−1|>1e−6, дополнительно считается null/negative ветка:

$$v=v_{null}+w(v_{cond}-v_{null}).$$

w=1 означает conditional generation с одним DiT forward, а не отсутствие текста. w=0 возвращает null prediction, но исходник всё равно сначала вычисляет conditional — итого два forward. `@torch.no_grad` отключает autograd: training loss находится вне этого inference helper.

Записывающий ProbeVelocity ниже возвращает известные числа для двух conditioning. Это позволяет проверить формулу, время и masks точно.'''),code('''source_excerpt('kandinsky/generation_utils.py','def get_velocity',42)
class ProbeVelocity:
    def __init__(self,visual_cond=False): self.visual_cond=visual_cond; self.calls=[]
    def __call__(self,x,text,pooled,time,visual_rope,text_rope,**kwargs):
        self.calls.append({'x':x.clone(),'time':time.clone(),'mask':kwargs.get('attention_mask'),
                           'scale_factor':kwargs.get('scale_factor'),'text_rope':text_rope.clone()})
        return x[...,:2]*0 + pooled.mean()  # без no_grad эта формула сохраняла бы связь с x
cond={'text_embeds':torch.ones(1,3,4),'pooled_embed':torch.tensor([[2.]])}
null={'text_embeds':torch.zeros(1,2,4),'pooled_embed':torch.tensor([[-1.]])}
positions=[torch.arange(2)]*3; cond_pos=torch.arange(3); null_pos=torch.arange(2)
cond_mask=torch.tensor([[True,True,False]]); null_mask=torch.tensor([[True,False]])
core_conf=make_generation_conf()
velocity_input=torch.randn(2,2,2,2,requires_grad=True)
for weight,expected,ncalls in [(0.,-1.,2),(1.,2.,1),(2.,5.,2)]:
    probe=ProbeVelocity()
    output=gu['get_velocity'](probe,velocity_input,torch.tensor([0.5]),cond,null,
        positions,cond_pos,null_pos,weight,core_conf,attention_mask=cond_mask,null_attention_mask=null_mask)
    assert torch.allclose(output,torch.full_like(output,expected)) and not output.requires_grad
    assert len(probe.calls)==ncalls and probe.calls[0]['time'].item()==500
    assert probe.calls[0]['mask'] is cond_mask
    if ncalls==2: assert probe.calls[1]['mask'] is null_mask
    print('guidance:',weight,'velocity:',output.mean().item(),'DiT forwards:',len(probe.calls))
'''),md(r'''## 5. generate: shifted grid, Euler и conditioning

Core sampler принимает уже созданный noise img. Он не генерирует случайные числа и не вызывает VAE или text encoder.

1. get_sparse_params строит static mask до возможного tensor-parallel split.
2. Grid t=1→0 сдвигается $f_s(t)=st/(1+(s-1)t)$.
3. На каждом шаге собирается model input, затем get_velocity.
4. Обновляются только первые C_out каналов: `img[..., :C_out] += delta_t * velocity`.
5. Возвращаются первые C_out channels как view изменённого img.

Delta_t отрицателен. При постоянной velocity весь update равен −velocity независимо от shift: sum(delta_t)=−1. `seed` здесь не используется; seed влияет на wrapper noise. `progress` также не используется для выбора tqdm. Исходник не проверяет num_steps>0 и scheduler_scale>0.

### 5.1. Проверяем время, NFE, мутацию и лишние каналы

Extra editing channels остаются неизменными в памяти, но не попадают в return. Для w=1,N=16 получается16 DiT forwards; для обычного CFG с w≠1 —32. Это подсчёт вызовов, не замер latency.'''),code('''source_excerpt('kandinsky/generation_utils.py','def generate(',60)
def run_core(state,model,steps=4,scale=1.,first=None,seed=1,tp_mesh=None):
    rope=[torch.arange(state.shape[0]),torch.arange(state.shape[1]),torch.arange(state.shape[2])]
    return gu['generate'](model,'cpu',state,steps,cond,null,rope,cond_pos,null_pos,1.,scale,first,
                          core_conf,progress=False,seed=seed,tp_mesh=tp_mesh,
                          attention_mask=cond_mask,null_attention_mask=null_mask)
initial=torch.randn(6,2,2,2,generator=torch.Generator().manual_seed(1205))
fig,axes=plt.subplots(1,2,figsize=(10,3))
for scale in (1.,5.):
    state=initial.clone(); probe=ProbeVelocity(); result=run_core(state,probe,scale=scale)
    assert torch.allclose(result,initial-2.,atol=1e-6)
    assert result.data_ptr()==state.data_ptr() and not torch.equal(state,initial)
    times=torch.tensor([c['time'].item()/1000 for c in probe.calls]+[0.])
    assert torch.allclose(times,shifted_schedule(4,scale))
    axes[0].plot(times,'o-',label=f'scale={scale:g}')
    axes[1].plot(torch.diff(times),'o-',label=f'scale={scale:g}')
axes[0].set(xlabel='grid index',ylabel='t'); axes[1].set(xlabel='step',ylabel='negative delta_t')
for ax in axes: ax.legend()
plt.tight_layout(); plt.show()
assert torch.equal(run_core(initial.clone(),ProbeVelocity(),seed=1),run_core(initial.clone(),ProbeVelocity(),seed=999))
extra=torch.randn(6,2,2,3)
editing_state=torch.cat([initial,extra],-1); editing_before=editing_state.clone()
trimmed=run_core(editing_state,ProbeVelocity())
assert trimmed.shape[-1]==2 and torch.equal(editing_state[...,2:],editing_before[...,2:])
try: run_core(initial.clone(),ProbeVelocity(),steps=0)
except UnboundLocalError: print('Expected num_steps=0 failure: pred_velocity is never assigned')
'''),md(r'''### 5.2. I2V mask: условие попадает в current state

При model.visual_cond=True input имеет [img, visual_cond, mask]. В этой ревизии visual_cond создаётся нулевым и остаётся нулевым. Если first_frames передан, **img[:1]** заменяется условием и mask[:1]=1. Поэтому первая часть input содержит condition, а отдельные visual_cond channels — нули.

Перед каждым forward первый frame закрепляется заново, но после forward Euler update снова его сдвигает. Wrapper I2V отдельно закрепляет frame после sampling, затем применяет normalize_first_frame. Проверим core поведение на простом поле.'''),code('''first_latent=torch.full((1,2,2,2),0.7)
conditioned_probe=ProbeVelocity(visual_cond=True)
conditioned_result=run_core(initial.clone(),conditioned_probe,first=first_latent)
for call in conditioned_probe.calls:
    model_input=call['x']
    assert model_input.shape[-1]==5
    assert torch.allclose(model_input[:1,...,:2],first_latent)
    assert torch.equal(model_input[...,2:4],torch.zeros_like(model_input[...,2:4]))
    assert model_input[:1,...,4:].eq(1).all() and model_input[1:,...,4:].eq(0).all()
assert not torch.allclose(conditioned_result[:1],first_latent)
print('First frame after final Euler update:',conditioned_result[:1].mean().item(),'condition:',first_latent.mean().item())
'''),md(r'''### 5.3. Tensor parallel: место разделения и ограничения проверки

Если tp_mesh задан и first_frames=None, img делится по **height dim1**, а не по time. get_sparse_params вызывается до split. generate_sample собирает shards через all_gather и cat(dim1), generate_sample_i2v такого gather не делает. Для реального distributed запуска нужны parallelized DiT, согласованные RoPE/masks и одинаковые shard shapes; простой torch.chunk не реализует collective communication.

Ниже выполняем исходную ветку split с имитацией rank1/world2 и constant ProbeVelocity. Проверяем только индекс разделения. Математика настоящего parallel DiT разобрана в лабораторной08.'''),code('''class MeshSliceProbe:
    def get_local_rank(self): return 1
    def size(self): return 2
shard_input=torch.randn(6,4,2,2)
shard_output=run_core(shard_input.clone(),ProbeVelocity(),tp_mesh={'tensor_parallel':MeshSliceProbe()})
assert shard_output.shape==(6,2,2,2)
assert torch.allclose(shard_output,shard_input[:,2:]-2.,atol=1e-6)
print('global THWC:',shard_input.shape,'rank1 THWC:',shard_output.shape)
'''),md(r'''### 5.4. Подключаем настоящий tiny DiT вместо ProbeVelocity

Лабораторная06 создаёт уменьшенные классы Kandinsky DiT. Подключим их к **настоящему generate**: теперь velocity зависит от x, времени и текстовых тензоров, а не задана константой. labkit.native использует отдельный CPU adapter для DiT (eager FP32 и SDPA); weights случайные. Проверяем интерфейс inference, формы и NFE, не качество видео.

Вход sampler T×4×4×16 превращается в T×4×4×33 через visual_cond; DiT возвращает16 velocity channels. Text/Qwen размеры уменьшены до32, pooled/CLIP до16.'''),code('''from labkit.native import tiny_dit
tiny_model=tiny_dit('cpu')
tiny_conf=make_generation_conf(patch_size=(1,2,2))
tiny_state=torch.randn(2,4,4,16)
tiny_cond={'text_embeds':torch.randn(1,6,32),'pooled_embed':torch.randn(1,16)}
tiny_null={key:torch.zeros_like(value) for key,value in tiny_cond.items()}
tiny_rope=[torch.arange(2),torch.arange(2),torch.arange(2)]
tiny_calls=[]
handle=tiny_model.register_forward_hook(lambda module,args,out: tiny_calls.append(tuple(out.shape)))
tiny_result=gu['generate'](tiny_model,'cpu',tiny_state.clone(),2,tiny_cond,tiny_null,
    tiny_rope,torch.arange(6),torch.arange(6),1.,5.,None,tiny_conf,
    attention_mask=torch.ones(1,6,dtype=torch.bool),null_attention_mask=torch.ones(1,6,dtype=torch.bool))
handle.remove()
assert tiny_result.shape==tiny_state.shape and torch.isfinite(tiny_result).all()
assert len(tiny_calls)==2 and all(shape==(2,4,4,16) for shape in tiny_calls)
print('Actual tiny DiT parameters:',sum(p.numel() for p in tiny_model.parameters()),'sampler forwards:',len(tiny_calls))
del tiny_model; release()
'''),md(r'''## 6. resize_video: cover resize и центральный crop

Функция сохраняет aspect ratio, увеличивая/уменьшая image до полного покрытия target rectangle, затем обрезает центр. Это **не letterbox**: поля не добавляются, часть краёв теряется.

Input torchvision tensor здесь имеет C,H,W или T,C,H,W; TI2I caller передаёт B=1,C,H,W. B,C,T,H,W codec layout не является допустимой автоматической video-обработкой этого helper. `visual_size=(target_height,target_width)`.

На примере H=12,W=20,target8×8 scale=min(12/8,20/8)=1.5; resize даёт8×13 (int округляет вниз), затем crop8×8. Нечётная разница ширин делает left/right crop различным на один пиксель. Геометрия одна для всех frames.'''),code('''source_excerpt('kandinsky/generation_utils.py','def resize_video',17)
y_coord=torch.linspace(0,1,12)[:,None].expand(12,20)
x_coord=torch.linspace(0,1,20)[None,:].expand(12,20)
coordinate_image=torch.stack([x_coord,y_coord,((x_coord>0.4)&(x_coord<0.6)).float()])
coordinate_video=torch.stack([coordinate_image,coordinate_image.flip(-1)])
resized=gu['resize_video'](coordinate_video,(8,8))
assert resized.shape==(2,3,8,8)
fig,axes=plt.subplots(1,2,figsize=(8,3))
axes[0].imshow(coordinate_image.permute(1,2,0)); axes[0].set_title('12×20 input')
axes[1].imshow(resized[0].permute(1,2,0)); axes[1].set_title('cover + center crop 8×8')
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()
'''),md(r'''## 7. encode_video: layout/scaling и два codec контракта

Input B,C,T,H,W → output B,T,h,w,Cz. Image branch требует T=1: удаляет time axis, берёт `.latent_dist.sample()` и возвращает time axis. Video branch берёт `vae.encode(data)[0]`, предполагая Tensor. Затем latent умножается inplace на scaling_factor и permute переносит channels в конец. Return может быть non-contiguous view.

**Проверенная несовместимость:** Hunyuan vae.py этой ревизии возвращает AutoencoderKLOutput с DiagonalGaussianDistribution. Поэтому `[0]` — posterior object, а не latent Tensor, и video branch падает на умножении. I2V caller обходит этот helper: отдельный get_first_frame_from_image использует `.latent_dist.sample()`, что соответствует posterior API.

Первый опыт ниже исследует ожидаемый Tensor-returning contract заглушки. Второй воспроизводит несовместимость на настоящих diffusers posterior/output classes без загрузки VAE weights. Явная формула исправленного caller — `posterior.sample()` или `posterior.mode()` до scaling; выбор должен соответствовать training/conditioning контракту. Vendored helper не меняем.'''),code('''source_excerpt('kandinsky/generation_utils.py','def encode_video',10)
codec_input=torch.randn(1,3,9,16,16)
for image_branch in (False,True):
    encode_log=[]; codec=FakeVAE(encode_log,image_vae=image_branch,channels=2,scaling_factor=0.5)
    pixels=codec_input[:,:,:1] if image_branch else codec_input
    encoded=gu['encode_video'](pixels,codec,image_vae=image_branch)
    assert encoded.shape==(1,1 if image_branch else 3,2,2,2)
    unscaled=encoded.permute(0,4,1,2,3)/codec.config.scaling_factor
    assert torch.allclose(encoded, (unscaled*codec.config.scaling_factor).permute(0,2,3,4,1))
    print('image_vae:',image_branch,'BCTHW:',pixels.shape,'→ BTHWC:',encoded.shape,'contiguous:',encoded.is_contiguous())
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
class PosteriorReturningVAE:
    config=SimpleNamespace(scaling_factor=0.5)
    def encode(self,pixels):
        return AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution(torch.zeros(1,4,3,2,2)))
try: gu['encode_video'](codec_input,PosteriorReturningVAE(),image_vae=False)
except TypeError as error: print('Expected actual posterior-contract failure:',error)
posterior=PosteriorReturningVAE().encode(codec_input).latent_dist
safe_mode=(posterior.mode()*0.5).permute(0,2,3,4,1)
assert safe_mode.shape==(1,3,2,2,2)
''')]
wrappers=json.loads((ROOT/'scripts/generation_wrappers.json').read_text())
for raw in wrappers:
    cells.append(md(raw['source']) if raw['cell_type']=='markdown' else code(raw['source']))
cells += [md(r'''## 11. Сводка для Lite Distill 5s и задания

Для 512×768, 5s настоящий wrapper получает shape=(1,31,64,96,16). Patch=(1,2,2) даёт31×32×48=47,616 visual tokens. При visual_cond=True input channels=16+16+1=33, output velocity channels=16. Hunyuan декодирует (31−1)×4+1=121 RGB frames. Для NABLA grid блоков31×4×6=744; mask744², не47,616².

Defaults helper и выбранная модель различаются: generate_sample25 steps/CFG5/scale1; Distill YAML16/1/scale5; T2V pipeline argument default scale10. Записывайте фактически переданные значения. Для w=1 core sampler делает один DiT forward за шаг; wrappers всё равно кодируют negative caption. NFE (number of function evaluations) — число вычислений поля velocity: здесь это число DiT forward. Оно не включает text/VAE расходы.

| Функция | Самая важная проверка |
|---|---|
| get_sparse_params | Spatial block dimensions и temporal patch=1 |
| adaptive_mean_std_normalization | std не нулевой, reference не пуст |
| normalize_first_frame | T≥5 для обычной ветки, Tensor return, первые4frames |
| get_velocity | t×1000 и CFG branch count |
| generate | Inplace img, negative dt, extra channels, seed вне sampler |
| resize_video | Cover + crop, input layout |
| encode_video | Posterior sampling/mode перед scaling для соответствующего codec |
| generate_sample | B=1, noise/text/RoPE/decode, CUDA/offload |
| generate_sample_ti2i | Source image channels+mask и 2D decode |
| generate_sample_i2v | Scaled latent input, persistent mode и normalization после repin |

1. Повторите mask опыт с wT=1 и3. Предскажите изменившиеся блоки.
2. Измените std source/reference и объясните, какие clamp ограничения сработали.
3. Предложите consistent Tensor-returning interface для normalize_first_frame и политику для T≤4. Отдельно обсудите epsilon/std=0. Не меняйте upstream молча.
4. При одинаковом noise сравните w=1 и2, steps4 и8. Разделите NFE, grid, velocity и decoder расходы.
5. В TI2I trace найдите, какие каналы меняет Euler и какие conditioning сохраняются.
6. Для I2V проверьте first frame до/после normalization. Объясните, почему repin перед normalization не гарантирует equality.
7. Составьте план реального CUDA замера: cold/warm, prompt expansion, text residency, peak VRAM/RAM, seed и sampler settings. Заглушки для performance не используйте.

Исходник закреплён в reference/kandinsky-5; все замечания относятся к этой ревизии. Полная генерация pretrained weights здесь не выполняется.'''),code('''coverage={n.name for n in function_nodes}
assert coverage==set(gu_cpu['_function_names'])
print('All source functions covered:',len(coverage))
print('Real upstream revision:',REVISION)
print('CPU facade events:',len(generation['_adapter_log']),'global torch.Generator unchanged:',torch.Generator is original_generator)
''')]
for i,c in enumerate(cells): c['id']=f'gen-{i:03}'
nb={'nbformat':4,'nbformat_minor':5,'metadata':base['metadata'],'cells':cells}
p=ROOT/'notebooks/12_generation_utils.ipynb'; p.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
print('Built',p.name,len(cells),'cells')
