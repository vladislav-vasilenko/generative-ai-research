"""Build a MIT-linked course from DiT primitives to K5/K6 CrossDiT and native π-Flow."""
from pathlib import Path
import textwrap
import nbformat as nbf
C=Path(__file__).resolve().parents[1]
cells=[];LAB=''
MIT={
 '00':'MIT Lecture 1, notes §2.1 pp.7–9; Lecture 2, notes §3 pp.14–24, Lab 2 Q2.1–Q2.3/Q3.1. Сначала выведите probability path и conditional target; затем измените направление времени.',
 '01':'MIT Lecture 4, notes §6.1 pp.41–46; Lab 3 Q3.1 (cells 38–39), Q3.2 (40–42), Q3.3 (43–44), Q3.4 (45–46), Q3.5 (47–48). Здесь Fourier time, patchifier, attention, modulation и output head.',
 '02':'Предпосылки MIT: Lab 3 Q3.3–Q3.5 и class-conditioning Parts 1–2. Text cross-attention / CrossDiT K5 — промышленное расширение; не выдаём его за отдельное MIT задание.',
 '03':'Предпосылки MIT: notes §6.1 / Lab 3 DiT. Асимметричные Audio/Video streams, fusion gates и раздельные time embeddings — дополнение K6, отсутствующее в исходном MNIST Lab 3.',
 '04':'MIT Lecture 3B, notes §5 pp.34–40; Lab 3 Q2.2 (27–31), Q2.3 (32–33), Sanity Check 2.4 (34–36). Token/pooled encoders и modality codecs — промышленное расширение class-conditioning.',
 '05':'MIT Lab 3 Part 4–5 объясняет latent space и frozen codec, но reference-tail layouts/role IDs/semantic RoPE не задаются исходным MNIST lab. Сначала [общий VAE маршрут](../../vae/README.md).',
 '06':'MIT Lab 1 Q1.1 (cells 9–13), notes §2.1 pp.7–9: Euler; Lab 2 Q2.3: поле; Lab 3 Part 2: CFG. Native denoise loop связывает эти три навыка.',
 '07':'MIT Lab 2 Q2.1–Q2.3 и notes §3.1–3.2 дают affine path и conditional velocity. DX reparameterization выводится из них; π-Flow policy, time warp и interpolation — отдельное исследовательское расширение.',
 '08':'MIT Lab 1 Euler + Lab 2 FM — предпосылки. Grid-head, MultimodalDXPolicy, короткий final segment и network NFE ≠ policy substeps изучаем по K6 source / π-Flow paper, не приписываем MIT.',
 '09':'MIT Lab 2 Q3.1, notes §3.3 pp.19–24: velocity regression. В π-ID target — frozen teacher на rollout student; это другой training distribution и отдельная статья.',
 '10':'MIT Lab 3 Part 5 / Q5.1 связывает frozen codec, latent field и sampling. ComfyUI observation и hardware measurements — следующий практический уровень, а не замена MIT вычислений.'}

def md(s):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()+'\n'))
def code(s):cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()+'\n'))
def start(slug,title,goal):
 global cells,LAB
 LAB=slug;cells=[]
 md(f'# {title}\n\n{goal}\n\n[Маршрут](../README.md) · [Точные задания MIT](../MIT_BRIDGE.md) · [Методы и source map](../SOURCE_MAP.md) · [Измерения](../reports/VALIDATION.md).\n\nКаждый опыт: **теория → формула → формы/единицы → прогноз → код → измерение → ограниченный вывод**. CPU по умолчанию; без pretrained weights, сети и ComfyUI jobs.')
 md('**Связь с MIT:** '+MIT[slug[:2]]+'\n\n[MIT course](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf).')
 code(f'''
 from pathlib import Path
 import sys,math,json,copy
 import numpy as np
 import torch
 from torch import nn
 import torch.nn.functional as F
 import matplotlib.pyplot as plt
 COURSE=next((p/'dit' for p in [Path.cwd(),*Path.cwd().parents] if (p/'dit/labkit/core.py').is_file()),None)
 assert COURSE is not None,'Clone the study repository first; see dit/README.md (Colab setup).'
 sys.path.insert(0,str(COURSE))
 from labkit.core import (native,tiny_model,tiny_inputs,activate_gates,fresh_forward,
   load_k5_block,manual_single,manual_fused,CountingLinearField,tiny_bundle,
   clone_bundle,scalar_bundle,vector_from_bundle,dummy_text,show_source,
   check_sources,save_metrics,figure_path)
 torch.set_num_threads(2);torch.manual_seed(42)
 LAB={slug!r};metrics={{}}
 print('Pinned source files checked:',check_sources())
 ''')
def finish(exercises,conclusion):
 md('## Измерение → вывод\n\n'+conclusion)
 md('## Самостоятельные опыты\n\n'+exercises+'\n\nПеред изменением запишите прогноз. Сохраняйте baseline, меняйте один фактор, отмечайте seed и tensor layout. Нулевая ошибка на oracle не равна pretrained quality.')
 code("result=save_metrics(LAB,metrics)\nprint(json.dumps(result,ensure_ascii=False,indent=2))")
 md('## Первичные источники\n\n[MIT course / notes](https://diffusion.csail.mit.edu/2026/index.html) · [DiT](https://arxiv.org/abs/2212.09748) · [Flow Matching](https://arxiv.org/abs/2210.02747) · [K5](https://arxiv.org/abs/2511.14993) · [K6 §3.2, §4.5](https://arxiv.org/html/2610.05608v1) · [π-Flow §3.1, §4 / Alg.1](https://arxiv.org/html/2510.14974v1) · [Source hashes / provenance](../sources.json).\n\nФормулы ниже/выше выводятся для явно заданных соглашений. Тела source methods сохраняются; adapters и doubles подписаны. Малые experiments не воспроизводят промышленное обучение.')
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (root .venv)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'dit_course':{'default_execution':'CPU','pretrained_weights':False}})
 for i,c in enumerate(nb.cells):c['id']=f'{LAB[:2]}-{i:03d}'
 nbf.write(nb,C/'notebooks'/f'{LAB}.ipynb')

start('00_probability_path_and_learning_route','00. Что предсказывает DiT: path, velocity и направление времени',
 'Отделим encoder/decoder, neural field, solver и policy. Выведем все соглашения до первого transformer forward.')
md(r'''
## 1. От распределения к траектории

MIT часто ведёт время s от шума к данным: y_s=(1−s)ε+s z. В этих K6 inference helpers удобна противоположная координата σ: чистые данные при 0, шум при 1.

$$
x_\sigma=(1-\sigma)z+\sigma\epsilon,\qquad
u=\frac{d x_\sigma}{d\sigma}=\epsilon-z,\qquad s=1-\sigma.
$$

Следовательно dx/ds=−u. Одинаковый tensor нельзя подставлять как velocity после смены time convention без знака. Исходный conditional target доступен при обучении, когда известны z и ε; на inference чистый z неизвестен.

$$
p_\sigma(x)=\int p_\sigma(x\mid z)p_{\mathrm{data}}(z)\,dz,\qquad
u_\star(x,\sigma,c)=\mathbb{E}[\epsilon-z\mid x_\sigma=x,c].
$$

Условное ожидание возникает как minimizer squared regression: разложим E||f−U||² на E||f−E[U|X]||² и conditional variance, не зависящую от f. Отсюда смысл FM training, а не «нейросеть просто угадывает noise».
''')
code('''
z=torch.tensor([[.3,-.2]],dtype=torch.float64);noise=torch.tensor([[1.1,.7]],dtype=torch.float64)
sigma=.6;u=noise-z;x=(1-sigma)*z+sigma*noise
assert torch.allclose(x-sigma*u,z)
h=1e-5
fd=(((1-sigma-h)*z+(sigma+h)*noise)-((1-sigma+h)*z+(sigma-h)*noise))/(2*h)
assert torch.allclose(fd,u,atol=1e-10)
metrics['affine_velocity_fd_max_error']=float((fd-u).abs().max())
metrics['clean_endpoint_reparameterization_error']=float((x-sigma*u-z).abs().max())
print('x_sigma:',x,'u_sigma:',u,'u_s:',-u)
''')
md(r'''
### Мост к MIT Lecture 3A: score не равен velocity

Для xσ=(1−σ)z+σε conditional covariance равна σ²I. Дифференцирование log Gaussian по x и posterior averaging дают:

$$
\nabla_x\log p_\sigma(x)=-\frac{x-(1-\sigma)\mathbb{E}[z\mid x]}{\sigma^2}.
$$

Подставим E[z|x]=x−σu⋆ и сократим σ:

$$
\nabla_x\log p_\sigma(x)=-\frac{x+(1-\sigma)u_\star(x,\sigma)}{\sigma},\qquad 0<\sigma<1.
$$

Это вывод для выбранного affine path, не универсальное тождество всех diffusion parameterizations. Проверим многомерную Gaussian с коррелированной covariance: z∼N(μ,Σ), Cσ=(1−σ)²Σ+σ²I. По Gaussian conditioning:

$$
\mathbb{E}[z\mid x]=\mu+(1-\sigma)\Sigma C_\sigma^{-1}(x-(1-\sigma)\mu).
$$

У density score напрямую равен −Cσ⁻¹(x−(1−σ)μ). Сравнение двух путей проверяет одновременно marginalization, covariance и velocity→score conversion. При σ=0 формулы с division нельзя подставлять без отдельного limit analysis.
''')
code('''
mu=torch.tensor([.2,-.3],dtype=torch.float64)
Sigma=torch.tensor([[1.,.4],[.4,2.]],dtype=torch.float64);sig=.4
cov=(1-sig)**2*Sigma+sig**2*torch.eye(2,dtype=torch.float64)
x=torch.randn(16,2,dtype=torch.float64);delta=x-(1-sig)*mu
post_mean=mu+delta@torch.linalg.solve(cov,(1-sig)*Sigma).T
u_marginal=(x-post_mean)/sig
score_from_velocity=-(x+(1-sig)*u_marginal)/sig
score_direct=-torch.linalg.solve(cov,delta.T).T
assert torch.allclose(score_from_velocity,score_direct,atol=1e-12)
metrics['correlated_Gaussian_score_bridge_max_error']=float((score_from_velocity-score_direct).abs().max())
''')
md(r'''
## 2. Почему solver не равен DiT

Neural field Gθ выдаёт velocity той же **физической** размерности, что latent. Euler использует его как правую часть ODE:

$$
x_{k+1}=x_k+(\sigma_{k+1}-\sigma_k)G_\theta(x_k,1000\sigma_k,c).
$$

При denoising Δσ<0. Множитель 1000 — conditioning convention сети; Euler всё равно шагает по σ, а не на 1000Δσ. VAE только задаёт пространство z. Он не выполняет эту ODE.

В π-Flow student вместо одного velocity выдаёт параметры π. Между двумя student вызовами solver многократно вычисляет **обычную tensor function policy**, а не transformer. Поэтому отдельно считаем network NFE, grid points G и policy evaluations.
''')
code('''
x=noise.clone()
for sigma_now,sigma_next in zip(torch.linspace(1,0,9)[:-1],torch.linspace(1,0,9)[1:]):
    x=x+(sigma_next-sigma_now)*u
assert torch.allclose(x,z,atol=1e-7)
metrics['constant_field_euler_error']=float((x-z).abs().max())
# A dimension/units ledger, not real inference.
metrics['contracts']={'video_latent':'B,T,H,W,Cv','audio_latent':'B,A,Ca',
 'standard_head':'same physical C','grid_head':'B,G,...,C; state C unchanged',
 'solver_time':'sigma in [0,1]','network_time':'1000*sigma'}
print(metrics['contracts'])
''')
md('''
## 3. Порядок нового маршрута

00 probability/ODE → 01 DiT primitives → 02 one generated visual stream + text CrossDiT → 03 Audio/Video fusion → 04 semantic/codec conditioning → 05 visual references → 06 Euler/CFG loop → 07 DX mathematics → 08 native π sampler → 09 miniature imitation distillation → 10 ComfyUI/manual experiment.

Существующие K6 labs 05–12 дают overview; здесь каждый из запрошенных файлов получает отдельный подробный разбор. Для codec вернитесь в [vae](../../vae/README.md). [MIT_BRIDGE](../MIT_BRIDGE.md) указывает исходные вопросы и границу дополнений.
''')
finish('1. Выведите target при s=1−σ, затем проверьте finite difference.\n2. Для xσ−σu объясните, когда это истинный z, а когда лишь predicted posterior mean.\n3. Запишите shape ledger для clip и audio разной длины.',
 'Алгебра affine path и знак подтверждены численно. Constant-field Euler exact не доказывает exactness для нелинейного trained field; следующий шаг — понять параметризацию Gθ.')

start('01_dit_primitives_and_tensor_geometry','01. DiT: patchifier, time, attention, modulation и output',
 'Соберём вычисление из простых операторов, проверим ручные формулы и исследуем первые градиенты adaLN-Zero.')
md(r'''
## 1. Patchification — перестановка индексов плюс проекция

Для latent X∈R^(B×T×H×W×C) и patch p=(p_t,p_h,p_w):

$$
N_v=\frac{T}{p_t}\frac{H}{p_h}\frac{W}{p_w},\quad
P_{in}=p_t p_h p_w C,\qquad E=\mathrm{patchify}(X)W_e+b_e.
$$

После reshape/permute каждый patch — вектор, последняя ось содержит p_t,p_h,p_w,C в этом порядке. Формы: B,N_v,P_in → B,N_v,D_v. Изменение patch size меняет token budget, attention N_v² и детали, но не даёт автоматически такого же изменения всей peak memory.
''')
code('''
x=torch.arange(1*2*4*4*3,dtype=torch.float32).reshape(1,2,4,4,3)
def patchify(x):
    B,T,H,W,C=x.shape
    return x.view(B,T,1,H//2,2,W//2,2,C).permute(0,1,3,5,2,4,6,7).flatten(4,7)
def unpatchify(p,C):
    B,T,H,W,_=p.shape
    return p.view(B,T,H,W,1,2,2,C).permute(0,1,4,2,5,3,6,7).reshape(B,T,H*2,W*2,C)
p=patchify(x);assert torch.equal(unpatchify(p,3),x)
embed=native().VisualEmbeddings(3,24,(1,2,2))
assert torch.equal(embed(x),embed.in_layer(p))
metrics['patch_shapes']={'input':list(x.shape),'patches':list(p.shape),'embedding':list(embed(x).shape)}
print(metrics['patch_shapes'])
''')
md(r'''
## 2. Time embedding и adaLN

Fourier features переводят scalar network time в вектор:

$$
\omega_j=\exp[-j\log(10000)/m],\qquad
f(t)=[\cos(t\omega_j),\sin(t\omega_j)]_j.
$$

TimeEmbeddings применяет Linear→SiLU→Linear; затем к time embedding добавляется projected pooled text. Modulation: SiLU→Linear выдаёт shift b, scale s и gate g. Для residual branch:

$$
\mathrm{M}(x;c)=(1+s(c))\odot\mathrm{LN}(x)+b(c),\qquad
y=x+g(c)\odot F(\mathrm{M}(x;c)).
$$

По chain rule:

$$
\frac{\partial y}{\partial\theta_F}=g\odot\frac{\partial F}{\partial\theta_F},\qquad
\frac{\partial y}{\partial g}=F(\mathrm{M}(x;c)).
$$

При g=0 первый gradient равен нулю, второй в общем случае ненулевой. У source Modulation выходная linear zero initialized. Значит сначала b=s=g=0, residual branch выключен. Ветвь может обучаться: gradient gate ненулевой, хотя gradient внутренних F weights на первом backward равен нулю. Не путайте identity отдельного transformer block с zero output всей сети.
''')
code('''
block=native().TransformerDecoderBlock(24,16,48,8,engine='sdpa')
x=torch.randn(1,6,24);text=torch.randn(1,4,24);c=torch.randn(1,16)
rope=native('core.components.rope').RoPE1D(8)(torch.arange(6))
y=block(x,text,c,rope);assert torch.equal(y,x)
loss=y.square().mean();loss.backward()
first_inner=float(block.self_attention.to_query.weight.grad.abs().max())
first_gate=float(block.visual_modulation.out_layer.weight.grad.abs().max())
assert first_inner==0 and first_gate>0
metrics['adaLN_zero']={'identity_error':float((y-x).abs().max()),'inner_weight_gradient':first_inner,'modulation_gradient':first_gate}
print(metrics['adaLN_zero'])
''')
md(r'''
## 2b. LayerNorm, RMSNorm и FFN без скрытых формул

LayerNorm вычисляет статистики по последней feature axis D отдельно для каждого token, а не по sequence length:

$$
\mu=\frac1D\sum_jx_j,\quad v=\frac1D\sum_j(x_j-\mu)^2,\qquad
\mathrm{LN}(x)_j=\frac{x_j-\mu}{\sqrt{v+\epsilon}}.
$$

У block pre-norm elementwise_affine=False: learned shift/scale поступают из modulation. У text projection LayerNorm affine=True. RMSNorm для Q/K не вычитает mean:

$$
\mathrm{RMSNorm}(x)_j=w_j\frac{x_j}{\sqrt{D^{-1}\sum_i x_i^2+\epsilon}}.
$$

Поскольку E[x²]=Var(x)+E[x]², RMSNorm и LN совпадают только при дополнительных условиях; сдвиг x константой не нейтрален для RMSNorm.

FFN source использует bias-free Linears и exact GELU; dropout=0:

$$
\mathrm{GELU}(x)=\frac{x}{2}\left[1+\mathrm{erf}\left(\frac{x}{\sqrt2}\right)\right],\qquad
\mathrm{FFN}(x)=W_2\,\mathrm{GELU}(W_1x).
$$

Для row-vector code запись эквивалентна xW₁ᵀ и затем W₂ᵀ. FFN действует независимо по tokens, attention связывает tokens. Проверим оба norm и GELU по формулам.
''')
code('''
x=torch.randn(2,5,8);ln=nn.LayerNorm(8,elementwise_affine=False);rms=nn.RMSNorm(8)
mean=x.mean(-1,keepdim=True);var=(x-mean).square().mean(-1,keepdim=True)
manual_ln=(x-mean)/torch.sqrt(var+ln.eps)
eps=rms.eps if rms.eps is not None else torch.finfo(x.dtype).eps
manual_rms=x/torch.sqrt(x.square().mean(-1,keepdim=True)+eps)*rms.weight
manual_gelu=.5*x*(1+torch.erf(x/math.sqrt(2)))
assert torch.allclose(ln(x),manual_ln,atol=1e-6)
assert torch.allclose(rms(x),manual_rms,atol=1e-6)
assert torch.allclose(F.gelu(x),manual_gelu,atol=1e-6)
metrics['normalization_formula_errors']={'LayerNorm':float((ln(x)-manual_ln).abs().max()),'RMSNorm':float((rms(x)-manual_rms).abs().max()),'GELU':float((F.gelu(x)-manual_gelu).abs().max())}
# Explicit modulation broadcast over tokens for B=2.
scale=torch.randn(2,8);shift=torch.randn(2,8)
actual=native('core.tensors').apply_scale_shift_norm(ln,x,scale,shift)
assert torch.allclose(actual,ln(x)*(1+scale[:,None])+shift[:,None])
''')
md(r'''
## 3. Attention от скалярных произведений к tensor axes

При h heads, d=D/h получаем Q,K,V размера B,h,N,d. Scores делятся на sqrt(d), чтобы при независимых unit-variance coordinates дисперсия суммы не росла как d:

$$
S_{ij}=\frac{q_i^\top k_j}{\sqrt d}+M_{ij},\quad
A_{ij}=\frac{\exp S_{ij}}{\sum_{l}\exp S_{il}},\quad Y_i=\sum_j A_{ij}v_j.
$$

В K6 Q/K дополнительно имеют RMSNorm по head dimension и RoPE. Orthogonal rotation сохраняет нормы, но dot product двух разных positions меняет через relative rotation. Это не добавление позиции к latent value и не attention mask.

Полный forward ниже: Linears → reshape → RMSNorm → rotate → scaled softmax → weighted V → output Linear. Сравниваем ручную формулу с оригинальной attention, использующей локальный dense-SDPA backend.
''')
code('''
attn=native().MultiheadSelfAttentionDec(24,8,engine='sdpa');x=torch.randn(1,6,24)
q=attn.query_norm(attn.to_query(x).reshape(1,6,3,8))
k=attn.key_norm(attn.to_key(x).reshape(1,6,3,8));v=attn.to_value(x).reshape(1,6,3,8)
rot=native('core.tensors').apply_rotary
qr,kr=rot(q,rope),rot(k,rope)
A=torch.softmax(qr.transpose(1,2)@kr.transpose(1,2).transpose(-1,-2)/math.sqrt(8),dim=-1)
manual=attn.out_layer((A@v.transpose(1,2)).transpose(1,2).flatten(-2,-1))
actual=attn(x,rope)
assert torch.allclose(manual,actual,atol=2e-6)
assert torch.allclose(qr.norm(dim=-1),q.norm(dim=-1),atol=2e-6)
metrics['attention_manual_error']=float((manual-actual).abs().max())
fig,ax=plt.subplots(figsize=(4.5,3.5));im=ax.imshow(A[0,0].detach());fig.colorbar(im,ax=ax);ax.set(xlabel='Key token',ylabel='Query token',title='One attention head; row sums = 1')
fig.tight_layout();fig.savefig(figure_path(LAB,'attention'));plt.show()
''')
md(r'''
## 4. OutLayer — learned prediction, не обратная embedding matrix

Hidden tokens проходят conditioned normalization и Linear в p_t p_h p_w C_out значений. Source video output layout раскладывает последнее измерение как **C_out,p_t,p_h,p_w**, затем переставляет в B,T,H,W,C_out. Это отличается от порядка входного patch vector. Learned output projection учитывает своё соглашение; нельзя объявить OutLayer inverse VisualEmbeddings с identity weights.

Audio head выдаёт B,A,C_a и содержит **два** применения norm. Второй norm есть в pinned training-compatible code; при переносе его нельзя удалять как «лишний» без parity test.

Выполним whole random tiny K6 forward: обычная голова сохраняет physical dimensions, grid-head позже добавит отдельную ось G. Это проверка shapes/function, не generation quality.
''')
code('''
model=tiny_model().eval();v,a,kw=tiny_inputs(model)
with torch.no_grad():pv,pa=fresh_forward(model,kw)
assert pv.shape==v.shape and pa.shape==a.shape
metrics['native_tiny_shapes']={'video':list(pv.shape),'audio':list(pa.shape),'parameters':sum(p.numel() for p in model.parameters())}
show_source('core/components/dit.py','OutLayerAudio')
print(metrics['native_tiny_shapes'])
''')
finish('1. Закончите MIT Q3.1–Q3.5 и докажите patch roundtrip для p_t=2.\n2. После одного optimizer step снова измерьте gradient attention weights.\n3. Увеличьте N в 2 раза и посчитайте score elements отдельно от total memory.\n4. Выведите relative-position property Rᵀ(i)R(j)=R(j−i).',
 'Patch indices, attention formula, rotation norms и zero-init gradient routing проверены. Full tiny model сохранил размеры; никакого видео/аудио качества случайной сети эти тесты не оценивают.')

start('02_single_stream_crossdit_k5','02. Однопоточный CrossDiT K5: visual state и текстовый context',
 'Разберём SA→text CA→FFN, воспроизведём оригинальный K5 block и сравним его с соответствующим K6 single-stream block при одинаковых весах.')
md(r'''
## 1. Что означает «один поток»

Здесь один **генерируемый** latent stream V; text tokens T — внешний conditioning context, который может дополнительно обрабатываться text transformer/LTF. Это не утверждение, что в модели всего один tensor, и не то же самое, что joint self-attention по concatenated visual+text tokens.

$$
V_1=V+g_{sa}\odot\mathrm{SA}(\mathrm{M}_{sa}(V;c)),
$$

$$
V_2=V_1+g_{txt}\odot\mathrm{CA}(Q=\mathrm{M}_{txt}(V_1;c),K=T,V=T),
$$

$$
V_3=V_2+g_{ff}\odot\mathrm{FFN}(\mathrm{M}_{ff}(V_2;c)).
$$

Text CA: V shape B,N_v,D; T shape B,N_t,D после text projection. Q имеет N_v positions, K/V — N_t. Output сохраняет N_v. Нет требования N_v=N_t. В K6 multimodal CA дополнительно разрешены разные feature widths.

История: [DiT](https://arxiv.org/abs/2212.09748) — исходный diffusion-transformer framework; [PixArt-α](https://arxiv.org/abs/2310.00426) — текстовый cross-attention precedent. Слово CrossDiT не обозначает одного универсального блока всех авторов; здесь определяем его по K5/K6 source и не заявляем абсолютное первое употребление термина.
''')
code('''
k5=load_k5_block();old=k5.TransformerDecoderBlock(48,16,96,8,attention_engine='sdpa',text_token_padding=True)
for m in old.modules():
    if isinstance(m,k5.Modulation):nn.init.normal_(m.out_layer.weight,std=.08);nn.init.normal_(m.out_layer.bias,std=.08)
new=native().TransformerDecoderBlock(48,16,96,8,engine='sdpa',text_token_padding=True)
# Source modules have different FFN key names; map ONLY the equivalent FFN Linears.
state={}
for key,value in old.state_dict().items():
    key=key.replace('feed_forward.in_layer.','feed_forward.net.0.proj.').replace('feed_forward.out_layer.','feed_forward.net.2.')
    state[key]=value
new.load_state_dict(state,strict=True)
rv=native('core.components.rope').RoPE1D(8)(torch.arange(8))
v=torch.randn(1,8,48);text=torch.randn(1,5,48);c=torch.randn(1,16)
mask=torch.tensor([[[[True,True,True,True,False]]]])
with torch.no_grad():a=old(v,text,c,rv,None,mask);b=new(v,text,c,rv,mask);manual=manual_single(new,v,text,c,rv,mask)
assert torch.allclose(a,b,atol=2e-6) and torch.allclose(b,manual,atol=2e-6)
metrics['k5_k6_dense_block_parity_error']=float((a-b).abs().max())
metrics['k6_manual_sequence_error']=float((b-manual).abs().max())
print(metrics)
''')
md(r'''
## 2. Что изменено в execution adapter

Исполняются native class bodies K5; CUDA compile/autocast decorators сняты через AST. Affine/residual/RoPE helpers заменены локальными FP32 helpers K6 с теми же математическими операциями. Dense attention backend реализует ожидаемый helper return protocol. Sparse/NABLA, BF16 rounding, distributed inference и whole-model K5↔K6 checkpoint parity не проверялись.

В одинаковой точности, с одинаковыми весами, порядок SA→CA→FFN даёт тот же small-block result. Это более узкое утверждение, чем «K5 и K6 полностью одинаковы».
''')
code('''
show_source('dit.py','TransformerDecoderBlock',family='k5')
show_source('core/components/dit.py','TransformerDecoderBlock')
changed=text.clone();changed[:,-1]+=1000
with torch.no_grad():masked=new(v,changed,c,rv,mask);unmasked=new(v,changed,c,rv,None)
assert torch.equal(masked,b)
metrics['masked_token_perturbation_error']=float((masked-b).abs().max())
metrics['unmasked_token_perturbation_error']=float((unmasked-b).abs().max())
assert metrics['unmasked_token_perturbation_error']>1e-5
''')
md(r'''
## 3. Cross-attention и условная зависимость

При фиксированных weights/normalization текст влияет через K/V. При закрытом text gate эта **ветвь** не меняет V. Однако whole model может получать тот же prompt ещё через pooled embedding в modulation — поэтому закрыть CA недостаточно для полного удаления text conditioning.

Проверим два разных утверждения: влияние text tokens на block при фиксированном c и влияние pooled c при фиксированных tokens. Это разные causal interventions, не оценка prompt adherence модели.
''')
code('''
v0=v.clone();text2=text+.3*torch.randn_like(text);c2=c+.2*torch.randn_like(c)
with torch.no_grad():
    base=new(v0,text,c,rv,mask)
    token_change=new(v0,text2,c,rv,mask)
    pooled_change=new(v0,text,c2,rv,mask)
metrics['token_path_change']=float((token_change-base).square().mean())
metrics['modulation_path_change']=float((pooled_change-base).square().mean())
assert metrics['token_path_change']>0 and metrics['modulation_path_change']>0
fig,ax=plt.subplots(figsize=(6,3));ax.bar(['Token context','Time/pooled vector'],[metrics['token_path_change'],metrics['modulation_path_change']]);ax.set(ylabel='Output MSE to baseline',title='Two distinct conditioning routes in random block')
fig.tight_layout();fig.savefig(figure_path(LAB,'conditioning_routes'));plt.show()
''')
finish('1. Сопоставьте MIT class embedding с pooled branch и объясните, чего ему недостаёт для token CA.\n2. Закройте только CA gate и повторите два interventions.\n3. Выпишите Q/K/V и complexity для N_v=100,N_t=20,D=48,h=6.\n4. Почему измеренная small-block parity не доказывает whole-checkpoint portability?',
 'Original dense K5 block и K6 single-stream counterpart совпали при явном parameter mapping/FP32 adapter. Masked key perturbation не повлияла на output. Text CA и pooled modulation являются разными маршрутами условия.')

start('03_dual_stream_crossdit_k6','03. Dual-stream CrossDiT K6: Audio ↔ Video по одному оператору',
 'Выведем asymmetric Q/K/V, точный порядок pre/post text states и modulation routing; проверим ручную сборку, gradients и directional cuts.')
md(r'''
## 1. Два генерируемых состояния и разные widths

$$
V\in\mathbb{R}^{B\times N_v\times D_v},\qquad
A\in\mathbb{R}^{B\times N_a\times D_a}.
$$

Released Lite config: D_v=1792, D_a=896, head dimension 64: h_v=28,h_a=14. Tiny test ниже: D_v=48,D_a=32,d=8. Video и audio имеют собственные self-attention, text refiners, time embeddings, FFN и output heads.

Для **обновления video из audio**:

$$
Q_v=V_qW_Q^{v},\quad K_{a\to v}=A_{pre}W_K^{a\to v},\quad
V_{a\to v}=A_{pre}W_V^{a\to v}.
$$

W_Q: D_v→D_v, W_K/W_V: D_a→D_v. Attention имеет B,h_v,N_v,N_a; результат снова B,N_v,D_v. Обратная ветвь проецирует D_v→D_a и сохраняет N_a. Не требуются одинаковые sequence lengths или hidden widths. Число parameters одной video-receiving CA: Q/O дают 2D_v², K/V — 2D_aD_v; четыре biases дают 4D_v; Q/K RMSNorm — 2d_v. В обеих directions:

$$
P_{cross}=2(D_v+D_a)^2+4(D_v+D_a)+2d_v+2d_a.
$$

Это только два CA modules: внешние modulation/norm и self-attention не включены. Имена source va/av не используем как доказательство направления: направление задаёт **получатель Q**.
''')
code('''
model=tiny_model();block=model.visual_transformer_blocks[0];activate_gates(block)
Dv,Da=48,32
v=torch.randn(1,8,Dv);a=torch.randn(1,6,Da)
tv=torch.randn(1,5,Dv);ta=torch.randn(1,5,Da);cv=torch.randn(1,16);ca=torch.randn(1,16)
rv=model.visual_rope((2,2,2),[torch.arange(2)]*3).flatten(0,2);ra=model.audio_rope(torch.arange(6))
params=sum(p.numel() for m in [block.va_cross_attention,block.av_cross_attention] for p in m.parameters())
expected=2*(Dv+Da)**2+4*(Dv+Da)+2*8+2*8 # biases + two Q/K RMSNorm sets
assert params==expected
metrics['cross_attention_parameters']={'measured':params,'formula':expected,'scope':'two cross-attention modules only, excludes gates/norm outside attention'}
print(metrics['cross_attention_parameters'])
''')
md(r'''
## 2. Порядок внутри FusedTransformerDecoderBlock

Сначала каждый поток делает self-attention. После normalized state перед text CA сохраняются V_pre и A_pre. Text update добавляется в каждый поток. Затем:

$$
Q_v\leftarrow\mathrm{M}_{va}(V_{after\ text}),\qquad
Q_a\leftarrow\mathrm{M}_{av}(A_{after\ text}),
$$

$$
\Delta V=\mathrm{CA}(Q_v,K/V=A_{pre}),\qquad
\Delta A=\mathrm{CA}(Q_a,K/V=V_{pre}).
$$

![Схема двухпоточного блока: Q после text update, K/V до него](../../kandinsky-6/reports/figures/components/dual_stream_architecture.png)

Схема из существующей K6 component lab08; arrows показывают сохранённые pre-CA states, а не уже обновлённые outputs.

Оба cross outputs вычисляются до применения любого из них. Следовательно A→V и V→A используют сохранённый pre-text контекст, а queries — post-text state. Простая схема «сначала обновим video, потом из обновлённого video возьмём audio K/V» была бы другой сетью.

Ручная реализация из math ledger сравнивается с original forward. Затем намеренно подставляем post-text K/V, чтобы проверить, что утверждение о порядке действительно влияет на результат.
''')
code('''
with torch.no_grad():
    original=block(v,a,tv,ta,(cv,ca),rv,ra)
    manual=manual_fused(block,v,a,tv,ta,cv,ca,rv,ra)
    wrong=manual_fused(block,v,a,tv,ta,cv,ca,rv,ra,kv_after_text=True)
parity=max(float((x-y).abs().max()) for x,y in zip(original,manual))
wrong_error=max(float((x-y).abs().max()) for x,y in zip(original,wrong))
assert parity<2e-6 and wrong_error>1e-5
metrics['manual_block_max_error']=parity;metrics['wrong_KV_order_max_error']=wrong_error
show_source('core/components/dit.py','FusedTransformerDecoderBlock')
''')
md(r'''
## 3. Cross gates и два времени

Shift/scale/gate приходят из va/av-modulation. В released flags fix_modulation=True, cross_gates=True:

- video query shift/scale зависит от c_v, а gate обновления video — от c_a;
- audio query shift/scale зависит от c_a, а gate обновления audio — от c_v.

Причина — не название flag, а конкретный split: va output содержит D_v+D_v+D_a, av output — D_a+D_a+D_v. Gate последней части уходит потоку соответствующей ширины. При других комбинациях flags routing меняется. Код требует согласованных time dimensions; released оба равны 512, в tiny оба 16.

Сравним original/manual при всех четырёх flags. Затем отключим только одну cross residual связь: audio perturbation больше не должна менять video внутри этого блока. В полном stack косвенные пути через следующие blocks надо анализировать отдельно.
''')
code('''
routing=[]
for fix in [False,True]:
    for cross in [False,True]:
        b=native().FusedTransformerDecoderBlock(48,16,96,8,32,16,64,8,engine='sdpa',ca_rope=True,cross_gates=cross,fix_modulation=fix)
        activate_gates(b)
        with torch.no_grad():
            actual=b(v,a,tv,ta,(cv,ca),rv,ra);manual=manual_fused(b,v,a,tv,ta,cv,ca,rv,ra)
        err=max(float((x-y).abs().max()) for x,y in zip(actual,manual));assert err<2e-6
        routing.append({'fix_modulation':fix,'cross_gates':cross,'manual_error':err})
metrics['flag_parity']=routing
with torch.no_grad():
    base=block(v,a,tv,ta,(cv,ca),rv,ra)
    a_changed=a+.3*torch.randn_like(a)
    perturbed=block(v,a_changed,tv,ta,(cv,ca),rv,ra)
    cut=block(v,a,tv,ta,(cv,ca),rv,ra,va_gate_scale=0.)
    cut_perturbed=block(v,a_changed,tv,ta,(cv,ca),rv,ra,va_gate_scale=0.)
assert torch.equal(cut[0],cut_perturbed[0])
metrics['audio_to_video_open_MSE']=float((base[0]-perturbed[0]).square().mean())
metrics['audio_to_video_cut_error']=float((cut[0]-cut_perturbed[0]).abs().max())
assert metrics['audio_to_video_open_MSE']>0
''')
md(r'''
## 4. Измеряем cross dependence градиентом

Скалярная функция выбранного video output имеет Jacobian по audio input. При открытой ветви norm градиента должен быть ненулевым; при cut — нулевым/unused. Это проверка вычислительной связи, а не доказательство lip-sync, семантической согласованности или качества audio.

Отдельная модальность может присутствовать в input, но не обновляться solver. Это conditioning на clean stream. Удаление её tensor — другая ветка forward, а обнуление cross gate — ещё другой intervention.
''')
code('''
a_probe=a.clone().requires_grad_(True)
y=block(v,a_probe,tv,ta,(cv,ca),rv,ra)[0]
g=torch.autograd.grad(y.square().mean(),a_probe)[0]
ycut=block(v,a_probe,tv,ta,(cv,ca),rv,ra,va_gate_scale=0.)[0]
gcut=torch.autograd.grad(ycut.square().mean(),a_probe,allow_unused=True)[0]
metrics['cross_jacobian_norm']=float(g.norm());metrics['cut_jacobian_norm']=0. if gcut is None else float(gcut.norm())
assert metrics['cross_jacobian_norm']>0 and metrics['cut_jacobian_norm']==0
fig,axes=plt.subplots(1,2,figsize=(10,3));axes[0].bar(['Correct manual','Wrong K/V order'],[parity,wrong_error]);axes[0].set(ylabel='Output max difference',title='Ordering control');axes[1].bar(['A→V open','A→V cut'],[metrics['cross_jacobian_norm'],metrics['cut_jacobian_norm']]);axes[1].set(ylabel='Audio→video Jacobian norm',title='Directional dependence')
fig.tight_layout();fig.savefig(figure_path(LAB,'fusion_controls'));plt.show()
''')
finish('1. Повторите directional cut для V→A (`av_gate_scale=0`).\n2. Найдите source inputs gates для всех flags без подглядывания в результат.\n3. Сравните h_v N_v N_a + h_a N_a N_v с joint SA на N_v+N_a.\n4. Объясните, почему random tiny Jacobian не доказывает trained AV synchronization.',
 'Ручная сборка совпала с native fused block при всех flags; неверный pre/post K/V порядок дал расхождение. Directional cut устранил выбранную cross зависимость. Проверены function/gradient contracts, не качество синхронизации.')

start('04_conditioning_text_and_clean_modalities','04. Conditioning: token/pooled text, clean audio/video, masks и cache',
 'Раскроем все входы условия, исполним encode_audio_condition/encode_video_condition с подписанными codec doubles и проверим собственно text-conditioning путь native DiT.')
md(r'''
## 1. Условная плотность и три маршрута информации

Цель — p(x|c), neural field uθ(x,σ,c). Условие может входить через token context, глобальный modulation vector и clean modality/reference state. Это разные computational edges.

$$
T_v=P_v(T_{enc}),\quad T_a=P_a(T_{enc}),\qquad
c_v=\mathrm{Time}_v(1000\sigma_v)+P_{pool,v}(e_{pool}),
$$

$$
c_a=\mathrm{Time}_a(1000\sigma_a)+P_{pool,a}(e_{pool}).
$$

Qwen token features и pooled CLIP features — outputs отдельных pretrained encoders; DiT содержит собственные projections/refiners. Заморозка внешнего encoder не означает заморозку text projection внутри DiT. В tiny опытах features synthetic, vocabulary/tokenizer не загружаются; поэтому semantic prompt quality не измеряется.

Conditioning encoders ниже не делают transfer к device сами: caller обязан подготовить input на устройстве codec. Video helper переводит uint8 RGB в [-1,1], берёт **posterior.mode**, масштабирует и переставляет BCTHW→BTHWC, затем BF16. Audio helper требует mono (N,), делает B,D,A→B,A,D и scale.
''')
code('''
from types import SimpleNamespace
cond=native('core.algo.conditioning')
class VideoCodecDouble(nn.Module):
    def __init__(self):
        super().__init__();self.anchor=nn.Parameter(torch.tensor(1.));self.config={'scaling_factor':.5}
    def encode(self,pixels):
        # Coordinate/scale contract only: NOT learned Hunyuan VAE.
        z=F.avg_pool3d(pixels.mean(1,keepdim=True),kernel_size=(1,8,8))[:,:,::4].repeat(1,2,1,1,1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:z))
class AudioCodecDouble:
    scaling_factor=.25
    def wrapped_encode(self,w):return w.unfold(-1,4,4).mean(-1)[:,None,:].repeat(1,4,1)
white=torch.full((1,3,9,16,16),255,dtype=torch.uint8)
vc=cond.encode_video_condition(white,VideoCodecDouble())
wave=torch.ones(16,requires_grad=True);ac=cond.encode_audio_condition(wave,AudioCodecDouble())
assert vc.shape==(1,3,2,2,2) and vc.dtype==torch.bfloat16 and vc.eq(.5).all()
assert ac.shape==(1,4,4) and ac.eq(.25).all() and not ac.requires_grad
metrics['codec_contracts']={'video':list(vc.shape),'audio':list(ac.shape),'dtype':str(vc.dtype),'grad_disabled':not ac.requires_grad}
for fn,arg,codec in [(cond.encode_audio_condition,torch.ones(2,16),AudioCodecDouble()),(cond.encode_video_condition,white.float(),VideoCodecDouble())]:
    try:fn(arg,codec)
    except ValueError:pass
    else:raise AssertionError('Invalid conditioning input should be rejected')
show_source('core/algo/conditioning.py')
''')
md(r'''
## 2. Mask polarity: boolean keep-mask не равна float 0/1 bias

При masked softmax запрещённые keys должны иметь −∞ в additive mask. PyTorch SDPA boolean mask использует True=разрешено. Float mask складывается со scores; float 0/1 ничего не запрещает.

$$
M_j=\begin{cases}0,&\mathrm{valid}_j,\\-\infty,&\mathrm{padding}_j.\end{cases}
$$

`_normalize_attn_mask` в K6 меняет shape B,S→B,1,1,S, но **не преобразует float 0/1 в bool**. Контракт caller должен определять dtype и семантику. Проверьте хотя бы один valid key на пример, особенно в generic softmax implementation.
''')
code('''
ca=native().MultiheadCrossAttention(24,8,engine='sdpa',text_token_padding=True)
q=torch.randn(1,3,24);text=torch.randn(1,4,24);mask=torch.tensor([[[[True,True,True,False]]]])
changed=text.clone();changed[:,-1]+=100
with torch.no_grad():
    base=ca(q,text,mask);correct=ca(q,changed,mask);wrong=ca(q,changed,mask.float())
assert torch.equal(base,correct)
metrics['bool_mask_error']=float((base-correct).abs().max())
metrics['float_01_mask_change']=float((base-wrong).abs().max())
assert metrics['float_01_mask_change']>1e-4
''')
md(r'''
## 3. Что можно кэшировать

Token/pooled **projections** не зависят от σ и могут быть переиспользованы внутри одной генерации, если inputs и weights неизменны. LTF text blocks используют time-conditioned modulation и пересчитываются на каждом time.

В source key cache использует data_ptr и shape. In-place изменение того же tensor не меняет data_ptr; caller должен считать conditioning immutable или вызывать clear_text_proj_cache. При training нельзя переиспользовать cached autograd graph между backward. Мы очищаем cache перед training forward.

Проверим вызовы projection и refiner hooks. Затем намеренно изменим text tensor in-place и сравним stale cache с clear; это иллюстрация контракта cache, не blanket bug claim для штатного immutable inference.
''')
code('''
model=tiny_model().eval();activate_gates(model);_,_,kw=tiny_inputs(model)
counts={'projection':0,'refiner':0}
def inc(name):
    def hook(*args):counts[name]+=1
    return hook
h1=model.video_text_embeddings.register_forward_hook(inc('projection'))
h2=model.video_text_transformer_blocks[0].register_forward_hook(inc('refiner'))
model.clear_text_proj_cache()
with torch.no_grad():
    first=model(**kw)
    later=dict(kw,time=torch.tensor([300.]))
    second=model(**later)
h1.remove();h2.remove()
assert counts=={'projection':1,'refiner':2}
with torch.no_grad():
    kw['text_embed'].add_(.5*torch.randn_like(kw['text_embed']))
    stale=model(**later)
    model.clear_text_proj_cache();fresh=model(**later)
metrics['cache_hook_counts']=counts
metrics['mutable_input_cache_difference']=float((stale[0]-fresh[0]).square().mean())
assert metrics['mutable_input_cache_difference']>0
print(metrics['cache_hook_counts'])
''')
md('''
## 4. Null conditions и wrapper boundaries

`_resolve_null_embeds` выбирает shared null или separate video/audio nulls. При отсутствии audio null/RoPE делает documented fallback на video null. Это branch input подготовка; CFG algebra будет в lab 06. `encode_*_condition` использует no_grad: подготовка clean condition и обучение codec — разные стадии.
''')
code('''
loop=native('core.algo.denoise_loop');pv=dummy_text(positive=0.);pa=dummy_text(positive=-1.)
r1=torch.zeros(3,1,4,2,2);r2=torch.ones_like(r1)
te,pe,rr=loop._resolve_null_embeds([pv,pa],[r1,r2]);assert isinstance(te,list) and len(te)==2
te2,pe2,rr2=loop._resolve_null_embeds([pv,None],[r1,None]);assert te2 is pv['text_embeds']
metrics['null_branches']={'separate_modalities':len(te),'fallback_shape':list(te2.shape)}
''')
finish('1. Вернитесь в MIT Lab 3 Q2.2: где возникает unconditional branch при обучении?\n2. Сравните mode/sample clean reference при одном seed и сформулируйте preprocessing contract.\n3. Проверьте cache после weight update, device/dtype change и нового prompt.\n4. Почему float mask нельзя исправить только reshape?',
 'Native conditioning helpers прошли scale/layout/validation проверки на явных codec doubles. Projection cache отделён от time-dependent refiners; masks и immutable-input предпосылка проверены. Семантические encoders и реальный codec не запускались.')

start('05_visual_conditioning_references_and_rope','05. Visual Conditioning: reference tails, masks, роли и RoPE',
 'Исполним каждый метод visual_conditioning.py на batched/packed tensor layouts и отделим storage positions от semantic temporal positions.')
md(r'''
## 1. Три схемы visual condition

Для physical C_v channels модель с visual_cond=True принимает 2C_v+1:

$$
X_{in}=\mathrm{concat}(X,C_{aux},M),\qquad
X_{in}\in\mathbb{R}^{B\times T\times H\times W\times(2C_v+1)}.
$$

`pretrain`: clean first frame в auxiliary channels на старте; noisy X не перезаписывается. `i2v`: X first frame перезаписывается clean latent, auxiliary channels нулевые, mask помечает known frame. `tail_cond_first_frame`: clean reference append в конец storage sequence; source восстанавливает его и маркирует mask.

Mask channel — условие для сети; generated_mask/role IDs — отдельные tensors с другой задачей. `visual_token_type_embeddings` добавляет learned role embedding после patch embedding. Канальный mask не заменяет attention padding mask и не описывает time σ.
''')
code('''
vc=native('core.algo.visual_conditioning');loop=native('core.algo.denoise_loop')
B,T,H,W,C=2,3,2,2,4
video=torch.randn(B,T,H,W,C);first=torch.randn(B,H,W,C);cu=torch.arange(B+1,dtype=torch.int32)*T
pre=loop._build_video_input(video.clone(),True,first,cu,'pretrain')
i2v=loop._build_video_input(video.clone(),True,first,cu,'i2v')
assert pre.shape==i2v.shape==(B,T,H,W,2*C+1)
assert torch.equal(pre[...,:C],video) and torch.equal(pre[:,0,...,C:2*C],first)
assert torch.equal(i2v[:,0,...,:C],first) and i2v[...,C:2*C].eq(0).all()
assert pre[:,0,-1,-1,-1].eq(1).all() and pre[:,1:,...,-1].eq(0).all()
metrics['visual_input_channels']={'physical':C,'model_input':2*C+1,'shape':list(pre.shape)}
''')
md(r'''
## 2. Append / restore / generated mask

При R=1 или 2 refs длина становится T+R, generated frames по-прежнему T. Role=0 для generated, role=1 для references. Packed layout объединяет B и T; cu_seqlens[b+1]−R даёт начало references каждого элемента. Нельзя взять последние R frames всего packed tensor: так будет испорчен только последний batch item.

`restore_tail_references` меняет переданный tensor in-place. `build_tail_condition` сначала восстанавливает refs, затем concat [video,zeros,mask]. Передайте clone, если нужен baseline до модификации. Ниже сравним batched и packed вычисления для B=2,R=2, где ошибка индексации была бы заметна.
''')
code('''
refs=torch.randn(B,2,H,W,C)
batched,roles,generated=vc.append_visual_references(video.clone(),refs,B,T)
packed,roles_p,generated_p=vc.append_visual_references(video.flatten(0,1).clone(),refs,B,T)
assert torch.equal(batched.flatten(0,1),packed)
assert roles.reshape(B,T+2)[:,-2:].eq(1).all() and generated.sum()==B*T
assert torch.equal(roles.flatten(),roles_p.flatten())
cu_ext=torch.arange(B+1,dtype=torch.int32)*(T+2)
corrupted=packed+10;vc.restore_tail_references(corrupted,refs,cu_ext)
assert torch.equal(corrupted.reshape(B,T+2,H,W,C)[:,-2:],refs)
inp=vc.build_tail_condition(packed.clone(),refs,cu_ext)
assert inp.shape[-1]==2*C+1
assert inp.reshape(B,T+2,H,W,-1)[:,-2:,...,-1].eq(1).all()
metrics['two_reference_contract']={'batched':list(batched.shape),'packed':list(packed.shape),'generated_frames':int(generated.sum()),'reference_restore_max_error':0.}
show_source('core/algo/visual_conditioning.py','append_visual_references')
''')
md(r'''
## 3. Semantic time и storage index различаются

Пусть generated RoPE содержит temporal positions 0,…,T−1. Первый reference физически стоит после T generated slots, но получает RoPE t=0. Второй reference — position T−1. Иначе модель сочла бы reference будущим кадром за пределами нужного video.

$$
\mathrm{RoPE}_{extended}=[R_0,\ldots,R_{T-1},R_0,R_{T-1}].
$$

Spatial coordinate rotations refs остаются теми же. `compute_reference_rope` сначала строит generated grid shape(T_total−R,H_p,W_p), затем append; count=0 делегирует обычный compute_visual_rope. Это reference position encoding, не доказательство perfect interpolation между двумя картинками.
''')
code('''
model=tiny_model();shape=(3,2,2);scale=(1.,1.,1.)
base=native('core.algo.prepare_ropes').compute_visual_rope(model.visual_rope,shape,scale)
ext=vc.append_reference_rope(base,2)
computed=vc.compute_reference_rope(model.visual_rope,(5,2,2),scale,2)
plain=vc.compute_reference_rope(model.visual_rope,shape,scale,0)
assert torch.equal(ext,computed) and torch.equal(plain,base)
assert torch.equal(ext[-2],base[0]) and torch.equal(ext[-1],base[-1])
metrics['reference_rope_exact']=True
fig,ax=plt.subplots(figsize=(7,2.8));ax.scatter(range(5),[0,1,2,0,2],s=120,c=['#2078b4']*3+['#e87924']*2)
for i,label in enumerate(['gen0','gen1','gen2','ref first','ref last']):ax.annotate(label,(i,[0,1,2,0,2][i]),xytext=(0,10),textcoords='offset points',ha='center')
ax.set(xlabel='Storage position',ylabel='Semantic temporal position',xticks=range(5),yticks=range(3),ylim=(-.5,3))
fig.tight_layout();fig.savefig(figure_path(LAB,'reference_positions'));plt.show()
''')
md(r'''
## 4. Constraint projection и solver

Reference restoration задаёт projector P на множество states с фиксированными clean refs. При tail scheme после Euler step выполняется:

$$
X_{k+1}=P\left(X_k+\Delta\sigma\,u(X_k,\sigma_k,c)\right).
$$

Это отличается от свободной ODE на всех coordinates. Generated mask нужен при decode/export: reference tail не должен автоматически увеличивать длительность сгенерированного видео. Low-level denoise loop может вернуть extended tensor; filtering принадлежит pipeline/postprocess, который надо проверить отдельно.

I2V first-frame scheme перезаписывает перед network forward и восстанавливает в конце; tail scheme дополнительно restore после каждого update. Это разные алгоритмические места применения ограничения.
''')
code('''
# Projection idempotence is an exact mathematical/control test.
x=batched+torch.randn_like(batched);once=x.clone();vc.restore_tail_references(once,refs,cu_ext)
twice=once.clone();vc.restore_tail_references(twice,refs,cu_ext)
assert torch.equal(once,twice)
metrics['reference_projector_idempotence_error']=float((once-twice).abs().max())
try:vc.append_reference_rope(base,3)
except ValueError:metrics['invalid_reference_count_rejected']=True
else:raise AssertionError('Only one/two references supported by this source API')
''')
finish('1. Выпишите индексы двух refs для каждого B в packed layout.\n2. Повторите masks/roles для R=1 и сравните с R=2.\n3. Что изменится, если refs получат storage-time RoPE?\n4. Нарисуйте места constraint projection в Euler и π sampler.',
 'Все пять visual_conditioning methods исполнены. Batched/packed transforms, projector idempotence и semantic RoPE совпали с ожидаемыми indices. Это preprocessing/indexing evidence, не image-to-video generation quality.')

start('06_denoise_loop_euler_cfg_and_partial_sampling','06. Denoise Loop: Euler, CFG, time warp и partial sampling',
 'От интеграла ODE перейдём к native denoise_loop и каждому helper; проверим convergence, число forward calls и замороженную модальность.')
md(r'''
## 1. Дискретизация интеграла и ошибка

$$
x(\sigma_{k+1})=x(\sigma_k)+\int_{\sigma_k}^{\sigma_{k+1}}u(x(s),s)\,ds
\approx x_k+\Delta\sigma_k u(x_k,\sigma_k).
$$

При гладком Lipschitz field local truncation error O(h²), global O(h) на конечном интервале. Neural approximation error — отдельная величина: уменьшение h не исправляет неверное поле. Taylor expansion exact solution:

$$
x(t+h)=x(t)+hu+\frac{h^2}{2}(\partial_tu+J_u u)+O(h^3).
$$

Euler отбрасывает h² term. При bound C на него и Lipschitz constant L ошибка удовлетворяет e_(k+1)≤(1+L|h|)e_k+C h². Суммирование по конечному interval даёт global O(max|h|); отрицательный denoising step не меняет порядок оценки.

В source schedule raw τ от 1 к 0 отображается σ=aτ/(1+(a−1)τ). Network time=1000σ. Euler использует **difference shifted σ**, не raw Δτ. Здесь scheduler_scale меняет сетку, а не само Gθ как функцию σ.
''')
code('''
loop=native('core.algo.denoise_loop');schedule=native('core.algo.flow_matching')
A=torch.tensor([[.7,.2],[.3,.4]]);start=scalar_bundle();x1=vector_from_bundle(start)
exact=torch.matrix_exp(-A)@x1;rows=[]
for steps in [4,8,16,32,64]:
    field=CountingLinearField(A)
    result=loop.denoise_loop(clone_bundle(start),field,dummy_text(),dummy_text(positive=0.),None,None,
      torch.zeros(3,1,4,2,2),torch.zeros(3,1,4,2,2),steps,1.,1.)
    error=float((vector_from_bundle(result)-exact).norm())
    assert field.calls==steps
    rows.append({'steps':steps,'NFE':field.calls,'error':error})
assert rows[-1]['error']<rows[0]['error']/8
metrics['native_euler_convergence']=rows
fig,ax=plt.subplots(figsize=(6,3.5));ax.loglog([r['steps'] for r in rows],[r['error'] for r in rows],'o-',label='Native Euler vs matrix exponential');ax.loglog([4,64],[rows[0]['error'],rows[0]['error']/16],'--',label='1/N guide')
ax.set(xlabel='Network evaluations',ylabel='Endpoint L2 error');ax.legend();fig.tight_layout();fig.savefig(figure_path(LAB,'euler_convergence'));plt.show()
''')
md(r'''
## 2. CFG — две оценки поля, одна Euler траектория

$$
u_w=u_{null}+w(u_{cond}-u_{null}).
$$

w=1 возвращает conditional velocity, поэтому source не вызывает null branch повторно (с допуском 1e−6). При w≠1 обычно два network forwards на один solver step. Это не два Euler шага. Negative-prompt baseline не обязательно truly unconditional model branch.

Для tiny analytic field положительное/нулевое text condition меняет постоянный bias. Проверим NFE отдельно от num_steps. Обычный CFG sweep не следует применять к distilled π checkpoint: там guided teacher behaviour уже могло быть distilled; native π sampler вообще не принимает CFG argument.
''')
code('''
rows=[]
for w in [0.,1.,3.]:
    field=CountingLinearField(torch.zeros(2,2),text_bias=.2)
    out=loop.denoise_loop(clone_bundle(start),field,dummy_text(positive=1.),dummy_text(positive=0.),None,None,
      torch.zeros(3,1,4,2,2),torch.zeros(3,1,4,2,2),8,w,1.)
    expected=x1-.2*w
    assert torch.allclose(vector_from_bundle(out),expected,atol=2e-6)
    assert field.calls==(8 if w==1 else 16)
    rows.append({'CFG':w,'steps':8,'NFE':field.calls,'endpoint_error':float((vector_from_bundle(out)-expected).abs().max())})
metrics['CFG_counts']=rows;print(rows)
''')
md(r'''
## 3. Partial sampling: modality остаётся в input

Если sample_audio=False, source не обновляет audio и передаёт audio time=0; video time меняется. Это условная генерация video на уже clean audio state. Удалить audio tensor — другая branch forward; заменить его noise и объявить clean — неверный conditioning protocol.

Проверим неизменность tensor и фактические time arguments. Оба потока остаются доступны oracle; настоящий multimodal DiT также может вычислять оба streams, несмотря на отключённый solver update.
''')
code('''
field=CountingLinearField(A);before=start.audio.clone()
out=loop.denoise_loop(clone_bundle(start),field,dummy_text(),dummy_text(positive=0.),None,None,
 torch.zeros(3,1,4,2,2),torch.zeros(3,1,4,2,2),8,1.,3.,sample_audio=False)
assert torch.equal(out.audio,before)
assert all(t[1]==[0.] for t in field.time_log)
metrics['partial_sampling']={'audio_unchanged':True,'audio_time_always_zero':True,'network_calls':field.calls}
show_source('core/algo/denoise_loop.py','denoise_loop')
''')
md(r'''
## 4. Helpers, RoPE caching и настоящее tiny forward

`_raw_dit` разворачивает cache wrapper только при наличии `.module` и `.set_cache`; не любую обёртку с `.module`. `_rebuild_text_rope` воспроизводит tables подходящей длины. `recompute_ropes_each_step` — диагностический A/B, он не меняет intended coordinates.

Сравним два native tiny прогонов на одном noise: precomputed RoPE и recomputation. Исходники network/loop настоящие, random weights уменьшены. Измеряем exact/numeric parity и NFE; не делаем вывода о perceptual quality.
''')
code('''
from types import SimpleNamespace
model=tiny_model().eval();v,a,kw=tiny_inputs(model)
wrapped=SimpleNamespace(module=model,set_cache=lambda:None)
assert loop._raw_dit(wrapped) is model
rebuilt=loop._rebuild_text_rope(model,kw['text_rope'])
assert all(torch.equal(x,y) for x,y in zip(rebuilt,kw['text_rope']))
B=native('core.types').LatentBundle
bundle=B(v.clone(),a.clone(),torch.tensor([0,2],dtype=torch.int32),torch.tensor([0,6],dtype=torch.int32))
def run(recompute):
    model.clear_text_proj_cache();calls=[];h=model.register_forward_hook(lambda *args:calls.append(1))
    out=loop.denoise_loop(clone_bundle(bundle),model,{'text_embeds':kw['text_embed'],'pooled_embed':kw['pooled_text_embed']},
      dummy_text(positive=0.),kw['visual_rope'],kw['audio_rope'],kw['text_rope'],kw['text_rope'],
      3,1.,3.,attention_mask=kw['attention_mask'],recompute_ropes_each_step=recompute)
    h.remove();return out,len(calls)
base,n=run(False);recomputed,m=run(True)
err=max(float((base.video-recomputed.video).abs().max()),float((base.audio-recomputed.audio).abs().max()))
assert err<2e-6 and n==m==3
metrics['native_tiny_rope_AB']={'max_error':err,'NFE_each':n}
''')
finish('1. Выполните MIT Lab 1 Q1.1 и сравните convergence с нашим oracle.\n2. При scheduler_scale=5 повторите convergence; сравните Δσ, а не raw Δτ.\n3. Запустите sample_video=False и объясните роли data/noise в frozen video.\n4. Добавьте first-frame reference и проверьте constraint после последнего update.',
 'Native Euler дал ожидаемое уменьшение ошибки и точные CFG branch counts. Partial modality tensor сохранился, frozen time был нулевым. Recomputed/precomputed RoPE дали совпадающий tiny result. Ни network error, ни production speed этим не измерены.')

start('07_piflow_math_dx_policy','07. Математика π-Flow: от affine path к DX policy',
 'Выведем u→x0, raw/shifted time, grid interpolation, polynomial mode, gradients/detach и discrete policy rollout для каждого метода piflow_math.py.')
md(r'''
## 1. Почему clean endpoint связан с velocity

Для conditional affine path xσ=z+σ(ε−z) имеем u=ε−z, значит z=xσ−σu. После conditional expectation:

$$
\widehat z(x,\sigma)=x-\sigma u_\star(x,\sigma)
=\mathbb{E}[z\mid x_\sigma=x].
$$

Это posterior mean только для соответствующего FM parameterization и точного marginal field. Generic neural output/другая schedule не получают этот смысл автоматически.

DX student в начале сегмента (x_s,σ_s) выдаёт G параметров U_g. Source строит:

$$
a_g=x_s-\sigma_s U_g.
$$

Все U_g reparameterized относительно **одного source state**, а не являются G истинными velocities в неизвестных future states. Interpolation a(τ) задаёт dynamic clean-anchor approximation, а policy использует текущее состояние:

$$
\pi(x,\sigma)=\frac{x-a(\tau)}{\max(\sigma,\varepsilon)}.
$$

Grid anchors фиксированы внутри сегмента, но π зависит от текущего x. При новом network call строится новая policy.
''')
code('''
pm=native('core.algo.piflow_math');xs=torch.tensor([[1.2,.8]],dtype=torch.float64)
x0=torch.tensor([[.3,-.2]],dtype=torch.float64);u=xs-x0
pol=pm.DXPolicy(u[:,None].expand(1,5,2).clone(),xs,torch.ones(1,dtype=torch.float64),segment_size=1.,shift=1.)
assert torch.allclose(pol.denoising_output_x_0,x0[:,None].expand(1,5,2))
assert torch.allclose(pol.pi(xs,torch.ones(1,dtype=torch.float64)),u)
metrics['u_to_x0_max_error']=float((pol.denoising_output_x_0-x0[:,None]).abs().max())
show_source('core/algo/piflow_math.py','DXPolicy')
''')
md(r'''
## 2. Raw τ и shifted σ: вывод inverse и Jacobian

$$
\sigma=f(\tau)=\frac{a\tau}{1+(a-1)\tau},\qquad a>0.
$$

Умножаем знаменатель и собираем τ: σ=τ[a−σ(a−1)]. Поэтому:

$$
\tau=f^{-1}(\sigma)=\frac{\sigma}{a+(1-a)\sigma},\qquad
f'(\tau)=\frac{a}{[1+(a-1)\tau]^2}.
$$

Если интегрировать напрямую по τ, dX/dτ=f'(τ)π(X,f(τ)). Source вместо этого выбирает raw grid, вычисляет **точное difference f(τ_next)−f(τ)** и умножает π на него. Дополнительно умножать на f' нельзя — это double counting.

`segment_size` задан по raw time. Из source τ_s получаем τ_d=max(τ_s−Δτ,0); внутренний denominator segment_size=max(τ_s−τ_d,eps). Shape broadcast добавляет singleton axes после batch.
''')
code('''
tau=torch.linspace(.01,.99,99,dtype=torch.float64);a=5.
sigma=pm.shift_timesteps(tau,a);back=sigma/(a+(1-a)*sigma)
assert torch.allclose(tau,back,atol=1e-14)
h=1e-6;fd=(pm.shift_timesteps(tau+h,a)-pm.shift_timesteps(tau-h,a))/(2*h)
jac=a/(1+(a-1)*tau).square();assert torch.allclose(fd,jac,atol=1e-8)
metrics['warp_inverse_error']=float((tau-back).abs().max());metrics['warp_derivative_error']=float((fd-jac).abs().max())
fig,axes=plt.subplots(1,2,figsize=(10,3));axes[0].plot(tau,sigma,label='sigma(tau), a=5');axes[0].plot(tau,tau,'--',label='identity');axes[0].legend();axes[0].set(xlabel='Raw tau',ylabel='Shifted sigma')
axes[1].plot(tau,jac);axes[1].set(xlabel='Raw tau',ylabel='d sigma / d tau');fig.tight_layout();fig.savefig(figure_path(LAB,'time_warp'));plt.show()
''')
md(r'''
## 3. Grid interpolation, error и endpoint amplification

Grid axis идёт от destination к source: r=(τ−τ_d)/(τ_s−τ_d), q=r(G−1). После clamp floor даёт i, сосед i+1 и λ=q−i:

$$
a(\tau)=(1-\lambda)a_i+\lambda a_{i+1}.
$$

При G=1 source возвращает единственный anchor. При G≥2 endpoint r=1 выбирает последнее значение через i=G−2,λ=1. Для дважды differentiable anchor и равномерного grid spacing h_g:

$$
\|a-a_{interp}\|_\infty\leq\frac{h_g^2}{8}\sup\|a''\|.
$$

Для scalar coordinate remainder линейной интерполяции имеет вид:

$$
a(\tau)-\ell(\tau)=\tfrac12 a''(\xi)(\tau-\tau_i)(\tau-\tau_{i+1}).
$$

Произведение по модулю не больше h_g²/4; отсюда коэффициент 1/8. Для vector coordinates bound применяется покомпонентно в max norm. Velocity error из anchor error равна −δa/max(σ,eps), поэтому near-clean endpoint чувствителен к ошибкам. Уменьшение solver step не убирает policy/grid approximation error. Это мотивация dense grid/обучения, а не гарантия качества конкретного числа G.
''')
code('''
rows=[];query=torch.linspace(0,1,1001,dtype=torch.float64)
for G in [2,3,5,9,17,33]:
    times=torch.linspace(0,1,G,dtype=torch.float64);anchors=.3+.2*torch.sin(math.pi*times)
    # A query is treated as a separate batch item; all share the same coefficients.
    grids=anchors[None,:,None].expand(len(query),G,1)
    got=pm.DXPolicy._interpolate(grids,query[:,None]).flatten()
    error=float((got-(.3+.2*torch.sin(math.pi*query))).abs().max())
    bound=.2*math.pi**2/(8*(G-1)**2)
    assert error<=bound+1e-12
    rows.append({'grid':G,'max_anchor_error':error,'curvature_bound':bound})
metrics['grid_interpolation']=rows
sigma_probe=torch.tensor([1.,.1,.01,.001]);amp=.001/sigma_probe.clamp(min=1e-4)
metrics['anchor_error_amplification']={'sigma':sigma_probe.tolist(),'velocity_error_for_delta_a_0_001':amp.tolist()}
fig,ax=plt.subplots(figsize=(6,3));ax.loglog([r['grid']-1 for r in rows],[r['max_anchor_error'] for r in rows],'o-',label='Measured');ax.loglog([r['grid']-1 for r in rows],[r['curvature_bound'] for r in rows],'--',label='Curvature bound');ax.set(xlabel='G−1',ylabel='Anchor max error');ax.legend();fig.tight_layout();fig.savefig(figure_path(LAB,'grid_error'));plt.show()
''')
md(r'''
### Чувствительность около clean endpoint

Ниже одна и та же anchor error 0.001 превращается в разные velocity errors. Важно показывать это отдельно от interpolation accuracy: даже маленькая grid error может быть значима при малом σ. Clamp ограничивает denominator снизу, а не исправляет predicted anchor.
''')
code('''
fig,ax=plt.subplots(figsize=(6,3));ax.loglog(sigma_probe,amp,'o-');ax.set(xlabel='Sigma',ylabel='Velocity error for anchor error 0.001',title='DX sensitivity: 1 / max(sigma, eps)');fig.tight_layout();fig.savefig(figure_path(LAB,'endpoint_sensitivity'));plt.show()
''')
md(r'''
## 4. Polynomial mode: coefficients, не grid samples

Source alternative `mode='polynomial'` вычисляет:

$$
a(\tau)=\sum_{i=0}^{G-1}c_i(\tau_s-\tau)^i,\qquad
c_i=x_s-\sigma_s U_i.
$$

Здесь c_i — coefficients. Если все c_i равны z, результат обычно z(1+δτ+δτ²+...), а не constant z. Для constant polynomial нужны c_0=z,c_i=0 при i>0. Native K6 piflow sampler выбирает **grid**, polynomial — отдельная поддержанная math branch, не default released sampling recipe.

`copy` shallow: tensor storage разделяется. `detach_` отсоединяет только denoising_output_x_0, а не все tensor attributes. Это достаточно для stop-gradient по predicted policy coefficients, но x_t_src как attribute остаётся тем же объектом. Не называйте это deep clone.
''')
code('''
xs=torch.tensor([[1.2,.8]],dtype=torch.float64);x0=torch.tensor([[.3,-.2]],dtype=torch.float64)
coeff=torch.stack([x0,torch.zeros_like(x0),torch.zeros_like(x0)],1)
U=xs[:,None]-coeff
poly=pm.DXPolicy(U,xs,torch.ones(1,dtype=torch.float64),mode='polynomial')
xq=torch.tensor([[.6,.1]],dtype=torch.float64);sq=torch.tensor([.4],dtype=torch.float64)
assert torch.allclose(poly.pi(xq,sq),(xq-x0)/.4)
U=torch.randn(1,3,2,requires_grad=True)
p=pm.DXPolicy(U,xs.float(),torch.ones(1));shallow=p.copy();det=p.detach()
assert shallow.denoising_output_x_0.data_ptr()==p.denoising_output_x_0.data_ptr()
assert p.denoising_output_x_0.requires_grad and not det.denoising_output_x_0.requires_grad
assert det.x_t_src is p.x_t_src
metrics['copy_detach_contract']={'shared_storage':True,'coefficient_grad_detached':True,'deep_copy':False}
''')
md(r'''
## 5. Native policy rollout: discretization vs representation

Возьмём anchor a(σ)=a₀+bσ² и dX/dσ=(X−a(σ))/σ. Разделим на σ:

$$
\frac{d}{d\sigma}\frac{X}{\sigma}=-\frac{a_0}{\sigma^2}-b.
$$

Интеграция от 1 до σ даёт exact trajectory:

$$
X(\sigma)=a_0+\sigma(X_1-a_0+b)-b\sigma^2.
$$

Сравним native Euler-policy rollout до σ=0.2, а не только до 0: endpoint DX может скрывать intermediate trajectory error. При fixed grid всё более мелкий solver сначала уменьшает discretization error, затем упирается в representation error.

Source выбирает n_b=max(1,round((τ_s−τ_d)·total_substeps)); разные элементы batch могут иметь разные n. Loop идёт до max(n_b), `torch.where(active_mask,...)` замораживает завершённые элементы.
''')
code('''
a0=.3;b=.2;xs=torch.tensor([[1.2]],dtype=torch.float64);rows=[]
for G in [3,9,33]:
    grid_t=torch.linspace(0,1,G,dtype=torch.float64)
    anchors=(a0+b*grid_t.square())[None,:,None]
    p=pm.DXPolicy(xs[:,None]-anchors,xs,torch.ones(1,dtype=torch.float64),segment_size=1.)
    for steps in [8,32,128,512]:
        out,sig,_=pm.policy_rollout_fm(xs,torch.ones(1,dtype=torch.float64),torch.ones(1,dtype=torch.float64),torch.tensor([.2],dtype=torch.float64),steps,p)
        exact=a0+.2*(1.2-a0+b)-b*.2**2
        rows.append({'grid':G,'density':steps,'error':abs(float(out)-exact)})
metrics['policy_rollout_error']=rows
assert next(r['error'] for r in rows if r['grid']==33 and r['density']==512)<next(r['error'] for r in rows if r['grid']==33 and r['density']==8)
fig,ax=plt.subplots(figsize=(6,3.5))
for G in [3,9,33]:
    rr=[r for r in rows if r['grid']==G];ax.loglog([r['density'] for r in rr],[r['error'] for r in rr],'o-',label=f'G={G}')
ax.set(xlabel='Policy substep density over unit raw time',ylabel='Error at sigma=0.2');ax.legend();fig.tight_layout();fig.savefig(figure_path(LAB,'rollout_error'));plt.show()
''')
md(r'''
## 6. Разная длина rollout в одном batch

Source выполняет max(n_b) vectorized policy calls, а updates для завершённых rows отключает mask. Это отличается от суммы scalar calls по всем batch rows. Для одного grid anchor G=1 interpolation возвращает его без gather. Проверим обе ветви и terminal σ каждого элемента.
''')
code('''
single=torch.tensor([[[.3,.4]]]);assert torch.equal(pm.DXPolicy._interpolate(single,torch.tensor([[.8]])),single[:,0])
ss=torch.tensor([1.,.5],dtype=torch.float64);end=torch.tensor([.4,.25],dtype=torch.float64)
x0=torch.tensor([[.3],[.1]],dtype=torch.float64);u=torch.tensor([[.8],[-.3]],dtype=torch.float64)
xs=x0+ss[:,None]*u
p=pm.DXPolicy(u[:,None].expand(2,3,1).clone(),xs,ss,segment_size=ss-end)
class CountPolicy:
    shift=1.
    def __init__(self,policy):self.policy=policy;self.calls=0
    def pi(self,x,sigma):self.calls+=1;return self.policy.pi(x,sigma)
proxy=CountPolicy(p)
out,sigma,last_time=pm.policy_rollout_fm(xs,ss,ss,end,20,proxy)
assert torch.allclose(out,x0+end[:,None]*u,atol=1e-12)
assert torch.allclose(sigma.flatten(),end) and proxy.calls==12
metrics['heterogeneous_rollout']={'vectorized_policy_calls':proxy.calls,'active_steps_per_row':[12,5],
 'endpoint_error':float((out-x0-end[:,None]*u).abs().max()),'network_time_at_end':last_time.tolist()}
''')
finish('1. Выведите inverse warp и dx/dτ заново; где должен стоять Jacobian?\n2. Проверьте G=1, r<0, r>1 и endpoint interpolation.\n3. Сравните grid и polynomial для constant anchor.\n4. Добавьте batch с разными durations rollout и проверьте active_mask.\n5. Почему policy approximation error усиливается около σ=0?',
 'Все методы DXPolicy и shift/rollout проверены по формулам. Grid error, solver error и near-endpoint amplification различаются. Полный π-Flow paper включает и GMFlow policy; данный K6 source реализует DX, поэтому GMFlow theorem на него автоматически не переносится.')

start('08_piflow_grid_head_and_native_sampler','08. π-Flow operation: grid head → policies → native sampler',
 'Проследим expanded channels, normalize layouts, schedule, packed/batched policy math, actual NFE и короткий финальный сегмент.')
md(r'''
## 1. Physical channels и expanded head

Student head предсказывает G наборов параметров на physical coordinate. При video C_v=16, audio C_a=40 и G=10 output widths=160/400; state по-прежнему содержит 16/40 channels.

$$
\mathrm{head}_v:\ D_v\to p_t p_h p_w(GC_v),\qquad
\mathrm{head}_a:\ D_a\to GC_a.
$$

`PiFlowDiffusionTransformer3D` принимает **per-grid** C_v,C_a и сам умножает на G. HF config может уже хранить expanded widths; повторное умножение дало бы G²C. После forward reshape/movedim создаёт B,G,T,H,W,C_v и B,G,A,C_a.

`_normalize_grid_prediction` принимает canonical grid-first, single-output без grid (добавляет G=1) и legacy grid-last. Некорректные shapes отвергает. При одинаковых размерах разных axes одна shape проверка не устанавливает смысл осей; нужны известные index values и metadata.
''')
code('''
model=tiny_model(grid=3).eval();v,a,kw=tiny_inputs(model)
with torch.no_grad():pv,pa=fresh_forward(model,kw)
assert pv.shape==(1,3,2,4,4,4) and pa.shape==(1,3,6,4)
ps=native('core.algo.piflow_sampler')
canonical=torch.arange(1*3*2*4*4*4).reshape(1,3,2,4,4,4)
legacy=canonical.movedim(1,-2)
assert torch.equal(ps._normalize_grid_prediction(legacy,v,'video'),canonical)
assert ps._normalize_grid_prediction(v,v,'video').shape==(1,1,2,4,4,4)
try:ps._normalize_grid_prediction(torch.zeros(1,7,3),a,'audio')
except ValueError:pass
else:raise AssertionError('Invalid grid must be rejected')
metrics['native_grid_shapes']={'video':list(pv.shape),'audio':list(pa.shape),'physical_video_channels':v.shape[-1]}
show_source('core/components/piflow_dit.py');show_source('core/algo/piflow_sampler.py','_normalize_grid_prediction')
''')
md(r'''
## 2. Почему final segment короче

Пусть S network evaluations и final_step_size_scale=r>0. Требуем, чтобы S−1 обычных raw segments длины h и последний rh покрыли unit interval:

$$
(S-1)h+rh=1\quad\Longrightarrow\quad h=\frac{1}{S-1+r}.
$$

Native default r=.5. При S=10 обычный segment≈0.105263, final≈0.052632. Destination clamp eps оставляет маленький ненулевой final time. Если r>1, final segment длиннее; source не ограничивает r сверху. Это schedule parameter, не математическая необходимость .5.

Важно: num_policy_substeps — density **по единичному raw interval**. Каждый segment получает max(1,round(Δτ·density)). Поэтому суммарное число evaluations близко density, но не обязано равняться ему из-за rounding и minimum. Grid points G, network evaluations S и substep density K независимы.
''')
code('''
rows=[]
for S in [1,2,4,10]:
    h=1/(S-1+.5);tau=torch.tensor(1.)
    counts=[];sizes=[]
    for i in range(S):
        size=h*(.5 if i==S-1 else 1.);dst=(tau-size).clamp(min=1e-6)
        n=int(((tau-dst)*128).round().clamp(min=1));counts.append(n);sizes.append(float(tau-dst));tau=dst
    rows.append({'network_NFE':S,'grid_points_example':10,'policy_density':128,'policy_evaluations_per_modality':sum(counts),'per_segment':counts,'raw_lengths':sizes})
metrics['policy_schedule_counts']=rows
assert rows[-1]['per_segment']==[13]*9+[7] and rows[-1]['policy_evaluations_per_modality']==124
print(json.dumps(rows,indent=2))
''')
md(r'''
## 3. MultimodalDXPolicy: независимые policies, совместно предсказанные параметры

Один fused DiT call видит video/audio и выдаёт обе grids. Далее policy_V и policy_A интегрируются отдельно с собственными shapes. Внутри cheap policy substeps заново cross-attention не вычисляется: cross-modal зависимость уже вошла в source prediction grids. На следующем segment она обновится новым fused call.

Batched video имеет B,T,H,W,C; audio — B,A,D. В packed layout first axis=sum_T/sum_A: σ_s и raw interval повторяются по sequence lengths каждого batch item. Это math broadcasting contract, не доказательство, что любой ragged packed attention runtime изолирует clips без подходящих masks.

`detach` multimodal policy возвращает новый wrapper с detached coefficient tensors. Рассмотрим один прямой instance без network.
''')
code('''
pm=native('core.algo.piflow_math')
xs_v=torch.randn(2,3,2,2,4);xs_a=torch.randn(2,5,4);uv=torch.randn(2,3,3,2,2,4,requires_grad=True);ua=torch.randn(2,3,5,4,requires_grad=True)
p=pm.MultimodalDXPolicy(uv,xs_v,ua,xs_a,torch.ones(2),torch.full((2,),.5),torch.ones(2),torch.full((2,),.5))
det=p.detach();assert not det.policy_V.denoising_output_x_0.requires_grad and not det.policy_A.denoising_output_x_0.requires_grad
same=p.detach_();assert same is p and not p.policy_V.denoising_output_x_0.requires_grad
metrics['multimodal_detach']=True
''')
md(r'''
## 4. Полный native sampler с random tiny grid-DiT

Теперь исполним оригинальный sampler. Начальный noise и веса фиксированы, меняется только policy density. Для S=2 hook должен насчитать ровно 2 transformer calls при любой density. Runtime finite/shape проверки не означают осмысленное audio/video из random model.

Native sampler требует оба latent tensors и multimodal transformer. sample_video=False не удаляет video input и не делает audio-only released sampler. CFG у этой функции отсутствует; обычный CFG argument нельзя незаметно добавлять.
''')
code('''
Bundle=native('core.types').LatentBundle
bundle=Bundle(v.clone(),a.clone(),torch.tensor([0,2],dtype=torch.int32),torch.tensor([0,6],dtype=torch.int32))
rows=[]
for density in [8,32,128]:
    calls=[];hook=model.register_forward_hook(lambda *args:calls.append(1));model.clear_text_proj_cache()
    out=ps.piflow_denoise_loop(clone_bundle(bundle),model,{'text_embeds':kw['text_embed'],'pooled_embed':kw['pooled_text_embed']},
      kw['visual_rope'],kw['audio_rope'],kw['text_rope'],2,5.,attention_mask=kw['attention_mask'],num_policy_substeps=density)
    hook.remove()
    assert len(calls)==2 and out.video.shape==v.shape and out.audio.shape==a.shape
    assert torch.isfinite(out.video).all() and torch.isfinite(out.audio).all()
    rows.append({'density':density,'network_NFE':len(calls),'video_norm':float(out.video.norm()),'audio_norm':float(out.audio.norm())})
metrics['native_sampler_counts']=rows
# Source guards execute before any model call.
try:ps.piflow_denoise_loop(Bundle(v,None,None,None),model,dummy_text(),None,None,kw['text_rope'],2,1.)
except ValueError:metrics['missing_modality_rejected']=True
else:raise AssertionError('Native sampler requires both modalities')
''')
md(r'''
## 5. Ошибка compute budget и корректный итог

При S segments стоимость в первом приближении S·cost(DiT)+Σn_i·cost(policy). Это accounting approximation; overhead, output-grid width, память и kernels мешают прямому обещанию speedup. Увеличение cheap substeps не увеличивает NFE, но их стоимость не буквально нулевая.

Policy method family, output layout, schedule и checkpoint должны согласоваться. Переключить обычный velocity checkpoint на π sampler без обучения expanded head нельзя. В частности, количество grid channels не равно количеству дополнительных audio/video frames.
''')
code('''
fig,ax=plt.subplots(figsize=(7,3));ax.bar([str(r['network_NFE']) for r in metrics['policy_schedule_counts']],[r['policy_evaluations_per_modality'] for r in metrics['policy_schedule_counts']]);ax.axhline(128,ls='--',c='black',label='Configured density 128');ax.set(xlabel='Network NFE',ylabel='Actual policy evaluations per modality',title='Native rounding and final-segment schedule');ax.legend();fig.tight_layout();fig.savefig(figure_path(LAB,'substep_accounting'));plt.show()
''')
finish('1. Проверьте S=1,r=.5 аналитически: почему segment покрывает весь interval?\n2. Измените r и получите per-segment counts, не умножая S×128.\n3. Реализуйте index-valued legacy/canonical roundtrip с неравными axis lengths.\n4. Повторите native sampler с frozen audio и проверьте неизменность state.\n5. Какие два source-level действия отличают сетку output head от latent state?',
 'Native grid wrapper и sampler сохранили physical shapes; NFE не зависел от policy density. Для S=10,K=128,r=.5 получено 124 cheap evaluations на каждую modality. Multimodal policies detach проверен. Это execution/accounting evidence, не measured production acceleration.')

start('09_miniature_policy_imitation_distillation','09. Мини-исследование π-ID: frozen teacher → обучаемая DX policy',
 'Проведём маленький воспроизводимый training experiment с известным 2D teacher field, held-out probe и stop-gradient rollout. Промышленные weights не нужны.')
md(r'''
## 1. FM regression и imitation — разные targets

MIT CFM regresses доступный conditional target ε−z на data/noise paths. Здесь teacher Gθ уже задаёт поле; student Gφ выдаёт policy parameters η. На начале segment строим η=Gφ(x_s,σ_s), а промежуточные states получаем по **detached student policy**:

$$
\tilde x_\sigma=\mathrm{Rollout}(\mathrm{stopgrad}(\eta),x_s,\sigma_s\to\sigma).
$$

Затем сравниваем live policy velocity с frozen teacher:

$$
L(\phi)=\mathbb{E}\left[\|\pi_\eta(\mathrm{stopgrad}(\tilde x_\sigma),\sigma)
-\mathrm{stopgrad}(G_\theta(\tilde x_\sigma,\sigma))\|^2\right].
$$

Gradient идёт к live η, но не через teacher или state-generating rollout. Это позволяет обучать поведение на student-induced distribution. Fixed off-policy states и on-policy states отвечают на разные вопросы; ниже training on-policy, evaluation — дополнительно fixed held-out teacher states, чтобы изменение distribution не маскировало loss.

Первичный источник: [π-Flow §4 / Algorithm 1](https://arxiv.org/html/2510.14974v1#S4). Наш optimizer/dataset/MLP — учебная постановка, не reproduction K6 recipe и не release training script. В paper есть также micro-window loss и GMFlow; здесь изучаем базовый instantaneous DX objective.
''')
code('''
pm=native('core.algo.piflow_math');torch.manual_seed(23)
A=torch.tensor([[.7,.2],[.3,.4]])
teacher=nn.Linear(2,2,bias=False);teacher.weight.data.copy_(A);teacher.requires_grad_(False)
G=5
student=nn.Sequential(nn.Linear(3,64),nn.SiLU(),nn.Linear(64,G*2))
opt=torch.optim.Adam(student.parameters(),lr=.003)
def parameters_for(xs,ss):return student(torch.cat((xs,ss[:,None]),-1)).reshape(-1,G,2)
def draw(batch,generator=None):
    xs=torch.randn(batch,2,generator=generator)
    ss=torch.tensor([.25,.5,.75,1.])[torch.randint(0,4,(batch,),generator=generator)]
    query=ss-.25*(.05+.9*torch.rand(batch,generator=generator))
    return xs,ss,query
# Held-out values fixed before training; independent generator.
xs_h,ss_h,tq_h=draw(512,torch.Generator().manual_seed(1001))
xq_h=(torch.matrix_exp((tq_h-ss_h)[:,None,None]*A)@xs_h[:,:,None]).squeeze(-1)
with torch.no_grad():target_h=teacher(xq_h)
def fixed_probe():
    with torch.no_grad():
        p=pm.DXPolicy(parameters_for(xs_h,ss_h),xs_h,ss_h,segment_size=.25)
        return float(F.mse_loss(p.pi(xq_h,tq_h),target_h))
before=fixed_probe();curve=[];teacher_before=teacher.weight.detach().clone()
''')
md(r'''
## 2. Train → measure, без утечки held-out target

Каждый training step получает новые source states и времена. Исходный `policy_rollout_fm` интегрирует detached policy до query time. Teacher evaluated там, где оказался student; training query states не берутся из аналитической teacher trajectory. Зафиксированный held-out probe используется только для оценки.

Проверяем routing градиентов: teacher.grad=None, rollout state requires_grad=False, live policy coefficients передают gradient student weights. Уменьшение loss свидетельствует об imitation на этом toy field, а не о generation quality K6.
''')
code('''
for step in range(500):
    xs,ss,tq=draw(64)
    p=pm.DXPolicy(parameters_for(xs,ss),xs,ss,segment_size=.25)
    with torch.no_grad():
        xq,sigma,_=pm.policy_rollout_fm(xs,ss,ss,tq,32,p.detach())
        target=teacher(xq)
    pred=p.pi(xq,sigma.flatten())
    loss=F.mse_loss(pred,target)
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),10.);opt.step()
    curve.append(float(loss.detach()))
after=fixed_probe()
assert not xq.requires_grad and teacher.weight.grad is None and torch.equal(teacher.weight,teacher_before)
assert after<before*.8 and np.isfinite(curve).all()
metrics['toy_pi_ID']={'steps':500,'grid_points':G,'heldout_before':before,'heldout_after':after,
 'improvement_factor':before/max(after,1e-30),'first_50_train_mean':float(np.mean(curve[:50])),
 'last_50_train_mean':float(np.mean(curve[-50:])),'teacher_frozen':True,'rollout_detached':True,
 'teacher_training_batch_calls':500,'teacher_training_states':500*64,'student_source_training_batch_calls':500}
print(metrics['toy_pi_ID'])
fig,axes=plt.subplots(1,2,figsize=(10,3));axes[0].plot(curve,lw=.8);axes[0].set(xlabel='Optimizer step',ylabel='On-policy velocity MSE')
axes[1].bar(['Before','After'],[before,after]);axes[1].set(yscale='log',ylabel='Fixed held-out velocity MSE');fig.tight_layout();fig.savefig(figure_path(LAB,'imitation_training'));plt.show()
''')
md(r'''
## 3. От velocity loss к trajectory error: что требует доказательства

Если teacher Lipschitz по x с константой L и policy error ≤ε на всём relevant rollout domain, Grönwall даёт на interval длины Δ:

$$
\|x_{student}(\sigma)-x_{teacher}(\sigma)\|
\leq e^{L\Delta}\|\delta x_s\|+\frac{e^{L\Delta}-1}{L}\varepsilon
+\mathrm{integration\ error}.
$$

Это bound с предпосылками, а не автоматическое следствие малого **среднего** train loss. Надо проверять held-out trajectories и near-endpoint error отдельно. У DX около σ=0 Lipschitz/anchor sensitivity может расти; eps задаёт numerical clamp, но не доказывает fidelity.

Измерим trajectory states на held-out sources до нескольких positive query times, используя source interval .25. Teacher exact matrix exponential. В отличие от только instantaneous regression, это проверяет накопленную ошибку rollout.
''')
code('''
rows=[]
with torch.no_grad():
    xs=xs_h[:64];ss=ss_h[:64]
    p=pm.DXPolicy(parameters_for(xs,ss),xs,ss,segment_size=.25)
    for fraction in [.25,.5,.9]:
        tq=ss-.25*fraction
        out,_,_=pm.policy_rollout_fm(xs,ss,ss,tq,128,p.detach())
        exact=(torch.matrix_exp((tq-ss)[:,None,None]*A)@xs[:,:,None]).squeeze(-1)
        rows.append({'fraction_of_segment':fraction,'state_MSE':float(F.mse_loss(out,exact)),
                     'network_calls_per_source_segment':1,'policy_density':128})
metrics['heldout_trajectory_probe']=rows
for row in rows:print(row)
''')
md('''
## 4. Что в этом эксперименте не воспроизведено

K6 §4.5 описывает initialization из pretrained backbone, отдельные training regimes и adversarial post-training. Здесь teacher — известная матрица, student — маленький MLP, нет text/video/audio данных, real teacher CFG, perceptual discriminator или Sim-LADD. Мы проверили механизм gradient/rollout/imitation и получили численные результаты с явным scope.

Для применения к CrossDiT понадобятся paired modalities, реальные pretrained heads/parameter layouts, teacher-query accounting, memory budget и held-out perceptual/regression metrics. Меньше NFE без соответствующего обучения не является distillation.
''')
finish('1. Повторите с 3 seeds и сравните fixed held-out loss, не только train curve.\n2. Измените G при том же optimizer budget.\n3. Сравните off-policy и on-policy state distributions на одной teacher matrix.\n4. Уберите detach только в отдельном опыте и измерьте memory/gradient changes; не называйте это тем же Alg.1.\n5. Почему held-out mean loss недостаточен для uniform Grönwall bound?',
 'Frozen teacher сохранил веса, rollout был detached, а live policy обучалась. Fixed held-out velocity loss уменьшился; trajectory error измерена отдельно. Это маленький numerical imitation experiment, не Kandinsky distillation training и не proof of pretrained quality.')

start('10_comfyui_and_manual_research_protocol','10. ComfyUI → code tracing → измеримый архитектурный эксперимент',
 'Свяжем визуальный graph с разобранными hidden blocks, составим ручной measurement protocol и отделим функциональные CPU проверки от real inference.')
md('''
## 1. Где блоки находятся в ComfyUI graph

Используйте существующий [K6 Lite Smoke workflow](../../kandinsky-6/workflows/k6_lite_distill_smoke.json) и [инструкцию Desktop](../../kandinsky-6/COMFYUI_AND_EXPERIMENT_RU.md). Graph визуализирует loader → text encode → joint latent → sampler → video/audio decode → mux. Внутренние SA/CA/gates/policy не становятся отдельными Comfy nodes автоматически.

| UI участок | Разобранный code path | Что можно наблюдать |
|---|---|---|
| Dual text loader / prompt nodes | Text encoder → DiT projections/refiners | Prompt, masks/source type; не внутренние trained gates |
| Empty joint latent / first image | Conditioning / visual_conditioning | Physical shapes, references, frame count |
| K6 Sampler | denoise_loop или π sampler по checkpoint | Calls/steps и pipeline time, при корректном runtime tracing |
| Video/audio VAE decode | Отдельные codecs | RGB/audio duration, reconstruction artifacts |
| Create/Save Video | Postprocess/mux | Container fps/sample rate, sync и trimming |

На данном Mac pretrained inference отложен до 50 GB+ SSD. Этот notebook не скачивает weights, не устанавливает MLX и не отправляет jobs в ComfyUI. Веса полного Pro/SR/Qwen не нужны для CPU labs 00–09.
''')
code('''
workflow=COURSE.parent/'kandinsky-6/workflows/k6_lite_distill_smoke.json'
graph=json.loads(workflow.read_text())
metrics['comfy_static']={'workflow':str(workflow.relative_to(COURSE.parent)),
 'nodes':len(graph['nodes']),'links':len(graph['links']),'jobs_submitted':0,'runtime_verified':False}
for node in graph['nodes']:print(node['id'],node['type'],node.get('title',''))
''')
md('''
## 2. Ручной baseline после resource gate

1. Зафиксируйте checkpoint SHA, source revision, architecture class, G/physical channels, text/codec versions, dtype/device. Используйте Lite и уже скачанные совместимые weights; не соглашайтесь автоматически на ненужные Pro/SR downloads.
2. Зафиксируйте seed42, prompt, clip size, FPS, recommended distilled steps/CFG=1. Число steps в UI не доказывает точный transformer NFE: проверьте исполненный sampler и forward hooks/logs.
3. Выполните warmup, затем 3 настоящих runs. Обойдите Comfy cache: неизменённые nodes могут не исполниться. Разделите text encode, transformer/cheap policy, VAE decode, mux и total wall time.
4. Запишите calls и policy counts отдельно. Для native DX schedule считайте rounding по каждому interval; не используйте S×128. В рамках cheap substeps network forward отсутствует, но runtime overhead всё равно измеряется.
5. Для architecture ablations используйте отдельную Python ветку с full weights и hooks. UI смена prompt не изолирует cross gate; CPU lab03 показывает контролируемое отключение A→V/V→A.

На CUDA измерьте synchronized latency, peak allocated и reserved. На MPS point counters before/after не называйте peak; 48 GB unified memory включает macOS/CPU/GPU. Не переносите H100 latency на M3 Max. MLX port потребует совместимости всего computation graph и numerical parity — checkpoint conversion сама её не гарантирует.
''')
md('''
## 3. Карточка одного исследования

Гипотеза A: при fixed weights/source inputs отключение A→V уничтожает локальную audio→video dependency, но не обязано улучшить/ухудшить global perceptual quality предсказуемым образом. CPU gradient evidence есть в lab03; реальное влияние на sync требует clips и метрик.

Гипотеза B: при фиксированном trained policy и network NFE увеличение policy substeps уменьшает discretization error до representation floor. На analytic case проверено в lab07; на real model нужны paired outputs и teacher-reference trajectory/quality evidence.

Не меняйте одновременно CFG, checkpoint, resolution и policy density. Не сравнивайте случайную tiny сеть с trained checkpoint как quality baseline. Каждое conclusion должно указывать dataset/input, intervention, budget и measurement units.
''')
code('''
columns=['run_id','source_revision','checkpoint_sha256','model_class','device','dtype','seed',
 'video_shape','audio_shape','grid_points','network_NFE','policy_evaluations_video',
 'policy_evaluations_audio','num_steps_UI','scheduler_shift','final_step_scale',
 'CFG','reference_count','cross_gate_intervention','text_seconds','DiT_seconds',
 'policy_seconds','decode_seconds','mux_seconds','total_seconds',
 'CUDA_peak_allocated_bytes','CUDA_peak_reserved_bytes','MPS_point_counters',
 'metric_name','metric_value','status','limits']
metrics['manual_record_columns']=columns
metrics['research_status']={'CPU_labs':'executed separately by validator','pretrained':'not_run','ComfyUI_runtime':'not_run','MLX_parity':'not_run'}
print(','.join(columns))
''')
finish('1. Объясните через UI graph, где заканчивается conditioning и начинается neural field evaluation.\n2. Напишите ожидаемые NFE/policy counts до запуска, затем сравните с logs.\n3. Для одной выбранной гипотезы заполните [общий research template](../../curriculum/EXPERIMENT_TEMPLATE.md).\n4. Выберите teacher-reference и no-regression критерии до получения результатов.',
 'Workflow прочитан статически, составлена схема измерений. Real generation не выполнена. Исследовательский результат сейчас — algebra, native tiny parity, controlled interventions и toy imitation; real AV quality/memory/latency — отдельный следующий этап.')

print('Built',len(list((C/'notebooks').glob('*.ipynb'))),'DiT/CrossDiT/π-Flow notebooks')
