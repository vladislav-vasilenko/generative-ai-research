"""Generate the sequential HunyuanVideo VAE course; preserve notebook code in this source."""
from pathlib import Path
import textwrap
import nbformat as nbf
C=Path(__file__).resolve().parents[1]
cells=[]
LAB=''
MIT_LINKS={
 '00':'**MIT до этой lab:** Lecture 4 slides 3–19; notes §6.2 pp.46–52. Посмотрите исходный [Lab 3 Part 4](../../lecture_04/references/lab_three_original.ipynb), zero-based cells 55–69; здесь мы строим общую карту.',
 '01':'**MIT параллельно:** Lab 3 Q4.7 (cells 68–69), `VAE.forward/compute_loss`; notes §6.2 pp.46–52, Appendix D pp.77–82. После KL и reparameterization допишите свою версию compute_loss; сверяйте reduction и смысл x_logvar. [Задание и checklist](../MIT_LAB3_BRIDGE.md).',
 '02':'**MIT мост:** Lab 3 Q4.1 ResidualBlock (56–57), Q4.3 EncoderBlock (60–61), Q4.5 DecoderBlock (64–65). В MIT — 2D MNIST; здесь добавлены time axis, causal padding и first-frame rule. Tiling/streaming — сверх MIT. [Точные соответствия](../MIT_LAB3_BRIDGE.md).',
 '03':'**MIT мост:** Q4.2 AttentionBlock (58–59), Q4.4 Encoder (62–63), Q4.6 Decoder (66–67), Q4.7 VAE (68–69), затем Q5.1 LatentCFGTrainer (76–77). Сравните tuple(mean,logvar) из MIT с distribution API промышленной VAE. [Контракты](../MIT_LAB3_BRIDGE.md).',
 '04':'**MIT предпосылка:** закончите Lab 3 Part 4 и получите full reconstruction. Spatial tiling отсутствует среди Q4.1–4.7 и является промышленным расширением. Сравниваем с тем же full baseline. [Переход к практике](../MIT_LAB3_BRIDGE.md).',
 '05':'**MIT предпосылка:** Lab 3 Q4.3/Q4.5 объясняют down/up sampling; переход от image к temporal video — дополнительная тема. Frame lattice и temporal context не следует приписывать исходному MNIST заданию. [Мост image→video](../MIT_LAB3_BRIDGE.md).',
 '06':'**MIT предпосылка:** Part 4–5 разделяет codec и latent model. Free-memory policy и GPU counters не являются задачами исходного Lab 3: здесь начинается deployment extension. [Карта границ](../MIT_LAB3_BRIDGE.md).',
 '07':'**MIT связь:** после VAE training (cells 70–74) зафиксируйте reconstruction baseline; после Q5.1 (76–77) отдельно оценивайте codec и DiT. Здесь benchmark codec, без generative ODE. [Критерии перехода](../MIT_LAB3_BRIDGE.md).',
 '08':'**MIT итог:** Part 4 VAE и Part 5 latent DiT → промышленный frozen codec. ComfyUI показывает reconstruction; он не вычисляет ELBO и не заменяет реализацию Q4.7. [Полный маршрут](../MIT_LAB3_BRIDGE.md).'}
def md(s):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()+'\n'))
def code(s):cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()+'\n'))
def start(slug,title,goal):
 global cells,LAB
 cells=[];LAB=slug
 md(f'# {title}\n\n{goal}\n\n[Порядок и источники](../README.md) · [Связь с MIT Lab 3](../MIT_LAB3_BRIDGE.md) · [Все компоненты и методы](../SOURCE_MAP.md) · [Результаты](../reports/VALIDATION.md).\n\nРаботаем по циклу **формула → прогноз → код → измерение → вывод**. Все ячейки по умолчанию используют CPU; никакие веса, датасеты или ComfyUI jobs не скачиваются/запускаются. Исходные алгоритмы и учебные адаптеры явно различаются. Перед запуском численного опыта сформулируйте ожидаемый результат.')
 md(MIT_LINKS[slug[:2]])
 code(f'''
 from pathlib import Path
 import sys, json, math
 import numpy as np
 import torch
 from torch import nn
 import torch.nn.functional as F
 import matplotlib.pyplot as plt
 candidates=[Path.cwd(),*Path.cwd().parents]
 COURSE=next((p/'vae' for p in candidates if (p/'vae/labkit').is_dir()),None)
 assert COURSE is not None, 'Clone the repository first; see the Colab instructions in the course README. No weights are needed.'
 sys.path.insert(0,str(COURSE))
 from labkit.vae_lab import (NATIVE,TinyCodec,DiagonalPosterior,oracle_codec,plan,
     axis_coverage,anchored_starts,synthetic_video,errors,measure,show_source,
     source_hash,save_metrics,figure_path)
 torch.set_num_threads(2)
 torch.manual_seed(42)
 LAB={slug!r};metrics={{}}
 print('Pinned source:',source_hash())
 ''')
def finish(exercises):
 md('## Упражнения и критерий перехода\n\n'+exercises+'\n\nСначала запишите прогноз, затем меняйте один фактор. Разделяйте ошибку аппроксимации, ошибку интерфейса/формы и аппаратное ограничение. Сохранённые метрики относятся к этому запуску.')
 code("result=save_metrics(LAB,metrics)\nprint(json.dumps(result,ensure_ascii=False,indent=2))")
 md('## Источники\n\n[HunyuanVideo, v6, §4.1 / Fig. 6 / Eq. 1, стр. 5–6](https://arxiv.org/pdf/2412.03603v6) · [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) · [Официальный HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) · [Закреплённый исследуемый исходник](../../kandinsky-6/reference/kandinsky-6/kandinsky/core/components/hunyuan_vae.py) · [Lecture 4 / исходный MIT Lab 3](../../lecture_04/README.md).\n\nМатематические выводы и малые опыты — авторские пояснения. Числа tiny-модели не являются оценкой pretrained HunyuanVideo/Kandinsky. Подробная provenance и ограничения — в SOURCE_MAP.')
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (root .venv)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'hunyuan_vae_lab':{'source_sha256':'f5e8dd402aba625f23aef86ab7d5324dc6a4e8585b72917a3bf27bd46745bbaf','default_execution':'CPU; no pretrained weights'}})
 for i,c in enumerate(nb.cells):c['id']=f'{LAB[:2]}-{i:03d}'
 nbf.write(nb,C/'notebooks'/f'{LAB}.ipynb')

start('00_route_and_source_contracts','00. Маршрут: видео → posterior → latent → видео',
      'Цель: отделить вероятностную модель, реализацию кодека, политику памяти и качество весов. После MIT Lecture 4 начните здесь; потом последовательно 01–08.')
md(r'''
## 1. Четыре разных объекта исследования

1. **Статья v6**: §4.1 описывает causal 3D VAE, факторы сжатия; §4.1.1 — обучение с perceptual/adversarial losses; §4.1.2 — перекрывающиеся тайлы и дополнительный finetuning с tiling.
2. **Приложенный `hunyuan_vae.py`**: адаптация из Kandinsky с выбором spatial tiles по CUDA free bytes. Он побайтово совпал с уже закреплённым K6 snapshot; повторной копии не нужно.
3. **Native tiny**: оригинальные encoder/decoder blocks с уменьшенными ширинами, случайными весами, без optional mid attention. Это проверка вычислений/форм и свойств операторов.
4. **Pretrained reconstruction**: требует настоящего codec checkpoint и отдельного аппаратного запуска. Он отложен; мы не выдаём tiny reconstruction за результат модели.

Исходные `tiled_*` методы исполняются из AST без изменения тела. Для posterior/output используются подписанные учебные dependency adapters. Для planner CUDA budget имитируется только в локальном namespace. Весь проект не патчит `torch` глобально.
''')
code('''
import hashlib
attached=Path('/Users/vladmac/Downloads/hunyuan_vae.py')
if attached.exists():
    assert hashlib.sha256(attached.read_bytes()).hexdigest()==source_hash()
    print('Attached file == pinned source')
manifest=json.loads((COURSE/'sources.json').read_text())
print(json.dumps(manifest['paper'],ensure_ascii=False,indent=2))
metrics['attached_source_sha256']=source_hash()
''')
md(r'''
## 2. Геометрия и единицы

В статье число кадров записано как T+1. Здесь используем F, чтобы не путать frame index, duration и коэффициент сжатия:

$$
X\in\mathbb{R}^{B\times3\times F\times H\times W},\qquad
M=E_\phi(X)\in\mathbb{R}^{B\times32\times F_z\times H_z\times W_z}.
$$

32 канала — два набора по 16: mean и log variance. Это **не 32 latent channels**.

$$
F_z=\lfloor(F-1)/4\rfloor+1,\quad H_z=H/8,\quad W_z=W/8.
$$

Равенство для H/W здесь предполагает делимость на 8; реальная свёртка с нечётным размером требует ceil и проверки. Для обратимого размера выбираем F=4k+1 и H/W кратные 8. Decoder возвращает 4(F_z−1)+1; произвольный F без padding/crop может не восстановить длину.

$$
N_X=3FHW,\qquad N_Z=16F_zH_zW_z,\qquad R=N_X/N_Z.
$$

R — сжатие числа **скаляров**, не ratio размера checkpoint и не число DiT tokens. Token patchification — следующий компонент.
''')
code('''
F,H,W=121,480,864
fz=(F-1)//4+1
nx,nz=3*F*H*W,16*fz*(H//8)*(W//8)
metrics['example']={'BCTHW':[1,3,F,H,W],'latent':[1,16,fz,H//8,W//8],
 'scalar_compression':nx/nz,'input_fp32_MiB':nx*4/2**20,'latent_fp32_MiB':nz*4/2**20}
assert 4*(fz-1)+1==F
print(metrics['example'])
''')
md(r'''
## 3. План накопления знаний

| Lab | Математика | Код / измерение | Что объяснить своими словами |
|---|---|---|---|
| 01 | ELBO, Gaussian KL, reparameterization | MC integral, gradient, small VAE | Почему sample и mode различаются |
| 02 | Conv3D, receptive field, normalization | Native conv, mask, future perturbation | Где кончается строгая причинность |
| 03 | Encoder/decoder, posterior API | Native tiny hooks, contract check | Где появляются 32 и 16 каналов |
| 04 | Partition of unity, halo | Native spatial stitch, border coverage | Почему blend не восстанавливает контекст |
| 05 | Temporal lattice, first frame | Native temporal stitch, frame counts | Зачем F=4k+1 и почему этого мало для tile loop |
| 06 | Memory model, integer partition | Original planner budget sweep | Почему «optimal» не доказывает OOM safety |
| 07 | Controlled comparison | Three seeds, seams, CPU time, plots | Что числа позволяют заключить |
| 08 | Pretrained/manual validation | No-download runner + ComfyUI protocol | Как получить реальные memory/quality results |

Для связи с промышленной генерацией: [K5 VAE](../../kandinsky-5/notebooks/02_vae.ipynb), [K6 video codec](../../kandinsky-6/notebooks/03_video_vae.ipynb), [K6 latents/wrappers](../../kandinsky-6/notebooks/12_latents_conditioning_and_output.ipynb). VAE не решает Flow Matching ODE; он задаёт пространство, в котором DiT учит поле скоростей.
''')
finish('1. Посчитайте формы для F=1, 33, 122 и объясните потерю длины без padding.\n2. Укажите, какие параметры обучаются при обучении VAE и какие frozen при DiT training.\n3. Прочитайте Fig. 6 и сопоставьте каждый блок с исходником. Переходите дальше, когда можете отделить posterior parameters от latent sample.')

start('01_elbo_and_probabilistic_codec','01. VAE: от вероятностной модели к обучению',
      'Выведем ELBO и diagonal Gaussian KL, свяжем непрерывную плотность с конечным тензором и проверим reparameterization градиентами.')
md(r'''
## 1. Почему нельзя просто интегрировать decoder

Порождающая модель: сначала z из prior, затем x из decoder likelihood. Encoder q — обучаемое приближение недоступного posterior.

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz,
\qquad q_\phi(z\mid x)\approx p_\theta(z\mid x).
$$

Домножим подынтегральную функцию на q/q (там, где q положительна) и применим вогнутость log:

$$
\log p_\theta(x)=\log\mathbb{E}_{q_\phi}\left[\frac{p_\theta(x\mid z)p(z)}{q_\phi(z\mid x)}\right]
\geq \mathbb{E}_{q_\phi}[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}(q_\phi\Vert p)=\mathcal{L}.
$$

Точная разница:

$$
\log p_\theta(x)-\mathcal{L}=D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p_\theta(z\mid x))\geq0.
$$

Чем точнее posterior, тем теснее bound. Для scalar примера p(z)=N(0,1), p(x|z)=N(z,s²) marginal известна: N(0,1+s²). Проверим bound и разрыв Monte Carlo, а не на словах.
''')
code('''
torch.manual_seed(1);dtype=torch.float64
x=torch.tensor(.7,dtype=dtype);s=torch.tensor(.4,dtype=dtype)
q=torch.distributions.Normal(torch.tensor(.2,dtype=dtype),torch.tensor(.6,dtype=dtype))
prior=torch.distributions.Normal(torch.tensor(0.,dtype=dtype),torch.tensor(1.,dtype=dtype))
z=q.sample((200000,))
term=torch.distributions.Normal(z,s).log_prob(x)+prior.log_prob(z)-q.log_prob(z)
elbo=float(term.mean());logp=float(torch.distributions.Normal(0.,torch.sqrt(1+s*s)).log_prob(x))
se=float(term.std()/math.sqrt(len(term)))
assert elbo < logp+5*se
metrics['elbo']={'MC':elbo,'standard_error':se,'exact_log_marginal':logp,'gap':logp-elbo}
print(metrics['elbo'])
''')
md(r'''
## 2. Gaussian posterior: вывод KL по координатам

Для каждого скаляра q=N(μ,σ²), p=N(0,1). Разность log densities:

$$
\log q-\log p=-\log\sigma-\frac{(z-\mu)^2}{2\sigma^2}+\frac{z^2}{2}.
$$

Под q имеем E[(z−μ)²]=σ² и E[z²]=μ²+σ². Поэтому:

$$
D_{\mathrm{KL}}(q\Vert p)=\tfrac12(\mu^2+\sigma^2-1-\log\sigma^2).
$$

Для диагональной ковариации многомерная log density — сумма, значит KL суммируется по C,F,H,W отдельно для каждого B:

$$
\ell=\log\sigma^2,\quad \sigma=\exp(\ell/2),\quad
D_{\mathrm{KL}}=\tfrac12\sum_i(\mu_i^2+\exp(\ell_i)-1-\ell_i).
$$

Проверьте: std не равна exp(logvar); mean reduction и sum reduction дают разные масштабы β. Код source clamp logvar ограничивает численную динамику; это не часть исходного вывода Gaussian KL.
''')
code('''
moments=torch.randn(2,32,3,4,4)*.2
posterior=DiagonalPosterior(moments)
q=torch.distributions.Normal(posterior.mean,posterior.std)
p=torch.distributions.Normal(torch.zeros_like(posterior.mean),torch.ones_like(posterior.std))
expected=torch.distributions.kl_divergence(q,p).flatten(1).sum(1)
assert torch.allclose(posterior.kl(),expected,atol=1e-4)
metrics['kl_max_error']=float((posterior.kl()-expected).abs().max())
print('moments / mean / KL:',moments.shape,posterior.mean.shape,posterior.kl())
''')
md(r'''
## 3. Как gradient проходит через sample

Замена переменной переносит случайность в независимый ε:

$$
z=\mu_\phi(x)+\exp(\ell_\phi(x)/2)\odot\epsilon,\qquad \epsilon\sim N(0,I).
$$

Для одного фиксированного ε:

$$
\frac{\partial z}{\partial\mu}=1,\qquad
\frac{\partial z}{\partial\ell}=\tfrac12\exp(\ell/2)\epsilon.
$$

Так pathwise estimator дифференцирует deterministic graph. Сравним autograd с finite difference при **том же ε**, иначе шум замаскирует derivative. При mode z=μ encoder uncertainty не используется; это полезный deterministic baseline реконструкции.
''')
code('''
mu=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
lv=torch.tensor(-.7,dtype=torch.float64,requires_grad=True);eps=torch.tensor(.8,dtype=torch.float64)
def loss(m,l):return (m+torch.exp(.5*l)*eps-1).square()
g=torch.autograd.grad(loss(mu,lv),(mu,lv));h=1e-5
fd=[(loss(mu+h,lv)-loss(mu-h,lv))/(2*h),(loss(mu,lv+h)-loss(mu,lv-h))/(2*h)]
assert all(torch.allclose(a,b,atol=1e-8) for a,b in zip(g,fd))
metrics['gradient_max_error']=max(float((a-b).abs().detach()) for a,b in zip(g,fd))
print('autograd:',[float(v) for v in g],'finite difference:',[float(v.detach()) for v in fd])
''')
md(r'''
## 4. От likelihood к loss HunyuanVideo

Независимая Gaussian likelihood с фиксированным σₓ даёт squared reconstruction:

$$
-\log p_\theta(x\mid z)=\frac{1}{2\sigma_x^2}\sum_i(x_i-\hat x_i)^2+\mathrm{const}.
$$

Для Laplace likelihood с scale b получается L1/b плюс константа. В отчёте HunyuanVideo v6 Eq. 1 используется:

$$
L=L_1+0.1L_{\mathrm{LPIPS}}+0.05L_{\mathrm{adv}}+10^{-6}L_{\mathrm{KL}}.
$$

Perceptual и adversarial terms расширяют цель за пределы простого ELBO. По одним коэффициентам нельзя воспроизвести весь training recipe: нужны reductions, discriminator, данные, optimizer. При обучении DiT pretrained VAE обычно frozen, его KL loss повторно не оптимизируется.

Обучим крошечный **linear VAE на 2D данных** с MSE+βKL: это упражнение на механизм, не реплика Hunyuan training. Данные x — конечномерный вектор вместо видеотензора; сумма формально та же.
''')
code('''
torch.manual_seed(6)
data=torch.stack((torch.linspace(-1,1,128),torch.linspace(-1,1,128)**2),1)
enc=nn.Linear(2,4);dec=nn.Linear(2,2);opt=torch.optim.Adam([*enc.parameters(),*dec.parameters()],lr=.025)
curve=[]
for step in range(300):
    mu,lv=enc(data).chunk(2,-1);lv=lv.clamp(-8,4)
    z=mu+torch.exp(.5*lv)*torch.randn_like(mu)
    rec=F.mse_loss(dec(z),data)
    kl=.5*(mu.square()+lv.exp()-1-lv).sum(-1).mean()
    loss=rec+.01*kl;opt.zero_grad();loss.backward();opt.step()
    curve.append(float(loss.detach()))
metrics['linear_vae']={'first_30_mean':float(np.mean(curve[:30])),'last_30_mean':float(np.mean(curve[-30:])),'beta':.01}
assert np.mean(curve[-30:])<np.mean(curve[:30])
fig,ax=plt.subplots(figsize=(7,3));ax.plot(curve);ax.set(xlabel='Optimizer step',ylabel='MSE + 0.01 KL',title='Educational 2D linear VAE')
fig.tight_layout();fig.savefig(figure_path(LAB,'training'));plt.show()
''')
md(r'''
## 5. Возвращаемся в MIT Q4.7: reconstruction с learned variance

В MIT Lab 3 `decode` выдаёт x_mean и x_logvar. Условная likelihood по каждому pixel — N(m_i(z),exp(r_i(z))). Тогда с учётом normalizing constant:

$$
-\log p(x\mid z)=\frac12\sum_i\left[\log(2\pi)+r_i+(x_i-m_i)^2\exp(-r_i)\right].
$$

Сначала суммируем по C/H/W (или C/F/H/W для видео), затем усредняем по batch. Это NLL на пример. KL также sum по latent dimensions, mean по batch:

$$
L=\mathbb{E}_{B}[\mathrm{NLL}(x\mid z)+\beta D_{\mathrm{KL}}(q(z\mid x)\Vert p(z))].
$$

MSE эквивалентна Gaussian NLL лишь при фиксированной variance и согласованном масштабе. Без члена r_i learned variance могла бы просто расти, уменьшая weighted error; normalizer штрафует это. x_logvar decoder — uncertainty likelihood, z_logvar encoder — uncertainty posterior. В HunyuanVideo применяется другая reconstruction objective (Eq. 1); одинаковой learned output variance требовать нельзя.

Теперь допишите `VAE.compute_loss` в **своей копии MIT Lab 3, cell 69**. Ниже проверка формулы, не замена canonical задания готовыми ответами.
''')
code('''
torch.manual_seed(12)
x_true=torch.randn(3,1,8,8,dtype=torch.float64)
x_mean=torch.randn_like(x_true);x_logvar=torch.randn_like(x_true)*.2
per_pixel=.5*(math.log(2*math.pi)+x_logvar+(x_true-x_mean).square()*torch.exp(-x_logvar))
nll=per_pixel.flatten(1).sum(1)
reference=-torch.distributions.Normal(x_mean,torch.exp(.5*x_logvar)).log_prob(x_true).flatten(1).sum(1)
assert torch.allclose(nll,reference,atol=1e-10)
metrics['MIT_Q4_7_NLL_max_error']=float((nll-reference).abs().max())
print('Gaussian NLL per example:',nll.tolist())
''')
finish('1. Выведите KL для prior N(m₀,s₀²), затем проверьте через torch.distributions.\n2. Увеличьте β в 10 раз: сравните reconstruction, KL и latent variance на held-out точках.\n3. Почему качество decoder(mode) не равно likelihood estimate? Укажите MC uncertainty. Переход: самостоятельно получите знак и все члены KL без запоминания.')

start('02_causal_convolution_and_geometry','02. Causal Conv3D, размеры и пределы причинности',
      'Из непрерывной идеи зависимости только от прошлого получим дискретную свёртку, padding, down/up sampling и проверим строгую причинность каждого оператора.')
md(r'''
## 1. Дискретная причинная свёртка

Для фиксированных каналов и spatial coordinates causal filter использует x[t], x[t−1], …:

$$
y_{b,o,t,h,w}=b_o+\sum_{c,i,j,k}W_{o,c,i,j,k}\,
 x_{b,c,\,s_t t+i-p_t,\,s_h h+j-p_h,\,s_w w+k-p_w}.
$$

При temporal kernel K и dilation d нужна левая добавка pₜ=d(K−1), правая 0. Тогда самый поздний input index равен sₜt. Spatial padding симметричен. `replicate` повторяет первый кадр, а не создаёт физически существовавшее прошлое.

Размер одной оси (Pₗ/Pᵣ — F.pad, p — внутренний padding Conv3d):

$$
L_{out}=\left\lfloor\frac{L+P_l+P_r+2p-d(K-1)-1}{s}\right\rfloor+1.
$$

Для K=3, d=1, temporal Pₗ=2, Pᵣ=0 получаем ceil(L/s). Два temporal stride=2 дают floor((F−1)/4)+1. **Не подменяйте Conv3d tuple (T,H,W) порядком F.pad (W,H,T)**.
''')
code('''
show_source('HunyuanVideoCausalConv3d')
conv=NATIVE.HunyuanVideoCausalConv3d(1,1,3,bias=False)
x=torch.randn(1,1,9,8,8);changed=x.clone();changed[:,:,5:]+=7
with torch.inference_mode():
    delta=(conv(x)[:,:,:5]-conv(changed)[:,:,:5]).abs().max()
assert delta==0
metrics['causal_conv_prefix_error']=float(delta)
for n in [1,2,5,9,10]:
    down=NATIVE.HunyuanVideoCausalConv3d(1,1,3,stride=(2,1,1))
    assert down(torch.zeros(1,1,n,8,8)).shape[2]==math.ceil(n/2)
''')
md(r'''
## 2. Проверяем условия реализации, а не имя класса

В приложенной версии padding построен из kernel tuple в порядке, который совпадает с нужным только для используемых изотропных kernels (1 или 3). Также dilation не входит в вычисление causal padding. Это ограничение general constructor, не доказательство поломки released architecture: в ней надо отдельно проверить фактические call sites.

Предскажите output shape для kernel=(3,3,5) и dilation=2. В корректном same-spatial/causal-time варианте H/W не меняются и future influence исключён. Ниже лишь конструкторный edge case; типовой k=3,d=1 уже прошёл.
''')
code('''
x=torch.zeros(1,1,9,12,12)
for kwargs in [{'kernel_size':(3,3,5)},{'kernel_size':3,'dilation':2}]:
    c=NATIVE.HunyuanVideoCausalConv3d(1,1,**kwargs)
    print(kwargs,'padding=',c.time_causal_padding,'output=',tuple(c(x).shape))
metrics['constructor_limits']='anisotropic kernel and dilation require a separate padding audit; default isotropic path tested'
''')
md(r'''
## 3. Почему special first frame меняет inverse geometry

Spatial interpolation удваивает каждый spatial index. Temporal interpolation удваивает только F−1 кадров после первого:

$$
U_r(F)=1+r(F-1),\qquad U_2(U_2(F))=1+4(F-1).
$$

Обычное F→2F дважды дало бы 4F и лишние три кадра. Native `HunyuanVideoUpsampleCausal3D.forward` отдельно обрабатывает first_frame. Проверим F=1 и F>1 на оригинальном классе.
''')
code('''
show_source('HunyuanVideoUpsampleCausal3D')
up=NATIVE.HunyuanVideoUpsampleCausal3D(1,upsample_factor=(2,2,2))
for n in [1,2,5]:
    y=up(torch.zeros(1,1,n,4,4));assert y.shape==(1,1,2*n-1,8,8)
    print(n,'->',tuple(y.shape))
''')
md(r'''
## 4. Receptive field и GroupNorm: где теряется локальность

Для последовательных convolution layers с jump J и receptive field R:

$$
J_0=1,\quad R_0=1,\qquad
R_l=R_{l-1}+(K_l-1)d_lJ_{l-1},\quad J_l=s_lJ_{l-1}.
$$

Это размер **convolutional path**. Его нельзя считать конечным receptive field всего VAE с глобальной нормализацией. GroupNorm на B,C,F,H,W вычисляет group mean/variance по channels group и всем F/H/W:

$$
\mu_g=\frac{1}{m}\sum_{i\in g}x_i,\qquad
\hat x_i=(x_i-\mu_g)/\sqrt{v_g+\varepsilon}.
$$

Изменение будущего кадра меняет μ и v для прошлого. `eval()` не замораживает эти статистики, как BatchNorm. Значит **causal conv ≠ strict online causality всей сети**. Аналогично spatial tile меняет статистики. Causal attention mask отдельно запрещает future attention, но не отменяет утечку через normalization.
''')
code('''
x=torch.randn(1,4,9,8,8);changed=x.clone();changed[:,:,5:]+=5
norm=nn.GroupNorm(2,4).eval()
with torch.inference_mode():delta=float((norm(x)[:,:,:5]-norm(changed)[:,:,:5]).abs().max())
assert delta>1e-3
metrics['groupnorm_future_prefix_error']=delta
mask=NATIVE.prepare_causal_attention_mask(3,2,torch.float32,torch.device('cpu'),1)[0]
assert torch.isneginf(mask[0,2:]).all() and torch.isfinite(mask[2,:4]).all()
fig,ax=plt.subplots(figsize=(4,3));ax.imshow(torch.isfinite(mask),cmap='Blues');ax.set(xlabel='Key token',ylabel='Query token',title='3 frames × 2 spatial tokens: allowed attention')
fig.tight_layout();fig.savefig(figure_path(LAB,'causal_mask'));plt.show()
J=R=1
for k,s in [(3,1),(3,2),(3,2)]:R+=(k-1)*J;J*=s
metrics['example_conv_path']={'receptive_field':R,'jump':J}
print('GroupNorm prefix change:',delta,'convolution path RF:',R)
''')
finish('1. Повторите future perturbation для native ResnetBlock, затем для блока без GroupNorm.\n2. Найдите halo для трёх k=3 conv со stride=1. Почему с GroupNorm этого halo недостаточно?\n3. Докажите temporal output formula с first-frame rule. Переход: различайте causal convolution, attention mask и strict streaming contract.')

start('03_architecture_posterior_and_api','03. Архитектура, posterior и API кодека',
      'Сопоставим все классы с вычислительным графом, проследим BCTHW hooks и проверим реальный return contract encode.')
md(r'''
## 1. Полный путь encoder и decoder

Encoder: causal input conv → четыре DownBlocks → MidBlock → GroupNorm/SiLU/output conv → quant_conv. Ширины released defaults: 128,256,512,512; residual blocks по 2 на stage. Spatial downsample у первых трёх stages; temporal — у второго и третьего.

| После stage | C defaults | F | H/W |
|---|---|---|---|
| Input / conv_in | 3 → 128 | F | H,W |
| Down 0 | 128 | F | H/2,W/2 |
| Down 1 | 256 | ceil(F/2) | H/4,W/4 |
| Down 2 | 512 | ceil(F/4) | H/8,W/8 |
| Down 3 / mid | 512 | F_z | H/8,W/8 |
| conv_out / quant_conv | 32 | F_z | H/8,W/8 |
| posterior mode/sample | 16 | F_z | H/8,W/8 |

`quant_conv` — learned 1×1×1 projection posterior moments, **не discrete quantizer и не VQ codebook**. `post_quant_conv` — learned projection перед decoder. Decoder идёт по reversed widths, использует layers_per_block+1 residuals, spatial upsampling на stages 0–2, temporal на 1–2. Optional MidBlock attention получает flattened FHW tokens и causal frame mask; quadratic attention и dense mask могут доминировать память.

Для измерений уменьшим widths до 4,8,8,8 и layers до 1, сохранив четыре stages и геометрию. Mid attention выключена; это явно ограничивает перенос результатов.
''')
code('''
codec=TinyCodec(seed=3);x=synthetic_video(9,32,32)
shapes=[];handles=[]
for side,net,blocks in [('enc',codec.encoder,codec.encoder.down_blocks),('dec',codec.decoder,codec.decoder.up_blocks)]:
    for i,block in enumerate(blocks):
        handles.append(block.register_forward_hook(lambda m,a,b,label=f'{side}.{i}':shapes.append((label,list(b.shape)))))
with torch.inference_mode():
    moments=codec._encode(x);posterior=DiagonalPosterior(moments)
    rec=codec._decode(posterior.mode()).sample
for h in handles:h.remove()
assert moments.shape==(1,4,3,4,4) and rec.shape==x.shape
metrics['stage_shapes']=shapes;metrics['tiny_parameters']=sum(p.numel() for p in codec.parameters())
print('native tiny widths: 4,8,8,8; latent channels: 2')
for row in shapes:print(row)
''')
md(r'''
## 1b. Mid attention: из feature volume в Q/K/V

При включённой attention X имеет B,C,F,H,W. Permute/flatten даёт B,N,C, N=FHW. Для одного head:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,\qquad
Y=\mathrm{softmax}(QK^\top/\sqrt d+M)V.
$$

Форма Q/K/V — B,h,N,d, где C=h·d. Mask M равен 0 для key frame≤query frame и −∞ для будущего; spatial positions внутри одного кадра видят друг друга. В source encoder/decoder передают attention_head_dim равным ширине mid, то есть при C=512 получается один head с d=512; не переносите multihead настройки DiT на VAE автоматически.

Полный score/mask tensor растёт как N². Tiling уменьшает N внутри одного вызова, но одновременно убирает взаимодействия за пределами tile. Ниже вычислим attention по формуле на маленьком тензоре и сравним с PyTorch SDPA. Это **учебная реализация attention formula**, не импорт Diffusers Attention и не numerical parity с pretrained mid block.
''')
code('''
q=torch.randn(1,1,6,4);k=torch.randn_like(q);v=torch.randn_like(q)
mask=NATIVE.prepare_causal_attention_mask(3,2,q.dtype,q.device,1).unsqueeze(1)
manual=torch.softmax(q@k.transpose(-1,-2)/math.sqrt(4)+mask,dim=-1)@v
sdpa=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
assert torch.allclose(manual,sdpa,atol=2e-6)
metrics['attention_formula_sdpa_max_error']=float((manual-sdpa).abs().max())
show_source('HunyuanVideoMidBlock3D')
''')
md(r'''
## 2. Return contract: encode не возвращает готовый tensor

В **этом** snapshot `encode(..., return_dict=True)` возвращает `AutoencoderKLOutput(latent_dist=posterior)`, а `return_dict=False` — tuple с posterior. Итак:

- `vae.encode(x)[0]` — distribution object, не latent tensor;
- deterministic latent: `vae.encode(x).latent_dist.mode()`;
- stochastic latent: `.latent_dist.sample(generator=...)`.

В K6 wrapper `prepare_latents.encode_video` ожидается tensor после `[0]`. Это отдельная несовместимость контрактов закреплённой пары caller/callee; она не доказывает, что любой runtime K6 broken (может использовать другую VAE/adapter). Старый tuple[tensor] double проверяет ожидаемый wrapper interface, а не настоящий VAE API.

Ниже body `encode` оригинальный, posterior/output dependency — учебный adapter с тем же смыслом. Актуальный Diffusers не установлен в root env; не выдаём adapter за его runtime.
''')
code('''
show_source('encode')
# opt_tiling=False avoids overriding explicit full-shape test configuration.
with torch.inference_mode():
    output=codec.encode(x,opt_tiling=False)
    wrong=output[0]
    assert isinstance(wrong,DiagonalPosterior) and not torch.is_tensor(wrong)
    mode=output.latent_dist.mode()
    sample=output.latent_dist.sample(generator=torch.Generator().manual_seed(9))
metrics['posterior_contract']={'getitem_is_tensor':torch.is_tensor(wrong),'mode_shape':list(mode.shape),
 'mode_sample_MSE':float((mode-sample).square().mean())}
print(metrics['posterior_contract'])
''')
md(r'''
## 3. Scaling: смена единиц перед DiT

Пусть s=0.476986 из config. В generative pipeline z_DiT=s·z_VAE; перед decoder делим обратно:

$$
z_{\mathrm{DiT}}=s z,\qquad z=z_{\mathrm{DiT}}/s,\qquad
\mathrm{Cov}(s z)=s^2\mathrm{Cov}(z).
$$

Scaling не делает распределение автоматически N(0,I), не заменяет KL и не входит сам собой в `decode`. В учебном вызове `_decode` подаём unscaled z. Проверим roundtrip единиц и ошибку двойного масштабирования; outputs случайной сети не оценивают perceptual quality.
''')
code('''
s=codec.config.scaling_factor
with torch.inference_mode():
    correct=codec._decode((s*mode)/s).sample
    wrong_scale=codec._decode(s*mode).sample
assert torch.allclose(correct,rec,atol=2e-6)
metrics['correct_scale_roundtrip']=errors(rec,correct)
metrics['missing_inverse_scale']=errors(rec,wrong_scale)
print(metrics['missing_inverse_scale'])
''')
md(r'''
## 4. Dispatch и stateful настройки

`_encode` проверяет `use_framewise_decoding`, несмотря на имя операции. `use_framewise_encoding` в этом body не участвует. Public `encode` при opt_tiling=True вызывает planner и `apply_tiling`; public `decode` также пересчитывает policy и игнорирует дополнительные kwargs. Поэтому ручные flags могут быть перезаписаны следующим public call.

В опытах 04–07 используем `_encode/_decode` с явной конфигурацией. Это диагностический private API, а не обещание стабильности библиотеки. Для приложения нужен adapter с версионированным контрактом, shape guards и проверкой postconditions.
''')
code('''
show_source('_encode');show_source('decode');show_source('apply_tiling')
metrics['dispatch_notes']='_encode uses use_framewise_decoding; public decode recomputes tile policy'
''')
finish('1. Нарисуйте tensor dimensions для released latent=16 и tiny latent=2; не смешивайте их.\n2. Создайте adapter caller→posterior.mode→scale→layout и проверьте batch=2.\n3. Какой дополнительный источник нужен, чтобы утверждать, что optional attention включена в конкретном checkpoint? Переход: объясните каждый тип значения на границе encode/DiT/decode.')

start('04_spatial_tiling_and_blending','04. Spatial tiling: контекст, blend и границы',
      'Выведем условия корректной сшивки, исполним оригинальные blend/tiled methods и отделим отсутствие швов от эквивалентности full inference.')
md(r'''
## 1. Зачем overlap и как устроен blend

Tile size Q, stride S, overlap O=Q−S. В latent coordinates encoder требует Q_z=Q/8, S_z=S/8. Смешиваем два предсказания одного глобального pixel/latent position:

$$
y_j=(1-\alpha_j)a_j+\alpha_j b_j,\qquad \alpha_j=j/O,\quad j=0,\ldots,O-1.
$$

Веса суммируются к единице. При a=b результат не меняется. Последняя α=(O−1)/O, а не 1; следующий несмешанный pixel уже берётся из нового tile. Source `blend_*` меняет b **in-place**. Vertical blending выполняется до horizontal; четыре tile contributions в углу зависят от этого порядка.

Обобщённая weighted overlap-add (WOLA) делит сумму weighted predictions на сумму weights. Это другая схема; оригинальные methods ниже не заменяем WOLA молча:

$$
\hat y(p)=\frac{\sum_i w_i(p)y_i(p)}{\sum_iw_i(p)},\qquad \sum_iw_i(p)>0.
$$
''')
code('''
obj=oracle_codec();show_source('blend_h');show_source('blend_v')
a=torch.full((1,1,1,1,8),2.);b=torch.full_like(a,6.)
y=obj.blend_h(a,b.clone(),4).flatten()
expected=torch.tensor([2.,3.,4.,5.,6.,6.,6.,6.]);assert torch.equal(y,expected)
metrics['native_blend_values']=y.tolist()
fig,ax=plt.subplots(figsize=(6,3));ax.plot(range(8),y,'o-');ax.set(xlabel='Tile b local index',ylabel='Blended value',title='Native linear overlap: 2 → 6')
fig.tight_layout();fig.savefig(figure_path(LAB,'blend'));plt.show()
''')
md(r'''
## 2. Exact coordinate oracle: сначала доказываем индексы

Возьмём encoder, который выбирает каждый четвёртый кадр и каждый восьмой spatial pixel, decoder с first-frame preserving repetition. Он не учился и не имеет context. Если tile origins кратны compression ratio и loop полностью покрывает кадр, tiled и full outputs должны совпасть.

Это сильный **index test**, но слабый quality test. Он не выявляет GroupNorm, attention или boundary padding artifacts. Original methods получают оригинальные global coordinate values, поэтому off-by-one станет виден без нейросети.
''')
code('''
x=torch.arange(1*1*17*48*48,dtype=torch.float64).reshape(1,1,17,48,48)
obj=oracle_codec()
full=obj.quant_conv(obj.encoder(x));tiled=obj.tiled_encode(x)
assert torch.equal(full,tiled)
full_dec=obj.decoder(full);tile_dec=obj.tiled_decode(full).sample
assert torch.equal(full_dec,tile_dec)
metrics['aligned_oracle_encode_max_error']=float((full-tiled).abs().max())
metrics['aligned_oracle_decode_max_error']=float((full_dec-tile_dec).abs().max())
show_source('tiled_encode');show_source('tiled_decode')
''')
md(r'''
## 3. Покрытие правого/нижнего края — отдельное условие

Native loop: starts=range(0,L−Q+1,S). Он генерирует только полные tiles. При L≥Q последний покрытый index exclusive:

$$
L_{covered}=Q+S\left\lfloor\frac{L-Q}{S}\right\rfloor.
$$

Missing tail равен (L−Q) mod S. Финальный slice `:latent_width` может отрезать лишнее, но не дорисует отсутствующее. При L<Q список tiles пуст. Сработает ли проблема в приложении, зависит от planner/caller dimensions; проверяем обе части, а не объявляем весь Hunyuan upstream ошибочным.

Ниже L=88, Q=32, S=24: 88 кратно 8, но coverage=80. Пример показывает ограничение **прикреплённой адаптации**, не утверждение статьи о невозможности arbitrary sizes.
''')
code('''
obj=oracle_codec();obj.apply_tiling((1,17,32,32),(8,24,24))
x=torch.zeros(1,1,17,88,88)
got=obj.tiled_encode(x)
coverage=axis_coverage(88,32,24)
assert coverage['missing']==8 and got.shape[-1]==10 # expected 88/8 = 11
metrics['tail_counterexample']={'coverage':coverage,'expected_latent_width':11,'actual':got.shape[-1]}
print(metrics['tail_counterexample'])
# Explicit alternative, not a silent patch of pinned code:
starts=anchored_starts(88,32,24,align=8)
assert starts==[0,24,48,56]
print('Anchored grid:',starts,'variable final overlap:',32-(starts[-1]-starts[-2]))
''')
md(r'''
## 3b. Точный halo+crop на локальной свёртке

Пусть y[i]=Σw[k]x[i+k−r], radius r=1. Для output interval [a,b) достаточно input [a−r,b+r). Добавим halo, применим тот же convolution и вырежем только valid interior. На внешней границе используем тот же zero padding, что и full conv. Тогда каждый выход получает ровно те же summands — это конструктивное доказательство exactness для локального linear operator.

Сравним без halo и с halo. Этот пример намеренно 1D, чтобы индексы можно было проверить вручную; spatial 2D получается декартовым произведением двух interval halos. Добавление GroupNorm нарушит предпосылку локальности.
''')
code('''
x=torch.randn(1,1,32);weight=torch.tensor([[[.2,.6,.2]]])
full=F.conv1d(x,weight,padding=1)
naive=torch.cat([F.conv1d(x[:,:,a:a+8],weight,padding=1) for a in range(0,32,8)],dim=2)
padded=F.pad(x,(1,1));parts=[]
for a in range(0,32,8):
    b=a+8
    parts.append(F.conv1d(padded[:,:,a:b+2],weight,padding=0))
with_halo=torch.cat(parts,dim=2)
assert torch.equal(full,with_halo) and not torch.equal(full,naive)
metrics['halo_local_conv']={'with_halo_max_error':float((full-with_halo).abs().max()),
                           'without_halo_max_error':float((full-naive).abs().max())}
print(metrics['halo_local_conv'])
''')
md(r'''
## 4. Почему overlap не гарантирует full equivalence

Для локального оператора с radius r пространственный halo r и cropping interiors могут дать exact full result (кроме внешних boundaries). Без halo у каждого tile искусственная граница. Blend сглаживает расхождения, но не восстанавливает отсутствующий контекст.

GroupNorm использует статистики всего tile. Mid attention видит только local tile tokens. Даже очень большой конечный convolutional halo не восстанавливает global operation. В статье §4.1.2 дополнительный finetuning с включением/отключением tiling как раз уменьшает этот train/inference mismatch; не делает произвольный checkpoint математически exact.

Проверим ненулевую разницу на random native encoder без attention, где остаётся GroupNorm. Фиксируем одни веса и input: меняется только spatial policy.
''')
code('''
codec=TinyCodec(seed=4);x=synthetic_video(9,48,48)
with torch.inference_mode():
    full=codec._encode(x)
    codec.set_tiles(spatial=True,temporal=False)
    tiled=codec._encode(x)
stat=errors(full,tiled);assert stat['mse']>0
metrics['native_tiny_spatial_difference']=stat
emap=(full-tiled).square().mean((0,1,2)).numpy()
fig,ax=plt.subplots(figsize=(4,3));im=ax.imshow(emap);fig.colorbar(im,ax=ax,label='Latent squared error');ax.set(title='Same random encoder: full vs spatial',xlabel='Latent W',ylabel='Latent H')
fig.tight_layout();fig.savefig(figure_path(LAB,'spatial_error'));plt.show()
print(stat)
''')
finish('1. Реализуйте WOLA и докажите, что constant field сохраняется.\n2. Сделайте halo+crop для одной обычной spatial convolution и получите exact interior equivalence; затем добавьте GroupNorm.\n3. Обработайте L<Q и нечётные H/W через явный pad/crop policy. Переход: oracle exactness, coverage и perceptual quality должны быть тремя отдельными проверками.')

start('05_temporal_tiling_and_first_frame','05. Temporal tiling: первый кадр, lattice и context',
      'Выведем temporal coordinate map и проверим длины/значения оригинальных encode/decode tile loops на известном временном сигнале.')
md(r'''
## 1. Temporal lattice

При F=4k+1 encoder выдаёт k+1 latents с nominal anchor frames 0,4,8,…,4k. Это lattice, а не утверждение, что latent зависит только от одного кадра. Conv history и normalization расширяют зависимость.

В исходнике sample tile содержит Q+1 кадров; его latent length Q/4+1. Начала temporal tiles идут с шагом S кадров, кратным 4. Для tiles после первого удаляется первый latent; decoder аналогично удаляет первый **sample frame**:

$$
F_z=1+(F-1)/4,\qquad Q_z=Q/4,\quad S_z=S/4,\quad O_z=(Q-S)/4.
$$

Source defaults Q=16,S=12; adaptive planner ставит S=8. Это разные policies. `apply_tiling` принимает ft=Q+1 и хранит min_num_frames=ft−1. Забыть −1 означает сдвинуть lattice.
''')
code('''
show_source('_temporal_tiled_encode');show_source('_temporal_tiled_decode');show_source('blend_t')
obj=oracle_codec();F=33
x=torch.arange(F,dtype=torch.float64).reshape(1,1,F,1,1).expand(1,1,F,8,8)
with torch.inference_mode():
    full=obj.encoder(x);encoded=obj._temporal_tiled_encode(x)
    # spatial tiling disabled: H/W smaller than spatial tile min.
    decoded=obj._temporal_tiled_decode(full).sample
    expected_dec=obj.decoder(full)
assert torch.equal(full,encoded) and torch.equal(decoded,expected_dec)
metrics['aligned_temporal_oracle']={'F':F,'Fz':encoded.shape[2],'encode_max_error':float((full-encoded).abs().max()),'decode_max_error':float((decoded-expected_dec).abs().max())}
print('latent time anchors:',encoded[0,0,:,0,0].tolist())
''')
md(r'''
## 2. Frame-count audit: F=4k+1 ещё недостаточно

Если last start не доходит до конца, temporal loop теряет tail; первый tile отдельно обрезается до S_z+1. Поэтому нельзя переносить geometry encoder на geometry stitching автоматически.

Будем вызывать `_encode`, то есть **реальный dispatch**, а не temporal private method на коротком input, для которого он не предназначен. Для F≤Q+1 пойдёт full encode. Для длинных F сравним original tile output с expected lattice. Shape failure — результат опыта, а не повод допадить output незаметно.
''')
code('''
rows=[]
for frames in [1,9,17,21,25,29,33,37,41,49]:
    obj=oracle_codec();obj.use_tiling=False
    x=torch.arange(frames,dtype=torch.float64).reshape(1,1,frames,1,1).expand(1,1,frames,8,8)
    out=obj._encode(x);expected=obj.encoder(x)
    rows.append({'frames':frames,'expected_Fz':expected.shape[2],'actual_Fz':out.shape[2],
                 'same_shape':out.shape==expected.shape})
assert any(not r['same_shape'] for r in rows)
metrics['temporal_encode_lengths']=rows
for row in rows:print(row)
fig,ax=plt.subplots(figsize=(7,3));ax.plot([r['frames'] for r in rows],[r['expected_Fz'] for r in rows],'o-',label='Expected lattice');ax.plot([r['frames'] for r in rows],[r['actual_Fz'] for r in rows],'x--',label='Original stitch')
ax.set(xlabel='Input F',ylabel='Latent Fz',title='Original adaptation: temporal coverage audit');ax.legend();fig.tight_layout();fig.savefig(figure_path(LAB,'length_audit'));plt.show()
''')
md(r'''
## 3. Проверяем decoder отдельно

Encoder и decoder loops не обязаны ошибаться на одних и тех же lengths. Decode принимает F_z, а temporal tile slice имеет Q_z+1. На выходе first frame special rule даёт 4(F_z−1)+1. Сравним ожидаемую и фактическую длину **для каждого пути отдельно**.

Это unit evidence о pinned algorithm. Возможная защитная обвязка в production — дополнить последовательность до поддерживаемой grid, выполнить encode/decode, затем явно обрезать original length, записав valid-frame mask. Нельзя без доказательства считать padding нейтральным: GroupNorm изменяет все значения.
''')
code('''
rows=[]
for fz in [1,3,5,6,7,8,9,10,11,13]:
    obj=oracle_codec();obj.use_tiling=False
    z=torch.arange(fz,dtype=torch.float64).reshape(1,1,fz,1,1)
    out=obj._decode(z).sample
    rows.append({'Fz':fz,'expected_frames':4*(fz-1)+1,'actual_frames':out.shape[2]})
metrics['temporal_decode_lengths']=rows
for row in rows:print(row)
''')
md(r'''
## 4. Reset history и временные швы

Каждый temporal tile заново делает replicate-padding своего первого кадра. Для causal conv receptive history h кадров предыдущего tile нужна как prefix, а не как новый «начальный момент мира». Можно переносить conv cache или передавать halo и отбрасывать невалидные outputs. Но GroupNorm и attention снова мешают exact streaming equivalence.

Измерим full vs temporal на длине с правильным coverage, чтобы не смешивать shape failure с approximate context. Random tiny encoder/decoder, без optional attention. Граф показывает mean squared difference по latent time; первый latent не обязан быть единственным изменившимся.
''')
code('''
codec=TinyCodec(seed=2);x=synthetic_video(33,32,32)
with torch.inference_mode():
    full=codec._encode(x)
    codec.set_tiles(spatial=False,temporal=True)
    tiled=codec._encode(x)
stat=errors(full,tiled);metrics['native_tiny_temporal_difference']=stat
profile=(full-tiled).square().mean((0,1,3,4)).numpy()
fig,ax=plt.subplots(figsize=(7,3));ax.plot(profile,'o-');ax.set(xlabel='Latent temporal index',ylabel='Squared difference',title='Same random encoder: full vs temporal')
fig.tight_layout();fig.savefig(figure_path(LAB,'temporal_error'));plt.show();print(stat)
''')
finish('1. Выпишите все tile slices и dropped elements для F=33,Q=16,S=8.\n2. Повторите oracle length sweep с S=12. Найдите условия отсутствия tail loss.\n3. Сравните input replication и conv-cache подход на простом 1D causal filter. Переход: объясните отдельно timestamp, context, coverage и sample count.')

start('06_free_vram_planner','06. Free-VRAM-aware planner: формула, округление, ограничения',
      'Разберём каждую строку get_enc/get_dec_optimal_tiling, затем проверим policy на сетке budgets без обращения к реальному CUDA GPU.')
md(r'''
## 1. Что на самом деле моделирует heuristic

Память процесса состоит из weights, input/output, live activations, workspace, allocator reserve. При inference autograd не хранит training graph, но промежуточные tensors и full stitching buffers остаются:

$$
M_{peak}\approx M_{weights}+M_{IO}+M_{live}(Q_t,Q_h,Q_w)+M_{workspace}+M_{allocator}.
$$

В приложенном planner:

$$
A_{max}=\frac{M_{free}}{256\cdot17\cdot8},\qquad
N_{vals}=256\cdot17\cdot(H+32)(W+32).
$$

M_free в **bytes**. Чтобы размерность A была pixels, произведение denominator интерпретируется как эмпирические bytes на spatial pixel. 256,17,8 — константы реализации, а не универсально выведенный tensor liveness model. В формуле не используются B, dtype, latent channels, weights или optional attention. 8 нельзя без источника уверенно объявить bytes-per-element: BF16 сам по себе занимает 2 bytes.

Вторая проверка N_vals<2³¹ похожа на ограничение размера промежуточного tensor/indexing; код не документирует точную операцию. Не путать с лимитом RAM 2 GB.
''')
code('''
show_source('get_enc_optimal_tiling');show_source('get_dec_optimal_tiling')
shape=[1,3,121,480,864]
# Original CPU/MPS branch uses infinity, no memory API call.
for device in ['cpu','mps']:
    tile,stride=plan(shape,device=device)
    print(device,tile,stride)
    assert tile==(1,17,480,864)
metrics['non_cuda_policy']='infinite budget; temporal tile 17, no free-unified-memory query'
''')
md(r'''
## 2. От числа tiles к двум осям

Если весь кадр не проходит, код выбирает:

$$
N=\max\left(\left\lceil HW/A_{max}\right\rceil,
\left\lceil N_{vals}/2^{31}\right\rceil\right),\qquad
k=\max(H/W,W/H).
$$

Пусть a·b≈N, b/a≈k. Подстановка b=ka даёт a≈sqrt(N/k), b≈sqrt(Nk). Обе величины округляются вверх. Большему размеру кадра назначается большее число splits. Это эвристика aspect ratio, не решение дискретной оптимизации peak memory или latency.

Для разделённой width:

$$
Q_w=8\left\lceil\frac{W}{8n_w}\right\rceil+16,\qquad S_w=Q_w-32.
$$

Получается nominal overlap 32 pixels. При слишком малом Q stride может стать нулевым/отрицательным. Округление tile может превысить short axis. Ни coverage, ни final memory bound после rounding код не доказывает. `get_dec` превращает latent shape обратно в sample dimensions (4(F_z−1)+1,8H_z,8W_z) и вызывает ту же encode heuristic; decoder cost profile отдельно не калибруется.
''')
code('''
rows=[]
for gib in [.125,.25,.5,1,2,4,8,16]:
    # Local torch namespace returns synthetic CUDA budget; no CUDA tensor/device is allocated.
    tile,stride=plan(shape,free_gib=gib,device='cuda:0')
    _,ft,ht,wt=tile;fs,hs,ws=stride
    hc,wc=axis_coverage(shape[3],ht,hs),axis_coverage(shape[4],wt,ws)
    rows.append({'synthetic_free_GiB':gib,'tile':list(tile),'stride':list(stride),
                 'height_missing':hc['missing'],'width_missing':wc['missing'],
                 'valid_spatial_grid':hc['valid'] and wc['valid'],
                 'tile_formula_GiB':256*17*8*ht*wt/2**30})
metrics['synthetic_budget_sweep']=rows
for row in rows:print(row)
fig,axes=plt.subplots(1,2,figsize=(10,3.2))
axes[0].plot([r['synthetic_free_GiB'] for r in rows],[r['tile'][2]*r['tile'][3] for r in rows],'o-');axes[0].set(xscale='log',xlabel='Synthetic CUDA free GiB',ylabel='Tile area',title='Original heuristic')
axes[1].plot([r['synthetic_free_GiB'] for r in rows],[max(r['height_missing'],r['width_missing']) for r in rows],'o-');axes[1].set(xscale='log',xlabel='Synthetic CUDA free GiB',ylabel='Missing edge pixels',title='Separate coverage postcondition')
fig.tight_layout();fig.savefig(figure_path(LAB,'planner_sweep'));plt.show()
''')
md(r'''
## 3. Memory guard ≠ OOM guarantee

Сначала проверяются 0<S≤Q≤L, spatial/temporal alignment и coverage. Затем реальный прогон измеряет peak allocated/reserved; free_mem до запуска может измениться из-за других процессов. Практическая policy оставляет safety reserve, калибрует memory vs tile area, уменьшает tile при OOM и логирует retry. Нельзя зацикливать retries с тем же tile.

M3 Max 48 GB — **общая память**, включая macOS, CPU и GPU. `torch.mps.current_allocated_memory()` и `driver_allocated_memory()` описывают PyTorch/Metal allocations, а не «свободную VRAM 48 GB». `recommended_max_memory()` — рекомендованный working-set bound, не гарантированная free memory. Нужны отдельные measurements и консервативный user budget; CUDA policy не переносится автоматически.

В source все decoded tiles сохраняются в `rows` до concat; значит tiling снижает size одной активации, но не делает общую память постоянной при неограниченной duration/resolution. Список slices может удерживать backing storage. `empty_cache()` не освобождает live tensors.
''')
code('''
# Show a low-budget invalid-stride case without invoking any tensor generation.
low=plan([1,3,121,480,864],free_gib=.0001,device='cuda:0')
metrics['very_low_budget_policy']={'tile':list(low[0]),'stride':list(low[1])}
assert low[1][1]<=0 or low[1][2]<=0
print('Guard must reject:',low)
# Dense mask alone, if attention enabled: B * (Fz*Hz*Wz)^2 * bytes.
fz,hz,wz=31,60,108;n=fz*hz*wz
metrics['dense_attention_mask_fp32_GiB']=n*n*4/2**30
print('Possible full-latent dense mask GiB:',metrics['dense_attention_mask_fp32_GiB'])
''')
finish('1. Выведите sqrt factorization без подсказки и объясните ceiling.\n2. Найдите input, где tile heuristic проходит memory approximation, но не coverage; запишите guard.\n3. Спроектируйте calibration table (dtype,device,shape,tile,peak,time) и safety factor. Переход: не называйте synthetic budget реальным VRAM measurement.')

start('07_controlled_benchmark','07. Контролируемый опыт: full / spatial / temporal / combined',
      'Проверим гипотезу: tiling сохраняет формы на выбранной grid, но меняет результат нелокального encoder/decoder. Измерим разницу и CPU latency на трёх случайных tiny checkpoints.')
md(r'''
## 1. Гипотеза, контроль и метрики

**H1:** на coverage-safe размерах original tiling сохраняет BCTHW, но native tiny reconstruction отличается от full из-за context/padding/GroupNorm. **Контроль:** те же random weights, input и deterministic posterior mode. **Фактор:** только tile policy; три seeds показывают устойчивость свойства, а не оценивают pretrained distribution.

Все результаты сравниваются с full и отдельно с input. Для нормированных RGB x∈[−1,1] dynamic range R=2:

$$
\mathrm{MSE}=\frac{1}{N}\sum_i(x_i-y_i)^2,\qquad
\mathrm{PSNR}=10\log_{10}\frac{R^2}{\mathrm{MSE}}.
$$

PSNR к input — reconstruction quality **случайного tiny codec**; PSNR к full — deviation от baseline. Они отвечают на разные вопросы. На латенте R=2 произволен, поэтому в выводах используем latent MSE, а PSNR только для RGB signal convention.

Вместо одного среднего дополнительно измерим spatial seams vs interior и temporal profile. Peak GPU memory не измеряется в CPU опыте. Warmup исключён; median трёх timing repeats уменьшает шум, не даёт производственный benchmark.
''')
code('''
rows=[];visual={}
x=synthetic_video(33,48,48,seed=101)
policies={'full':(False,False),'spatial':(True,False),'temporal':(False,True),'combined':(True,True)}
for seed in [0,1,2]:
    codec=TinyCodec(seed)
    def reconstruct():
        moments=codec._encode(x)
        return codec._decode(DiagonalPosterior(moments).mode()).sample
    codec.set_tiles(spatial=False,temporal=False)
    baseline,_=measure(reconstruct,repeats=3)
    for name,(spatial,temporal) in policies.items():
        codec.set_tiles(spatial=spatial,temporal=temporal)
        out,timing=measure(reconstruct,repeats=3)
        assert out.shape==x.shape
        delta=(out-baseline).square().mean((0,1,2))
        seam=torch.zeros_like(delta,dtype=torch.bool)
        for boundary in [16,32]:
            seam[max(0,boundary-1):boundary+1,:]=True
            seam[:,max(0,boundary-1):boundary+1]=True
        row={'seed':seed,'policy':name,**timing,
             'difference_to_full':errors(baseline,out),'reconstruction_vs_input':errors(x,out),
             'seam_MSE':float(delta[seam].mean()),'interior_MSE':float(delta[~seam].mean())}
        rows.append(row)
        if seed==0:visual[name]=out.detach().clone()
metrics['runs']=rows
assert all(r['difference_to_full']['mse']==0 for r in rows if r['policy']=='full')
assert all(r['difference_to_full']['mse']>0 for r in rows if r['policy']!='full')
for r in rows:print(r['seed'],r['policy'],'MSE to full=',round(r['difference_to_full']['mse'],6),'CPU s=',round(r['seconds_median'],3))
''')
md(r'''
## 2. Графики и диагностическая локализация

Первый график сопоставляет MSE и CPU time. Если tiny tiling медленнее full, это ожидаемо: повторные overlapping convolutions и Python loops. Это не доказывает замедление на real GPU или невозможность больших videos; цель tiling — вместимость, которую этот CPU опыт не измеряет.

Seam mask — двухpixel полоса вокруг nominal spatial boundaries, одинаковая для всех policies. Ненулевая interior error говорит, что расхождение не ограничено линией сшивки. Сравнение маленькой seam area и большой interior требует means по каждой области, не суммы.
''')
code('''
fig,axes=plt.subplots(1,3,figsize=(12,3.4))
for seed in [0,1,2]:
    rr=[r for r in rows if r['seed']==seed]
    axes[0].plot([r['policy'] for r in rr],[r['difference_to_full']['mse'] for r in rr],'o-',label=f'seed {seed}')
    axes[1].plot([r['policy'] for r in rr],[r['seconds_median'] for r in rr],'o-')
rr=[r for r in rows if r['seed']==0]
axes[2].plot([r['policy'] for r in rr],[r['seam_MSE'] for r in rr],'o-',label='Seam band')
axes[2].plot([r['policy'] for r in rr],[r['interior_MSE'] for r in rr],'x--',label='Interior')
axes[0].set(ylabel='RGB MSE to full',title='Identical input and weights');axes[0].legend()
axes[1].set(ylabel='CPU median seconds',title='Warmup + 3 timed runs')
axes[2].set(ylabel='RGB MSE to full',title='Seed 0 spatial localization');axes[2].legend()
for ax in axes:ax.tick_params(axis='x',rotation=25)
fig.tight_layout();fig.savefig(figure_path(LAB,'benchmark'));plt.show()
fig,axes=plt.subplots(1,4,figsize=(12,3))
for ax,(name,out) in zip(axes,visual.items()):
    delta=(out-visual['full']).square().mean((0,1,2));im=ax.imshow(delta.numpy(),vmin=0,vmax=max(float((visual['combined']-visual['full']).square().mean((0,1,2)).max()),1e-9));ax.set(title=name,xlabel='W',ylabel='H')
fig.tight_layout();fig.savefig(figure_path(LAB,'benchmark_error_maps'));plt.show()
''')
md(r'''
## 3. Overlap work и retained buffers

При большом regular 3D grid отношение обработанного объёма к полезному приблизительно:

$$
R_{work}\approx\frac{Q_t+1}{S_t}\cdot\frac{Q_h}{S_h}\cdot\frac{Q_w}{S_w}.
$$

Это оценка числа повторных input positions, не точная FLOPs/latency ratio: первый/последний tile, компрессия, attention, backend и launch overhead меняют стоимость. Здесь spatial Q=32,S=16 и temporal Q+1=17,S=8 дают большой overlap. Для реального codec размер tile и overlap выбирают по budget/качество/latency совместно.
''')
code('''
metrics['asymptotic_processed_volume_ratio']=(17/8)*(32/16)**2
metrics['conclusion']={
 'shape_invariance_on_test_grid':True,
 'tiling_is_full_equivalent':False,
 'native_mid_attention_tested':False,
 'pretrained_quality_tested':False,
 'peak_VRAM_tested':False,
 'CPU_timing_is_production_latency':False}
print(metrics['conclusion'])
''')
finish('1. Добавьте overlap=8/16 и сравните quality deviation/time при корректном coverage.\n2. Выполните encoder-only и decoder-only ablations: какой из них вносит больше MSE?\n3. Для pretrained phase выберите non-overlapping video split и бюджет повторов; не используйте seed как замену video diversity. Переход: сформулируйте вывод отдельно по гипотезе H1, latency и памяти.')

start('08_pretrained_and_comfyui_protocol','08. Настоящий VAE: ручной протокол PyTorch и ComfyUI',
      'Подготовим измеримый pretrained эксперимент без автоматической загрузки весов или запуска ComfyUI. Сейчас выполняется только dry-run плана.')
md(r'''
## 1. Исследовательская карточка

**Вопрос:** при заданном budget какой spatial/temporal tile даёт минимальный peak memory, сохраняя размер/длительность и допустимую reconstruction quality?

**Материалы:** 3–5 локальных коротких clips с согласием на использование: static texture/text, motion, scene transition. Не нужно качать DiT, Qwen, CLIP, SR, audio VAE: только соответствующий video VAE. Tiny experiments уже есть в 01–07; pretrained checkpoint должен быть идентифицирован SHA/config/provenance отдельно.

**Порядок:** decode video→RGB float [-1,1]→BCTHW→pad/alignment с valid mask→posterior.mode→decode→unpad→metrics. Сначала F=1, затем 17 и 33; 32×32 / 64×64 для первого shape smoke, затем 240×432 (H кратна 8). Для F=33/17 допустима grid из предыдущих labs, но конкретный planner всё равно проверяем.

**Контроль:** один checkpoint, dtype/device, input, posterior.mode, weights и warmup. Начинайте с full inference на малом input, затем spatial, temporal, combined. Масштабируйте input только после корректного full baseline. Если full на большом input OOM, quality baseline берите на меньшем clip; не выдавайте другой shape за paired full comparison.

**Ограничение станции:** pretrained phase отложен пользователем до освобождения 50 GB+ SSD. Этот notebook не проверяет gate вместо человека и не запускает загрузку. MPS support требует отдельного smoke; CUDA free-memory heuristic в attached code для MPS не действует.
''')
code('''
import subprocess
runner=COURSE/'scripts/benchmark_local_codec.py'
p=subprocess.run([sys.executable,str(runner),'--help'],capture_output=True,text=True,check=True)
print(p.stdout)
plan_record={'execute':False,'codec':'local Diffusers-format Hunyuan VAE directory',
 'input':'local BCTHW float tensor saved as .npy, RGB [-1,1]',
 'policies':['full','spatial','temporal','combined'],
 'device':'cpu for smoke / mps experimental / cuda for measured VRAM',
 'no_network':True,'no_weights_loaded':True}
metrics['manual_plan']=plan_record
print(json.dumps(plan_record,ensure_ascii=False,indent=2))
''')
md(r'''
## 2. Команда ручного запуска после resource gate

Runner использует установленный **Diffusers HunyuanVideo VAE** и local_files_only=True. Это отдельная reference реализация; он **не утверждает**, что её tiling совпадает с attached adaptation. Версия, class и VAE config сохраняются. При недоступном требуемом flag runner останавливается, а не делает silent fallback.

```bash
# Из корня репозитория, в отдельно подготовленном пользователем окружении с Diffusers.
# Сначала dry-run: не загружает веса и не импортирует Diffusers.
python vae/scripts/benchmark_local_codec.py \
  --model /ABS/LOCAL/vae --input /ABS/LOCAL/clip.npy --device mps
# После проверки путей/места добавьте --execute, например для одной policy:
python vae/scripts/benchmark_local_codec.py \
  --model /ABS/LOCAL/vae --input /ABS/LOCAL/clip.npy --device cuda \
  --dtype float16 --policy spatial --repeats 3 --execute
```

Нужен Diffusers-format folder с config.json и весами. ComfyUI single `.safetensors` не следует просто переименовывать: architecture/config/key mapping проверяются отдельно. Не устанавливайте пакеты в Desktop .venv ради этого notebook. Adapter к attached native class — следующий отдельный runtime experiment после проверки import/weight format/API; source-level code в 00–07 уже изучен.

Runner рассчитывает PSNR/MSE на floating RGB, **до MP4 compression**, запускает deterministic mode, синхронизирует device вокруг timing. На CUDA сбрасывает peak counters после загрузки модели и warmup; peak absolute allocated включает резидентные веса, incremental peak считается относительно baseline. Reserved показывается отдельно. На MPS записывает before/after allocation counters: **это не peak**. RSS не называем VRAM.
''')
md(r'''
## 3. ComfyUI как визуальный контроль

Откройте существующий [workflow codec reconstruction](../../curriculum/workflows/04_hunyuan_vae_reconstruction.json), инструкция [C4](../../curriculum/COMFYUI_LABS.md). Он работает с image F=1: позволяет увидеть normalization/color/tiling artifacts без DiT. Для temporal опыта нужен nodes graph, который сохраняет **целый video batch** и вызывает temporal VAE interface; серия независимых image encode не проверяет causal video codec.

1. Загрузите [test chart](../../curriculum/assets/curriculum_test_chart.png), выберите **уже имеющийся** Hunyuan-compatible VAE. Зафиксируйте ComfyUI revision/custom nodes и SHA checkpoint. Откажитесь от автоматической догрузки ненужных pipeline моделей.
2. Запустите decode(encode(image)); сохраните исходный PNG и reconstruction PNG. Сначала обычные VAEEncode/VAEDecode; tiled nodes добавляйте только при наличии в вашей версии. Запишите реальные tile/overlap settings.
3. Очистите/обойдите кэш для измерения: неизменённый graph может не вызвать VAE повторно. Сначала warmup, затем 3 фактических repetitions; следите, что nodes исполнялись.
4. Сравните через [compare_images.py](../../curriculum/scripts/compare_images.py), не screenshot preview. Учтите RGB/sRGB, clipping и PNG quantization: это другой comparison stage, чем float tensors runner.
5. Для видео сначала проверяйте F=17/33, FPS и число output frames. MP4 используйте для просмотра после численных измерений; отдельно экспортируйте float/PNG frames для точного сравнения.

ComfyUI обычно использует собственный loader/wrapper. Результат графа не является автоматически runtime test именно attached `get_enc_optimal_tiling`. Зафиксируйте исполненный class/source revision, а не только имя nodes.
''')
code('''
# Template only: no HTTP calls to Desktop and no queued generation.
columns=['run_id','implementation','revision','checkpoint_sha256','input_sha256','device','dtype',
 'F','H','W','policy','tile_F','tile_H','tile_W','stride_F','stride_H','stride_W',
 'posterior','warmup','repeats','seconds_median','peak_allocated_bytes','peak_reserved_bytes',
 'mps_current_before','mps_current_after','mse_input','psnr_input','mse_full',
 'seam_mse','interior_mse','output_F','output_H','output_W','status','notes']
metrics['measurement_columns']=columns
print(','.join(columns))
''')
md(r'''
## 4. Как сделать вывод, когда появятся реальные числа

- **Корректность:** все lengths/layout, отсутствие NaN и aligned coverage. OOM — отдельный статус, не качество=0.
- **Качество:** paired MSE/PSNR to full и input; seam/interior; temporal consistency на motion clips. Не сравнивайте только один static image.
- **Память:** CUDA peak allocated и reserved в bytes; MPS counters before/after не дают пика. Измерение должно указывать implementation, shape, dtype и policy.
- **Скорость:** warmup, синхронизация, median и разброс, отдельно encode и decode. ComfyUI cache/download/load times не смешиваем с kernel inference.
- **Ограничения:** small clips и fixed hardware; результат не распространяется автоматически на 1080p/длинное видео или другой checkpoint.

Для честного сравнения adaptive vs fixed tiling нужны оба runtime на одних данных и budgets. Source header обещает lower peak относительно Diffusers, но наша CPU серия это **не проверяет**. Исследовательский вывод пока: algebra/indexing/runtime interfaces изучены; pretrained resource/quality evidence ожидает ручного запуска.
''')
finish('1. Заполните один dry-run record до покупки GPU.\n2. Выберите допустимую degradation threshold и memory reserve **до** просмотра результатов.\n3. После запуска сравните 3–5 clips и сформулируйте минимальный поддержанный вывод. Критерий завершения: таблица real runs с provenance, корректными units и отдельно описанными failures.')

print('Built',len(list((C/'notebooks').glob('*.ipynb'))),'Hunyuan VAE notebooks')
