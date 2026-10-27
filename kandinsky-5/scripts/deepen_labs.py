"""Add reproducible theory chapters to the two existing labs after base generation."""
from pathlib import Path
import json,textwrap
ROOT=Path(__file__).resolve().parents[1]
def md(s): return {'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip()+'\n'}
def code(s): return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':textwrap.dedent(s).strip()+'\n'}
def pair(m,c):return [md(m),code(c)]
vae_intro=[md(r'''
## Часть A. Теория VAE: от сжатия к вероятностной модели

После этой части вы сможете вывести ELBO, отличить prior от posterior, написать differentiable sampling и объяснить, почему готовый VAE и Flow Matching учатся разным loss.

**Обозначения:** $x$ — наблюдаемые пиксели, $z$ — скрытый непрерывный код; $\phi$ — веса encoder, $\theta$ — веса decoder. $q_\phi(z\mid x)$ — приближённый posterior, $p(z)$ — prior, $p_\theta(x\mid z)$ — модель наблюдения. Вероятностное условие $\mid$ читается «при известном».

Обычный autoencoder кодирует картинку одной точкой. VAE кодирует её распределением: центр говорит «какой код подходит», разброс — «насколько можно его менять». Decoder должен восстанавливать изображение из соседних точек, а не только из одного запомненного кода. Это полезная интуиция; математическое основание — вероятностная цель ниже.

### A1. Откуда берётся задача обучения

Порождающая модель: сначала $z\sim p(z)=\mathcal N(0,I)$, затем $x\sim p_\theta(x\mid z)$. Хотим увеличить likelihood наблюдаемых данных:

$$p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz.$$

Интеграл суммирует объяснения картинки всеми возможными кодами. При нейросетевом decoder его обычно нельзя вычислить аналитически. Encoder не заменяет этот интеграл: он даёт распределение, из которого удобно брать подходящие $z$ для обучения.

**Bayes:** $p_\theta(z\mid x)=p_\theta(x\mid z)p(z)/p_\theta(x)$. Это истинный posterior; он тоже зависит от трудного знаменателя. Поэтому учим доступное $q_\phi(z\mid x)$, обычно Gaussian с diagonal covariance.

### A2. Вывод ELBO по шагам

Умножим и разделим под интегралом на $q_\phi$ (при подходящей поддержке распределений):

$$\log p_\theta(x)=\log\mathbb E_{q_\phi(z\mid x)}\left[\frac{p_\theta(x\mid z)p(z)}{q_\phi(z\mid x)}\right].$$

Логарифм — вогнутая функция. Jensen позволяет перенести его внутрь ожидания и получить **нижнюю границу**:

$$\log p_\theta(x)\geq\mathbb E_q[\log p_\theta(x\mid z)]-D_{KL}(q_\phi(z\mid x)\parallel p(z))=\mathrm{ELBO}.$$

Точнее, $\log p_\theta(x)=\mathrm{ELBO}+D_{KL}(q_\phi(z\mid x)\parallel p_\theta(z\mid x))$. Второй KL — зазор до истинного posterior; первый KL внутри ELBO — регуляризация к prior. **Это два разных KL.**

Минимизируем negative ELBO: reconstruction negative log-likelihood + KL. Реконструкция требует информативный код; KL ограничивает, как сильно posterior отклоняется от prior. [Первичная статья VAE](https://arxiv.org/abs/1312.6114).
''')]
vae_intro+=pair(r'''
### A3. Небольшой exact пример: проверяем нижнюю границу

Возьмём линейную Gaussian модель: $z\sim N(0,1)$, $x\mid z\sim N(z,\sigma_x^2)$. Тогда $x\sim N(0,1+\sigma_x^2)$, а истинный posterior имеет mean $x/(1+\sigma_x^2)$ и variance $\sigma_x^2/(1+\sigma_x^2)$.

Здесь интеграл известен: можно проверить ELBO численно без Monte Carlo. Для плохого $q$ ELBO ниже log likelihood; для точного posterior граница совпадает. Decoder в этом опыте не нейросеть: мы изолировали одну математическую идею.
''',r'''
x_obs=1.3; obs_var=0.25
true_mean=x_obs/(1+obs_var); true_var=obs_var/(1+obs_var)
log_evidence=-0.5*(np.log(2*np.pi*(1+obs_var))+x_obs**2/(1+obs_var))
def exact_elbo(mean,var):
    expected_nll=0.5*(np.log(2*np.pi*obs_var)+((x_obs-mean)**2+var)/obs_var)
    kl_prior=0.5*(mean**2+var-1-np.log(var))
    return -expected_nll-kl_prior
for name,mean,var in [('bad q',0.,1.),('exact posterior',true_mean,true_var)]:
    bound=exact_elbo(mean,var)
    kl_gap=0.5*(var/true_var+(mean-true_mean)**2/true_var-1+np.log(true_var/var))
    print(name,'ELBO:',round(bound,5),'log p(x):',round(log_evidence,5),'gap:',round(kl_gap,5))
    assert bound<=log_evidence+1e-7
    assert np.isclose(log_evidence-bound,kl_gap)
''')
vae_intro+=pair(r'''
### A4. Diagonal Gaussian: почему encoder выдаёт 2C каналов

$q_\phi(z\mid x)=N(\mu_\phi(x),\operatorname{diag}(\sigma^2_\phi(x)))$. Encoder выдаёт mean и **log variance**, а не mean и standard deviation: $\ell=\log\sigma^2$, $\sigma=\exp(\ell/2)$. Так variance остаётся положительной.

$$D_{KL}(q\parallel N(0,I))=\frac12\sum_j(\mu_j^2+\exp\ell_j-1-\ell_j).$$

Это сумма по latent coordinates одного примера, затем обычно среднее по batch. Смена `sum` на `mean` меняет относительную силу KL. У Hunyuan 16 latent channels, но posterior parameters имеют **32 канала**. Это continuous KL-VAE; название `quant_conv` не означает discrete VQ codebook.
''',r'''
mu=torch.tensor([[0.2,-0.3]]); logvar=torch.tensor([[-1.,0.5]])
manual=0.5*(mu.square()+logvar.exp()-1-logvar).sum(-1)
q=torch.distributions.Normal(mu,(0.5*logvar).exp())
p=torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(mu))
api=torch.distributions.kl_divergence(q,p).sum(-1)
assert torch.allclose(manual,api)
print('KL per example:',manual,'exact q=prior KL:',0.5*(torch.zeros(2)+torch.ones(2)-1).sum())
''')
vae_intro+=pair(r'''
### A5. Reparameterization: как градиент проходит через случайность

Пишем $\epsilon\sim N(0,I)$ и $z=\mu+\exp(\ell/2)\epsilon$. Шум независим от весов; $z$ — differentiable функция mean/logvar. `Normal.sample()` обычно не сохраняет нужный путь gradient; `rsample()` реализует reparameterization.

Для одного фиксированного noise sample $\partial z/\partial\mu=1$, $\partial z/\partial\ell=\frac12\sigma\epsilon$. Проверим autograd. Зажим logvar предотвращает переполнение; слишком сильное clipping меняет модель.
''',r'''
mu=torch.tensor([0.2,-0.3],requires_grad=True)
lv=torch.tensor([-1.,0.5],requires_grad=True); eps=torch.tensor([0.4,-0.8])
std=(lv/2).exp(); z=mu+std*eps
z.sum().backward()
assert torch.allclose(mu.grad,torch.ones_like(mu))
assert torch.allclose(lv.grad,0.5*std.detach()*eps)
print('mean gradient:',mu.grad,'logvar gradient:',lv.grad)
dist=torch.distributions.Normal(mu,std)
print('sample differentiable:',dist.sample().requires_grad,'rsample:',dist.rsample().requires_grad)
''')
vae_intro+=pair(r'''
### A6. Как reconstruction loss связан с вероятностью

При Gaussian decoder с фиксированной observation variance NLL пропорционален squared error. При Bernoulli decoder для binary pixels получаем binary cross-entropy. Выбор **не взаимозаменяем**: Bernoulli хорош для учебных бинарных фигур, но не утверждается как loss Hunyuan.

$L_\beta=\mathbb E_q[-\log p_\theta(x\mid z)]+\beta KL$. При $\beta=1$ и согласованном likelihood/reduction это negative ELBO; произвольный $\beta$ — другая регуляризованная цель. Предыдущий простой MiniVAE с mean-MSE + 0.001·mean-KL будет лишь наглядной аппроксимацией. Ниже добавим loss с явными единицами **на один пример**.
''',r'''
x=torch.tensor([[0.,1.,1.,0.]])
logits=torch.tensor([[-1.,1.,0.3,-0.5]])
bernoulli_nll=torch.nn.functional.binary_cross_entropy_with_logits(logits,x,reduction='none').sum(-1)
prob=torch.sigmoid(logits)
explicit=-(x*prob.log()+(1-x)*(1-prob).log()).sum(-1)
assert torch.allclose(bernoulli_nll,explicit)
print('Bernoulli NLL per image:',bernoulli_nll)
''')

vae_extra=[]
vae_extra+=pair(r'''
## Часть C. Более строгий учебный VAE: сравниваем β, prior и posterior

Для 8×8 фигур обучим encoder → mean/logvar → reparameterized latent → decoder logits. Суммируем reconstruction по **64 пикселям**, KL по **2 координатам**, затем усредняем batch. Две модели начинают с одинаковых весов и обучаются на одном наборе данных; сравнение β остаётся малым педагогическим экспериментом, а не benchmark.

Отдельно сохраняем reconstruction NLL и KL: общий loss без разложения не показывает, использует ли encoder латент. При posterior collapse $q(z\mid x)$ близок к prior и decoder может игнорировать $z$. Маленький KL сам по себе не доказывает collapse; смотрим также разброс mean между входами и зависимость decoder от z.
''',r'''
class ELBOVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=torch.nn.Sequential(torch.nn.Linear(64,48),torch.nn.SiLU(),torch.nn.Linear(48,4))
        self.decoder=torch.nn.Sequential(torch.nn.Linear(2,48),torch.nn.SiLU(),torch.nn.Linear(48,64))
    def encode(self,x):
        mu,lv=self.encoder(x).chunk(2,-1)
        return mu,lv.clamp(-10,5)
    def forward(self,x):
        mu,lv=self.encode(x)
        z=mu+torch.exp(lv/2)*torch.randn_like(mu)
        return self.decoder(z),mu,lv
binary=data.flatten(1)
models={}; histories={}
for beta in [0.01,1.0]:
    torch.manual_seed(7); network=ELBOVAE(); optimizer=torch.optim.Adam(network.parameters(),lr=0.01)
    history=[]
    for step in range(400):
        logits,mu,lv=network(binary)
        nll=F.binary_cross_entropy_with_logits(logits,binary,reduction='none').sum(-1).mean()
        kl=0.5*(mu.square()+lv.exp()-1-lv).sum(-1).mean()
        objective=nll+beta*kl
        optimizer.zero_grad(); objective.backward(); optimizer.step()
        history.append((nll.item(),kl.item()))
    models[beta]=network.eval(); histories[beta]=np.array(history)
fig,axes=plt.subplots(1,2,figsize=(10,3))
for beta,history in histories.items():
    axes[0].plot(history[:,0],label=f'beta={beta}')
    axes[1].plot(history[:,1],label=f'beta={beta}')
axes[0].set_title('Reconstruction NLL / image'); axes[1].set_title('KL / image')
for ax in axes: ax.set_xlabel('step'); ax.legend()
plt.tight_layout(); plt.show()
''')
vae_extra+=pair(r'''
### C2. Реконструкция и генерация — разные операции

**Reconstruction:** $x\to q(z\mid x)\to z\to decoder$. **Prior generation:** $z\sim N(0,I)\to decoder$, без входного изображения. Posterior mode полезен для детерминированного сравнения, но это не случайная генерация. Aggregate posterior $q(z)=\mathbb E_{data}q(z\mid x)$ не обязан точно совпадать с prior даже после KL-регуляризации.

Ниже сравним input, reconstruction и prior samples; scatter posterior means покажет геометрию кодов. Если β слишком мал, reconstruction может быть хорошей, а случайные prior samples — плохими: prior посещает области, с которыми decoder мало работал.
''',r'''
fig,axes=plt.subplots(2,6,figsize=(10,4))
for row,beta in enumerate([0.01,1.0]):
    net=models[beta]
    with torch.no_grad():
        means,lv=net.encode(binary)
        restored=net.decoder(means).sigmoid().reshape(-1,8,8)
        prior=net.decoder(torch.randn(4,2)).sigmoid().reshape(-1,8,8)
    images=[data[0],restored[0],*list(prior)]
    for col,img in enumerate(images):
        axes[row,col].imshow(img,vmin=0,vmax=1,cmap='gray'); axes[row,col].axis('off')
        axes[row,col].set_title(('input' if col==0 else 'mode' if col==1 else 'prior')+f' b={beta}')
    print('beta',beta,'variation of posterior mean:',means.std(0).tolist())
plt.tight_layout(); plt.show()
fig,ax=plt.subplots(figsize=(4,4))
for beta,net in models.items():
    with torch.no_grad(): means,_=net.encode(binary)
    ax.scatter(means[:,0],means[:,1],label=f'beta={beta}',alpha=0.6)
ax.set_xlabel('mu_1'); ax.set_ylabel('mu_2'); ax.legend(); plt.show()
''')
vae_extra+=pair(r'''
## Часть D. Прямой разбор настоящего vae.py Kandinsky

Используем **классы именно сохранённого upstream vae.py**, а не наш MiniVAE и не только библиотечный decoder. Уменьшим widths, latent channels и размеры видео; весов не скачиваем. `quant_conv` 1×1×1 смешивает 2C posterior parameters, `post_quant_conv` — C latent channels. Никакого argmin/codebook здесь нет.

Для CPU вызываем encoder/quant_conv и post_quant_conv/decoder напрямую. Публичный `decode()` upstream всегда обращается к CUDA memory planner: обход planner здесь явный и ограничен маленькой архитектурной проверкой. Структура основной сети сохраняется; полная эквивалентность tiling не утверждается.
''',r'''
import importlib.util
path=SOURCE/'kandinsky/models/vae.py'
spec=importlib.util.spec_from_file_location('_k5_vae_theory',path)
vae_module=importlib.util.module_from_spec(spec); spec.loader.exec_module(vae_module)
small_vae=vae_module.AutoencoderKLHunyuanVideo(
    latent_channels=4,block_out_channels=(8,16,16,16),layers_per_block=1,
    norm_num_groups=4,mid_block_add_attention=False).eval()
video=torch.randn(1,3,9,32,32)
shapes={}
def shape_hook(name):
    def record(module,args,out):
        if isinstance(out,torch.Tensor): shapes[name]=tuple(out.shape)
    return record
handles=[]
for name,layer in small_vae.named_modules():
    if name in ['encoder.conv_in','encoder.down_blocks.0','encoder.down_blocks.1','encoder.down_blocks.2','encoder.down_blocks.3','encoder.conv_out','quant_conv','post_quant_conv','decoder']:
        handles.append(layer.register_forward_hook(shape_hook(name)))
with torch.no_grad():
    moments=small_vae.quant_conv(small_vae.encoder(video))
    posterior=vae_module.DiagonalGaussianDistribution(moments)
    mode=posterior.mode()
    rec=small_vae.decoder(small_vae.post_quant_conv(mode))
for h in handles: h.remove()
print('posterior parameters:',moments.shape,'latent:',mode.shape,'RGB:',rec.shape)
for name,shape in shapes.items(): print(name,'→',shape)
assert moments.shape[1]==2*mode.shape[1]
assert rec.shape==video.shape
''')
vae_extra+=pair(r'''
### D2. Почему время сжимается иначе, чем пространство

Для causal codec первый кадр имеет особую роль: $T_{latent}=1+(T_{RGB}-1)/4$ для совместимых длин $T_{RGB}=4k+1$. Пространство сжимается обычными уровнями downsample. В исходном encoder при temporal compression 4 временные stride 2 стоят на промежуточных уровнях; spatial stride 2 — на первых трёх.

CausalConv3d добавляет padding только в прошлое; это свойство convolution, **не автоматическое доказательство strict causality всей сети**. Нормализации, mid attention и обработка сегментов тоже нужно проверять. GroupNorm может агрегировать по временной оси, поэтому изменение будущего потенциально влияет на нормализованные признаки. Ранее проверенный локальный conv-test проверяет только conv.
''',r'''
for i in range(4):
    spatial=i<3
    temporal=i>=1 and i<3
    print('encoder stage',i,'stride',(2 if temporal else 1,2 if spatial else 1,2 if spatial else 1))
for frames in [1,9,121]:
    latent=(frames-1)//4+1
    print(frames,'RGB →',latent,'latent →',4*(latent-1)+1,'RGB')
source_excerpt('kandinsky/models/vae.py','if temporal_compression_ratio == 4',22)
''')
vae_extra+=pair(r'''
### D3. Стыковка с DiT: VAE prior и noise sampler — не одно распределение

При T2V inference VAE encoder не нужен: DiT начинает с Gaussian noise и выдаёт **распределение data latents**, после чего VAE decoder переводит их в RGB. Encoder нужен для получения data latents при обучении и для image/video conditioning.

Важно: DiT noise $\epsilon\sim N(0,I)$ — начальная точка flow. VAE prior $p(z)=N(0,I)$ — регуляризатор codec. Финальные DiT latents не обязаны оставаться нормальными. $z_{DiT}=s z_{VAE}$; decoder получает $z_{DiT}/s$. Масштабирование не добавляет информационного содержания, но меняет численные масштабы обучения генератора.
''',r'''
scale=small_vae.config.scaling_factor
z_dit=mode*scale
layout=z_dit[0].permute(1,2,3,0).contiguous()
back=layout.permute(3,0,1,2).unsqueeze(0)/scale
assert torch.allclose(mode,back,atol=1e-6)
print('VAE:',mode.shape,'→ DiT:',layout.shape,'scale:',scale)
source_excerpt('kandinsky/generation_utils.py','images = (images / vae.config.scaling_factor)',8)
del small_vae,posterior; release()
''')
vae_extra.append(md(r'''
### D4. Что мы можем утверждать о настоящем обучении VAE

ELBO выше — общая теория. В `vae.py` есть архитектура и posterior, но нет VAE trainer и полной weighting map reconstruction/perceptual/adversarial/KL losses. Нельзя выводить исторический loss Hunyuan только из имени `AutoencoderKL`. Продвинутые visual codecs могут сочетать несколько целей; наша Bernoulli mini-задача не воспроизводит их качество.

**Задания с проверяемым результатом:**

1. Выведите KL для diagonal Gaussian на бумаге и сравните с API для 100 случайных mean/logvar.
2. Проверьте ELBO gap в exact Gaussian примере при q variance 0.01/0.2/2.
3. Повторите β experiment на новых бинарных фигурах и отдельно измерьте validation NLL/KL.
4. Разделите интерполяцию posterior means и prior sampling: почему это разные проверки?
5. Сравните whole-video и tiled reconstruction настоящих весов, измерьте ошибку на границах отдельно.
6. Проверьте воздействие изменения будущих кадров после conv, normalization и encoder целиком; не переносите результат одного слоя на всю сеть.

Источники: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), [HunyuanVideo](https://arxiv.org/abs/2412.03603), pinned `vae.py` в `reference`. Первые источники объясняют метод/codec; конкретные формы и API проверяем по локальному коду.
'''))

flow_intro=[md(r'''
## Часть A. Что значит Flow Matching «внутри DiT»

**DiT — архитектура функции** $v_\theta(x_t,t,c)$; **Flow Matching — способ её обучения**; **Euler — алгоритм интегрирования после обучения**. В `dit.py` нет optimizer или FM loss. Сеть получает промежуточный латент, время и conditioning, а возвращает velocity. Построение training пары и loss находится в trainer, а ODE шаг — в `generation_utils.py`.

В сохранённом репозитории есть inference implementation, а полный pretraining trainer не опубликован. Ниже воспроизводим прозрачную conditional straight-path FM постановку, согласованную со знаком upstream sampler, и отдельно показываем подтверждённый execution path. Не приписываем свой uniform time sampling всей истории обучения/дистилляции Kandinsky.

### A1. Непрерывная картина: множество частиц движется по полю

Представьте облако точек noise. В каждой точке сеть задаёт стрелку; интегратор многократно двигает частицы. Совокупность траекторий преобразует одно распределение в другое. ODE описывает отдельную траекторию:

$$\frac{dx_t}{dt}=v_\theta(x_t,t,c).$$

На уровне плотности это соответствует continuity equation $\partial_t p_t+\nabla\cdot(p_t v_t)=0$: масса перераспределяется, а не исчезает. Для обычной генерации не нужно вычислять плотность или divergence; достаточно velocity forward.

### A2. Соглашение времени именно в нашем Kandinsky sampler

$x_0$ — data latent, $x_1=\epsilon$ — noise. Строим $x_t=(1-t)x_0+t\epsilon$ и target $u=dx_t/dt=\epsilon-x_0$. Training draw может идти по t в любом порядке. **Генерация идёт t=1→0**, поэтому Euler имеет отрицательное $\Delta t$.

В части FM литературы время задаётся noise→data. Это эквивалентное соглашение после $s=1-t$, но скорость меняет знак. Нельзя взять target одного соглашения и sampler другого. [Flow Matching primary paper](https://arxiv.org/abs/2210.02747).
''')]
flow_intro+=pair(r'''
### A3. Визуализируем path и проверяем производную

Пока одна пара data/noise. Straight path — учебная пара, не обещание, что learned sampler по неизвестному noise попадёт именно в этот data sample. Все пары смешиваются в обучении, и сеть не знает $x_0$ напрямую.
''',r'''
x0=torch.tensor([2.,-1.]); eps=torch.tensor([-1.,2.]); ts=torch.linspace(0,1,21)
path=(1-ts[:,None])*x0+ts[:,None]*eps
plt.plot(path[:,0],path[:,1],'.-'); plt.scatter(*x0,label='data t=0'); plt.scatter(*eps,label='noise t=1')
plt.quiver(path[::5,0],path[::5,1],*[torch.full((5,),float(v)) for v in eps-x0],angles='xy',scale_units='xy',scale=5)
plt.axis('equal'); plt.legend(); plt.show()
t=0.4; h=1e-3
finite_difference=(((1-(t+h))*x0+(t+h)*eps)-((1-(t-h))*x0+(t-h)*eps))/(2*h)
assert torch.allclose(finite_difference,eps-x0,atol=1e-3)
print('dx/dt:',finite_difference,'reverse increment has negative dt')
''')
flow_intro+=pair(r'''
### A4. Почему обучение не требует ODE solver

Для каждого training sample известны $x_0$ и выбранный noise. Поэтому $x_t$ и target получаются одной формулой, без 16-step sampling. Обучаем regression:

$$L_{CFM}(\theta)=\mathbb E_{x_0,\epsilon,t,c}\|v_\theta(x_t,t,c)-(\epsilon-x_0)\|^2.$$

Математический optimum squared loss в одной точке — conditional average всех подходящих targets: $v^*(x,t,c)=\mathbb E[\epsilon-x_0\mid x_t=x,t,c]$. Поэтому **условные straight paths могут пересекаться**, а learned marginal trajectories не обязаны быть прямыми. Noise-target pair не может быть восстановлена однозначно по $x_t$.
''',r'''
# Две разные пары пересекаются в одной точке при t=0.5.
a_data=torch.tensor([-2.]); a_noise=torch.tensor([2.])
b_data=torch.tensor([2.]); b_noise=torch.tensor([-2.])
assert torch.equal((a_data+a_noise)/2,(b_data+b_noise)/2)
targets=torch.stack([a_noise-a_data,b_noise-b_data]).flatten()
predictions=torch.linspace(-5,5,101)
loss=((predictions[:,None]-targets[None,:])**2).mean(-1)
print('targets:',targets,'best shared prediction:',predictions[loss.argmin()].item())
plt.plot(predictions,loss); plt.xlabel('shared velocity'); plt.ylabel('MSE at crossing'); plt.show()
''')
flow_intro+=pair(r'''
### A5. Velocity, noise prediction и clean prediction

Для выбранного linear path $x_t=x_0+t u$. Поэтому $\hat x_0=x_t-t\hat v$ и $\hat\epsilon=x_t+(1-t)\hat v$. Velocity — не изображение и не обязательно noise prediction. Это соотношения параметризаций данного path; они не означают, что неполно обученный network выдаст точный $x_0$ за один шаг.

Проверка на известной паре позволяет поймать ошибку знака независимо от нейросети. Затем source показывает: OutLayer действительно возвращает latent channels, а sampler использует их как velocity.
''',r'''
t=torch.tensor(0.37); xt=(1-t)*x0+t*eps; u=eps-x0
assert torch.allclose(xt-t*u,x0,atol=1e-6)
assert torch.allclose(xt+(1-t)*u,eps,atol=1e-6)
print('x0 recovered:',xt-t*u,'noise recovered:',xt+(1-t)*u)
''')
flow_extra=[]
flow_extra+=pair(r'''
## Часть C. Разложим настоящий Kandinsky forward на этапы

Выполним **те же методы**, что вызывает `DiffusionTransformer3D.forward`:

1. Qwen hidden states → Linear/LayerNorm; CLIP pooled → projection; t → sinusoidal/MLP; time+CLIP складываются.
2. Visual latent → patches; text → 1D RoPE и Linguistic Token Refiner.
3. Visual grid → flatten + 3D RoPE → CrossDiT blocks.
4. Modulated OutLayer → unpatchify → 16-channel velocity.

Время ODE t измеряется от 0 до 1, но network получает **t×1000** — это отдельный contract embeddings, не 1000 sampling steps. В input T2V есть дополнительные zero visual channels/mask. На маленьком примере сравним ручное исполнение stages с единым forward.
''',r'''
from labkit.native import tiny_dit
trace_model=tiny_dit('cpu')
trace_x=torch.randn(1,4,4,33); trace_text=torch.randn(1,6,32); trace_clip=torch.randn(1,16)
trace_t=torch.tensor([0.5])*1000
trace_pos=[torch.arange(1),torch.arange(2),torch.arange(2)]; text_pos=torch.arange(6)
with torch.no_grad():
    te,time_emb,text_rope,ve=trace_model.before_text_transformer_blocks(trace_text,trace_t,trace_clip,trace_x,text_pos)
    print('projected text:',te.shape,'time+CLIP:',time_emb.shape,'visual grid:',ve.shape)
    for block in trace_model.text_transformer_blocks: te=block(te,time_emb,text_rope)
    ve,visual_shape,to_fractal,visual_rope=trace_model.before_visual_transformer_blocks(ve,trace_pos,(1,1,1),None)
    print('visual sequence:',ve.shape,'3D RoPE:',visual_rope.shape)
    for block in trace_model.visual_transformer_blocks: ve=block(ve,te,time_emb,visual_rope,None)
    staged=trace_model.after_blocks(ve,visual_shape,to_fractal,te,time_emb)
    direct=trace_model(trace_x,trace_text,trace_clip,trace_t,trace_pos,text_pos)
assert torch.allclose(staged,direct,atol=1e-6)
print('velocity:',direct.shape,'stage/forward difference:',(staged-direct).abs().max().item())
source_excerpt('kandinsky/models/dit.py','text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks',18)
''')
flow_extra+=pair(r'''
### C2. Как время и текст меняют velocity

Cross-attention вводит детальные Qwen tokens; CLIP pooled меняет modulation вместе со временем. В fresh модели modulation gates zero-initialized: некоторые residual ветви сначала могут не влиять на результат. После обучения dependence проверяется экспериментом, а не предполагается по названию attention.

На tiny DiT, обученном ранее для одного фиксированного latent, сравним ответы при разных t. Различие outputs показывает dependence; оно не доказывает точность и не показывает семантическое понимание Qwen. Случайный conditioning в учебном training не эквивалентен настоящим embeddings.
''',r'''
probe=torch.randn_like(x)
probe_input=torch.cat([probe,torch.zeros_like(probe),torch.zeros_like(probe[...,:1])],-1)
with torch.no_grad():
    v_early=model(probe_input,text,pooled,torch.tensor([900.],device=LAB_DEVICE),pos,torch.arange(6,device=LAB_DEVICE))
    v_late=model(probe_input,text,pooled,torch.tensor([100.],device=LAB_DEVICE),pos,torch.arange(6,device=LAB_DEVICE))
print('mean |velocity(t=.9)-velocity(t=.1)|:',(v_early-v_late).abs().mean().item())
''')
flow_extra+=pair(r'''
## Часть D. Полный маленький conditional flow: учим два распределения

Дополнительная 2D задача отделяет статистический смысл FM от сложности video Transformer. Вместо DiT используем маленький MLP, но path, target и solver одинакового типа. Conditioning class=0/1 играет роль двух разных промптов. Data distribution — Gaussian clouds вокруг двух центров, noise — один стандартный Gaussian.

Обучение: случайные data/noise/t → один network forward → MSE → update. Генерация: только noise + выбранный class → повторные velocity forwards; data sample на вход не подаётся. Это принципиальное различие с реконструкцией VAE.
''',r'''
class ToyVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(5,64),torch.nn.SiLU(),torch.nn.Linear(64,64),torch.nn.SiLU(),torch.nn.Linear(64,2))
    def forward(self,x,t,c):
        return self.net(torch.cat([x,t,torch.nn.functional.one_hot(c,2).float()],-1))
torch.manual_seed(12)
field=ToyVelocity(); optimizer=torch.optim.Adam(field.parameters(),lr=0.003)
centers=torch.tensor([[-2.,-0.7],[2.,0.7]]); flow_losses=[]
for step in range(650):
    classes=torch.randint(0,2,(128,)); data2=centers[classes]+0.3*torch.randn(128,2)
    noise2=torch.randn_like(data2); times=torch.rand(128,1)
    mixed=(1-times)*data2+times*noise2
    loss=torch.nn.functional.mse_loss(field(mixed,times,classes),noise2-data2)
    optimizer.zero_grad(); loss.backward(); optimizer.step(); flow_losses.append(loss.item())
plt.plot(flow_losses); plt.xlabel('training step'); plt.ylabel('CFM velocity MSE'); plt.show()
''')
flow_extra+=pair(r'''
### D2. Генерируем без encoder и без data samples

Для каждого class используем одинаковый initial noise: так разница результата зависит от conditioning. Сначала сохраняем intermediate trajectory, потом сравниваем final samples с известными training distribution centers. Ошибка mean и spread — учебные проверки; они не являются image/video quality metrics.

Euler идёт по decreasing t. Uniform grid здесь выбран для ясности; Kandinsky distill grid исследуем следующим разделом. Даже если training paths прямые, sampled trajectory learned field может изгибаться.
''',r'''
field.eval(); initial=torch.randn(300,2); generated={}; traces={}
times=torch.linspace(1,0,41)
with torch.no_grad():
    for class_id in [0,1]:
        particle=initial.clone(); labels=torch.full((len(initial),),class_id,dtype=torch.long)
        trajectory=[particle[0].clone()]
        for current,next_t in zip(times[:-1],times[1:]):
            particle=particle+(next_t-current)*field(particle,current.expand(len(particle),1),labels)
            trajectory.append(particle[0].clone())
        generated[class_id]=particle; traces[class_id]=torch.stack(trajectory)
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].scatter(initial[:,0],initial[:,1],s=6,alpha=.3,label='initial noise')
for class_id,points in generated.items():
    axes[0].scatter(points[:,0],points[:,1],s=6,alpha=.5,label=f'class {class_id}')
    axes[1].plot(traces[class_id][:,0],traces[class_id][:,1],'.-',label=f'class {class_id}')
    print('class',class_id,'mean:',points.mean(0).tolist(),'target:',centers[class_id].tolist(),'std:',points.std(0).tolist())
for ax in axes: ax.set_aspect('equal'); ax.legend()
axes[0].set_title('Generated conditional distributions'); axes[1].set_title('Same noise, different conditions')
plt.tight_layout(); plt.show()
''')
flow_extra+=pair(r'''
## Часть E. Что точно делает Kandinsky 5.0 inference

`get_velocity` → DiT(t×1000) → optional CFG. `generate` → shifted timesteps → visual conditioning → `img += timestep_diff * pred_velocity`. После этого latent делится на VAE scaling factor и декодируется. **Flow Matching не запускается заново при inference:** применяем обученное поле.

Для 5s distill checkpoint config содержит 16 steps и guidance=1.0: conditional velocity без второй uncond ветви. `scheduler_scale` влияет на time grid, а не на архитектуру. Ниже проверяем endpoint/monotonicity и печатаем именно source update. YAML scale=5 и pipeline default=10 расходятся; курс передаёт scale явно.
''',r'''
from omegaconf import OmegaConf
cfg=OmegaConf.load(SOURCE/'configs/k5_lite_t2v_5s_distil_sd.yaml')
grid=shifted_schedule(cfg.model.num_steps,cfg.metrics.scheduler_scale)
assert grid[0]==1 and grid[-1]==0 and torch.all(grid.diff()<0)
print('steps:',cfg.model.num_steps,'guidance:',cfg.model.guidance_weight,'scale:',cfg.metrics.scheduler_scale)
print('first timesteps:',grid[:5],'sum dt:',grid.diff().sum().item())
source_excerpt('kandinsky/generation_utils.py','pred_velocity = dit(',15)
source_excerpt('kandinsky/generation_utils.py','img[..., :pred_velocity.shape[-1]] +=',4)
''')
flow_extra.append(md(r'''
### E2. Дистилляция: почему 16 шагов — отдельные обученные веса

Distilled weights учат student эффективно воспроизводить сокращённую trajectory. На inference архитектура всё равно выдаёт velocity и sampler делает Euler updates. Один только аргумент `steps=16` не превращает SFT weights в distilled model. Снижение NFE не гарантирует меньший размер весов или latent activation peak.

Технический отчёт описывает CFG и trajectory distillation, затем post-training; этот notebook не воспроизводит полный distillation trainer. Прямой simple CFM loss выше объясняет foundation training, а не все цели distill checkpoint. [Kandinsky report, training stages](https://arxiv.org/html/2511.14993v1#S6).

### E3. Объединяем VAE и FM теорию

| Этап | Вход | Цель | Какие веса меняются |
|---|---|---|---|
| VAE training | pixels x | reconstruction NLL + KL; у real codecs возможны дополнительные terms | VAE encoder/decoder |
| DiT FM training | scaled VAE data latent, noise, t, text | velocity regression | DiT; external codec/encoders frozen в учебной схеме |
| Distillation | teacher trajectories / conditions | отдельная student objective | Student DiT |
| Generation | noise + text | forward + ODE integration, без training loss | Никакие |

VAE не «убирает Gaussian noise DiT» итерациями: он один раз кодирует/декодирует. DiT не вычисляет ELBO encoder: он учится transport в уже определённом latent space.

**Задания:**

1. Измените time convention на s=1−t, измените target и sampling direction вместе, сравните результаты.
2. Попробуйте убрать time из ToyVelocity; объясните, почему одна стрелка в x может быть недостаточна.
3. Уберите class conditioning и сравните с двумя conditional clouds.
4. Замерьте Euler 8/16/40 steps при одинаковом noise; считайте NFE отдельно от training iterations.
5. В trained tiny DiT сохраните промежуточные latents и оцените distance до known fixed data latent; сравните с random weights.
6. Проследите source: frozen Qwen → trainable LTF → cross-attention → velocity → Euler → frozen decoder.

Итог проверки: вы должны суметь объяснить каждую переменную в Euler строке upstream и воспроизвести forward через отдельные DiT stages без скрытого trainer.
'''))

for name,intro,extra in [('02_vae.ipynb',vae_intro,vae_extra),('06_dit_and_flow_matching.ipynb',flow_intro,flow_extra)]:
    path=ROOT/'notebooks'/name; nb=json.loads(path.read_text())
    # Existing tutorial remains; insert theory before its first practical section,
    # append deeper experiments after it and before the shared final notes.
    nb['cells'][3]['source']='## Часть B. Базовые опыты и связь с исходниками\n\n'+nb['cells'][3]['source']
    nb['cells']=nb['cells'][:3]+intro+nb['cells'][3:-1]+extra+nb['cells'][-1:]
    for i,cell in enumerate(nb['cells']): cell['id']=f'cell-{i:03}'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    print(name,len(nb['cells']),'cells')
