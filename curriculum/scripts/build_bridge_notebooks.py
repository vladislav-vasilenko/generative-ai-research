"""Build two small teaching notebooks. Never touches original labs or user answers."""
import json
from pathlib import Path
from textwrap import dedent
ROOT=Path(__file__).resolve().parents[2]

def build(path, blocks):
    cells=[]
    for i,(kind,source) in enumerate(blocks):
        cell={'id':f'bridge-{i:03d}','cell_type':kind,'metadata':{},'source':dedent(source).strip()+'\n'}
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python','version':'3.12'}},'cells':cells}
    output=ROOT/path;output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    print(path,len(cells),'cells')

common=r"""
from pathlib import Path
import json, os
import numpy as np
import torch

def find_root():
    for p in (Path.cwd(),*Path.cwd().parents):
        if (p/'curriculum/references/materials_manifest.json').is_file():return p
    raise FileNotFoundError('Open this notebook from generative-ai-research.')
ROOT=find_root();OUT=ROOT/'curriculum/artifacts';OUT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl-cache'))
import matplotlib.pyplot as plt
torch.set_num_threads(4);torch.manual_seed(184)
rng=np.random.default_rng(184)
DEVICE='cpu'  # All default experiments are small and use no downloaded weights.
results={'scope':'small educational CPU experiments; no pretrained inference','mlx':'not_run'}
print('Device:',DEVICE,'Torch:',torch.__version__)
"""
M='markdown';C='code'
foundation=[
(M,r"""
# Мост: многомерные вероятности → Flow Matching → контракты видеомодели

Работайте в паре с [картой курса](../README.md). Здесь нет скачивания датасетов/весов и полной генерации.
Это маленькие проверяемые примеры на CPU PyTorch; они готовят к Lab 1–3, ComfyUI и чтению Hunyuan/Wan/Kandinsky.

Перед каждой ячейкой предскажите результат. После неё запишите, что проверка доказывает и чего не доказывает.
MLX — последний optional этап после корректности PyTorch; по умолчанию не импортируется.
"""),
(C,common),
(M,r"""
## 1. Один объект, batch и совместная вероятность

Одна случайная точка $X\in\mathbb R^d$ имеет covariance matrix $\Sigma$.
Batch `[B,d]` содержит B реализаций такого вектора. Независимость batch samples не означает независимость d координат.

Для Gaussian $s(x)=\nabla_x\log p(x)=-\Sigma^{-1}(x-\mu)$.
Проверим moments выборки и аналитический score через autograd.
"""),
(C,r"""
mu=torch.tensor([1.,-1.],dtype=torch.float64)
cov=torch.tensor([[1.,0.75],[0.75,2.]],dtype=torch.float64)
noise=torch.randn(10000,2,dtype=torch.float64)
samples=mu+noise@torch.linalg.cholesky(cov).T
empirical_cov=torch.cov(samples.T)
assert torch.max(torch.abs(empirical_cov-cov))<0.1
query=samples[:32].clone().requires_grad_(True)
distribution=torch.distributions.MultivariateNormal(mu,covariance_matrix=cov)
grad=torch.autograd.grad(distribution.log_prob(query).sum(),query)[0]
analytic=-(query-mu)@torch.linalg.inv(cov)
score_error=float((grad-analytic).abs().max().detach())
assert score_error<1e-10
print('Sample mean:',samples.mean(0));print('Sample covariance:',empirical_cov)
print('Score max error:',score_error)
results['gaussian_score_error']=score_error
plt.figure(figsize=(5,3));plt.scatter(*samples[:1500].T,s=3,alpha=.2)
plt.xlabel('x1');plt.ylabel('x2');plt.title('Коррелированные координаты одной Gaussian');plt.tight_layout();plt.show()
"""),
(M,r"""
**Вопрос:** как изменится плотность при замене covariance на её diagonal?
Почему вероятность конкретного вектора равна нулю для continuous law, хотя density положительна?
Не интерпретируйте один pixel как независимую случайную величину без такого предположения модели.

## 2. Два time conventions описывают один путь

В MIT: $x_s=(1-s)\epsilon+s\,x_{data}$, velocity $x_{data}-\epsilon$, sampling $s:0\to1$.
В data-to-noise convention: $x_t=(1-t)x_{data}+t\epsilon$, velocity $\epsilon-x_{data}$, sampling $t:1\to0$.
Подставьте $s=1-t$. Ниже проверяем состояния и знак скорости; это не утверждение об одинаковом scheduler всех моделей.
"""),
(C,r"""
data=torch.tensor([2.,-1.],dtype=torch.float64)
noise=torch.tensor([-1.,1.5],dtype=torch.float64)
s=torch.linspace(0,1,41,dtype=torch.float64)[:,None];t=1-s
mit_path=(1-s)*noise+s*data
industrial_path=(1-t)*data+t*noise
assert torch.allclose(mit_path,industrial_path)
assert torch.allclose(data-noise,-(noise-data))
x=noise.clone()
for _ in range(20):x+=(-1/20)*(noise-data)
assert torch.allclose(x,data)
results['time_reversal_max_error']=float((mit_path-industrial_path).abs().max())
plt.figure(figsize=(5,3));plt.plot(*mit_path.T,'o-',markersize=3)
plt.scatter(*noise,label='noise');plt.scatter(*data,label='data')
plt.legend();plt.title('Один путь, две параметризации времени');plt.tight_layout();plt.show()
"""),
(M,r"""
## 3. Conditional endpoint не равен marginal predictor

Пусть чистый endpoint принимает значения $z\in\{-2,2\}$ с probabilities .3/.7.
При фиксированном t: $p_t(x\mid z)=\mathcal N(tz,(1-t)^2)$.
Conditional velocity $u(x,t\mid z)=(z-x)/(1-t)$ известна. Sampler не знает z;
ему нужно $u(x,t)=\sum_z p(z\mid x_t=x)u(x,t\mid z)$.

Posterior weights зависят от x. Проверим также Gaussian-path relation:
$u=x/t+(1-t)s/t$ для interior t. Формула не предназначена для t=0 или 1 без анализа пределов.
"""),
(C,r"""
time=0.6;beta=1-time
endpoints=torch.tensor([-2.,2.],dtype=torch.float64)
prior=torch.tensor([.3,.7],dtype=torch.float64)
x=torch.linspace(-3,3,201,dtype=torch.float64).requires_grad_(True)
log_components=-.5*((x[:,None]-time*endpoints)/beta)**2+prior.log()
posterior=log_components.softmax(-1)
conditional_velocity=(endpoints-x[:,None])/beta
marginal_velocity=(posterior*conditional_velocity).sum(-1)
wrong_prior_average=(prior*conditional_velocity).sum(-1)
log_marginal=torch.logsumexp(log_components,dim=-1)-np.log(beta*np.sqrt(2*np.pi))
score=torch.autograd.grad(log_marginal.sum(),x)[0]
conversion=x/time+beta/time*score
error=float((marginal_velocity-conversion).abs().max().detach())
assert error<1e-10
results['marginal_velocity_score_conversion_error']=error
fig,axes=plt.subplots(1,2,figsize=(9,3))
axes[0].plot(x.detach(),posterior.detach());axes[0].set_title('Posterior weights зависят от x')
axes[1].plot(x.detach(),marginal_velocity.detach(),label='posterior average')
axes[1].plot(x.detach(),wrong_prior_average.detach(),'--',label='prior average: неверно')
axes[1].legend();axes[1].set_title('Поле sampler');plt.tight_layout();plt.show()
"""),
(M,r"""
**Связь с обучением:** MSE regression на endpoint-specific targets имеет оптимум в виде conditional expectation.
В реальном Lab 2 marginal field аппроксимирует сеть. Этот пример вычисляет его точно и не выполняет training.

## 4. CFG: две ветви — не две разные архитектуры

Для одной сети, обученной с condition dropout:
$u_w=u_\varnothing+w(u_y-u_\varnothing)$.
Проверим w=0/1 и equivalence batched/separate evaluation на маленькой случайной MLP.
Здесь **не обучалось условное изображение**: проверяются tensor contracts и affine combination.
"""),
(C,r"""
net=torch.nn.Sequential(torch.nn.Linear(5,16),torch.nn.SiLU(),torch.nn.Linear(16,2)).eval()
x=torch.randn(8,2);time=torch.ones(8,1)*.4
null=torch.zeros(8,2);condition=torch.ones(8,2)
with torch.no_grad():
    u0=net(torch.cat([x,time,null],-1));uy=net(torch.cat([x,time,condition],-1))
    both=net(torch.cat([torch.cat([x,time,null],-1),torch.cat([x,time,condition],-1)],0))
assert torch.allclose(both[:8],u0) and torch.allclose(both[8:],uy)
assert torch.allclose(u0+0*(uy-u0),u0)
assert torch.allclose(u0+1*(uy-u0),uy)
print('Two separate batch-8 calls = one batch-16 call; branch work remains two samples per original sample.')
results['cfg_identities']='passed; random network, no conditional generation claim'
"""),
(M,r"""
## 5. Patchify и стоимость tokens

Чистая перестановка pixels в patches обратима. Learned projection в hidden dimension сама по себе не обязана быть обратимой.
Проверим перестановку через unfold/fold. Затем **только арифметически** посчитаем размер naive attention scores;
большой tensor не выделяется. Flash/SDPA kernels могут не материализовывать всю эту матрицу.
"""),
(C,r"""
image=torch.arange(2*3*8*12,dtype=torch.float32).reshape(2,3,8,12)
patches=torch.nn.functional.unfold(image,kernel_size=2,stride=2).transpose(1,2)
restored=torch.nn.functional.fold(patches.transpose(1,2),output_size=(8,12),kernel_size=2,stride=2)
assert torch.equal(image,restored)
print('Image:',tuple(image.shape),'Patches:',tuple(patches.shape))
# Illustrative codec t4/s8 + patch (1,2,2), NOT a universal shape rule.
for frames,height,width in [(33,240,432),(121,480,864)]:
    assert (frames-1)%4==0 and height%16==0 and width%16==0
    latent_t=1+(frames-1)//4
    n=latent_t*(height//16)*(width//16)
    naive_bytes=1*16*n*n*2  # B=1, heads=16, fp16 scores only
    print((frames,height,width),'tokens=',n,'naive scores GiB=',round(naive_bytes/2**30,2))
results['patch_roundtrip']='exact; no learned projection involved'
"""),
(M,r"""
## 6. VAE: std, KL и заморозка

Если encoder выдаёт logvar, $z=\mu+\exp(\tfrac12\,logvar)\epsilon$.
Для diagonal Gaussian KL к standard normal:
$\frac12\sum_j(\mu_j^2+\exp(logvar_j)-1-logvar_j)$.
Сначала суммируем latent dimensions, затем усредняем batch; другое reduction меняет относительный вес KL.

Во второй части замораживаем toy encoder/decoder и проверяем gradient только у latent predictor.
Это проверка wiring будущего LDM, **не обучение настоящего VAE/DiT**.
"""),
(C,r"""
mu=torch.zeros(4,6);logvar=torch.zeros_like(mu)
kl=.5*(mu.square()+logvar.exp()-1-logvar).sum(-1).mean()
assert kl.item()==0
std=(.5*torch.tensor(np.log(4.))).exp()
assert torch.allclose(std,torch.tensor(2.,dtype=std.dtype))
codec=torch.nn.ModuleDict({'encoder':torch.nn.Linear(8,4),'decoder':torch.nn.Linear(4,8)}).requires_grad_(False)
predictor=torch.nn.Linear(4,4)
before={k:v.detach().clone() for k,v in codec.state_dict().items()}
with torch.no_grad():latent=codec['encoder'](torch.randn(16,8))
loss=(predictor(latent)-torch.randn_like(latent)).square().mean();loss.backward()
assert all(p.grad is None for p in codec.parameters())
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in predictor.parameters())
assert all(torch.equal(v,before[k]) for k,v in codec.state_dict().items())
results['frozen_codec_gradients']='none; predictor gradients finite'
print('KL(q=N(0,I) || N(0,I)) =',kl.item(),'; frozen codec wiring passed')
"""),
(M,r"""
## 7. Optional MLX parity — после PyTorch correctness

По умолчанию пропущено; не устанавливает MLX и не скачивает weights.
Если позже создадите отдельное MLX окружение с Torch, передавайте одинаковые массивы,
а не просто одинаковый seed. Здесь только Linear+MSE, не порт K6.
Latency сравнивайте отдельно после warm-up с принудительной evaluation/synchronization.
"""),
(C,r"""
RUN_MLX=False
if RUN_MLX:
    import mlx.core as mx
    xx=rng.normal(size=(8,4)).astype(np.float32)
    ww=rng.normal(size=(3,4)).astype(np.float32);bb=np.zeros(3,dtype=np.float32)
    pt_w=torch.tensor(ww,requires_grad=True);pt_b=torch.tensor(bb,requires_grad=True)
    pt_y=torch.tensor(xx)@pt_w.T+pt_b;pt_loss=pt_y.square().mean()
    pt_grads=torch.autograd.grad(pt_loss,[pt_w,pt_b])
    params={'w':mx.array(ww),'b':mx.array(bb)}
    def objective(p):return mx.mean((mx.array(xx)@p['w'].T+p['b'])**2)
    mx_loss,mx_grads=mx.value_and_grad(objective)(params);mx.eval(mx_loss,mx_grads)
    error=max(float(np.max(np.abs(np.array(mx_grads[k])-g.detach().numpy()))) for k,g in zip(['w','b'],pt_grads))
    assert error<1e-5
    results['mlx']={'linear_mse_gradient_max_error':error,'scope':'tiny Linear only'}
else:
    print('MLX skipped. CPU educational checks remain usable without it.')
(OUT/'probability-and-contracts-results.json').write_text(json.dumps(results,indent=2)+chr(10))
print(json.dumps(results,indent=2))
"""),
(M,r"""
## Что делать дальше

- Перед Lab 1: объяснить batch vs dimensions и Gaussian score.
- Перед Lab 2: вывести reverse-time sign, conditional/marginal field и область определения conversion.
- Перед Lab 3: проверить CFG, patch shapes, logvar и frozen-stage optimizer.
- Для ComfyUI: выполнить [C0/C4](../COMFYUI_LABS.md), сохранив те же assumptions и честные ограничения измерений.
- Для реальной модели: читать [архитектурные точки входа](../INDUSTRY_BRIDGE.md), а не переносить toy dimensions автоматически.

Эти numerical checks не заменяют самостоятельную реализацию и обучение исходных Lab 1–3.
""")]

ctmc=[
(M,r"""
# Lecture 5: CTMC, дискретные paths и маленький denoiser

Дополнение к [Lecture 5](README.md), созданное по §7 notes / slides pp.5–28.
Это новая учебная работа, не официальный четвёртый MIT lab.
Все default cells работают на CPU, без downloads и больших весов.

Сначала сравним exact CTMC law с численным и Monte Carlo расчётом. Затем обучим маленький
categorical posterior для двух зависимых tokens. Это ещё не генеративная language model.
"""),
(C,common),
(M,r"""
## 1. Rate matrix — не transition probability matrix

Фиксируем row convention: Q[i,j] — rate из i в j; off-diagonal entries неотрицательны,
диагональ равна минус сумме outgoing rates, строки суммируются в 0.
Для row probability vector: $dp/dt=pQ$, $p(t)=p(0)\exp(tQ)$.
Вероятности переходов за малый h: $P(h)\approx I+hQ$; rate может быть больше 1.
"""),
(C,r"""
a,b=1.2,.7
Q=torch.tensor([[-a,a],[b,-b]],dtype=torch.float64)
p0=torch.tensor([1.,0.],dtype=torch.float64)
assert torch.allclose(Q.sum(-1),torch.zeros(2,dtype=Q.dtype))
assert (Q[~torch.eye(2,dtype=torch.bool)]>=0).all()
T=2.;exact=p0@torch.matrix_exp(T*Q)
stationary=torch.tensor([b,a],dtype=Q.dtype)/(a+b)
assert torch.allclose(stationary@Q,torch.zeros(2,dtype=Q.dtype),atol=1e-12)
assert torch.allclose(exact.sum(),torch.tensor(1.,dtype=Q.dtype))
ts=torch.linspace(0,4,101,dtype=Q.dtype)
curve=torch.stack([p0@torch.matrix_exp(t*Q) for t in ts])
plt.figure(figsize=(6,3));plt.plot(ts,curve[:,1],label='P(Xt=1) exact')
plt.axhline(float(stationary[1]),ls='--',label='stationary');plt.legend();plt.xlabel('time');plt.tight_layout();plt.savefig(OUT/'ctmc_probability.png');plt.show()
print('Exact at T:',exact,'Stationary:',stationary)
"""),
(M,r"""
## 2. Ошибка шага и нарушение positivity

При слишком большом h матрица I+hQ имеет отрицательные элементы и не является transition matrix.
Не исправляйте её clipping без объявления другого алгоритма.
Сравним допустимые Euler grids с exact matrix exponential.
"""),
(C,r"""
euler_rows=[]
for steps in [1,4,16,64,256]:
    h=T/steps;transition=torch.eye(2,dtype=Q.dtype)+h*Q
    valid=bool((transition>=0).all())
    state=p0@torch.linalg.matrix_power(transition,steps)
    error=float((state-exact).abs().max())
    euler_rows.append({'steps':steps,'h':h,'transition_valid':valid,'max_probability_error':error})
    print(euler_rows[-1])
assert not euler_rows[0]['transition_valid']
valid_rows=[r for r in euler_rows if r['transition_valid']]
assert valid_rows[-1]['max_probability_error']<valid_rows[0]['max_probability_error']
results['finite_step']=euler_rows
"""),
(M,r"""
## 3. Gillespie: точные времена скачков для постоянного Q

В state i время ожидания имеет Exponential(rate=-Qii); destination распределён пропорционально outgoing rates.
В двух состояниях destination единственный. Exact относится к simulation law при постоянном Q;
finite number of trajectories всё равно даёт Monte Carlo error.
"""),
(C,r"""
N=12000
states=np.zeros(N,dtype=np.int64);times=np.zeros(N);jumps=np.zeros(N,dtype=np.int64)
while True:
    active=np.flatnonzero(times<T)
    if not len(active):break
    rates=np.where(states[active]==0,a,b)
    next_times=times[active]+rng.exponential(1/rates)
    take=next_times<T;moving=active[take]
    states[moving]=1-states[moving];jumps[moving]+=1
    times[active]=next_times
empirical=float(states.mean());truth=float(exact[1])
standard_error=np.sqrt(truth*(1-truth)/N)
assert abs(empirical-truth)<5*standard_error+1/N
results['gillespie']={'trajectories':N,'empirical_probability_state_1':empirical,
    'exact_probability_state_1':truth,'binomial_standard_error':standard_error,'total_jumps':int(jumps.sum())}
print(results['gillespie'])
plt.figure(figsize=(5,3));plt.bar(['exact','Monte Carlo'],[truth,empirical],yerr=[0,1.96*standard_error])
plt.ylim(0,1);plt.ylabel('P(XT=1)');plt.tight_layout();plt.show()
"""),
(M,r"""
## 4. Factorized masking path для зависимых tokens

Данные: с вероятностью 1/2 строка `00`, с вероятностью 1/2 строка `11`.
Marginal каждого token одинакова, но tokens зависимы. Corruption независимо оставляет каждый token
с вероятностью kappa(t)=t, иначе заменяет на MASK=2.

Denoising target — categorical posterior clean token. Для `[MASK,MASK]` он равен [.5,.5];
если увидели хотя бы один clean token, второй однозначно известен.
Обучим MLP с cross-entropy. Независимый corruption не делает data tokens независимыми.
Для полностью masked input правильный posterior остаётся неопределённым: population loss не должен стремиться к нулю.
На фиксированном validation вычислим NLL аналитического Bayes posterior как ориентир, а не будем оценивать успех по нулевому loss.
"""),
(C,r"""
MASK=2
class TokenPosterior(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(3,8)
        self.net=torch.nn.Sequential(torch.nn.Linear(17,32),torch.nn.SiLU(),torch.nn.Linear(32,4))
    def forward(self,x,t):return self.net(torch.cat([self.embedding(x).flatten(1),t],-1)).reshape(-1,2,2)

def batch(n,generator=None):
    bit=torch.randint(2,(n,1),generator=generator)
    clean=bit.repeat(1,2);t=.05+.9*torch.rand(n,1,generator=generator)
    keep=torch.rand(n,2,generator=generator)<t
    noisy=torch.where(keep,clean,torch.full_like(clean,MASK))
    return noisy,t,clean

model=TokenPosterior();optimizer=torch.optim.Adam(model.parameters(),lr=.02)
val=batch(4096,torch.Generator().manual_seed(900))
oracle_validation_ce=float((val[0]==MASK).all(dim=1).float().mean())*np.log(2)
losses=[];validation=[]
with torch.no_grad():initial=float(torch.nn.functional.cross_entropy(model(val[0],val[1]).reshape(-1,2),val[2].reshape(-1)))
for step in range(140):
    noisy,t,clean=batch(256)
    logits=model(noisy,t)
    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,2),clean.reshape(-1))
    optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
    if step%5==0:
        with torch.no_grad():validation.append((step,float(torch.nn.functional.cross_entropy(model(val[0],val[1]).reshape(-1,2),val[2].reshape(-1)))))
with torch.no_grad():final=float(torch.nn.functional.cross_entropy(model(val[0],val[1]).reshape(-1,2),val[2].reshape(-1)))
assert np.isfinite(final) and final<initial
results['toy_denoising']={'parameters':sum(p.numel() for p in model.parameters()),'steps':140,
 'initial_validation_cross_entropy':initial,'final_validation_cross_entropy':final,
 'bayes_posterior_validation_cross_entropy':oracle_validation_ce,
 'scope':'two-token synthetic distribution only; not natural text or full CTMC generation'}
print(results['toy_denoising'])
plt.figure(figsize=(6,3));plt.plot(losses,alpha=.5,label='train');plt.plot(*zip(*validation),label='validation')
plt.axhline(oracle_validation_ce,ls='--',label='Bayes posterior on validation')
plt.xlabel('training step');plt.ylabel('token cross-entropy');plt.legend();plt.tight_layout();plt.show()
"""),
(M,r"""
## 5. Сверка с Bayes posterior и построение локальных rates

При interior t outgoing rates для masked token равны posterior/(1-t).
Диагональ устанавливается как минус outgoing sum. Проверка одной локальной rate row не заменяет
sampling полной sequence CTMC по временам.
"""),
(C,r"""
cases=torch.tensor([[2,2],[0,2],[2,1],[0,0],[1,1]])
expected=torch.tensor([[.5,.5],[1.,1.],[0.,0.],[1.,1.],[0.,0.]]) # P(clean token = 0), both positions
with torch.no_grad():predicted=model(cases,torch.full((5,1),.5)).softmax(-1)[...,0]
posterior_error=float((predicted-expected).abs().max())
print('Exact P(token=0):',expected);print('Predicted:',predicted)
print('Max posterior error:',posterior_error)
# For masked state at t=.5, destination probabilities are probabilities for 0/1.
prob=torch.tensor([.5,.5],dtype=torch.float64);t=.5
rate_row=torch.cat([prob/(1-t),torch.tensor([-1/(1-t)],dtype=prob.dtype)])
assert rate_row[:2].min()>=0 and torch.allclose(rate_row.sum(),torch.tensor(0.,dtype=prob.dtype))
results['toy_posterior_max_error']=posterior_error
results['full_learned_ctmc_sampling']='not_implemented_in_this_intro_lab'
(OUT/'ctmc-results.json').write_text(json.dumps(results,indent=2)+chr(10))
print('Saved:',OUT/'ctmc-results.json')
"""),
(M,r"""
## Упражнения и вывод

1. Изменить rates при том же stationary ratio: как меняется скорость mixing?
2. Сравнить Monte Carlo uncertainty при N и 4N. Почему error не обязан падать в каждом отдельном run?
3. Сделать initial distribution произвольной; проверить exact law и Gillespie.
4. Заменить зависимые data tokens независимыми. Что меняется в posterior для частично masked input?
5. Реализовать генерацию sequence CTMC на t<1, отдельно контролируя growing rates возле endpoint.
6. Написать вывод по сохранённым числам: отдельно numerical integration, Monte Carlo и learned posterior errors.

Этот notebook показывает CTMC law и denoising objective. Он не утверждает обучение language model,
не реализует production masked-language sampler и не превращает continuous video DiT в CTMC.
Для ComfyUI можно импортировать сохранённый plot как image; стандартный KSampler не вычисляет этот алгоритм.
""")]
build('curriculum/notebooks/00_probability_and_model_contracts.ipynb',foundation)
build('lecture_05/05_ctmc_and_discrete_flow.ipynb',ctmc)
