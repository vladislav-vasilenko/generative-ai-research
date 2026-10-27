"""Build the K6 component course. Only source generation, no network/weights."""
from pathlib import Path
from textwrap import dedent
import json

ROOT = Path(__file__).resolve().parents[1]
CELLS = []
CATALOG = []
REV = '7ceeefad7fb625011f7db602c43893a989f19c15'
PAPER = 'https://huggingface.co/papers/2610.05608'
ARXIV = 'https://arxiv.org/html/2610.05608v1'


def md(text):
    CELLS.append({'cell_type': 'markdown', 'metadata': {}, 'source': dedent(text).strip()+'\n'})


def code(text):
    CELLS.append({'cell_type': 'code', 'metadata': {}, 'source': dedent(text).strip()+'\n',
                  'execution_count': None, 'outputs': []})


def start(stem, title, goal, prerequisites, sources, section, k5):
    CELLS.clear()
    CATALOG.append({'stem': stem, 'title': title, 'goal': goal, 'sources': sources,
                    'paper_section': section, 'prerequisites': prerequisites})
    links = '\n'.join(f'- [`{p}`](https://github.com/kandinskylab/kandinsky-6/blob/{REV}/kandinsky/{p})' for p in sources)
    md(f'''# {title}

    **Результат:** {goal}

    **Перед началом:** {prerequisites}. [Порядок всех материалов](../../curriculum/CONTENTS.md).
    [Курс K6](../README.md) · [Покомпонентная карта](../COMPONENT_GUIDE.md) · [Предыдущий опыт K5]({k5}).

    Основной архитектурный источник: [Kandinsky 6 на Hugging Face Papers]({PAPER}),
    [полный отчёт, §{section}]({ARXIV}). Открывайте статью рядом с notebook.
    Мы различаем опубликованную архитектуру, конкретный checkpoint config и результат нашего опыта.

    **Код для сравнения (закреплённая revision `{REV[:8]}`):**
    {links}

    Run All выполняет небольшие CPU-опыты. Случайная уменьшенная сеть проверяет вычисления;
    pretrained генерацию, качество голоса и производительность M3 она не измеряет.
    Большие веса не скачиваются, ComfyUI jobs не отправляются. В Colab первая ячейка получает
    исходники курса из GitHub, если их ещё нет; пакеты не устанавливает. На Mac используется существующий checkout.
    Ниже объяснение идёт перед каждым опытом.
    '''.replace('\n    ', '\n'))
    code(f'''
    from pathlib import Path
    import json, sys, os, math
    for parent in (Path.cwd(), *Path.cwd().parents):
        candidates = (parent, parent / 'kandinsky-6')
        found = next((p for p in candidates if (p / 'labkit/components.py').is_file()), None)
        if found is not None:
            ROOT = found.resolve()
            break
    else:
        if not Path('/content').is_dir():
            raise FileNotFoundError('Откройте notebook внутри checkout generative-ai-research.')
        import subprocess
        checkout=Path('/content/generative-ai-research-k6-components')
        if not checkout.exists():
            subprocess.run(['git','clone','--depth','1','--branch',
                os.environ.get('K6_LABS_REF','kandinsky-6-components'),
                'https://github.com/vladislav-vasilenko/generative-ai-research.git',str(checkout)],check=True)
        ROOT=checkout/'kandinsky-6'
        if not (ROOT/'labkit/components.py').is_file():
            raise FileNotFoundError('В выбранной revision нет component labs; проверьте K6_LABS_REF.')
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts/component_labs/mpl-cache'))
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    import torch.nn.functional as F
    from labkit.components import (config, native, show_source, native_causal_conv,
        tiny_model, tiny_inputs, activate_gates, fresh_forward, write_metrics, figure_path)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    LAB = {stem!r}
    metrics = {{}}
    print('CPU teaching mode; PyTorch', torch.__version__)
    ''')


def finish(exercises, conclusion):
    md('## Самостоятельное исследование\n\n'+dedent(exercises).strip()+
       '\n\nПеред изменением запишите гипотезу. Сохраните baseline, меняйте один фактор и повторяйте по 3–5 seeds. '
       'Файл metrics.json содержит измерение конкретного запуска, а не статистическую оценку качества модели.')
    code('write_metrics(LAB, metrics)')
    md('## Критерий понимания\n\n'+dedent(conclusion).strip()+
       '\n\n[Вернуться к маршруту K6](../README.md) · [Полное содержание](../../curriculum/CONTENTS.md) · '
       '[Шаблон отчёта](../../curriculum/EXPERIMENT_TEMPLATE.md).')
    item=CATALOG[-1]
    for i,cell in enumerate(CELLS):cell['id']=f'cell-{i:03d}'
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'display_name':'Python 3 (K6 CPU labs)','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'},
        'k6_component_lab':{'source_revision':REV,'pretrained':False,'paper':PAPER}},'cells':list(CELLS)}
    (ROOT/'notebooks'/f"{item['stem']}.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')


start('01_system_and_training_map', '01. K6 Lite: вся система и training map',
      'восстановить состав checkpoint и посчитать параметры каждого потока без выделения памяти под 3B весов',
      'вводный probability notebook; K5 00',
      ['core/components/dit.py','core/components/piflow_dit.py','runtime/sampler.py'], '3.1–3.2, 4',
      '../../kandinsky-5/notebooks/00_start_and_training_map.ipynb')
md(r'''
## 1. Модель из 3B параметров — это ещё не весь pipeline

Сначала текст превращается в Qwen token states и CLIP pooled state. Два **разных обучаемых projection/refiner**
внутри DiT готовят их для видео и аудио. Gaussian noise живёт в двух разных latent spaces.
CrossDiT обновляет оба состояния; затем Hunyuan VAE декодирует видео, MMAudio VAE — mel,
а отдельный vocoder — waveform. Mux только объединяет результаты.

Это совместная модель распределения видео и аудио при данном тексте. Отдельный audio-forward
не содержит межмодального обмена. Он полезен для контроля кода, но не воспроизводит полную AV систему.

### Что именно обучается

В нашем полном FM-опыте обучаем DiT; внешние Qwen/CLIP/кодеки frozen. Это явный контракт опыта.
Исторические стадии в статье отличаются: separate → joint pretraining → SFT → RL → distillation.
Особенно сравните §4.3: в SFT stage 1 заморожены audio и fusion, в stage 2 DiT-модули размораживаются.
Не переносите слово «frozen» с одного этапа на всю историю модели. Inference вообще не имеет optimizer.
''')
code('''
index = json.loads((ROOT/'reference/hf/Kandinsky-6.0-Lite-distill-5s-Diffusers/model_index.json').read_text())
for name in ['transformer','text_encoder','tokenizer','text_encoder_2','tokenizer_2','vae','audio_vae','vocoder','scheduler']:
    print(f'{name:16s}', index[name])
cfg=config(); print('video/audio widths:',cfg['model_dim'],cfg['model_dim_a'])
assert cfg['out_visual_dim']==160 and cfg['out_audio_dim']==400
assert cfg['out_visual_dim']//cfg['in_visual_dim']==cfg['out_audio_dim']//cfg['in_audio_dim']==10
''')
md(r'''
## 2. Считать параметры можно без загрузки весов

`meta` хранит формы без значений. Конструктор создаёт правильную структуру, а `numel()` считает элементы.
Из checkpoint config убираем `scale_factor`: это параметр RoPE при вызове, не аргумент конструктора.
Используем **обычный native DiT с уже расширенными выходами 160/400**. Если передать эти выходы
PiFlow-wrapper, который сам умножает на 10, получится ошибочное двойное расширение.

Всё, что начинается с `audio_` или принадлежит `audio_dec_block`, относим к аудиотракту;
`va_`/`av_` — к fusion; остальное — к видео. Это прозрачная группировка state_dict,
а не заявление о независимых файлах весов. В опубликованном transformer они объединены.
Обзорные 2B/0.6B/0.4B из статьи — округлённые архитектурные оценки. Наш numel включает
все instantiated parameters и явно относит cross-modal modulation к fusion; эти группировки не тождественны.
''')
code('''
kwargs={k:v for k,v in cfg.items() if k!='scale_factor'}
kwargs['attention_engine']='sdpa'
with torch.device('meta'):
    full=native().DiffusionTransformer3D(**kwargs)
counts={'video':0,'audio':0,'fusion':0}
for name,p in full.named_parameters():
    group=('fusion' if any(s in name for s in ['.va_','.av_']) else
           'audio' if name.startswith('audio_') or '.audio_dec_block.' in name else 'video')
    counts[group]+=p.numel()
print('Exact parameter count:',counts,'total=',sum(counts.values()))
metrics['parameters']=counts;metrics['total_parameters']=sum(counts.values())
metrics['bf16_transformer_only_GiB']=sum(counts.values())*2/2**30
print('BF16 weights only GiB:',metrics['bf16_transformer_only_GiB'])
fig,ax=plt.subplots(figsize=(7,3));ax.bar(counts.keys(),[n/1e9 for n in counts.values()])
ax.set_ylabel('B parameters');ax.set_title('Lite Distill: counted from native constructor')
fig.tight_layout();fig.savefig(figure_path(LAB,'parameter_groups'));plt.show()
del full
''')
md('''
## 3. Связь с ресурсами

BF16-байты параметров — только нижняя оценка хранения DiT. Активации, attention, Qwen,
VAE, allocator и optimizer добавляют память. Unified Memory — общий бюджет GPU/CPU,
а не сумма независимых VRAM и RAM. Adam training может быть существенно тяжелее inference.
SR и prompt beautifier — дополнительные ветви, не обязательные для минимального Lite baseline.
''')
finish('''
1. Сравните plain Lite и Distill configs: какие формы головы изменились, какие backbone widths остались?
2. Составьте три optimizer maps: полный FM, SFT stage 1, наш frozen-FiLM эксперимент.
3. Добавьте память gradients и двух Adam moments в FP32; подпишите, что активации ещё не учтены.
''','''
Вы можете пройти путь от prompt до MP4 и назвать для каждого узла источник весов,
вход/выход, trainable/frozen статус в выбранном опыте. Число «3B» не является оценкой размера всего pipeline.
''')

start('02_text_conditioning', '02. Текст: tokenizer, Qwen, CLIP и два refiner',
      'различать IDs, hidden states, pooled condition и padding mask; проверить crop и text-projection cache',
      'K6 01; attention из Lecture 4', ['core/components/text_embedder.py','core/components/dit.py'], '3.1–3.2',
      '../../kandinsky-5/notebooks/01_text_embedders.ipynb')
md(r'''
## 1. Четыре разных представления текста

Tokenizer возвращает целые IDs; embedding lookup превращает ID в вектор; Qwen contextualizes
вектор с учётом других токенов; CLIP даёт pooled representation. K6 использует Qwen states ширины 3584
и CLIP pooled 768. После crop доступно до 1024 токенов; CLIP ограничен 77.
Текстовый VLM используется как encoder. Для T2AV его image/video inputs пустые.

У frozen Qwen один набор весов, но DiT содержит **два собственных trainable refiner**:
`video_text_transformer_blocks` и `audio_text_transformer_blocks`. Выход общего encoder
не равен выходу этих refiner. `text_embedder.py` native пути передаёт Qwen attention mask.

Ниже искусственные IDs показывают только crop/padding. Это **не реализация Qwen tokenizer**:
мы не имитируем его BPE/byte vocabulary обычным split по пробелу.
''')
code('''
crop,max_length=129,8
ids=torch.arange(2*(crop+max_length)).reshape(2,crop+max_length)
valid_lengths=torch.tensor([crop+6,crop+3])
mask=torch.arange(crop+max_length)[None,:]<valid_lengths[:,None]
kept_ids=ids[:,crop:];kept_mask=mask[:,crop:]
assert kept_ids.shape==(2,8) and kept_mask.sum(1).tolist()==[6,3]
print('IDs after crop:',kept_ids);print('valid:',kept_mask)
fig,ax=plt.subplots(figsize=(7,2));ax.imshow(kept_mask,cmap='Blues',vmin=0,vmax=1)
ax.set_xlabel('Token after system-template crop');ax.set_yticks([0,1]);plt.show()
metrics['valid_tokens_after_crop']=kept_mask.sum(1).tolist()
''')
md(r'''
## 2. Маска ограничивает доступ к K/V

В PyTorch SDPA boolean `True` означает разрешённый key. Мы изменим padded value на очень большое
число. С правильной маской результат не меняется. Это проверка контракта downstream attention,
не достаточная проверка tokenizer/template: реальный crop зависит от конкретных processor assets.
''')
code('''
q=torch.randn(2,2,3,4);k=torch.randn(2,2,8,4);v=torch.randn(2,2,8,4)
attention_mask=kept_mask[:,None,None,:]
a=F.scaled_dot_product_attention(q,k,v,attn_mask=attention_mask)
v2=v+((~kept_mask)[:,None,:,None])*1000
b=F.scaled_dot_product_attention(q,k,v2,attn_mask=attention_mask)
error=(a-b).abs().max().item();assert error==0
metrics['masked_value_perturbation_max_error']=error
print('Masked value perturbation error:',error)
show_source('core/components/text_embedder.py','encode_qwen')
''')
md('''
## 3. Что можно кэшировать, а что зависит от времени

Native DiT кэширует time-independent token/pooled projections, затем на каждом шаге
прибавляет time embedding и выполняет time-modulated text blocks. Проверим hit по hooks.
Ключ использует data pointer и shape: нельзя менять cached tensor in-place и ожидать автоматической
инвалидации. Кэш очищают перед новой генерацией. В учебном training очищаем перед каждым forward,
чтобы не переиспользовать старый autograd graph.
''')
code('''
model=tiny_model().eval();_,_,kw=tiny_inputs(model)
hits=[]
hook=model.video_text_embeddings.register_forward_hook(lambda *args:hits.append(1))
with torch.no_grad():
    model.clear_text_proj_cache()
    p1=model._project_text_tokens('video',kw['text_embed'],kw['pooled_text_embed'])
    p2=model._project_text_tokens('video',kw['text_embed'],kw['pooled_text_embed'])
    time1=model._time_embed('video',torch.tensor([200.]),p1[1])
    time2=model._time_embed('video',torch.tensor([800.]),p2[1])
hook.remove()
assert len(hits)==1 and torch.equal(p1[0],p2[0])
metrics['projection_calls_two_reads']=len(hits)
metrics['time_condition_change']=(time1-time2).abs().max().item()
print('projection calls:',len(hits),'time delta:',metrics['time_condition_change'])
''')
finish('''
1. Уберите mask, повторите padded-value perturbation и объясните изменение результата.
2. Найдите в исходнике поведение при одном и при нескольких prompts: packed и padded shapes.
3. Почему `eval()` само по себе не замораживает веса? Покажите разницу с `requires_grad_(False)`.
''','''
Вы отличаете frozen text encoder от trainable text-refiner внутри DiT. Можете указать,
какая маска требуется для прямоугольного batch и какие части text conditioning зависят от diffusion time.
''')

start('03_video_vae', '03. Видео VAE: posterior, compression, causality и scaling',
      'вывести VAE objective, обучить маленький codec и проверить causal Conv3d/latent normalization',
      'Lecture 4 / Lab 3; подробная теория VAE в K5 02',
      ['core/components/hunyuan_vae.py','core/components/vae_video.py','core/algo/postprocess_video.py'], '3.1',
      '../../kandinsky-5/notebooks/02_vae.ipynb')
md(r'''
## 1. Зачем распределение вместо одной точки

Encoder оценивает Gaussian posterior с диагональной covariance. Декодер задаёт likelihood наблюдения.
Reparameterization отделяет случайность от обучаемых mean/std:

$$
q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\mathrm{diag}(\exp(\ell_\phi(x)))),\qquad
z=\mu_\phi(x)+\exp(\ell_\phi(x)/2)\epsilon.
$$

Negative ELBO состоит из reconstruction NLL и KL к prior. Для standard Gaussian prior:

$$
\mathrm{KL}(q\Vert p)=\frac12\sum_j\left(\mu_j^2+\exp(\ell_j)-1-\ell_j\right).
$$

MSE соответствует Gaussian reconstruction likelihood при фиксированной дисперсии с соответствующим
коэффициентом и константой. Меняя вес KL или редукцию MSE, вы меняете objective. Ниже честно используем
**учебный beta-VAE objective** `mean_MSE + 0.001 * mean_batch(sum_latent_KL)`,
а не утверждаем, что воспроизвели все perceptual/adversarial losses промышленного Hunyuan VAE.
''')
code('''
vae_cfg=config('vae')
T,H,W=121,480,864
latent_shape=((T-1)//4+1,H//8,W//8,vae_cfg['latent_channels'])
assert latent_shape==(31,60,108,16)
print('RGB video -> latent THWC:',(T,H,W,3),'->',latent_shape)
metrics['full_video_latent_shape']=list(latent_shape)
print('Scaling factor:',vae_cfg['scaling_factor'])
''')
md('''
## 2. Миниатюрное обучение с известными данными

Создадим 12 коротких клипов движущегося Gaussian blob. Encoder/decoder — маленькие MLP,
чтобы видеть posterior и loss на CPU. Они не заменяют causal convolutional Hunyuan architecture.
Разделение цели и реализации позволяет сначала проверить математику.
''')
code('''
y,x=torch.meshgrid(torch.linspace(-1,1,8),torch.linspace(-1,1,8),indexing='ij')
clips=torch.stack([torch.stack([torch.exp(-((x-(t/4-0.5)*0.8)**2+(y-s)**2)/0.12)
                               for t in range(5)]) for s in torch.linspace(-.5,.5,12)])
flat=clips.flatten(1)
class MiniVAE(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(320,32),nn.SiLU(),nn.Linear(32,16))
        self.decoder=nn.Sequential(nn.Linear(8,32),nn.SiLU(),nn.Linear(32,320),nn.Sigmoid())
    def forward(self,x):
        mu,logvar=self.encoder(x).chunk(2,-1)
        z=mu+torch.exp(logvar/2)*torch.randn_like(mu)
        return self.decoder(z),mu,logvar
vae=MiniVAE();opt=torch.optim.Adam(vae.parameters(),lr=.01);history=[]
for step in range(100):
    recon,mu,logvar=vae(flat)
    reconstruction=F.mse_loss(recon,flat)
    kl=.5*(mu.square()+logvar.exp()-1-logvar).sum(-1).mean()
    loss=reconstruction+.001*kl
    opt.zero_grad();loss.backward();opt.step()
    history.append([loss.item(),reconstruction.item(),kl.item()])
assert np.isfinite(history).all() and history[-1][0]<history[0][0]
metrics['toy_objective_initial']=history[0][0];metrics['toy_objective_final']=history[-1][0]
with torch.no_grad():
    mu,logvar=vae.encoder(flat).chunk(2,-1);mean_recon=vae.decoder(mu).reshape_as(clips)
fig,axes=plt.subplots(1,3,figsize=(10,3));axes[0].plot(np.array(history)[:,:2])
axes[0].legend(['objective','MSE']);axes[1].imshow(clips[0,2]);axes[2].imshow(mean_recon[0,2])
axes[1].set_title('Input');axes[2].set_title('Decode posterior mean')
fig.tight_layout();fig.savefig(figure_path(LAB,'vae_training'));plt.show()
''')
md('''
## 3. Causal Conv3d и нормализация латентов

Кодек с causal temporal padding не использует будущие кадры. Возьмём **оригинальное определение**
`HunyuanVideoCausalConv3d`, изолированно от тяжёлых Diffusers imports, и изменим только будущее.
Проверяем конкретный слой при kernel=3/stride=1; это не доказательство causality всего генератора:
video attention в DiT может видеть всю последовательность.

DiT обучен на масштабированных латентах. При encode умножаем на scale, перед decode делим.
Пропущенный scale меняет distribution, хотя shape остаётся правильной.
''')
code('''
Causal=native_causal_conv();conv=Causal(1,1,kernel_size=3,bias=False)
video=torch.randn(1,1,6,5,5);future=video.clone();future[:,:,4:]+=100
with torch.no_grad():
    left=conv(video);right=conv(future)
causal_error=(left[:,:,:4]-right[:,:,:4]).abs().max().item();assert causal_error==0
z=torch.randn(1,16,3,4,4);scale=vae_cfg['scaling_factor']
roundtrip=((z*scale)/scale-z).abs().max().item()
metrics['native_causal_conv_past_error']=causal_error;metrics['latent_scale_roundtrip_error']=roundtrip
print('Causal past error:',causal_error,'scale roundtrip:',roundtrip)
''')
finish('''
1. Сравните beta=0, 0.001, 0.1: MSE, KL, posterior variance. Не делайте вывод по одному loss.
2. Сопоставьте 33/121 RGB frames с latent frames. Почему формула содержит T-1, а не просто T/4?
3. Измените последний кадр и сравните past outputs causal и обычной symmetric convolution.
4. Для настоящего VAE reconstruction используйте ComfyUI C4 и существующий Hunyuan codec после resource gate.
''','''
Вы различаете posterior mean/sample, reconstruction loss/KL, pixel/latent space и причинность кодека/DiT.
Другой VAE с теми же размерами не гарантирует совместимость с pretrained CrossDiT.
''')

start('04_audio_vae_and_vocoder', '04. Аудио: waveform → mel → VAE → vocoder',
      'понять частотное представление, роль audio VAE и vocoder, проверить duration/scaling/clipping',
      'K6 03; базовая вероятность Gaussian',
      ['core/components/vae_audio.py','core/components/mmaudio/features_utils.py','core/algo/postprocess_audio.py'], '3.1',
      '../../kandinsky-5/notebooks/11_kvae_separate.ipynb')
md(r'''
## 1. Три пространства и три разные задачи

Первичные статьи компонентов: [MMAudio](https://arxiv.org/abs/2412.15322),
[BigVGAN](https://arxiv.org/abs/2206.04658). Их полный generation backbone не загружается вместо K6.

Waveform — амплитуда во времени. STFT — комплексная частотно-временная матрица; mel-проекция
сжимает частотные полосы. VAE преобразует mel в 40-channel latent. Decoder VAE восстанавливает mel,
но не waveform: эту задачу решает BigVGAN-style vocoder.

Код K6: `FeaturesUtils.wrapped_encode` возвращает mean posterior; `wrapped_decode` — mel.
Режим checkpoint — 44100 Hz, downsample factor 1024, scale Lite Distill=0.417.
Обычный Lite config использует другой scale (0.5302). Не копируйте нормализацию между checkpoints.
KVAE-Audio — отдельный codec, не автоматическая замена MMAudio latent contract.

### Почему STFT inverse ещё не vocoder

Если сохранить phase, inverse STFT может точно вернуть синтетический waveform.
Mel magnitude сам по себе фазу не сохраняет; обученный vocoder восполняет недостающую информацию.
Наш опыт ниже не воспроизводит MMAudio preprocessing целиком.
''')
code('''
sr=44100;time=torch.arange(sr//4)/sr
wave=.2*torch.sin(2*torch.pi*440*time)+.1*torch.sin(2*torch.pi*880*time)
window=torch.hann_window(1024)
stft=torch.stft(wave,n_fft=1024,hop_length=256,window=window,return_complex=True)
inverse=torch.istft(stft,n_fft=1024,hop_length=256,window=window,length=wave.numel())
error=(wave-inverse).abs().max().item();assert error<1e-5
# Educational triangular mel filters; not the exact checkpoint feature extractor.
freqs=torch.linspace(0,sr/2,513);mel=lambda f:2595*torch.log10(1+f/700)
points=700*(10**(torch.linspace(0,mel(torch.tensor(sr/2)).item(),130)/2595)-1)
left=(freqs[None,:]-points[:-2,None])/(points[1:-1,None]-points[:-2,None])
right=(points[2:,None]-freqs[None,:])/(points[2:,None]-points[1:-1,None])
bank=torch.minimum(left,right).clamp(min=0)
mel_power=bank@stft.abs().square()
metrics['stft_inverse_max_error']=error
fig,axes=plt.subplots(1,2,figsize=(10,3));axes[0].plot(time[:500],wave[:500])
axes[0].set_title('Synthetic waveform');axes[1].imshow(torch.log1p(mel_power),origin='lower',aspect='auto')
axes[1].set_title('Educational mel energy, 128 bands');fig.tight_layout()
fig.savefig(figure_path(LAB,'audio_spaces'));plt.show()
''')
md('''
## 2. Сколько audio tokens соответствует видео

Длительность код вычисляет через число RGB frames/fps, затем берёт ceil(samples/1024).
Это даёт запас меньше одного audio latent step; он нужен для decoder/mux boundaries.
Интервал между первым и последним video timestamp и длительность хранения N frames/fps отличаются.
''')
code('''
prepare=native('core.algo.prepare_latents')
rows=[]
for latent_frames in [1,9,31]:
    frames=(latent_frames-1)*4+1
    n=prepare.audio_latent_duration(latent_frames)
    delta=n*1024/44100-frames/24
    assert 0<=delta<1024/44100
    rows.append({'video_frames':frames,'audio_latents':n,'padding_seconds':delta})
print(rows);metrics['duration_contract']=rows
''')
md('''
## 3. Настоящий postprocess с явно учебными decoder/vocoder

Перед decoder выполняется обратная нормализация. Чтобы проверить эту обвязку без весов,
подставим test doubles: mel decoder возвращает вход, vocoder — заданный waveform.
Так исполняется настоящий `postprocess_audio`, но нет заявления о качестве звука.
`clip` сохраняет масштаб до насыщения; `normalize` меняет громкость всего примера по peak.
''')
code('''
post=native('core.algo.postprocess_audio');Bundle=native('core.types').LatentBundle
class DecoderDouble:
    scaling_factor=.417
    def wrapped_decode(self,z):
        self.seen=z.clone();return z
class VocoderDouble:
    def __call__(self,mel):return torch.tensor([[[-2.,-.2,0.,.2,2.]]])
decoder=DecoderDouble();bundle=Bundle(None,torch.ones(1,5,40)*.417,None,torch.tensor([0,5],dtype=torch.int32))
clipped=post.postprocess_audio(bundle,decoder,VocoderDouble(),'clip')[0]
normalized=post.postprocess_audio(bundle,decoder,VocoderDouble(),'normalize')[0]
assert torch.allclose(decoder.seen,torch.ones_like(decoder.seen))
assert clipped[1]!=normalized[1]
metrics['clip_pcm']=clipped.tolist();metrics['normalize_pcm']=normalized.tolist()
print('clip:',clipped,'normalize:',normalized)
''')
finish('''
1. Удалите phase и объясните, почему mel→waveform не единственное обратное преобразование.
2. Измените scale с .417 на .5302 в одном месте и измерьте ошибку latent roundtrip.
3. Сравните peak и RMS до/после normalize. Почему peak normalization осложняет A/B громкости?
''','''
Вы можете назвать, что возвращает каждый audio component. 400 выходов DiT — не 400 channels audio VAE:
это 10 grid points × 40 channels, которые разберём в π-Flow лабораторной.
''')

start('05_nn_patches_and_modulation', '05. Строительные блоки: patches, time, AdaLN и gates',
      'воспроизвести patch layout и проверить native modulation/residual на маленьких тензорах',
      'K6 01–03; linear layers и broadcasting',
      ['core/components/dit.py','core/tensors.py'], '3.2',
      '../../kandinsky-5/notebooks/03_nn_building_blocks.ipynb')
md(r'''
## 1. Из локальной ячейки latent в visual token

Visual input содержит state, conditioning channels и mask: 16+16+1=33.
Patch 1×2×2 собирает 4 ячейки в вектор длины 132, затем Linear переводит его в width 1792.
Audio tokens не проходят spatial patching: 40→896. В tiny опыте ширины меньше, операции те же.

Unpatchify в output голове имеет собственный порядок C/patch axes. Нельзя предположить,
что inverse input permutation восстанавливает output без проверки порядка выходных features.
''')
code('''
D=native();x=torch.arange(1*2*4*4*3,dtype=torch.float32).reshape(1,2,4,4,3)
layer=D.VisualEmbeddings(3,12,(1,2,2))
with torch.no_grad():layer.in_layer.weight.copy_(torch.eye(12));layer.in_layer.bias.zero_()
y=layer(x)
manual=x.reshape(1,2,1,2,2,2,2,3).permute(0,1,3,5,2,4,6,7).flatten(4,7)
assert torch.equal(y,manual)
print('Native patches:',tuple(y.shape),'first vector:',y[0,0,0,0])
metrics['visual_patch_max_error']=(y-manual).abs().max().item()
''')
md(r'''
## 2. AdaLN — condition управляет обработкой state

Time sinusoid проходит MLP. CLIP pooled projection прибавляется к time representation.
Modulation выдаёт shift, scale и residual gate:

$$
h' = h + g(c) F\left((1+s(c))\mathrm{LN}(h)+b(c)\right).
$$

При нулевом gate residual branch не меняет h. Это свойство forward; градиенты gate всё ещё могут
быть ненулевыми. Нельзя сказать «нулевой слой никогда не учится». Если занулить и предшествующее
представление, градиент может исчезнуть по другой причине.
''')
code('''
tensors=native('core.tensors');h=torch.randn(2,5,8);condition=torch.randn(2,6)
mod=D.Modulation(6,8,3);shift,scale,gate=mod(condition).chunk(3,-1)
branch=nn.Linear(8,8);norm=nn.LayerNorm(8,elementwise_affine=False)
out=tensors.apply_gate_sum(h,branch(tensors.apply_scale_shift_norm(norm,h,scale,shift)),gate)
identity=(out-h).abs().max().item();assert identity==0
out.square().mean().backward()
metrics['zero_gate_identity_error']=identity
metrics['modulation_gradient_norm']=mod.out_layer.weight.grad.norm().item()
assert metrics['modulation_gradient_norm']>0
print(metrics)
''')
md('''
## 3. OutLayerAudio и маленькие на вид особенности

Audio output выполняет modulated LayerNorm, затем ещё одну LayerNorm, затем Linear.
Это буквальное поведение pinned source. Удалять второй norm «для красоты» при переносе нельзя:
проверка parity должна сравнивать именно вычисляемую функцию.
''')
code('''
audio=D.OutLayerAudio(8,6,4);a=torch.randn(1,5,8);t=torch.randn(1,1,6)
with torch.no_grad():
    actual=audio(a,t)
    shift,scale=audio.modulation(t).chunk(2,-1)
    m=tensors.apply_scale_shift_norm(audio.norm,a,scale,shift)
    expected=audio.out_layer(audio.norm(m))
assert torch.allclose(actual,expected)
metrics['audio_output_manual_max_error']=(actual-expected).abs().max().item()
print('Audio output shape:',actual.shape)
''')
finish('''
1. Посчитайте input projection dimensions для 16-channel и 33-channel входа.
2. Сравните gradient modulation и gradient residual branch на первом zero-init шаге.
3. Найдите в DiT FFN две bias-free Linears и GELU; посчитайте их параметры вручную.
''','''
Вы прослеживаете формы от latent cells до tokens и обратно, не смешивая число heads,
model width, latent channels и число grid points. Нулевой gate и frozen parameters — разные понятия.
''')

start('06_attention_and_rope', '06. Attention и RoPE: текст, видео и аудио',
      'сравнить ручной attention с SDPA, измерить память и проверить геометрию вращений',
      'K6 05; матричное умножение и softmax',
      ['core/components/dit.py','core/components/rope.py','core/tensors.py','runtime/kernels/sdpa.py'], '3.2',
      '../../kandinsky-5/notebooks/04_attention.ipynb')
md(r'''
## 1. Attention — выбор источников информации

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(QK^\top/\sqrt{d_k}+M\right)V.
$$

Q определяет, **какие tokens обновляем**. K/V определяют, **откуда берём информацию**.
Поэтому video queries + audio K/V дают обновление video, независимо от имени `va_cross_attention`.
У asymmetric widths проекции K/V переводят 896→1792 для video-query ветви и 1792→896 для audio-query.
K6 также делает RMSNorm Q/K до RoPE. Ниже проверяем само attention на уже готовых Q/K/V.

[Attention Is All You Need](https://arxiv.org/abs/1706.03762),
[RoFormer / RoPE](https://arxiv.org/abs/2104.09864) — первичные источники операций.
''')
code('''
q=torch.randn(1,2,7,8,dtype=torch.float64);k=torch.randn(1,2,5,8,dtype=torch.float64);v=torch.randn_like(k)
weights=(q@k.transpose(-1,-2)/math.sqrt(8)).softmax(-1)
manual=weights@v;sdpa=F.scaled_dot_product_attention(q,k,v)
error=(manual-sdpa).abs().max().item();assert error<1e-12
metrics['manual_sdpa_max_error']=error
fig,ax=plt.subplots(figsize=(6,3));ax.imshow(weights[0,0],aspect='auto');ax.set_xlabel('Source K/V token')
ax.set_ylabel('Updated Q token');plt.show()
''')
md(r'''
## 2. Вращение сохраняет норму; позиции меняют сравнение

RoPE поворачивает пары feature coordinates, а не переставляет tokens.
Общий сдвиг позиций Q и K сохраняет их inner product в 1D случае; относительный сдвиг меняет его.
В K6 video RoPE разделяет head_dim=16+24+24=64 на T/H/W, audio использует 1D RoPE,
а `audio_freqs_scaling=0.144` задаётся checkpoint config. Позиционный scale — не sampler shift.
''')
code('''
R=native('core.components.rope');rot=native('core.tensors').apply_rotary
rope=R.RoPE1D(8,max_pos=64);q=torch.randn(4,1,8);k=torch.randn_like(q)
qr=rot(q,rope(torch.arange(4)));kr=rot(k,rope(torch.arange(4)+2))
qshift=rot(q,rope(torch.arange(4)+5));kshift=rot(k,rope(torch.arange(4)+7))
norm_error=(qr.norm(dim=-1)-q.norm(dim=-1)).abs().max().item()
shift_error=((qr*kr).sum(-1)-(qshift*kshift).sum(-1)).abs().max().item()
assert norm_error<2e-6 and shift_error<2e-6
metrics['rope_norm_error']=norm_error;metrics['rope_relative_shift_error']=shift_error
r3=R.RoPE3D((2,2,4));grid=r3((2,3,4),[torch.arange(2),torch.arange(3),torch.arange(4)])
print('3D rotary matrices shape:',grid.shape,'not a learnable embedding table')
''')
md('''
## 3. Почему video attention дорого

Считаем логическую score matrix, не выделяя её память. 121 RGB frames при 480×864 дают
31×30×54=50220 visual tokens. Аудио — около 218 tokens. Self-attention video квадратично по Nv;
cross-modal attention пропорционально Nv×Na. FlashAttention меняет способ вычисления и хранения,
не автоматически pattern связей. Это не оценка реальной peak memory конкретного backend.
''')
code('''
Nv=31*30*54;Na=native('core.algo.prepare_latents').audio_latent_duration(31)
logical={'video self':28*Nv*Nv*2/2**30,'video queries/audio keys':28*Nv*Na*2/2**30,
         'audio queries/video keys':14*Na*Nv*2/2**30}
metrics['logical_bf16_score_GiB']=logical
print('Nv,Na:',Nv,Na,'logical BF16 score matrices GiB:',logical)
fig,ax=plt.subplots(figsize=(8,3));ax.bar(logical.keys(),logical.values());ax.set_yscale('log')
ax.set_ylabel('Logical score storage GiB, not SDPA peak');fig.tight_layout()
fig.savefig(figure_path(LAB,'attention_budget'));plt.show()
''')
finish('''
1. Покажите на shape, почему Q длиной Nv и K длиной Na возвращают Nv output tokens.
2. Удвойте H/W и посчитайте рост Nv и логической self-attention matrix.
3. Сравните audio RoPE scales 1 и .144 на одинаковых позициях; это не качество AV синхронизации.
''','''
Вы объясняете источник и получателя cross-attention по Q и K/V. Causal VAE, dense attention,
masking и memory-efficient attention решают разные задачи.
''')

start('07_dit_and_text_crossdit', '07. От DiT к CrossDiT: как denoiser устроен изнутри',
      'проследить путь latent→patches→blocks→velocity, отделить time/pooled modulation от token-level text cross-attention и отличить velocity head от π-Flow grid head',
      'K6 05–06 (patches, AdaLN, Q/K/V, RoPE); Lecture 4 / Lab 3',
      ['core/components/dit.py','core/components/piflow_dit.py','core/tensors.py','pipeline/config.py'], '3.1–3.2',
      '../../kandinsky-5/notebooks/06_dit_and_flow_matching.ipynb')
md(r'''
## 0. Вопрос, вход и выход

**Вопрос лабораторной:** сеть получает шумный latent, время и текст. Что она возвращает,
и через какую ветвь каждое условие влияет на ответ?

Denoiser — функция $u_\theta(x_t, t, y)$, которая возвращает tensor **той же формы, что физический latent**.
У base checkpoint это velocity, у distilled checkpoint — набор grid predictions (раздел 6).
Этот notebook рассматривает video-only T2V ветвь класса `DiffusionTransformer3D`; аудио и обмен между потоками — в 08.

| Объект | Форма (batched, channel-last) | Смысл |
|---|---|---|
| $x_t$ | `[B, T, H, W, 16]` | шумный video latent: B примеров, T latent frames, сетка H×W, 16 каналов VAE |
| visual input | `[B, T, H, W, 33]` | $x_t$ ‖ condition latent ‖ mask = 16+16+1 (`_build_video_input`) |
| text tokens | `[B, L, 3584]` | Qwen2.5-VL hidden states, по вектору на token; `attention_mask [B, L]` отмечает padding |
| pooled | `[B, 768]` | CLIP pooled vector: **один** вектор на prompt |
| time | `[B]` | `1000·t`, где в K6 t=1 — шум, t=0 — данные |
| output | `[B, T, H, W, 16]` | velocity только для физических каналов, без condition/mask |

**Соглашение о времени.** Pinned `flow_match_timesteps` идёт от t=1 к t=0. В конспекте MIT t=0 — шум, t=1 — данные.
Если $t_{MIT}=1-t_{K6}$, то $dx/dt_{MIT}=-dx/dt_{K6}$: тот же путь, противоположный знак velocity.
Вывод target и проверка знака — в [09](09_flow_matching_training.ipynb).
''')
code('''
import re
lite=json.loads((ROOT/'reference/hf/Kandinsky-6.0-Lite-5s-Diffusers/transformer/config.json').read_text())
distill=config('transformer');vae=config('vae')
recipe=(ROOT/'reference/kandinsky-6/kandinsky/configs/checkpoints/lite-distill.yaml').read_text()
gen={k:int(re.search(rf'^\\s*{k}:\\s*(\\d+)',recipe,re.M).group(1)) for k in ['height','width','sample_frames','latent_frames']}
T=(gen['sample_frames']-1)//vae['temporal_compression_ratio']+1
H,W=gen['height']//vae['spatial_compression_ratio'],gen['width']//vae['spatial_compression_ratio']
assert T==gen['latent_frames']
pT,pH,pW=lite['patch_size'];C_in=2*lite['in_visual_dim']+1 if lite['visual_cond'] else lite['in_visual_dim']
head_dim=sum(lite['axes_dims'])
contract={'recipe (pixels)':[gen['sample_frames'],gen['height'],gen['width']],
          'latent [T,H,W,C]':[T,H,W,lite['in_visual_dim']],'visual input channels':C_in,
          'token grid [T/pT,H/pH,W/pW]':[T//pT,H//pH,W//pW],'video tokens N_v':(T//pT)*(H//pH)*(W//pW),
          'patch features':pT*pH*pW*C_in,'model width D / heads':[lite['model_dim'],lite['model_dim']//head_dim],
          'time_dim':lite['time_dim'],'output per token, base velocity':pT*pH*pW*lite['out_visual_dim'],
          'output per token, distill HF config':pT*pH*pW*distill['out_visual_dim']}
assert contract['video tokens N_v']==31*30*54
metrics['lite_contract']=contract
for k,v in contract.items():print(f'{k:36s}',v)
''')
md(r'''
Число tokens определяется сеткой latent и patch size, а не числом каналов:
$N_v=\frac{T}{p_T}\cdot\frac{H}{p_H}\cdot\frac{W}{p_W}$. Каналы и patch определяют только длину вектора до проекции:
$p_Tp_Hp_W C_{in}=4\cdot33=132\to D=1792$. Head width на выходе — $p_Tp_Hp_W C_{out}$.

## 1. Что означает DiT и почему K6 использует два канала conditioning

В [DiT (Peebles & Xie)](https://arxiv.org/abs/2212.09748) transformer обрабатывает patches latent image вместо U-Net.
Авторы сравнили четыре способа подать timestep и class label: in-context tokens, cross-attention block,
adaLN и adaLN-Zero. Для **одного** вектора условия (time + class) лучшим оказался adaLN-Zero.

Текст — последовательность из L векторов. Один pooled vector теряет порядок слов и детали,
поэтому Kandinsky ([K5 report](https://arxiv.org/abs/2511.14993), K6 §3.1) использует оба механизма:

- **global path:** time + CLIP pooled → modulation (shift, scale, gate) каждого sub-layer;
- **token path:** Qwen tokens → text refiner → K/V в cross-attention, Q — visual tokens.

Training target — отдельное решение: DiT предсказывал noise, Kandinsky FM — velocity, distilled K6 — grid policy.
Слово CrossDiT здесь означает DiT с text cross-attention, а не image-classification CrossViT с patches двух размеров.

## 2. Полный forward одной формулой

$$
c = \mathrm{MLP}\big(\mathrm{sincos}(1000t)\big) + \mathrm{LN}(W_p\,\mathrm{pooled}),
\qquad e' = \mathrm{Refiner}_c\big(\mathrm{LN}(W_e\,\mathrm{tokens})\big),
$$

$$
h_0 = W_{in}\,\mathrm{patchify}(x_{in}),\qquad h_{k+1}=\mathrm{Block}_k(h_k;\,e',c),\qquad
u=\mathrm{unpatchify}\big(W_{out}((1+s(c))\,\mathrm{LN}(h_K)+b(c))\big).
$$

Text refiner тоже модулируется $c$: refined tokens зависят от времени. Поэтому кэшируются только
time-independent проекции $W_e$ и $W_p$ ([02](02_text_conditioning.ipynb)), а не выход refiner.
Синусоида строится на ширине `model_dim`, MLP переводит её в `time_dim`; порядок признаков — сначала cos, затем sin.
Проверим это на native классе.
''')
code('''
D=native();tensors=native('core.tensors')
time_encoder=D.TimeEmbeddings(48,16);t=torch.tensor([500.,520.])
with torch.no_grad():
    native_c=time_encoder(t)
    args=torch.outer(t,tensors.get_freqs(24));features=torch.cat([args.cos(),args.sin()],-1)
    emb=time_encoder.timestep_embedder;manual_c=emb.linear_2(emb.act(emb.linear_1(features)))
error=(native_c-manual_c).abs().max().item();assert error<1e-6
metrics['time_embedding_manual_error']=error
dt=np.linspace(-500,500,401)
freqs=tensors.get_freqs(lite['model_dim']//2).double()
def sincos(value):
    a=torch.as_tensor(value,dtype=torch.float64)[...,None]*freqs
    return torch.cat([a.cos(),a.sin()],-1)
similarity=F.cosine_similarity(sincos(500.).expand(len(dt),-1),sincos(500.+torch.tensor(dt)),dim=-1)
fig,ax=plt.subplots(figsize=(7,3));ax.plot(dt,similarity)
ax.set_xlabel('Time offset Δ in DiT units (1000·t), reference t=500')
ax.set_ylabel('Cosine similarity')
ax.set_title(f'Sinusoidal features before MLP, width {lite["model_dim"]}; no learned weights')
fig.tight_layout();fig.savefig(figure_path(LAB,'time_feature_similarity'));plt.show()
metrics['time_features_cos_at_delta_10']=float(similarity[np.argmin(abs(dt-10))])
print('manual time-embedding error:',error,'cos(Δ=10):',metrics['time_features_cos_at_delta_10'])
''')
md(r'''
Признаки различают даже близкие времена (при Δ=10 cosine заметно меньше 1), затем сходство медленно убывает.
Синусоида — фиксированное кодирование без весов; как время меняет обработку, учат MLP и modulation.

## 3. adaLN-Zero: свежая модель не видит ни текст, ни время

Modulation vectors идут в порядке shift, scale, gate ([05](05_nn_patches_and_modulation.ipynb)).
Text refiner block имеет 2 ветви × 3 = 6 векторов, visual block — 3 ветви × 3 = 9, `OutLayer` — 2 (shift, scale, без gate).
Все `Modulation` в K6 инициализируются нулями. Поэтому при инициализации:

- каждый residual gate равен 0, каждый block — тождественное отображение;
- shift=scale=0 в `OutLayer`, значит выход не зависит от $c$;
- выход **не** равен нулю: в отличие от официального кода DiT, где final linear тоже обнулён,
  `OutLayer.out_layer` в K6 получает обычную инициализацию.

Следовательно, свежая сеть — $u=\mathrm{unpatchify}(W_{out}\,\mathrm{LN}(W_{in}\,\mathrm{patchify}(x)))$.
Проверим три предсказания: равенство этой композиции и нулевую чувствительность к text tokens, pooled и времени.
''')
code('''
from labkit.torch_reference import tiny_config
cfg=tiny_config();cfg.update(is_multimodal=False,out_audio_dim=4)
fresh=D.DiffusionTransformer3D(**cfg).eval()
video=torch.randn(1,2,4,4,4)
kw=dict(x_video=torch.cat((video,torch.zeros_like(video),torch.zeros_like(video[...,:1])),-1),
        x_audio=None,text_embed=torch.randn(1,5,12),pooled_text_embed=torch.randn(1,8),
        time=torch.tensor([500.]),visual_rope=fresh.visual_rope((2,2,2),[torch.arange(2)]*3),
        audio_rope=None,text_rope=fresh.text_rope(torch.arange(5)),
        attention_mask=torch.ones(1,5,dtype=torch.bool))
def changed(**update):
    other=dict(kw);other.update(update)
    with torch.no_grad():return (fresh_forward(fresh,kw)-fresh_forward(fresh,other)).abs().max().item()
with torch.no_grad():
    out=fresh_forward(fresh,kw)
    composition=fresh.out_layer(fresh.visual_embeddings(kw['x_video']),torch.zeros(1,cfg['time_dim']))
fresh_rows={'composition error':(out-composition).abs().max().item(),
            'text tokens':changed(text_embed=kw['text_embed']+1),
            'pooled':changed(pooled_text_embed=kw['pooled_text_embed']+1),
            'time':changed(time=torch.tensor([100.])),'output mean |u|':out.abs().mean().item()}
assert fresh_rows['composition error']<1e-6 and fresh_rows['output mean |u|']>0
assert fresh_rows['text tokens']==fresh_rows['pooled']==fresh_rows['time']==0
metrics['fresh_model_invariants']=fresh_rows
print(json.dumps(fresh_rows,indent=2))
''')
md(r'''
Обучение должно открыть gates, иначе condition не влияет на velocity. Нулевой gate не означает нулевой gradient:
производная по весам `Modulation` ненулевая ([05](05_nn_patches_and_modulation.ipynb), раздел 2).

### Обратный путь: unpatchify имеет свой порядок признаков

Вход патча раскладывается как $(p_T,p_H,p_W,C)$, выход `OutLayer` читает признаки как $(C,p_T,p_H,p_W)$.
Если взять единичные $W_{in}$ и $W_{out}$ и убрать LayerNorm, round-trip **не** восстановит latent.
Перестановочная матрица в $W_{out}$ восстанавливает его точно. Обученная сеть выучивает любую такую перестановку,
поэтому parity переноса проверяется по полной функции, а не по предположению о layout.
''')
code('''
C=3;P=(1,2,2);width=math.prod(P)*C
x=torch.arange(1*2*4*4*C,dtype=torch.float32).reshape(1,2,4,4,C)
emb=D.VisualEmbeddings(C,width,P);head=D.OutLayer(width,8,C,P);head.norm=nn.Identity()
order=torch.arange(width).reshape(*P,C).permute(3,0,1,2).flatten()
with torch.no_grad():
    emb.in_layer.weight.copy_(torch.eye(width));emb.in_layer.bias.zero_();head.out_layer.bias.zero_()
    head.out_layer.weight.copy_(torch.eye(width));naive=head(emb(x),torch.zeros(1,8))
    head.out_layer.weight.copy_(torch.eye(width)[order]);permuted=head(emb(x),torch.zeros(1,8))
roundtrip={'identity W_out max error':(naive-x).abs().max().item(),
           'permuted W_out max error':(permuted-x).abs().max().item()}
assert roundtrip['identity W_out max error']>0 and roundtrip['permuted W_out max error']==0
metrics['unpatchify_roundtrip']=roundtrip;print(roundtrip)
''')
md(r'''
## 4. Text CrossDiT: visual query, text K/V

Visual decoder block делает три gated residual updates: self-attention, cross-attention к refined text, FFN.

$$
h \leftarrow h + g_{ca}\odot \mathrm{Attn}\big(Q=\tilde h W_Q,\;K=e'W_K,\;V=e'W_V\big),\qquad
\tilde h=(1+s_{ca})\,\mathrm{LN}(h)+b_{ca}.
$$

Q приходит от video tokens, поэтому выход имеет $N_v$ строк и обновляет video. K/V — от $L$ text tokens;
`attention_mask` закрывает padded keys. Сначала посмотрим формы через hooks на native video-only CrossDiT
(уменьшенные размеры, случайные веса), затем включим малые случайные modulation weights для sensitivity probe.
''')
code('''
model=D.DiffusionTransformer3D(**cfg).eval();activate_gates(model)
shapes={};hooks=[]
def capture(name):
    def hook(module,inputs,output):
        shapes[name]=list(output.shape)
    return hook
for name,mod in [('patches',model.visual_embeddings),('text_projection',model.text_embeddings),
                 ('text_refiner',model.text_blocks[0]),('visual_block',model.visual_transformer_blocks[0]),
                 ('out_layer',model.out_layer)]:
    hooks.append(mod.register_forward_hook(capture(name)))
with torch.no_grad():out=fresh_forward(model,kw)
for h in hooks:h.remove()
assert out.shape==video.shape
metrics['native_shapes']=shapes
print(json.dumps(shapes,indent=2))
''')
md('''
Patches выходят как `[B, T', H', W', D]`, внутри blocks они развёрнуты в `[B, N_v, D]`, `OutLayer` возвращает latent shape.

### Какая ветвь переносит какое условие

Опыт 2×2: возмущаем token states **или** pooled vector; text cross-attention работает **или** её выход занулён hook-ом во всех blocks.
Предсказание: token perturbation проходит только через cross-attention; pooled perturbation проходит через modulation и после cut.
Третий контроль: изменяем только padded tokens, закрытые маской, — выход не должен измениться.
''')
code('''
token_kw=dict(kw,text_embed=kw['text_embed']+torch.randn_like(kw['text_embed']))
pooled_kw=dict(kw,pooled_text_embed=kw['pooled_text_embed']+torch.randn_like(kw['pooled_text_embed']))
def delta(other,cut=False):
    handles=[b.cross_attention.register_forward_hook(lambda m,i,o:torch.zeros_like(o))
             for b in model.visual_transformer_blocks] if cut else []
    with torch.no_grad():value=(fresh_forward(model,kw)-fresh_forward(model,other)).abs().max().item()
    for h in handles:h.remove()
    return value
paths={'tokens, CA active':delta(token_kw),'tokens, CA cut':delta(token_kw,True),
       'pooled, CA active':delta(pooled_kw),'pooled, CA cut':delta(pooled_kw,True)}
mask=torch.tensor([[True,True,True,False,False]])
padded=dict(kw,attention_mask=mask);noisy_pad=dict(padded,text_embed=kw['text_embed'].clone())
noisy_pad['text_embed'][:,3:]+=10
with torch.no_grad():
    paths['padded tokens changed']=(fresh_forward(model,padded)-fresh_forward(model,noisy_pad)).abs().max().item()
assert paths['tokens, CA active']>0 and paths['tokens, CA cut']<1e-6
assert paths['pooled, CA cut']>0 and paths['padded tokens changed']<1e-6
metrics['condition_paths']=paths
fig,ax=plt.subplots(figsize=(8,3));ax.barh(list(paths),[max(v,1e-12) for v in paths.values()]);ax.set_xscale('log')
ax.set_xlabel('Max |Δ velocity| (log; values below 1e-12 drawn at 1e-12)')
ax.set_title('Random-weight tiny CrossDiT: which branch carries which condition')
fig.tight_layout();fig.savefig(figure_path(LAB,'condition_paths'));plt.show();print(paths)
''')
md(r'''
Это проверка wiring на случайных весах. Она показывает, **через какую ветвь** возможна зависимость,
но не измеряет, насколько хорошо pretrained модель следует prompt.

## 5. Velocity head и π-Flow grid head

Base checkpoint возвращает на token $p_Tp_Hp_W\cdot16=64$ числа: velocity в каждой ячейке patch.
Distilled Lite возвращает `n_grid` предсказаний на ячейку, из них sampler строит DX policy ([10](10_piflow_distillation.ipynb)).
Архитектура blocks та же; меняется только ширина `OutLayer`/`OutLayerAudio` и интерпретация выхода.

Тонкость config: HF config distill уже хранит **расширенную** ширину (160=16·10, 400=40·10),
а `PiFlowDiffusionTransformer3D` умножает базовую ширину на `n_grid` сам. Поэтому pinned loader
(`_per_grid_overrides`) отбрасывает расширенный override и оставляет базовую ширину module config.
Иначе голова получилась бы в `n_grid` раз шире.
''')
code('''
base_cfg=lite;grid_cfg=distill
n_grid=int(re.search(r'^\\s*n_grid:\\s*(\\d+)',recipe,re.M).group(1))
heads={'base out_visual_dim':base_cfg['out_visual_dim'],'distill out_visual_dim (HF)':grid_cfg['out_visual_dim'],
       'base out_audio_dim':base_cfg['out_audio_dim'],'distill out_audio_dim (HF)':grid_cfg['out_audio_dim'],'n_grid':n_grid}
assert grid_cfg['out_visual_dim']==n_grid*base_cfg['out_visual_dim']
assert grid_cfg['out_audio_dim']==n_grid*base_cfg['out_audio_dim']
show_source('pipeline/config.py','_per_grid_overrides')
grid_model=tiny_model(grid=3).eval();_,_,grid_kw=tiny_inputs(grid_model)
with torch.no_grad():gv,ga=fresh_forward(grid_model,grid_kw)
heads['tiny grid video [B,n_grid,T,H,W,C]']=list(gv.shape);heads['tiny grid audio [B,n_grid,N_a,C]']=list(ga.shape)
assert gv.shape[1]==ga.shape[1]==3
metrics['velocity_vs_grid_head']=heads;print(json.dumps(heads,indent=2))
''')
md(r'''
## 6. Сопоставление архитектур по контрактам

| Архитектура | Обновляемые tokens | Где condition | Корректное сравнение |
|---|---|---|---|
| [DiT](https://arxiv.org/abs/2212.09748) | image latent patches | time + class → adaLN-Zero (лучший из 4 вариантов) | patching, time conditioning, residual gating |
| Kandinsky text CrossDiT ([K5](https://arxiv.org/abs/2511.14993)) | video patches + отдельный text refiner | Qwen tokens → text cross-attention; time+pooled → modulation | native video-only forward выше |
| K6 dual-stream | video и audio tokens разной width | два text refiner + двунаправленная AV fusion | [08](08_dual_stream_audio_video_crossdit.ipynb) |
| [SD3 MMDiT](https://arxiv.org/abs/2403.03206) | text и image tokens с отдельными весами | совместное self-attention по объединённой последовательности | сравнение дизайна, не буквальный K6 код |

Слово «multimodal» само по себе не определяет порядок операций: сверяйте Q, K/V и residual target в коде.
''')
finish('''
1. Для 960×1728 при тех же 121 frames предскажите $N_v$ и рост логической self-attention matrix; проверьте кодом из раздела 0.
2. Гипотеза: в свежей модели gradient loss по весам `cross_attention.to_query` равен нулю, а по `visual_modulation` — нет. Проверьте и объясните, почему.
3. Перенесите опыт 2×2 на `pooled` при `activate_gates(std=0.01)` и `0.2`: как масштаб gates меняет чувствительность? Отчёт — таблица из 4 чисел на std.
4. Откройте DiT, K5 report и `TransformerDecoderBlock`: отметьте общие primitives, различие conditioning и prediction head.
''','''
Вы можете по config посчитать tokens и ширины голов, записать forward формулой с $c$ и $e'$,
объяснить, почему свежая K6 сеть не зависит от condition, и назвать ветвь, через которую проходят tokens и pooled.
Архитектура блока не определяет ни solver, ни target. **Следующий шаг:** [08 — два потока и межмодальный обмен](08_dual_stream_audio_video_crossdit.ipynb).
''')

start('08_dual_stream_audio_video_crossdit', '08. Dual-stream CrossDiT K6: два потока Audio + Video',
      'по исходному коду восстановить порядок FusedTransformerDecoderBlock, источники Q и K/V обоих межмодальных attention и роль флагов, затем проверить wiring причинными опытами',
      'K6 07; Q/K/V и RoPE из 06',
      ['core/components/dit.py','core/components/piflow_dit.py','core/algo/denoise_loop.py','core/algo/piflow_sampler.py'], '3.2, Figures 1–2, Table 2',
      '../../kandinsky-5/notebooks/06_dit_and_flow_matching.ipynb')
md(r'''
## 0. Вопрос, вход и выход

**Вопрос:** как два latent потока разной ширины и частоты обмениваются информацией внутри одного блока,
и какое время управляет каждым обменом?

«Dual-stream» означает **две модальности**, а не stereo audio. Вход одного `FusedTransformerDecoderBlock`:

| Аргумент | Форма | Смысл |
|---|---|---|
| `vis` | `[B, N_v, D_v]` | video tokens после patch embedding или предыдущего блока |
| `aud` | `[B, N_a, D_a]` | audio tokens: по одному на audio latent frame, без spatial patching |
| `text_v`, `text_a` | `[B, L, D_v]`, `[B, L, D_a]` | выходы **двух разных** text refiners из одного prompt |
| `time_embed` | `(c_v, c_a)`, каждый `[B, time_dim]` | time + pooled для каждой модальности; t_v и t_a могут различаться |
| `vis_rope`, `aud_rope` | rotations | 3D RoPE для video, 1D RoPE для audio |

Выход: `(vis', aud')` тех же форм. Основное чтение: [K6 §3.2, Figures 1–2](https://arxiv.org/html/2610.05608v1#S3.SS2),
[страница статьи](https://huggingface.co/papers/2610.05608). Статья задаёт порядок sub-layers; точный источник K/V
и роль флагов — детали pinned кода, которые проверяем ниже.
''')
code('''
D=native();tensors=native('core.tensors');tr=json.loads((ROOT/'reference/hf/Kandinsky-6.0-Lite-5s-Diffusers/transformer/config.json').read_text())
hd_v,hd_a=sum(tr['axes_dims']),sum(tr['axes_dims_a'])
Nv=31*30*54;Na=native('core.algo.prepare_latents').audio_latent_duration(31)
with torch.device('meta'):
    big=D.FusedTransformerDecoderBlock(tr['model_dim'],tr['time_dim'],tr['ff_dim'],hd_v,
        tr['model_dim_a'],tr['time_dim_a'],tr['ff_dim_a'],hd_a,engine='sdpa',text_token_padding=True,
        ca_rope=tr['ca_rope'],cross_gates=tr['cross_gates'],fix_modulation=tr['fix_modulation'])
count=lambda *mods:sum(p.numel() for m in mods for p in m.parameters())
per_block={'video stream (SA, text CA, FFN, modulation)':count(big.video_dec_block),
           'audio stream (SA, text CA, FFN, modulation)':count(big.audio_dec_block),
           'fusion attention (va + av)':count(big.va_cross_attention,big.av_cross_attention),
           'fusion modulation + norms':count(big.va_modulation,big.av_modulation)}
lite_streams={'D_v / heads':[tr['model_dim'],tr['model_dim']//hd_v],'D_a / heads':[tr['model_dim_a'],tr['model_dim_a']//hd_a],
              'head_dim v/a':[hd_v,hd_a],'ffn v/a':[tr['ff_dim'],tr['ff_dim_a']],'fused blocks':tr['num_visual_blocks'],
              'N_v (121 frames, 480x864)':Nv,'N_a (same clip)':Na,
              'flags':{k:tr[k] for k in ['ca_rope','cross_gates','fix_modulation']},
              'params per block, M':{k:round(v/1e6,1) for k,v in per_block.items()},
              'all fused blocks, B':round(sum(per_block.values())*tr['num_visual_blocks']/1e9,2)}
metrics['lite_streams']=lite_streams;print(json.dumps(lite_streams,indent=2))
''')
md(r'''
Асимметрия видна в числах: audio stream имеет половину width и FFN video stream и в сотни раз меньше tokens.
Attention между потоками прямоугольная, $N_v\times N_a$; одного audio token на video token не требуется.

## 1. Направление определяется по query

| Статья (Fig. 1) | Атрибут кода | Q (получатель) | K/V (источник) | Что обновляется |
|---|---|---|---|---|
| V2A block: video queries attend to audio keys/values | `va_cross_attention` | video | audio | video |
| A2V block: audio queries attend to video keys/values | `av_cross_attention` | audio | video | audio |

Имя «V2A» легко прочитать как «video → audio», хотя этот блок обновляет **video**. Поэтому направление проверяем по Q:
число выходных строк равно числу queries, а K/V проекции переводят ширину источника в ширину получателя.
''')
code('''
v_from_a=D.MultiheadCrossAttention(48,8,32,engine='sdpa')
a_from_v=D.MultiheadCrossAttention(32,8,48,engine='sdpa')
v=torch.randn(1,8,48);a=torch.randn(1,6,32)
with torch.no_grad():dv=v_from_a(v,a);da=a_from_v(a,v)
assert dv.shape==v.shape and da.shape==a.shape
metrics['fusion_projection_shapes']={
    'video_Q [out,in]':list(v_from_a.to_query.weight.shape),'audio_K_to_video [out,in]':list(v_from_a.to_key.weight.shape),
    'audio_Q [out,in]':list(a_from_v.to_query.weight.shape),'video_K_to_audio [out,in]':list(a_from_v.to_key.weight.shape)}
print(json.dumps(metrics['fusion_projection_shapes'],indent=2))
''')
md(r'''
## 2. Точный порядок внутри `FusedTransformerDecoderBlock`

Обозначим $\mathrm{M}(h;b,s)=(1+s)\,\mathrm{LN}(h)+b$; $c_v,c_a$ — time embeddings потоков.

**Video, часть 1.** Self-attention с 3D RoPE: $V\leftarrow V+g^v_{sa}\odot \mathrm{SA}_v(\mathrm{M}(V))$.
Затем $\tilde V=\mathrm{M}(V;b^v_{ca},s^v_{ca})$ — сохранённый `vis_pre_ca` — и $T_v=\mathrm{CA}_v(\tilde V,\,\text{text}_v)$.
Text update video **вычислен, но ещё не применён**.

**Audio.** $A\leftarrow A+g^a_{sa}\odot\mathrm{SA}_a(\mathrm{M}(A))$; $\tilde A$ = `aud_pre_ca`; сразу
$A\leftarrow A+g^a_{ca}\odot \mathrm{CA}_a(\tilde A,\text{text}_a)$.

**Fusion.**

1. $V\leftarrow V+g^v_{ca}\odot T_v$ — теперь text update применён к video.
2. Query states: $\hat V=\mathrm{M}(V;b_{va},s_{va})$, $\hat A=\mathrm{M}(A;b_{av},s_{av})$ — **после** text update.
3. $\Delta V=\mathrm{Attn}(Q=\hat V,\;K,V=\tilde A)$ и $\Delta A=\mathrm{Attn}(Q=\hat A,\;K,V=\tilde V)$ — K/V **до** text update.
   Оба выхода вычисляются до применения любого из них.
4. $V\leftarrow V+g_{\to v}\odot\Delta V$, $A\leftarrow A+g_{\to a}\odot\Delta A$; затем отдельные FFN.

Q/K каждой attention проходят RMSNorm по head; при `ca_rope=True` video side получает 3D, audio side — 1D rotations.
Схема ниже рисует именно это: K/V берутся из `vis_pre_ca`/`aud_pre_ca`, Q — из состояния после text update.
Затем — полное тело блока из pinned source и ручная сборка по этой записи.
''')
code('''
from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(12,4.6));ax.set_xlim(0,12.2);ax.set_ylim(0,4.9);ax.axis('off')
boxes=[('Self-attn',.2,1.6),('M(·): pre-CA\\nsaved state',2.25,1.7),('Text CA\\n(cross-attn)',4.45,1.6),
       ('text residual,\\nthen M(·): Q',6.55,1.6),('Cross-modal\\nattention',8.65,1.6),('FFN',10.75,1.2)]
rows={'video':(3.3,'#d8eafa','VIDEO  D_v=1792, 28 heads, 3D RoPE'),'audio':(.9,'#fbe3cc','AUDIO  D_a=896, 14 heads, 1D RoPE')}
for name,(y,color,label) in rows.items():
    ax.text(.2,y+1.0,label,fontsize=10,weight='bold')
    for i,(title,x,w) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x,y),w,.75,boxstyle='round,pad=.06',facecolor=color,edgecolor='#555555'))
        ax.text(x+w/2,y+.37,title,ha='center',va='center',fontsize=8.5)
        if i<len(boxes)-1:ax.annotate('',xy=(boxes[i+1][1],y+.37),xytext=(x+w,y+.37),arrowprops={'arrowstyle':'->'})
ax.annotate('',xy=(9.2,1.65),xytext=(3.1,3.3),arrowprops={'arrowstyle':'->','color':'#2f6db0','lw':1.6})
ax.annotate('',xy=(9.2,3.3),xytext=(3.1,1.65),arrowprops={'arrowstyle':'->','color':'#b0662f','lw':1.6})
ax.plot([],[],color='#2f6db0',label='vis_pre_ca → K/V of audio queries')
ax.plot([],[],color='#b0662f',label='aud_pre_ca → K/V of video queries')
ax.legend(loc='center right',fontsize=9,frameon=False)
ax.text(.2,.15,'Q comes from the destination stream after its text update; K/V come from the other stream before it. '
        'Both cross outputs are computed before either is added.',fontsize=9)
fig.tight_layout();fig.savefig(figure_path(LAB,'dual_stream_architecture'));plt.show()
''')
code("show_source('core/components/dit.py','FusedTransformerDecoderBlock')")
code('''
model=tiny_model().eval();block=model.visual_transformer_blocks[0];activate_gates(block)
Dv,Da=block.model_dim,block.model_dim_a
vr=model.visual_rope((2,2,2),[torch.arange(2)]*3).flatten(0,2);ar=model.audio_rope(torch.arange(6))
v=torch.randn(1,8,Dv);a=torch.randn(1,6,Da);tv=torch.randn(1,16);ta=torch.randn(1,16)
text_v=torch.randn(1,5,Dv);text_a=torch.randn(1,5,Da)
M,G=tensors.apply_scale_shift_norm,tensors.apply_gate_sum
def manual_block(vis,aud,variant='source'):
    vb,ab=block.video_dec_block,block.audio_dec_block
    sa,ca,ff=vb.visual_modulation(tv).chunk(3,-1)
    b,s,g=sa.chunk(3,-1);vis=G(vis,vb.self_attention(M(vb.self_attention_norm,vis,s,b),vr),g)
    b_v,s_v,g_v=ca.chunk(3,-1);vis_pre=M(vb.cross_attention_norm,vis,s_v,b_v);text_out=vb.cross_attention(vis_pre,text_v)
    sa,ca,ff_a=ab.visual_modulation(ta).chunk(3,-1)
    b,s,g=sa.chunk(3,-1);aud=G(aud,ab.self_attention(M(ab.self_attention_norm,aud,s,b),ar),g)
    b,s,g=ca.chunk(3,-1);aud_pre=M(ab.cross_attention_norm,aud,s,b);aud=G(aud,ab.cross_attention(aud_pre,text_a),g)
    vis=G(vis,text_out,g_v)
    # Released flags: fix_modulation=True -> destination time for shift/scale; cross_gates=True -> gates swapped.
    va_b,va_s,va_g=block.va_modulation(tv).split([Dv,Dv,Da],-1)
    av_b,av_s,av_g=block.av_modulation(ta).split([Da,Da,Dv],-1)
    q_v=M(block.va_normalization,vis,va_s,va_b);q_a=M(block.av_normalization,aud,av_s,av_b)
    kv_a,kv_v=(q_a,q_v) if variant=='post-text K/V' else (aud_pre,vis_pre)
    dv=block.va_cross_attention(q_v,kv_a,rope_q=vr,rope_kv=ar)
    if variant=='sequential V then A':
        vis=G(vis,dv,av_g);kv_v=M(vb.cross_attention_norm,vis,s_v,b_v)
    da=block.av_cross_attention(q_a,kv_v,rope_q=ar,rope_kv=vr)
    if variant!='sequential V then A':vis=G(vis,dv,av_g)
    aud=G(aud,da,va_g)
    b,s,g=ff.chunk(3,-1);vis=G(vis,vb.feed_forward(M(vb.feed_forward_norm,vis,s,b)),g)
    b,s,g=ff_a.chunk(3,-1);aud=G(aud,ab.feed_forward(M(ab.feed_forward_norm,aud,s,b)),g)
    return vis,aud
with torch.no_grad():
    nv,na=block(v,a,text_v,text_a,(tv,ta),vr,ar)
    order_rows={}
    for variant in ['source','post-text K/V','sequential V then A']:
        mv,ma=manual_block(v,a,variant)
        order_rows[variant]={'video max error':(mv-nv).abs().max().item(),'audio max error':(ma-na).abs().max().item()}
assert max(order_rows['source'].values())<1e-5
assert max(order_rows['post-text K/V'].values())>1e-4 and order_rows['sequential V then A']['audio max error']>1e-4
metrics['manual_block_vs_native']=order_rows;print(json.dumps(order_rows,indent=2))
''')
md(r'''
Ручная сборка по записи выше совпадает с native forward до округления float32. Две правдоподобные, но неверные схемы
дают заметную ошибку: «K/V после text update» и «сначала V, затем A из обновлённого V».
Схема из статьи, «self → text → cross-modal → FFN», верна для residual updates, но не говорит, из какого состояния берутся K/V.

## 3. Флаги: какое время управляет fusion

`va_modulation` строит shift/scale для video queries, `av_modulation` — для audio queries.
`fix_modulation` выбирает, **чьё время** подаётся в эти modulations. `cross_gates` меняет split:
video-sized gate берётся из `av_modulation`, audio-sized — из `va_modulation`.

| `fix_modulation` | `cross_gates` | shift/scale video query | gate video update | shift/scale audio query | gate audio update |
|---|---|---|---|---|---|
| False | False | $c_a$ | $c_a$ | $c_v$ | $c_v$ |
| False | True | $c_a$ | $c_v$ | $c_v$ | $c_a$ |
| True | False | $c_v$ | $c_v$ | $c_a$ | $c_a$ |
| **True** | **True** (Lite и Pro configs) | $c_v$ | $c_a$ | $c_a$ | $c_v$ |

В released configs нормализация query управляется временем получателя, а gate его обновления — временем **источника**.
Ветка `fix_modulation=False` подаёт $c_a$ в `va_modulation`, созданную с `time_dim`; это работает потому, что `time_dim == time_dim_a`.
Таблицу проверяем hooks: какое время реально получает каждая modulation и из какой modulation приходит gate каждого потока.
''')
code('''
def routing(fix,cross):
    b=D.FusedTransformerDecoderBlock(48,16,96,8,32,16,64,8,engine='sdpa',ca_rope=True,cross_gates=cross,fix_modulation=fix)
    activate_gates(b);seen={}
    def tag(name):
        def hook(module,inputs,output):
            seen[name]='c_v' if inputs[0] is tv else 'c_a' if inputs[0] is ta else '?'
        return hook
    hs=[b.va_modulation.register_forward_hook(tag('va')),b.av_modulation.register_forward_hook(tag('av'))]
    with torch.no_grad():b(v,a,text_v,text_a,(tv,ta),vr,ar)
    for h in hs:h.remove()
    gate_v=seen['av'] if cross else seen['va'];gate_a=seen['va'] if cross else seen['av']
    return {'video query':seen['va'],'video gate':gate_v,'audio query':seen['av'],'audio gate':gate_a}
flag_rows={f'fix={fix}, cross={cross}':routing(fix,cross) for fix in (False,True) for cross in (False,True)}
assert flag_rows['fix=True, cross=True']=={'video query':'c_v','video gate':'c_a','audio query':'c_a','audio gate':'c_v'}
metrics['fusion_time_routing']=flag_rows;print(json.dumps(flag_rows,indent=2))
''')
md(r'''
Gate source для `cross=True` выводится из split в коде; его правильность для released флагов уже подтвердила ручная сборка в разделе 2.

## 4. Zero-init invariance, затем причинное отключение связей

Свежий блок — тождественное отображение: все gates нулевые. Затем включим малые случайные modulation weights,
чтобы связи стали наблюдаемыми. Это исследование функции, а не синхронизации pretrained звука.

Предсказание: изменение A влияет на V при открытой ветви video-query; при `va_gate_scale=0` это влияние исчезает **точно**,
потому что внутри одного блока другого пути A→V нет. Аналогично для V→A и `av_gate_scale`.
Эти аргументы — probe-параметры блока: `DiffusionTransformer3D.forward` их не передаёт, в pipeline они всегда равны 1.
''')
code('''
fresh_block=tiny_model().eval().visual_transformer_blocks[0]
with torch.no_grad():v0,a0=fresh_block(v,a,text_v,text_a,(tv,ta),vr,ar)
identity=max((v0-v).abs().max().item(),(a0-a).abs().max().item());assert identity==0
def run(vv,aa,va=1.,av=1.):
    return block(vv,aa,text_v,text_a,(tv,ta),vr,ar,va_gate_scale=va,av_gate_scale=av)
perturb_a=torch.randn_like(a);perturb_v=torch.randn_like(v)
with torch.no_grad():
    base_v,base_a=run(v,a)
    changed_v,_=run(v,a+perturb_a)
    _,changed_a=run(v+perturb_v,a)
    off_v,_=run(v,a,va=0);off_v2,_=run(v,a+perturb_a,va=0)
    _,off_a=run(v,a,av=0);_,off_a2=run(v+perturb_v,a,av=0)
rows={'A affects V: active':(changed_v-base_v).abs().max().item(),
      'A affects V: va_gate_scale=0':(off_v2-off_v).abs().max().item(),
      'V affects A: active':(changed_a-base_a).abs().max().item(),
      'V affects A: av_gate_scale=0':(off_a2-off_a).abs().max().item()}
assert rows['A affects V: active']>0 and rows['V affects A: active']>0
assert rows['A affects V: va_gate_scale=0']==0 and rows['V affects A: av_gate_scale=0']==0
metrics['zero_gate_identity_error']=identity;metrics['directional_perturbations']=rows
fig,ax=plt.subplots(figsize=(8,3));ax.barh(list(rows),[max(x,1e-12) for x in rows.values()]);ax.set_xscale('log')
ax.set_xlabel('Max |Δ output| after perturbing the other stream (log; exact 0 drawn at 1e-12)')
ax.set_title('One tiny fused block, random weights: directional cuts')
fig.tight_layout();fig.savefig(figure_path(LAB,'cross_modal_ablation'));plt.show();print(rows)
''')
md(r'''
## 5. Gradient показывает путь обучения, не качество синхронизации

Пусть scalar loss зависит только от audio output. При открытой ветви audio-query он даёт gradient на video input.
Поэтому совместное обучение отличается от двух независимых T2V/T2A. Но наличие gradient само по себе
не доказывает lip-sync или сходство голоса: для этого нужны pretrained веса и внешние метрики (SyncNet-подобные, SECS, WER).
''')
code('''
vg=v.clone().requires_grad_()
_,audio_on=run(vg,a)
g_on=torch.autograd.grad(audio_on.square().mean(),vg)[0]
_,audio_off=run(vg,a,av=0)
g_off=torch.autograd.grad(audio_off.square().mean(),vg)[0]
metrics['audio_loss_gradient_to_video_on']=g_on.norm().item()
metrics['audio_loss_gradient_to_video_off']=g_off.norm().item()
assert g_on.norm()>0 and g_off.norm()==0
print('gradient norms on/off:',g_on.norm().item(),g_off.norm().item())
''')
md(r'''
## 6. Unimodal forward, joint forward, partial sampling и decode

| Режим | Вызов | Fusion | Время |
|---|---|---|---|
| video-only | `forward(x_video, x_audio=None)` | нет; text update video применяется в `else`-ветви | `video_*` refiner и $c_v$ |
| audio-only | `forward(None, x_audio)` | нет; video модули не читаются | `audio_*` refiner и $c_a$ |
| joint | оба latent | в каждом блоке | `time=[t_v, t_a]` могут различаться (статья §4.2.2: independent_time) |
| partial sampling | joint forward; известная модальность получает `timesteps[-1]·1000` = 0, т.е. чистые данные | да: чистая модальность — условие | Euler обновляет только генерируемый поток |
| π-Flow distilled | `PiFlowDiffusionTransformer3D` требует fused tuple | да, оба latent всегда | частичность задаётся time scale 0 в `piflow_sampler` |
| decode одного выхода | VAE/vocoder после sampling | не влияет на DiT | — |

Предсказание для первой строки: video-only forward **равен** joint forward, в котором закрыта ветвь video-query
во всех блоках. Audio влияет на video только через неё. Проверим на tiny native модели, установив `va_gate_scale=0`
через forward pre-hook. Это probe, а не опция pipeline.
''')
code('''
joint=tiny_model().eval();activate_gates(joint);_,_,jkw=tiny_inputs(joint)
def close_video_query(module,args,kwargs):
    kwargs['va_gate_scale']=0.;return args,kwargs
with torch.no_grad():
    joint_v,joint_a=fresh_forward(joint,jkw)
    video_only=fresh_forward(joint,dict(jkw,x_audio=None,audio_rope=None))
    hs=[b.register_forward_pre_hook(close_video_query,with_kwargs=True) for b in joint.visual_transformer_blocks]
    cut_v,_=fresh_forward(joint,jkw)
    for h in hs:h.remove()
    t_grid=native('core.algo.flow_matching').flow_match_timesteps(10,5.0,'cpu')
modes={'joint vs video-only, open fusion':(joint_v-video_only).abs().max().item(),
       'joint with video-query cut vs video-only':(cut_v-video_only).abs().max().item(),
       'frozen-modality model time (1000*t_final)':float(t_grid[-1]*1000),
       'joint output shapes':[list(joint_v.shape),list(joint_a.shape)]}
assert modes['joint vs video-only, open fusion']>1e-4 and modes['joint with video-query cut vs video-only']<1e-5
assert modes['frozen-modality model time (1000*t_final)']==0
metrics['forward_modes']=modes;print(json.dumps(modes,indent=2))
''')
md(r'''
## 7. Почему эти опыты ещё не lip-sync, и с чем сравнивать K6

Cuts и gradients доказывают **wiring**: связь существует, направлена так, как записано, и закрывается указанным gate.
Они не говорят, использует ли обученная модель эту связь для синхронизации губ, ритма или тембра.
Для таких выводов нужны pretrained веса, held-out prompts и внешние метрики — см. [протокол исследования](../../curriculum/RESEARCH_PROTOCOL.md).

Сравнение с первичными работами (по их abstracts; детали проверяйте в статьях):

| Система | Потоки | Обмен между потоками |
|---|---|---|
| K6 | асимметричные: video $D_v$=1792, audio $D_a$=896 (Lite), одинаковое число блоков | bidirectional cross-attention в каждом блоке; K/V из pre-text-CA states; 3D/1D RoPE |
| [Ovi](https://arxiv.org/abs/2510.01284) | twin backbones с **идентичной** архитектурой | blockwise bidirectional cross-attention + scaled RoPE для timing |
| [LTX-2](https://arxiv.org/abs/2601.03233) | асимметричные: 14B video, 5B audio | bidirectional AV cross-attention с temporal positional embeddings; cross-modality AdaLN |

Общая идея — blockwise fusion; codec, text encoder, normalization и источник K/V у каждой системы свои.
Не считайте Ovi или LTX-2 буквальной реализацией K6, и не путайте LTX-2 с LTX-Video.

Необычная деталь, которую нужно сохранять при переносе: `OutLayerAudio` применяет LayerNorm дважды
(см. [05](05_nn_patches_and_modulation.ipynb), раздел 3). Комментарий в pinned source прямо называет это поведением
reference training. Без отдельного доказательства это контракт, а не bug.
''')
finish('''
1. Повторите ручную сборку раздела 2 для `fix_modulation=False, cross_gates=False`: какие строки `manual_block` нужно изменить? Совпадение с native — критерий.
2. Повторите directional ablation при `ca_rope=False` с 5 seeds; отчёт — медиана и разброс sensitivity, без вывода о lip-sync.
3. Переставьте audio tokens и их RoPE одновременно, затем только tokens. Какие инвариантности вы ожидаете для video output и почему?
4. Гипотеза для partial sampling: при известном чистом audio ($t_a=0$) gate audio→video в released конфигурации управляется $c_a$. Проверьте hook-ом в joint forward.
5. Для voice-FiLM гипотезы отметьте insertion point и проверьте путь gradient к адаптеру при frozen backbone.
''','''
Вы можете нарисовать оба потока одного блока, подписать для каждой attention Q, K/V, время shift/scale и время gate,
воспроизвести блок вручную и объяснить, почему directional cut подтверждает wiring, но не AV synchronization.
**Следующий шаг:** [09 — joint Flow Matching](09_flow_matching_training.ipynb), где эти потоки обучаются на общей velocity loss.
''')

start('09_flow_matching_training', '09. Joint Flow Matching: path, velocity и учебное обучение',
      'вывести target в noise→data sampling convention и обучить настоящий tiny joint CrossDiT',
      'Lecture 2 / Lab 2; K6 08',
      ['core/components/dit.py','core/algo/flow_matching.py','core/algo/denoise_loop.py'], '4.2',
      '../../kandinsky-5/notebooks/06_dit_and_flow_matching.ipynb')
md(r'''
## 1. Сначала соглашение о времени

В этом опыте t=0 — data, t=1 — noise; sampler идёт от 1 к 0.

$$
x_t=(1-t)x_0+t\epsilon,\qquad u^*(x_t,t\mid x_0,\epsilon)=\epsilon-x_0.
$$

Target задан для каждой пары endpoints. Модель видит только current state, time и condition;
минимум MSE соответствует усреднённому conditional target при данном current state.

$$
\mathcal L=\lambda_v\mathbb E\|u_\theta^v-u_v^*\|^2+
\lambda_a\mathbb E\|u_\theta^a-u_a^*\|^2.
$$

Модальности могут иметь разные t. `model_time=1000*t` — input convention сети,
но Euler update использует разности **нормированного t**, без дополнительного множителя 1000.
[Flow Matching](https://arxiv.org/abs/2210.02747) и
[Rectified Flow](https://arxiv.org/abs/2209.03003) — первичные математические источники.
''')
code('''
x0=torch.randn(4,3);eps=torch.randn_like(x0);t=torch.tensor(.4);h=1e-3
path=lambda time:(1-time)*x0+time*eps
fd=(path(t+h)-path(t-h))/(2*h);target=eps-x0
error=(fd-target).abs().max().item();assert error<3e-4
metrics['finite_difference_target_error']=error
print('dx/dt error:',error,'sampler dt sign:',float(torch.tensor(0.)-torch.tensor(1.)))
''')
md('''
## 2. Данные с известной общей причиной

Создадим синтетические video/audio latents от scalar c; text embeddings также несут c.
Это позволяет учить conditioning без dataset/codec weights. Все данные — случайные числа,
не речь и не изображения. Веса внешних encoders здесь вообще не создаются.

Учебные lambda_v=lambda_a=0.5 относятся только к этому toy objective. Мы не воспроизводим
исторический industrial trainer, face loss, RL, model soup или distillation.
Native text projection cache очищается **перед каждым training forward**.
''')
code('''
model=tiny_model().train();_,_,template=tiny_inputs(model,batch=8)
opt=torch.optim.Adam(model.parameters(),lr=.002)
def batch(seed):
    g=torch.Generator().manual_seed(seed);B=8
    c=torch.randn(B,1,generator=g)
    v0=c.view(B,1,1,1,1)+.1*torch.randn(B,2,4,4,4,generator=g)
    a0=c.view(B,1,1)+.1*torch.randn(B,6,4,generator=g)
    ev=torch.randn(v0.shape,generator=g);ea=torch.randn(a0.shape,generator=g)
    tv=.1+.8*torch.rand(B,generator=g);ta=.1+.8*torch.rand(B,generator=g)
    vt=tv.view(B,1,1,1,1);at=ta.view(B,1,1)
    v=(1-vt)*v0+vt*ev;a=(1-at)*a0+at*ea
    kw=dict(template,x_video=torch.cat((v,torch.zeros_like(v),torch.zeros_like(v[...,:1])),-1),
            x_audio=a,text_embed=c[:,None,:].expand(B,5,12).clone(),
            pooled_text_embed=c.expand(B,8).clone(),time=[tv*1000,ta*1000])
    return kw,(ev-v0,ea-a0)
def losses(seed):
    kw,target=batch(seed);v,a=fresh_forward(model,kw)
    return F.mse_loss(v,target[0]),F.mse_loss(a,target[1])
with torch.no_grad():initial=[x.item() for x in losses(999)]
history=[]
for step in range(100):
    lv,la=losses(step+100);loss=.5*lv+.5*la
    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
    opt.step();history.append([lv.item(),la.item()])
with torch.no_grad():final=[x.item() for x in losses(999)]
assert np.isfinite(history).all() and sum(final)<sum(initial)
metrics['heldout_initial_video_audio']=initial;metrics['heldout_final_video_audio']=final
metrics['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
fig,ax=plt.subplots(figsize=(8,3));ax.plot(np.array(history));ax.legend(['video MSE','audio MSE'])
ax.set_xlabel('Toy training step');ax.set_ylabel('MSE');fig.tight_layout()
fig.savefig(figure_path(LAB,'joint_fm_training'));plt.show();print('Held-out:',initial,'->',final)
''')
md('''
## 3. Почему шумы и времена требуют отдельного контроля

Сходные значения MSE двух ветвей ещё не означают одинаковое perceptual качество:
одна ошибка измерена на 16-channel video latents, другая — на 40-channel audio latents,
масштаб и размерность зависят от codecs. При смене codec scale velocity target тоже меняется.
Для полной модели нужны waveform/video metrics и resource logs; toy validation проверяет только обучение функции.
''')
finish('''
1. Сравните shared t и independent t при фиксированных data/seeds/optimizer.
2. Заморозьте audio backbone и fusion: что останется trainable и какие loss имеют gradient?
3. Смените время s=1-t и выведите знак нового velocity, прежде чем менять код sampler.
4. Почему endpoint x0 можно использовать в target при training, но нельзя передавать sampler как неизвестное условие?
''','''
Вы можете выписать training batch и optimizer contract, назвать model inputs и target,
а также объяснить, почему маленький native CrossDiT учится без загрузки pretrained weights.
''')

start('10_piflow_distillation', '10. Lite Distill: grid-head, DX policy и π-Flow',
      'разобрать 160/400 outputs, выполнить native grid forward и отдельно network-free policy rollout',
      'K6 08–09; ODE и Euler',
      ['core/components/piflow_dit.py','core/algo/piflow_math.py','core/algo/piflow_sampler.py','runtime/sampler.py'], '4.5, 5',
      '../../kandinsky-5/notebooks/07_sampling.ipynb')
md(r'''
## 1. Distillation меняет контракт головы

Обычный FM backbone возвращает один velocity per modality. PiFlow wrapper предсказывает
G grid outputs: video G×16, audio G×40. При G=10 это 160/400, но физическое состояние остаётся 16/40.
Wrapper делает reshape и movedim: `[B,G,...,C]`.

**Два способа конфигурации нельзя смешивать:** HF config уже хранит расширенные 160/400;
native PiFlow constructor принимает per-grid 16/40 и сам умножает на G. Наш tiny constructor
использует per-grid 4/4 и G=3; это другой размер, но тот же native wrapper.

Первичный источник: [π-Flow / Policy-Based Few-Step Generation](https://arxiv.org/abs/2510.14974).
Distillation обучает student воспроизводить поведение teacher. Простое уменьшение числа Euler steps
без обучения student не является distillation. История K6 §4.5 включает π-Flow и adversarial post-training;
здесь исследуем released inference math, не воспроизводим industrial distillation training.
''')
code('''
model=tiny_model(grid=3).eval();v,a,kw=tiny_inputs(model)
with torch.no_grad():pv,pa=fresh_forward(model,kw)
assert pv.shape==(1,3,2,4,4,4) and pa.shape==(1,3,6,4)
metrics['native_grid_shapes']=[list(pv.shape),list(pa.shape)]
print('Physical state:',v.shape,a.shape,'Grid predictions:',pv.shape,pa.shape)
show_source('core/components/piflow_dit.py','PiFlowDiffusionTransformer3D')
''')
md(r'''
## 2. Из grid prediction в DX policy

Network вычисляется один раз на начале сегмента. Для каждого grid output строится candidate clean endpoint:

$$
\widehat x_0^{(g)}=x_{\sigma_s}-\sigma_s\widehat u^{(g)}.
$$

`DXPolicy` интерполирует эти endpoints по **raw time** внутри сегмента и возвращает velocity:

$$
\pi(x,\sigma)=\frac{x-\widehat x_0(\tau)}{\max(\sigma,\varepsilon)},\qquad
\sigma=s(\tau)=\frac{a\tau}{1+(a-1)\tau}.
$$

Raw tau и shifted sigma — разные координаты времени. `policy_rollout_fm` делает много дешёвых
подшагов этой функции **без нового transformer forward на каждом подшаге**.
Число model NFE, grid points и policy substeps — три разных числа.
''')
code('''
mathmod=native('core.algo.piflow_math')
x0=torch.tensor([[.3,-.2]],dtype=torch.float64);noise=torch.tensor([[1.2,.8]],dtype=torch.float64)
u=noise-x0;grid=u[:,None,:].expand(1,10,2).clone()
policy=mathmod.DXPolicy(grid,noise,torch.ones(1,dtype=torch.float64),segment_size=1.,shift=5.)
end,_,_=mathmod.policy_rollout_fm(noise,torch.ones(1,dtype=torch.float64),
     torch.ones(1,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),128,policy)
error=(end-x0).abs().max().item();assert error<1e-10
metrics['oracle_constant_grid_endpoint_error']=error
raw=torch.linspace(0,1,101);shifted=mathmod.shift_timesteps(raw,5.)
fig,ax=plt.subplots(figsize=(6,3));ax.plot(raw,raw,label='raw');ax.plot(raw,shifted,label='shifted sigma')
ax.legend();ax.set_xlabel('Raw tau');fig.tight_layout();fig.savefig(figure_path(LAB,'policy_time'));plt.show()
print('Oracle grid control endpoint error:',error)
''')
md('''
## 3. Native sampler + настоящий случайный tiny grid-DiT

Теперь исполним полный released **tensor sampler**, но на уменьшенной сети и случайных весах.
Это проверяет стыковку grid layout, двух policies и state shapes; осмысленное видео/аудио не синтезируется.
Stock loop проверяет, что присутствуют оба latent tensors. Отключить обновление video не равнозначно
удалить video input и fusion. Runtime запрещает PiFlow + CFG != 1 и PiFlow + MagCache/NaviCache.
''')
code('''
Bundle=native('core.types').LatentBundle
bundle=Bundle(v.clone(),a.clone(),torch.tensor([0,2],dtype=torch.int32),torch.tensor([0,6],dtype=torch.int32))
count=[];hook=model.register_forward_hook(lambda *args:count.append(1))
model.clear_text_proj_cache()
result=native('core.algo.piflow_sampler').piflow_denoise_loop(
    bundle,model,{'text_embeds':kw['text_embed'],'pooled_embed':kw['pooled_text_embed']},
    kw['visual_rope'],kw['audio_rope'],kw['text_rope'],num_steps=2,scheduler_scale=5.,
    attention_mask=kw['attention_mask'],num_policy_substeps=16)
hook.remove()
assert result.video.shape==v.shape and result.audio.shape==a.shape
assert torch.isfinite(result.video).all() and torch.isfinite(result.audio).all() and len(count)==2
metrics['native_tiny_sampler_model_NFE']=len(count)
metrics['native_tiny_sampler_finite']=True
print('2 transformer calls; policy substeps are not extra transformer calls')
''')
finish('''
1. В oracle опыте сделайте grid points различными и нарисуйте интерполируемый x0(tau).
2. Сравните 1, 16, 128 policy substeps при фиксированных network predictions; запишите NFE отдельно.
3. Объясните, почему переключить обычный Lite checkpoint на PiFlow scheduler недостаточно.
4. Найдите в sampler final_step_size_scale=.5: почему последний сегмент короче остальных?
''','''
Вы отличаете физические latent channels, expanded output-head width, grid axis, network NFE
и network-free substeps. Для distill checkpoint используете соответствующий sampler contract.
''')

start('11_sampling_cfg_and_nfe', '11. Sampling: Euler, CFG, schedules и NFE',
      'измерить ошибки solver и фактическое число calls в настоящем native denoise_loop',
      'K6 09–10; Lecture 1 и Lecture 3',
      ['core/algo/flow_matching.py','core/algo/guidance.py','core/algo/denoise_loop.py'], '5',
      '../../kandinsky-5/notebooks/07_sampling.ipynb')
md(r'''
## 1. Solver интегрирует поле, но не обучает его

$$
x_{t+\Delta t}=x_t+\Delta t\,u_\theta(x_t,t,c),\qquad \Delta t<0.
$$

Для u(x,t)=x, x(1)=1, точное x(0)=exp(-1). Это удобный контроль: известны поле и ответ,
поэтому вся ошибка здесь — numerical integration. Для trained модели появится ещё approximation error.
Scheduler shift меняет узлы интегрирования; он не создаёт новую обученную модель.
''')
code('''
schedule=native('core.algo.flow_matching').flow_match_timesteps
rows=[]
for shift in [1.,5.]:
    for steps in [4,8,16,32,64]:
        ts=schedule(steps,shift,'cpu').double();x=torch.tensor(1.,dtype=torch.float64)
        for dt in torch.diff(ts):x=x+dt*x
        rows.append({'shift':shift,'NFE':steps,'error':abs(x.item()-math.exp(-1))})
fig,ax=plt.subplots(figsize=(7,3))
for shift in [1.,5.]:
    sel=[r for r in rows if r['shift']==shift]
    ax.loglog([r['NFE'] for r in sel],[r['error'] for r in sel],'o-',label=f'shift={shift}')
ax.legend();ax.set_xlabel('Field evaluations');ax.set_ylabel('Known-ODE endpoint error')
fig.tight_layout();fig.savefig(figure_path(LAB,'euler_error'));plt.show();metrics['known_ode_errors']=rows
''')
md(r'''
## 2. Guidance — комбинация двух predictions

$$
u_{\mathrm{cfg}}=u_{\varnothing}+w(u_c-u_{\varnothing}).
$$

w=0 возвращает unconditional, w=1 — conditional. Native non-distilled Euler loop при w=1
пропускает второй forward. При w!=1 выполняет оба. Мы проверим это **настоящим loop**,
подставив явно аналитический constant predictor: он не умеет генерировать AV.
Для Lite Distill используем π-Flow из предыдущей лабораторной; CFG sweep ниже относится к обычному FM loop.
''')
code('''
Bundle=native('core.types').LatentBundle
class ConstantFieldDouble(nn.Module):
    visual_cond=True
    def __init__(self):super().__init__();self.calls=0
    def forward(self,**kw):
        self.calls+=1
        level=kw['text_embed'].mean()
        return torch.ones_like(kw['x_video'][...,:4])*level,torch.ones_like(kw['x_audio'])*level
base_v=torch.zeros(1,2,4,4,4);base_a=torch.zeros(1,6,4)
cu_v=torch.tensor([0,2],dtype=torch.int32);cu_a=torch.tensor([0,6],dtype=torch.int32)
cond={'text_embeds':torch.ones(1,3,12),'pooled_embed':torch.zeros(1,8)}
null={'text_embeds':torch.ones(1,3,12)*.25,'pooled_embed':torch.zeros(1,8)}
results=[]
for w in [0.,1.,3.]:
    field=ConstantFieldDouble();bundle=Bundle(base_v.clone(),base_a.clone(),cu_v,cu_a)
    out=native('core.algo.denoise_loop').denoise_loop(bundle,field,cond,null,None,None,
        torch.empty(3,1),torch.empty(3,1),num_steps=4,guidance_weight=w,scheduler_scale=5.)
    expected=-(.25+w*(1-.25))
    error=max((out.video-expected).abs().max().item(),(out.audio-expected).abs().max().item())
    assert error<1e-5 and field.calls==(4 if w==1 else 8)
    results.append({'guidance':w,'model_NFE':field.calls,'endpoint_error':error})
metrics['native_euler_cfg_checks']=results;print(results)
''')
md('''
## 3. Что замерять в промышленном опыте

Указывайте checkpoint, sampler implementation, steps, model NFE, shift, dtype и cache.
Отдельно измеряйте weight loading, text encoding, denoising, VAE decode, vocoder и mux.
Warmup/compile не смешивайте с steady-state generation; median нескольких прогонов лучше одного числа.
Наш constant-field контроль не оценивает speedup большого checkpoint и не подбирает оптимальный CFG.
''')
finish('''
1. При одной trained модели меняйте только solver steps; затем при одном solver сравните training checkpoints.
2. Добавьте отдельный счетчик policy.pi evaluations в π-Flow и сравните с model NFE.
3. На constant field проверьте, почему Euler точен для любого partition времени, а на u=x — уже нет.
''','''
Вы различаете solver error, model error, guidance strength и число transformer calls.
Фраза «10 steps» без названия checkpoint/sampler не определяет вычислительный эксперимент.
''')

start('12_latents_conditioning_and_output', '12. Latents, I2AV references, decode и output contracts',
      'проверить shapes/offsets, воспроизвести visual reference channels и разобрать encode/decode API',
      'K6 03–04, 08 и 11',
      ['core/algo/prepare_latents.py','core/algo/visual_conditioning.py','core/algo/postprocess_video.py',
       'core/algo/postprocess_audio.py','core/algo/mux.py'], '4.2.3, 5',
      '../../kandinsky-5/notebooks/12_generation_utils.ipynb')
md('''
## 1. Shape сам по себе не задаёт смысл

Video codec ожидает BCTHW, native DiT — BTHWC или packed THWC. Audio decoder ожидает BCA,
DiT — BAC или packed AC. `cu_seqlens` содержит границы последовательностей, а не всегда token counts:
у video это latent frames, которые после spatial patching разворачиваются в большее число tokens.
Native helpers создают reproducible CPU noise; межплатформенная RNG parity отдельно не предполагается.
''')
code('''
p=native('core.algo.prepare_latents');Bundle=native('core.types').LatentBundle
b=p.prepare_video_latents(2,3,4,4,4,42,'cpu',dtype=torch.float32)
b=p.prepare_audio_latents(b,7,4,42,'cpu',dtype=torch.float32)
b2=p.prepare_video_latents(2,3,4,4,4,42,'cpu',dtype=torch.float32)
assert torch.equal(b.video,b2.video)
positions=p.prepare_audio_rope_pos(b.audio_cu_seqlens)
assert positions.tolist()==list(range(7))*2
print('Packed shapes:',b.video.shape,b.audio.shape,'video offsets:',b.video_cu_seqlens)
metrics['packed_shapes']=[list(b.video.shape),list(b.audio.shape)]
metrics['audio_positions_reset_each_sequence']=True
''')
md('''
## 2. Tail conditioning и token roles

I2AV reference может добавляться **во временной оси**, а не только в каналах.
K6 помечает generated/reference frame через channel mask и Token Role Embedding.
RoPE задаёт позиционный смысл reference; TRE — дополнительный обучаемый сигнал.
Проверим append и restore на настоящих tensor helpers. Сам restore не доказывает,
что reference уже повлиял на denoising других кадров: это задача learned attention.
''')
code('''
visual=native('core.algo.visual_conditioning')
video=torch.randn(1,3,4,4,4);refs=torch.randn(1,1,4,4,4)
appended,roles,generated=visual.append_visual_references(video,refs,1,3)
assert appended.shape==(1,4,4,4,4) and roles.reshape(1,-1).tolist()==[[0,0,0,1]]
cu=torch.tensor([0,4],dtype=torch.int32)
inp=visual.build_tail_condition(appended.clone(),refs,cu)
assert inp.shape[-1]==9 and inp[0,-1,...,-1].eq(1).all()
assert inp[0,:-1,...,-1].eq(0).all()
corrupted=appended+10;visual.restore_tail_references(corrupted,refs,cu)
assert torch.equal(corrupted[:,-1:],refs)
metrics['tail_condition_shape']=list(inp.shape);metrics['reference_restore_error']=0.
print('Tail condition shape:',inp.shape,'role ids:',roles)
''')
md('''
## 3. Проверяем encode wrapper без выдуманного VAE API

Pinned custom Hunyuan encode возвращает tensor внутри tuple; Diffusers VAE может возвращать
posterior object. Нельзя заменять одно другим, не посмотрев return contract.
В учебном test double явно моделируем tuple[tensor]. Так проверяем layout/scaling wrapper,
а не качество настоящего Hunyuan reconstruction.
''')
code('''
from types import SimpleNamespace
class TupleEncoderDouble:
    config=SimpleNamespace(scaling_factor=.476986)
    def encode(self,frames):
        B,C,T,H,W=frames.shape
        return (torch.ones(B,16,(T-1)//4+1,H//8,W//8),)
encoded=p.encode_video(torch.zeros(2,3,9,32,32),TupleEncoderDouble())
assert encoded.shape==(6,4,4,16)
assert torch.allclose(encoded,torch.full_like(encoded,.476986))
metrics['encode_wrapper_output_shape']=list(encoded.shape)
print('Packed wrapper output:',encoded.shape)
show_source('core/algo/postprocess_video.py','postprocess_video')
''')
md('''
## 4. Последний этап — mux

Decode возвращает video uint8 и audio PCM; mux принимает уже сгенерированные streams.
Container fps, sample rate, число frames/samples и trimming определяют длительность файла.
Рассинхрон может появиться на этом этапе даже при корректных neural outputs.
Поэтому сохраняйте raw durations до mux и после чтения MP4; существующий ручной протокол содержит команды.
''')
finish('''
1. Нарисуйте BCTHW→BTHWC→packed THWC и обратный путь для batch=2; подпишите offsets.
2. Добавьте два references first/last и проверьте role mask, restore и positional correspondence.
3. Подставьте posterior object вместо tuple[tensor] и сформулируйте нужный adapter, не меняя source snapshot.
4. Перед полноценным MP4 запишите video/audio durations и допуск меньше одного audio latent step.
''','''
Вы можете обнаружить shape/API mismatch до загрузки больших весов и объяснить,
какой helper меняет layout, какой нормализует значения, а какой только объединяет streams.
''')

start('13_runtime_memory_and_mlx', '13. Runtime: память, offload, cache и путь к Apple MLX',
      'составить ресурсный бюджет и план переноса с измеримыми parity gates',
      'K6 01, 07–12',
      ['runtime/offload/block.py','runtime/offload/module.py','runtime/cache/magcache.py',
       'runtime/sampler.py','runtime/weights.py'], '6',
      '../../kandinsky-5/notebooks/09_local_m3_max.ipynb')
md('''
## 1. Offload, quantization и cache делают разные вещи

Offload перемещает веса между памятью/устройствами; quantization меняет представление весов;
approximate cache пропускает часть вычислений. Сравнивать их надо при одинаковом output contract.
CPU tensor parallel algebra не означает наличие distributed runtime на одном Apple GPU.
40 GPU cores — внутренние вычислительные блоки M3 Max, не 40 отдельных CUDA devices.

Apple MLX порт не требует обучения весов с нуля. Он требует воспроизвести операции, layouts,
normalization, time/grid semantics и загрузку конвертированных весов, затем пройти parity.
До этого нельзя объявлять полный pretrained K6 MLX inference работающим.
''')
code('''
N=3_000_000_000
rows={
 'BF16 weights GiB':N*2/2**30,
 'FP32 weights GiB':N*4/2**30,
 'FP32 gradients GiB':N*4/2**30,
 'Two FP32 Adam moments GiB':N*8/2**30}
metrics['illustrative_3B_memory_terms']=rows
print('Illustrative exact-3B terms; excludes Qwen, activations, allocator, copies:',rows)
fig,ax=plt.subplots(figsize=(8,3));ax.barh(list(rows),list(rows.values()));ax.set_xlabel('GiB')
fig.tight_layout();fig.savefig(figure_path(LAB,'memory_terms'));plt.show()
''')
md('''
## 2. Измеряем только то, что действительно запущено

Ниже native tiny **CPU** forward, 2 warmups и 5 измерений. Эти числа описывают данный CPU-процесс,
не latency M3 GPU, не прогноз времени pretrained модели и не результат MLX.
При GPU-замерах нужна synchronization перед/после интервала; lazy MLX требует явного evaluation.
''')
code('''
import time,statistics,importlib.util,shutil
model=tiny_model().eval();_,_,kw=tiny_inputs(model)
with torch.no_grad():
    for _ in range(2):fresh_forward(model,kw)
    elapsed=[]
    for _ in range(5):
        start=time.perf_counter();fresh_forward(model,kw);elapsed.append(time.perf_counter()-start)
metrics['tiny_cpu_forward_median_seconds']=statistics.median(elapsed)
metrics['mps_available']=torch.backends.mps.is_available()
metrics['mlx_installed']=importlib.util.find_spec('mlx') is not None
metrics['pretrained_or_mlx_parity_run']=False
free=shutil.disk_usage(ROOT).free
print('CPU median:',statistics.median(elapsed),'seconds; free SSD GB:',free/1e9)
print('Weight-download gate >=50 GB:',free>=50_000_000_000,'(does not trigger a download)')
''')
md('''
## 3. Parity gates: по слоям, затем по pipeline

| Gate | Сравниваем | Что сохраняем |
|---|---|---|
| P0 | Linear, LayerNorm/RMSNorm, GELU/SiLU | max abs + relative L2, dtype |
| P1 | patch/unpatch, 1D/3D RoPE | roundtrip и позиционные tensors |
| P2 | text/video/audio block | inputs, outputs, gate states |
| P3 | joint CrossDiT + grid-head | оба outputs, одинаковые noise tensors |
| P4 | DXPolicy + rollout | raw tau, sigma, grid axis, endpoint |
| P5 | VAE/vocoder + end-to-end | decoded raw streams и fixed-seed baseline |

`labkit/mlx_audio.py` и `scripts/mlx_parity.py` — существующий audio-only прототип.
Он не выполняет P3–P5 полного совместного pipeline. Даже если локальные audio weights можно прочитать
из общего safetensors, удаление video/fusion меняет модель. Оптимизацию добавляют после reference parity.
''')
code("show_source('runtime/sampler.py','reject_incompatible_cache')")
finish('''
1. Составьте таблицу resident modules по стадиям encode text / denoise / decode; добавьте activation budget.
2. Для MLX переноса выберите 3 primitives и выпишите tolerances до первого сравнения.
3. Объясните, почему независимые RNG seeds недостаточны для межфреймворковой parity: нужны одинаковые input tensors.
4. Почему PiFlow checkpoint нельзя включить с calibrated MagCache обычного Lite без отдельной проверки?
''','''
Вы отличаете перенос вычислений от обучения, model memory от полной peak memory и
CPU microbenchmark от pretrained latency. Следующий шаг после освобождения SSD — controlled baseline,
а не установка сразу всех Pro/SR моделей.
''')

start('14_comfyui_component_experiments', '14. ComfyUI: от визуального baseline к отдельному исследованию',
      'связать UI nodes с кодовыми компонентами и подготовить измеряемые однородные A/B опыты',
      'K6 01–13; ComfyUI C0–C5 общего маршрута',
      ['core/components/dit.py','core/algo/piflow_sampler.py','core/algo/postprocess_video.py','core/algo/postprocess_audio.py'], '5–6',
      '../../kandinsky-5/notebooks/10_colab_cuda_distill.ipynb')
md('''
## 1. UI показывает wiring, а не все внутренние слои

Loader DiT подключает joint transformer; DualCLIP loader — Qwen+CLIP; VAE video и audio-vocoder
обслуживают разные decoding paths. Один sampler node скрывает много CrossDiT forwards и policy updates.
Внутренние gates нельзя честно «выключить» простым отсоединением audio decoder:
decoder находится **после** совместного denoising.

Наши Smoke/Baseline workflows используют Lite Distill без SR/beautifier/MagCache.
Node names/widgets зависят от версии ComfyUI и extension. Перед исполнением сверяйте template/node implementation.
Следующая ячейка только читает локальный JSON; в ComfyUI ничего не отправляет.
''')
code('''
workflow=json.loads((ROOT/'workflows/k6_lite_distill_smoke.json').read_text())
rows=[{'id':n['id'],'type':n['type'],'title':n.get('title',n['type']),
       'widgets':n.get('widgets_values',[])} for n in workflow['nodes']]
for row in rows:print(row)
metrics['smoke_node_count']=len(rows);metrics['smoke_link_count']=len(workflow['links'])
assert len(rows)==12 and len(workflow['links'])==14
''')
md('''
## 2. Последовательность опытов и измерений

| Опыт | Что меняем | Что фиксируем | Что измеряем |
|---|---|---|---|
| C0 | Только image IO без моделей | PNG размеры/цвета | exact roundtrip / MSE |
| C4 | Video VAE encode→decode | input, codec scale, dtype | reconstruction MSE/PSNR, decode time |
| K6-S | Первый Lite Smoke | seed=42, 432×240, 33 frames, 10 steps | success, stage times, system memory |
| K6-B | Полный Baseline 5s | 864×480, 121 frames; те же weights | время/память, frames/audio samples |
| K6-A | Только один prompt factor | все остальные widgets/seeds | text adherence, failures, audio transcript |
| K6-R | Адаптер speaker condition | held-out speaker/text protocol | SECS/WER/CER, no-reference regression |

[C0/C4 workflows и инструкции](../../curriculum/COMFYUI_LABS.md),
[ручной K6 notebook](00_manual_experiment_comfy_mlx_film.ipynb),
[протокол измерений](../COMFYUI_AND_EXPERIMENT_RU.md).
K6-S и дальше выполняются **после выбранного SSD gate и проверки моделей**, вручную.
До этого таблицы содержат `not_run`, не выдуманные числа.
''')
code('''
import csv
folder=ROOT/'artifacts/component_labs'/LAB;folder.mkdir(parents=True,exist_ok=True)
columns=['run_id','experiment','checkpoint','workflow','seed','steps','width','height','frames',
         'load_s','text_s','denoise_s','video_decode_s','audio_decode_s','mux_s','peak_memory_bytes','status','notes']
path=folder/'manual_runs.csv'
if not path.exists():
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        writer.writerow({'run_id':'smoke-42','experiment':'K6-S','checkpoint':'Lite-distill-5s',
             'workflow':'k6_lite_distill_smoke.json','seed':42,'steps':10,'width':432,'height':240,
             'frames':33,'status':'not_run','notes':'Await SSD/model gate; empty metrics are not zero'})
print('Локальная таблица:',path)
metrics['prepared_experiments']=['C0','C1','C4','K6-S','K6-B','K6-A','K6-R']
metrics['comfyui_jobs_submitted']=0
''')
md('''
## 3. Как перейти к исследованию голоса

Для адаптера нужны прочитанные codec/conditioning/fusion contracts, baseline исходной модели,
разделение VCTK по текстам и схема insertion point. Zero-init должен сохранять исходную функцию
на шаге 0; frozen backbone не означает запрет gradient через него к адаптеру.

[Наш исследовательский протокол](../../curriculum/RESEARCH_PROTOCOL.md) и
[K6 status](../STATUS.md) перечисляют незавершённые части. Статья о voice adaptation
[arXiv:2608.15690](https://arxiv.org/abs/2608.15690) обучает также audio backbone;
наш frozen-FiLM вариант — отдельная гипотеза. Toy sensitivity из notebook 08 помогает проверить
wiring, но не заменяет SECS/WER и не подтверждает voice cloning.
''')
finish('''
1. На скриншоте workflow подпишите, где text encoder, DiT, policy, video codec, mel codec и vocoder.
2. Составьте A/B из 3 одинаковых seeds и одного изменённого prompt factor; заранее определите метрики.
3. Почему выключение mux audio input не является audio-stream ablation внутри CrossDiT?
4. Укажите stop rule: ресурсный лимит, недостоверная метрика или regression исходной модели.
''','''
У вас есть маршрут от формулы и native CPU опыта к UI baseline и отдельной исследовательской гипотезе,
с явным разделением выполненных численных контролей и будущих pretrained замеров.
''')

for index,item in enumerate(CATALOG):
    path=ROOT/'notebooks'/f"{item['stem']}.ipynb"
    nb=json.loads(path.read_text())
    nav=[]
    if index:nav.append(f"[← {CATALOG[index-1]['title']}]({CATALOG[index-1]['stem']}.ipynb)")
    if index+1<len(CATALOG):nav.append(f"[{CATALOG[index+1]['title']} →]({CATALOG[index+1]['stem']}.ipynb)")
    else:nav.append('[Далее: отдельное исследование →](00_manual_experiment_comfy_mlx_film.ipynb)')
    nb['cells'].append({'cell_type':'markdown','id':'cell-navigation','metadata':{},'source':' · '.join(nav)+'\n'})
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')

(ROOT/'reports/component-catalog.json').write_text(json.dumps({
    'source_revision':REV,'paper':PAPER,'notebooks':CATALOG},ensure_ascii=False,indent=2)+'\n')
print('Built',len(CATALOG),'K6 component notebooks')
