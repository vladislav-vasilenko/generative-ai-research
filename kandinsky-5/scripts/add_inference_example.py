from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'notebooks/10_colab_cuda_distill.ipynb';nb=json.loads(p.read_text())
def md(s): return {'cell_type':'markdown','metadata':{},'source':s+'\n'}
def code(s): return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s+'\n'}
extra=[md('''## Официальный inference_examples_t2v.ipynb: читаем перед запуском

В reference сохранён оригинальный notebook того же pinned commit. В нём показаны SFT, noCFG, Distil, 5s/10s и Pro; для нашей цели нужна только **Lite Distil 5s**. Не выполняйте весь notebook: он последовательно создаёт разные большие pipelines.

Оригинал использует `os.chdir("..")`, подразумевая запуск из examples/. Наш курс использует абсолютные пути. Загрузка весов и установка зависимостей в оригинальном notebook не организованы: это примеры вызова уже подготовленной среды, а не самостоятельный Colab quickstart.'''),code('''import json
example=json.loads((SOURCE/'examples/inference_examples_t2v.ipynb').read_text())
distill_code=next(''.join(c['source']) for c in example['cells'] if c['cell_type']=='code' and 'k5_lite_t2v_5s_distil_sd.yaml' in ''.join(c['source']))
print(distill_code)
assert 'get_video_pipeline' in distill_code and 'cuda:0' in distill_code
'''),md('''### Какие defaults скрывает короткая ячейка

| Параметр | В оригинале | В исследовательском запуске ниже |
|---|---|---|
| Backbone | distill config | Тот же distill 5s, strict checkpoint |
| Steps / guidance | Берутся из config: 16 / 1 | Задаются явно: 16 / 1 |
| Seed | Не задан, выбирается случайно | 42 для повторяемого сравнения |
| Prompt expansion | Pipeline default True | False: исследуем точный исходный prompt |
| Scheduler scale | Pipeline default 10 | 5 явно; отдельный опыт сравнивает 5/10 |
| Attention engine | auto | SDPA: без обязательной установки FlashAttention |
| Offload | По умолчанию False | True: экономим VRAM |
| Placement | Все модели cuda:0 | DiT/VAE CUDA; Qwen/CLIP CPU в baseline |

Первое воспроизведение можно сделать со scale=10, чтобы сохранить original default; scale=5 — значение YAML и наш явный исследовательский baseline. Различие фиксируйте в отчёте, не считайте два результата одинаковой конфигурацией.

Веса Qwen на CPU требуют порядка 13 GiB только для BF16 параметров и дополнительную память. Успех GPU проверки не гарантирует достаточную системную RAM. Вариант GPU Qwen + offload/NF4 — отдельная CUDA оптимизация, а не обязательная исходная зависимость.

### Mac или Colab?

Оригинальный pipeline вызывает CUDA set_device, CUDA generator/autocast и CUDA-aware VAE tiling. Замены строк cuda:0 на mps недостаточно. Mac notebook 09 использует отдельный experimental adapter; small MPS FP32/BF16 проверены, pretrained 5s пока не проверен. Для первого full-scale baseline выбираем CUDA, затем сравниваем с Mac port.

Colab GPU model и RAM не гарантированы. Проверяйте BF16 и ресурсы runtime; T4 не поддерживает выбранную native BF16 ветвь аппаратно и не является базовым маршрутом данного курса. Предпочтительный первый опыт — A100 и high-memory runtime, если доступны. [Colab resource FAQ](https://research.google.com/colaboratory/faq.html#resource-limits).'''),code('''for relative,needle in [('kandinsky/utils.py','torch.cuda.set_device'),('kandinsky/generation_utils.py','g = torch.Generator(device="cuda")'),('kandinsky/models/vae.py','free_mem = torch.cuda.mem_get_info')]:
    print('CUDA dependency:',relative)
    source_excerpt(relative,needle,5)
''')]
# Idempotent insertion for regeneration and direct updates.
if not any('## Официальный inference_examples_t2v.ipynb' in c['source'] for c in nb['cells']):
 nb['cells']=nb['cells'][:3]+extra+nb['cells'][3:]
 for i,c in enumerate(nb['cells']): c['id']=f'cell-{i:03}'
 p.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
