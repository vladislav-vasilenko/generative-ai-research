from pathlib import Path
import gc
import json
import platform
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reference' / 'kandinsky-5'
REVISION = (SOURCE / 'SOURCE_REVISION').read_text().strip()
TARGET = 'kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s'

def device_info():
    import torch
    import psutil
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    return dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                device=device, total_memory_GiB=round(psutil.virtual_memory().total / 2**30, 1),
                available_memory_GiB=round(psutil.virtual_memory().available / 2**30, 1))

def release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def synchronize(device):
    import torch
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    elif str(device) == 'mps': torch.mps.synchronize()

def source_excerpt(relative, symbol, lines=35):
    text = (SOURCE / relative).read_text().splitlines()
    start = next(i for i, line in enumerate(text) if symbol in line)
    print('\n'.join(f'{i+1:4}: {text[i]}' for i in range(start, min(start+lines, len(text)))))
    return f'https://github.com/kandinskylab/kandinsky-5/blob/{REVISION}/{relative}#L{start+1}'

def shifted_schedule(steps, scale=5.0):
    import torch
    if steps < 1 or scale <= 0: raise ValueError('steps >= 1 и scale > 0')
    t = torch.linspace(1, 0, steps+1)
    return scale*t / (1+(scale-1)*t)

def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
