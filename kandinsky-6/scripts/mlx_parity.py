"""Compare MLX against a saved PyTorch reference and record actual measurements."""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mlx.core as mx
from labkit.mlx_audio import AudioCrossDiT,audio_key

parser=argparse.ArgumentParser();parser.add_argument('--pretrained',action='store_true');args=parser.parse_args()
tag='pretrained' if args.pretrained else 'random_fixture'
folder=ROOT/'artifacts/parity';meta=json.loads((folder/f'{tag}.json').read_text())
data=np.load(folder/f'{tag}.npz')
mx.set_default_device(mx.gpu)
if args.pretrained:
    from labkit.check_weights import assert_complete
    assert_complete(ROOT/meta['weights'])
loaded=mx.load(str(ROOT/meta['weights']))
weights={k:v.astype(mx.float32) for k,v in loaded.items() if audio_key(k)}
del loaded
mx.eval(weights);mx.clear_cache()
model=AudioCrossDiT(weights,meta['config'])
inputs=[mx.array(data[k]) for k in ['x','text','pooled','time','mask']]
start=time.perf_counter();actual=model(*inputs);mx.eval(actual);cold=time.perf_counter()-start
actual_np=np.array(actual);expected=data['expected']
error=float(np.max(np.abs(expected-actual_np)))
relative=float(np.linalg.norm((expected-actual_np).ravel())/max(np.linalg.norm(expected.ravel()),1e-12))
film={'weight':mx.zeros((2*model.width,16)), 'bias':mx.zeros((2*model.width,))}
speaker=mx.ones((inputs[0].shape[0],16))
with_adapter=model(*inputs,film=film,speaker=speaker);mx.eval(with_adapter)
identity_error=float(mx.max(mx.abs(actual-with_adapter)).item())
timings=[]
for _ in range(3):
 start=time.perf_counter();value=model(*inputs);mx.eval(value);timings.append(time.perf_counter()-start)
report=dict(pretrained_weights=args.pretrained,scope='audio path only; not joint PiFlow generation',
 max_absolute_error=error,relative_l2_error=relative,zero_film_max_absolute_error=identity_error,
 output_shape=list(actual.shape),cold_seconds=cold,warm_seconds=timings,
 active_memory_bytes=mx.get_active_memory(),peak_memory_bytes=mx.get_peak_memory(),
 reference=meta,passed=bool(error<1e-3 and relative<1e-4 and identity_error==0.))
(ROOT/'reports').mkdir(exist_ok=True)
(ROOT/f'reports/mlx-{tag}-parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
if not report['passed']:raise SystemExit('Numerical parity check failed; do not proceed to training')
