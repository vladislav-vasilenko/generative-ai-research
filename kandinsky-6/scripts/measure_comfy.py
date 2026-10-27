"""Record read-only ComfyUI memory snapshots during a manually queued generation.

No prompts are submitted. MPS memory is unified and is not dedicated VRAM.
The sampling interval produces observed maxima, not a guaranteed true peak.
"""
import argparse,csv,json,time,urllib.request
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8188')
parser.add_argument('--seconds',type=int,default=600);parser.add_argument('--interval',type=float,default=1.)
parser.add_argument('--output',type=Path,default=Path('reports/comfy-memory-samples.csv'))
args=parser.parse_args()
if args.seconds<1 or args.interval<=0:raise ValueError('Positive duration and interval required')
args.output.parent.mkdir(exist_ok=True,parents=True)
with args.output.open('w',newline='') as stream:
    fields=['elapsed_seconds','device_type','ram_total_bytes','ram_free_bytes','vram_total_bytes',
            'vram_free_bytes','torch_vram_total_bytes','torch_vram_free_bytes']
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();start=time.perf_counter()
    while time.perf_counter()-start<args.seconds:
        try:
            with urllib.request.urlopen(args.url.rstrip('/')+'/system_stats',timeout=5) as response:stats=json.load(response)
        except Exception as error:
            raise SystemExit(f'Cannot read ComfyUI at {args.url}: {type(error).__name__}. Copy its actual server URL from the app.')
        system=stats.get('system',{})
        for device in stats.get('devices',[]):
            row={k:device.get(k) for k in fields[4:]}
            row.update(elapsed_seconds=round(time.perf_counter()-start,3),device_type=device.get('type'),
                       ram_total_bytes=system.get('ram_total'),ram_free_bytes=system.get('ram_free'))
            writer.writerow(row)
        stream.flush();time.sleep(args.interval)
print('Saved observed memory samples:',args.output)
