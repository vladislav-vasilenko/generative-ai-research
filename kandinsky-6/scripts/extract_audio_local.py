"""Copy original audio tensor bytes from a local ComfyUI checkpoint.

Does not modify the source model, quantize tensors or claim an equivalent joint
audio-video generator. The result is for audio-path parity/ablation research.
"""
from pathlib import Path
import argparse,hashlib,json,math,struct
from inspect_comfy_models import inspect_safetensors

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    src=args.checkpoint.resolve();out=args.output.resolve()
    if src==out:raise ValueError('Source checkpoint must not be overwritten')
    inspection=inspect_safetensors(src)
    if not inspection['structurally_complete']:raise ValueError('Source checkpoint is still incomplete')
    if any(t not in ['F32','F16','BF16'] for t in inspection['dtypes']):
        raise ValueError('This extractor requires original floating weights, not quantized/FP8 weights')
    with src.open('rb') as stream:
        n=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(n))
    keys=sorted([k for k in header if k!='__metadata__' and (k.startswith('audio_') or '.audio_dec_block.' in k)],key=lambda k:header[k]['data_offsets'][0])
    if not keys:raise ValueError('No native K6 audio keys found; inspect checkpoint format first')
    new={};cursor=0
    for key in keys:
        entry=header[key];count=entry['data_offsets'][1]-entry['data_offsets'][0]
        new[key]={**entry,'data_offsets':[cursor,cursor+count]};cursor+=count
    new['__metadata__']={'scope':'native K6 audio path only','source_filename':src.name,'conversion':'none'}
    data=json.dumps(new,separators=(',',':')).encode();data+=b' '*((-len(data))%8)
    out.parent.mkdir(exist_ok=True,parents=True);tmp=out.with_suffix('.partial')
    with src.open('rb') as source,tmp.open('wb') as target:
        target.write(struct.pack('<Q',len(data)));target.write(data)
        for key in keys:
            a,b=header[key]['data_offsets'];source.seek(8+n+a)
            remaining=b-a
            while remaining:
                chunk=source.read(min(8*2**20,remaining))
                if not chunk:raise ValueError('Unexpected source end')
                target.write(chunk);remaining-=len(chunk)
    tmp.replace(out)
    digest=hashlib.sha256()
    with out.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*2**20),b''):digest.update(chunk)
    report={'output':out.name,'bytes':out.stat().st_size,'sha256':digest.hexdigest(),
            'tensor_count':len(keys),'parameter_count':sum(math.prod(header[k]['shape']) for k in keys),
            'source_filename':src.name,'source_bytes':src.stat().st_size,
            'source_full_sha256_verified':False,'dtype_conversion':False,'scope':'audio path only'}
    out.with_suffix('.verified.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
