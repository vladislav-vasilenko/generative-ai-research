"""Read model paths and safetensors headers, without importing ComfyUI or loading weights."""
from pathlib import Path
import argparse,json,struct

def inspect_safetensors(path):
    with path.open('rb') as stream:
        header_size=struct.unpack('<Q',stream.read(8))[0]
        if header_size>64*2**20:raise ValueError('Unexpectedly large safetensors header')
        header=json.loads(stream.read(header_size))
    tensors={k:v for k,v in header.items() if k!='__metadata__'}
    data_size=max((v['data_offsets'][1] for v in tensors.values()),default=0)
    complete=(8+header_size+data_size)==path.stat().st_size
    audio=[k for k in tensors if k.startswith('audio_') or '.audio_dec_block.' in k]
    return dict(file=path.name,bytes=path.stat().st_size,structurally_complete=complete,
                tensor_count=len(tensors),audio_tensor_count=len(audio),
                dtypes=sorted(set(v['dtype'] for v in tensors.values())),
                audio_output_shape=tensors.get('audio_out_layer.out_layer.weight',{}).get('shape'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',action='append',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=Path('reports/comfy-model-inventory.json'))
    args=parser.parse_args();rows=[]
    for root in args.models:
        if not root.exists():continue
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix not in ['.safetensors','.pth','.pt','.gguf']:continue
            row={'relative_path':str(path.relative_to(root)),'bytes':path.stat().st_size}
            if path.suffix=='.safetensors':
                try:row.update(inspect_safetensors(path))
                except Exception as error:row['header_error']=str(error)
            rows.append(row)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps({'model_files':rows,'weights_loaded':False},indent=2)+'\n')
    for row in rows:print(row)
    print('Files:',len(rows),'; report:',args.output)
if __name__=='__main__':main()
