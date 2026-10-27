"""Download a small, pinned VCTK pilot through remote ZIP byte ranges.

The full corpus is not fetched. Audio/text splits are by utterance, not by crop.
The mirror remains an independently recorded source; it is not an official K6 asset.
"""
from pathlib import Path
import argparse,concurrent.futures,hashlib,io,json,re,struct,urllib.request,zipfile,zlib
ROOT=Path(__file__).resolve().parents[1]
DATASET='meritondd/vctk-data';REVISION='f39449cbbb0a5ee4879635dffb6c07b510a49e63'
SIZE=6279578615
URL=f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/VCTK-Corpus-mic1.zip'
def byte_range(start,length):
    if length==0:return b''
    end=start+length-1
    request=urllib.request.Request(URL+f'?download=true&vctk_range={start}',headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(request,timeout=180) as response:
        if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
            raise RuntimeError('Server did not return requested ZIP byte range')
        value=response.read()
    if len(value)!=length:raise RuntimeError('Incomplete ZIP byte range')
    return value

class RemoteZip(io.RawIOBase):
    def __init__(self):self.position=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else SIZE+offset
        return self.position
    def read(self,n=-1):
        length=SIZE-self.position if n<0 else min(n,SIZE-self.position)
        result=byte_range(self.position,length);self.position+=len(result);return result

def extract(info):
    header=byte_range(info['header_offset'],30)
    if header[:4]!=b'PK\x03\x04':raise RuntimeError('Invalid ZIP member header')
    name_len,extra_len=struct.unpack_from('<HH',header,26)
    start=info['header_offset']+30+name_len+extra_len
    compressed=byte_range(start,info['compress_size'])
    raw=zlib.decompress(compressed,-15) if info['compress_type']==8 else compressed
    if len(raw)!=info['file_size'] or zlib.crc32(raw)!=info['CRC']:raise RuntimeError('ZIP member checksum mismatch')
    return raw

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true')
    parser.add_argument('--speakers',nargs='+',default=['p225','p226','p227','p228'])
    args=parser.parse_args();out=ROOT/'artifacts/vctk';out.mkdir(exist_ok=True,parents=True)
    index_file=out/'zip-index.json'
    if index_file.exists():index=json.loads(index_file.read_text())
    else:
        print('Reading ZIP directory only...',flush=True)
        with zipfile.ZipFile(RemoteZip()) as archive:
            index=[{field:getattr(info,field) for field in ['filename','header_offset','compress_size','compress_type','file_size','CRC']}
                   for info in archive.infolist()]
        index_file.write_text(json.dumps(index,indent=1))
    rows=[];chosen=[]
    for speaker in args.speakers:
        files=sorted([r for r in index if Path(r['filename']).name.startswith(speaker+'_')
                      and r['filename'].lower().endswith(('.wav','.flac'))],key=lambda r:r['filename'])
        # Skip common introductory sentences shared across VCTK speakers.
        files=[r for r in files if int(Path(r['filename']).stem.split('_')[1])>=20][:30]
        if len(files)<30:raise RuntimeError(f'{speaker}: found {len(files)} audio files, need 30. Inspect zip-index.json first.')
        for i,info in enumerate(files):
            split='reference' if i<4 else 'train' if i<20 else 'validation' if i<25 else 'test'
            basename=Path(info['filename']).name
            utterance='_'.join(Path(basename).stem.split('_')[:2])
            texts=[r for r in index if Path(r['filename']).name==utterance+'.txt']
            row={'speaker_id':speaker,'utterance_id':utterance,'split':split,
                 'source_member':info['filename'],'audio_file':f'audio/{basename}',
                 'text_member':texts[0]['filename'] if texts else None}
            rows.append(row);chosen.append((info,out/row['audio_file']))
            if texts:chosen.append((texts[0],out/'text'/f'{utterance}.txt'))
    manifest={'dataset':DATASET,'revision':REVISION,'source_zip_bytes':SIZE,
              'scope':'small VCTK pilot, not the paper 674-pair benchmark',
              'split_status':'provisional_requires_text_audit','rows':rows}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Planned rows:',len(rows),'selected compressed MiB:',round(sum(x[0]['compress_size'] for x in chosen)/2**20,1),flush=True)
    print('First selected member:',rows[0]['source_member'],flush=True)
    if args.download:
        def run(item):
            info,path=item
            if path.exists() and path.stat().st_size==info['file_size'] and zlib.crc32(path.read_bytes())==info['CRC']:return
            data=extract(info);path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(data)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,chosen))
        for row in rows:
            path=out/row['audio_file'];row['audio_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            text=out/'text'/f"{row['utterance_id']}.txt"
            if text.exists():row['text']=text.read_text().strip()
        # Identical text must never cross train/validation/test, even for different speakers.
        priority={'reference':0,'train':1,'validation':2,'test':3}
        by_text={}
        for row in rows:
            key=' '.join(re.sub(r'[^a-z0-9 ]','',row.get('text','').lower()).split())
            if not key:raise RuntimeError('A target transcript is missing; text-disjoint audit cannot pass')
            row['text_sha256']=hashlib.sha256(key.encode()).hexdigest()
            by_text.setdefault(key,[]).append(row)
        reassigned=0
        for group in by_text.values():
            destination=max((r['split'] for r in group),key=priority.get)
            for row in group:
                if row['split']!=destination:
                    row['initial_split']=row['split'];row['split']=destination;reassigned+=1
        manifest['split_status']='text_disjoint_verified'
        manifest['split_policy']='whole utterance + global normalized-text groups; test > validation > train > reference'
        manifest['reassigned_rows']=reassigned
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Downloaded selected files and saved SHA256.',flush=True)
if __name__=='__main__':main()
