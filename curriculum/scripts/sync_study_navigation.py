"""Keep MIT-first navigation in authored study docs; canonical originals/user answers untouched."""
import argparse,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
COURSE='https://diffusion.csail.mit.edu/2026/index.html'
NOTES='https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf'
SLIDES={1:'20260120_Lecture_01.pdf',2:'20260122_Lecture_02.pdf',3:'20260123_Lecture_03.pdf',4:'20260128_Lecture_04_edited.pdf',5:'20260130_Lecture_05.pdf'}
K5={0:[2,4],1:[3,4],2:[4],3:[4],4:[4],5:[2,4],6:[2],7:[1,2,3],8:[4],9:[4],10:[4],11:[4],12:[1,2,3],13:[4]}
START='<!-- study-navigation:start -->';END='<!-- study-navigation:end -->'
parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');parser.add_argument('--check',action='store_true');args=parser.parse_args()

def navigation(path,lectures):
    import os
    roadmap=os.path.relpath(ROOT/'curriculum/README.md',path.parent).replace(os.sep,'/')
    contents=os.path.relpath(ROOT/'curriculum/CONTENTS.md',path.parent).replace(os.sep,'/')
    links=' · '.join(f'[Lecture {n}](https://diffusion.csail.mit.edu/2026/docs/{SLIDES[n]})' for n in lectures)
    return START+'\n'+f'**Основа:** [MIT 6.S184 / 2026]({COURSE}) · [Основной конспект]({NOTES}) · [Учебная карта]({roadmap}) · [Полное содержание]({contents}).'+('\n\n**Связанные лекции:** '+links+'.' if links else '')+'\n'+END

def insert(source,header):
    source=re.sub(re.escape(START)+r'.*?'+re.escape(END)+r'\n*','',source,flags=re.S)
    title,separator,rest=source.partition('\n')
    return title+'\n\n'+header+'\n\n'+rest.lstrip('\n') if title.startswith('#') else header+'\n\n'+source

rows=[]
mds=[ROOT/'README.md',*sorted(ROOT.glob('lecture_*/*.md')),*sorted((ROOT/'curriculum').glob('*.md')),
     ROOT/'kandinsky-5/README.md',ROOT/'kandinsky-6/README.md',ROOT/'kandinsky-6/COMPONENT_GUIDE.md',ROOT/'kandinsky-6/COMFYUI_AND_EXPERIMENT_RU.md',ROOT/'kandinsky-6/SUPER_RESOLUTION_RU.md',ROOT/'hunyuan-video/README.md',ROOT/'vae/README.md',ROOT/'vae/SOURCE_MAP.md',ROOT/'vae/MIT_LAB3_BRIDGE.md',ROOT/'dit/README.md',ROOT/'dit/MIT_BRIDGE.md',ROOT/'dit/SOURCE_MAP.md']
for path in mds:
    match=re.match(r'lecture_(\d+)',path.parent.name)
    lectures=[int(match.group(1))] if match else []
    old=path.read_text();new=insert(old,navigation(path,lectures))
    if args.write and new!=old:path.write_text(new)
    rows.append({'file':str(path.relative_to(ROOT)),'kind':'markdown','needs_update':new!=old})
notebooks=[*sorted((ROOT/'curriculum/notebooks').glob('*.ipynb')),
           *sorted((ROOT/'lecture_05').glob('*.ipynb')),*sorted((ROOT/'kandinsky-5/notebooks').glob('*.ipynb')),
           *sorted((ROOT/'kandinsky-6/notebooks').glob('*.ipynb')),*sorted((ROOT/'vae/notebooks').glob('*.ipynb')),*sorted((ROOT/'dit/notebooks').glob('*.ipynb'))]
for path in notebooks:
    nb=json.loads(path.read_text());cell=next(c for c in nb['cells'] if c['cell_type']=='markdown')
    before=json.dumps([c['source'] for c in nb['cells'] if c['cell_type']=='code'])
    lectures=[5] if path.parent.name=='lecture_05' else [1,2,3,4]
    if 'vae' in path.parts:lectures=[4]
    if 'dit' in path.parts:lectures={0:[1,2,3],1:[4],2:[3,4],3:[4],4:[3,4],5:[4],6:[1,2,3],7:[1,2,3],8:[1,2],9:[2],10:[1,2,3,4]}[int(path.name[:2])]
    if 'kandinsky-5' in path.parts:lectures=K5[int(path.name[:2])]
    if 'kandinsky-6' in path.parts:
        number=int(path.name[:2])
        lectures={0:[2,3,4],1:[2,4],2:[3,4],3:[4],4:[4],5:[4],6:[4],7:[4],8:[4],9:[2],10:[1,2],11:[1,2,3],12:[4],13:[4],14:[1,2,3,4],15:[1,2,3,4]}[number]
    source=cell['source'];text=''.join(source) if isinstance(source,list) else source
    new=insert(text,navigation(path,lectures));changed=new!=text
    if changed:
        cell['source']=new.splitlines(keepends=True) if isinstance(source,list) else new
        nb.setdefault('metadata',{})['study_navigation']={'course':COURSE,'notes':NOTES,'lecture_numbers':lectures}
    assert before==json.dumps([c['source'] for c in nb['cells'] if c['cell_type']=='code'])
    if args.write and changed:path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    rows.append({'file':str(path.relative_to(ROOT)),'kind':'notebook','needs_update':changed})
print(json.dumps({'mode':'write' if args.write else 'check','files':len(rows),'notebooks':len(notebooks),'pending':[r['file'] for r in rows if r['needs_update']],
 'scope':'navigation text only; originals and Lecture 1 student notebooks excluded'},ensure_ascii=False,indent=2))
if args.check and any(r['needs_update'] for r in rows):raise SystemExit(1)
