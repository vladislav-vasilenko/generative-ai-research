"""Check portable math style in authored Markdown guides; no notebook execution."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
paths=[*sorted((ROOT/'curriculum').glob('*.md')),*sorted(ROOT.glob('lecture_*/*.md')),
       ROOT/'kandinsky-6/README.md',ROOT/'kandinsky-6/COMPONENT_GUIDE.md',ROOT/'kandinsky-6/reports/COMPONENT_VALIDATION.md',*sorted((ROOT/'hunyuan-video').glob('*.md')),*sorted((ROOT/'vae').glob('*.md')),ROOT/'vae/reports/VALIDATION.md']
errors=[];rows=[]
for path in paths:
    fenced=False;math=False;math_fence=False;block=[];count=0;start=0
    for n,line in enumerate(path.read_text().splitlines(),1):
        if line.lstrip().startswith('```'):
            if line.strip()=='```math' and not fenced and not math:
                math_fence=True;line='$$'
            elif line.strip()=='```' and math_fence:
                math_fence=False;line='$$'
            else:
                fenced=not fenced;continue
        if fenced:continue
        if line.strip()=='$$':
            if math:
                content='\n'.join(block)
                if content.count('{')!=content.count('}'):errors.append((str(path.relative_to(ROOT)),start,'unbalanced math braces'))
                if re.search(r'\\(?:operatorname\b|rm\b|[,;:!])',content):errors.append((str(path.relative_to(ROOT)),start,'legacy or Markdown-sensitive macro'))
                if content.count(r'\left')!=content.count(r'\right'):errors.append((str(path.relative_to(ROOT)),start,'unbalanced left/right'))
                count+=1;block=[]
            else:start=n
            math=not math;continue
        if math:block.append(line);continue
        plain=re.sub(r'`[^`]*`','',line)
        if '$' in plain:errors.append((str(path.relative_to(ROOT)),n,'put math in a separate display block'))
    if math:errors.append((str(path.relative_to(ROOT)),start,'unclosed display block'))
    if fenced:errors.append((str(path.relative_to(ROOT)),n,'unclosed code fence'))
    rows.append({'file':str(path.relative_to(ROOT)),'display_blocks':count})
report={'scope':'source style and delimiters; GitHub rendering is verified separately','files':rows,'errors':errors,'passed':not errors}
out=ROOT/'curriculum/reports/markdown-math.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
if errors:raise SystemExit(1)
