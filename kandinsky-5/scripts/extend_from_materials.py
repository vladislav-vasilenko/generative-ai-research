"""Rebuild the source-grounded MIT additions in labs 02 and 06."""
from pathlib import Path
import json,re,ast
ROOT=Path(__file__).resolve().parents[1]
blocks=json.loads((ROOT/'scripts/materials_extensions.json').read_text())
for item in blocks:
    path=ROOT/'notebooks'/item['notebook']
    nb=json.loads(path.read_text())
    nb['cells']=[c for c in nb['cells'] if not c.get('metadata',{}).get('course_materials_extension')]
    where=next(i for i,c in enumerate(nb['cells']) if item['before'] in ''.join(c['source']))
    cells=[]
    for i,raw in enumerate(item['cells']):
        c={'cell_type':raw['cell_type'],'source':raw['source'],'metadata':{'course_materials_extension':True},'id':f'mit-{item["prefix"]}-{i:03}'}
        if c['cell_type']=='code':
            ast.parse(c['source'])
            c.update(execution_count=None,outputs=[])
        cells.append(c)
    nb['cells'][where:where]=cells
    for c in nb['cells']:
        if c['cell_type']=='markdown':
            c['source']=re.sub(r'\\operatorname\{([^}]+)\}',r'\\mathrm{\1}', ''.join(c['source']))
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    print('Enhanced',item['notebook'],'with',len(cells),'material cells')
