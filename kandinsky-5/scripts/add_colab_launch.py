"""Add reproducible Colab entry points without changing local lab behavior."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
bootstrap='''# Этот шаг нужен только в Google Colab; локально используется установленное окружение.
from pathlib import Path
import sys, subprocess
if 'google.colab' in sys.modules:
    checkout=Path('/content/generative-ai-research')
    if not checkout.exists():
        subprocess.run(['git','clone','--depth','1','https://github.com/vladislav-vasilenko/generative-ai-research.git',str(checkout)],check=True)
    course=checkout/'kandinsky-5'
    if not (course/'labkit').is_dir():
        # Пока PR не влит, берём его ветку. После merge используется main.
        subprocess.run(['git','-C',str(checkout),'fetch','--depth','1','origin','kandinsky-5-labs'],check=True)
        subprocess.run(['git','-C',str(checkout),'checkout','--detach','FETCH_HEAD'],check=True)
    course=checkout/'kandinsky-5'
    expected={'torch':'2.8.0','transformers':'4.56.2','diffusers':'0.35.1'}
    loaded={name:str(getattr(sys.modules[name],'__version__','')) for name in expected if name in sys.modules}
    subprocess.run([sys.executable,'-m','pip','install','-r',str(course/'requirements-labs.txt')],check=True)
    if any(version.split('+')[0]!=expected[name] for name,version in loaded.items()):
        raise RuntimeError('Библиотека уже была импортирована до установки. Перезапустите сессию Colab и выполните notebook с первой ячейки.')
    if str(course) not in sys.path: sys.path.insert(0,str(course))
    print('Course:',course)
    print('Git revision:',subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip())
else:
    print('Локальный запуск: зависимости устанавливаются по README.')
'''
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=json.loads(path.read_text())
    if any(c.get('metadata',{}).get('kandinsky_colab_setup') for c in nb['cells']): continue
    nb['cells'].insert(1,{'cell_type':'markdown','metadata':{'kandinsky_colab_setup':True},'source':'### Подготовка Google Colab\n\nПервая ячейка загружает код курса и устанавливает зависимости. Большие веса моделей она не скачивает. Для учебных опытов достаточно CPU; полная генерация в notebook 10 требует NVIDIA GPU. Если Colab попросит перезапуск сессии после установки, перезапустите и снова выполните ячейки сверху вниз.\n'})
    nb['cells'].insert(2,{'cell_type':'code','metadata':{'kandinsky_colab_setup':True},'execution_count':None,'outputs':[],'source':bootstrap})
    for c in nb['cells']:
        if c['cell_type']=='code' and 'candidates=[Path.cwd()' in ''.join(c['source']):
            c['source']=''.join(c['source']).replace("Path('/content/KandinskyLab')", "Path('/content/KandinskyLab'), Path('/content/generative-ai-research/kandinsky-5')")
    for i,c in enumerate(nb['cells']): c['id']=f'cell-{i:03}'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
