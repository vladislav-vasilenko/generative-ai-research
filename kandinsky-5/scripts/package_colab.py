from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
items=[ROOT/'README.md',ROOT/'MODEL_CATALOG.md',ROOT/'THIRD_PARTY.md',ROOT/'requirements-labs.txt',ROOT/'.gitignore']
for folder in ['notebooks','labkit','scripts','reference']:
    items.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
items.extend(p for p in (ROOT/'reports').glob('*') if p.is_file())
with zipfile.ZipFile(ROOT/'KandinskyLab-colab.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in items: z.write(p,Path('KandinskyLab')/p.relative_to(ROOT))
print('Colab package bytes:',(ROOT/'KandinskyLab-colab.zip').stat().st_size)
