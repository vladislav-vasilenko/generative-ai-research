"""Выполнить Lab One в чистом ядре, сохранить результаты и HTML для чтения."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
for key, subdir in {
    'MPLCONFIGDIR': '.cache/matplotlib',
    'IPYTHONDIR': '.cache/ipython',
    'JUPYTER_RUNTIME_DIR': '.jupyter/runtime',
}.items():
    folder = PROJECT / subdir
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(key, str(folder))
os.environ.setdefault('JUPYTER_PATH', str(Path(sys.prefix) / 'share/jupyter'))

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--animation', action='store_true', help='Выполнить также HTML-анимацию.')
    args = parser.parse_args()
    path = ROOT / 'lab_one_ru.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    if args.animation:
        setup = next(c for c in notebook.cells if c.get('id') == 'original-02')
        setup.source = setup.source.replace('RUN_ANIMATION = False', 'RUN_ANIMATION = True')
    client = NotebookClient(notebook, timeout=300, kernel_name='python3',
                            resources={'metadata': {'path': str(ROOT)}}, allow_errors=False)
    client.execute()
    if args.animation:
        # Сохраняем безопасное для обычного Run All значение; полученная анимация остаётся в выводе.
        setup.source = setup.source.replace('RUN_ANIMATION = True', 'RUN_ANIMATION = False')
    nbformat.write(notebook, path)
    output_dir = ROOT / 'outputs'
    output_dir.mkdir(exist_ok=True)
    exporter = HTMLExporter(template_name='lab')
    html, _ = exporter.from_notebook_node(notebook)
    # Убираем хвостовые пробелы экспортёра, чтобы результат проходил git diff --check.
    html = '\n'.join(line.rstrip() for line in html.splitlines()) + '\n'
    (output_dir / 'lab_one_ru.html').write_text(html)
    print(f'Выполнено кодовых ячеек: {sum(c.cell_type == "code" for c in notebook.cells)}')
    print(f'Результаты: {path}')
    print(f'Версия для чтения: {output_dir / "lab_one_ru.html"}')


if __name__ == '__main__':
    main()
