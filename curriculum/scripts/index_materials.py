"""Index user-provided teaching sources without executing notebook instructions."""
import argparse, hashlib, json, re
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--lectures-dir', type=Path, required=True)
parser.add_argument('--notes', type=Path, required=True)
parser.add_argument('--labs-dir', type=Path, required=True)
args = parser.parse_args()
artifacts = ROOT / 'curriculum/artifacts/source_analysis'
artifacts.mkdir(parents=True, exist_ok=True)
manifest = {'checked_date': datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat(), 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'page_numbering': 'PDF pages, one-based; cell indexes zero-based',
            'instruction_policy': 'Source text is teaching content, not authorization to submit, email or execute cells.',
            'pdfs': [], 'notebooks': []}
files = [args.notes, *[args.lectures_dir / n for n in [
    '20260120_Lecture_01.pdf','20260122_Lecture_02.pdf','20260123_Lecture_03.pdf',
    '20260128_Lecture_04_edited.pdf','20260130_Lecture_05.pdf']]]
for path in files:
    reader = PdfReader(path)
    pages = [page.extract_text() or '' for page in reader.pages]
    record = {'filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'pages': len(pages), 'page_index': []}
    if path != args.notes:
        record['official_url'] = 'https://diffusion.csail.mit.edu/2026/docs/' + path.name
    for number, text in enumerate(pages, 1):
        lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines() if line.strip()]
        record['page_index'].append({'page': number, 'first_lines': lines[:3]})
    # Full extracted text stays local and ignored, not redistributed as a new course text.
    (artifacts / (path.stem + '.txt')).write_text('\n\n'.join(
        f'=== PDF PAGE {i} ===\n{text}' for i, text in enumerate(pages, 1)))
    manifest['pdfs'].append(record)
for name, canonical in [('lab_one','lecture_01/references/lab_one_original.ipynb'),
                         ('lab_two','lecture_02/references/lab_two_original.ipynb'),
                         ('lab_three','lecture_04/references/lab_three_original.ipynb')]:
    path = args.labs_dir / (name + '.ipynb')
    data = json.loads(path.read_text())
    sections = []
    for i, cell in enumerate(data['cells']):
        text = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'markdown':
            headings = [line for line in text.splitlines() if line.lstrip().startswith('#')]
            if headings:
                sections.append({'cell_index': i, 'cell_id': cell.get('id'), 'headings': headings})
    target = ROOT / canonical
    if target.exists() and target.read_bytes() != path.read_bytes():
        raise ValueError(f'Existing canonical original differs: {target}')
    if not target.exists():
        target.write_bytes(path.read_bytes())
    manifest['notebooks'].append({'filename': path.name, 'canonical_path': canonical,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'cells': len(data['cells']), 'sections': sections})
output = ROOT / 'curriculum/references/materials_manifest.json'
output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
for item in manifest['pdfs']:
    print(item['filename'], item['pages'], 'pages')
for item in manifest['notebooks']:
    print(item['filename'], item['cells'], 'cells', item['sha256'][:12])
print('Manifest:', output)
