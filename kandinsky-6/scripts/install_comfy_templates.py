"""Install research templates locally; default is a dry run, no model download."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--replace', action='store_true', help='Replace an existing different research template')
    args = parser.parse_args()
    plugin = args.comfy_root / 'custom_nodes/kandinsky6'
    if not plugin.is_dir():
        raise FileNotFoundError(f'Kandinsky 6 extension is not installed at {plugin}')
    destinations = [plugin / 'example_workflows',
                    args.comfy_root / 'user/default/workflows/K6 Research']
    plan = []
    for profile, label in [('smoke', 'Smoke'), ('baseline', 'Baseline')]:
        source = ROOT / f'workflows/k6_lite_distill_{profile}.json'
        for directory in destinations:
            target = directory / f'Kandinsky 6 Lite - {label}.json'
            if target.exists() and target.read_bytes() != source.read_bytes() and not args.replace:
                raise FileExistsError(f'{target}: different contents; inspect before using --replace')
            plan.append((source, target))
    for source, target in plan:
        print(('INSTALL' if args.install else 'DRY RUN'), target)
        if args.install:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_bytes() != source.read_bytes():
                shutil.copyfile(source, target)
    print('Refresh/restart Comfy Desktop. No weights are downloaded by this script.')


if __name__ == '__main__':
    main()
