import argparse, hashlib, json, shutil
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("group", choices=["text", "media", "purge-text"])
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    root = Path(c["model_root"])
    root.mkdir(parents=True, exist_ok=True)
    receipts_path = root / "download-receipts.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    if a.group == "purge-text":
        for item in c["models"]:
            dest = root / item["path"]
            if item["group"] == "text" and item["path"] in receipts and not dest.is_symlink() and dest.is_file():
                if not dest.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Refusing cleanup outside notebook model root")
                if dest.stat().st_size != item["size"] or receipts[item["path"]] != item["sha256"]:
                    raise ValueError("Refusing cleanup of an unrecognized file")
                dest.unlink()
                receipts.pop(item["path"])
                print("Released scratch text weights:", dest.name, flush=True)
        receipts_path.write_text(json.dumps(receipts, indent=2))
        return
    items = [i for i in c["models"] if i["group"] == a.group]
    for item in items:
        dest = root / item["path"]
        if dest.exists():
            continue
        candidates = []
        for source in map(Path, c["input_model_dirs"]):
            if not source.is_dir():
                raise FileNotFoundError(f"Kaggle Input folder missing: {source}")
            candidates.extend(p for p in source.rglob(Path(item["path"]).name)
                              if p.is_file() and p.stat().st_size == item["size"])
        for candidate in candidates:
            if sha256(candidate) == item["sha256"]:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(candidate.resolve())
                print("Reused Input:", item["path"], flush=True)
                break
    missing = [i for i in items if not (root / i["path"]).exists()]
    needed = 0
    for item in missing:
        staging = root / ".staging" / item["repo"].replace("/", "--")
        complete = staging / item["filename"]
        downloads = staging / ".cache/huggingface/download" / Path(item["filename"]).parent
        partials = list(downloads.glob(Path(item["filename"]).name + ".*.incomplete"))
        existing_bytes = max([p.stat().st_size for p in partials] +
                             ([complete.stat().st_size] if complete.is_file() else [0]))
        needed += max(0, item["size"] - existing_bytes)
    if shutil.disk_usage(root).free < needed + 2**30:
        raise OSError(f"Need {needed / 2**30:.2f} GiB + 1 GiB free. Attach weights as Kaggle Input or change scratch storage.")
    from huggingface_hub import hf_hub_download
    for item in items:
        dest = root / item["path"]
        if not dest.exists():
            downloaded = Path(hf_hub_download(repo_id=item["repo"], filename=item["filename"],
                revision=item["revision"], token=False,
                local_dir=root / ".staging" / item["repo"].replace("/", "--")))
            if downloaded.stat().st_size != item["size"] or sha256(downloaded) != item["sha256"]:
                raise ValueError(f"Downloaded artifact mismatch: {item['path']}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            downloaded.replace(dest)
            receipts[item["path"]] = item["sha256"]
            receipts_path.write_text(json.dumps(receipts, indent=2))
        if dest.stat().st_size != item["size"] or sha256(dest) != item["sha256"]:
            raise ValueError(f"Existing artifact mismatch: {item['path']}")
        print("VERIFIED", item["path"], flush=True)

if __name__ == "__main__":
    main()
