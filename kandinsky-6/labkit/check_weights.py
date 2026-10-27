from pathlib import Path
import hashlib,json

def assert_complete(path):
    """Never mistake a resumable sparse partial file for pretrained weights."""
    path=Path(path);stamp=path.with_suffix('.verified.json')
    if stamp.is_file():record=json.loads(stamp.read_text())
    else:
        # Allow the exact full public checkpoint directly, without a second SSD copy.
        # Other revisions/quantizations require their own reviewed provenance entry.
        record={'bytes':6381599096,'sha256':'b1b52668561d9138d93bb4da13c9999e615bf569be9f8cf65732d54652447f53',
                'scope':'pinned original full Lite Distill checkpoint, read-only'}
        if path.stat().st_size!=record['bytes']:
            raise RuntimeError('No completion stamp and file is not the pinned full Lite Distill checkpoint')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*2**20),b''):digest.update(chunk)
    if digest.hexdigest()!=record['sha256'] or path.stat().st_size!=record['bytes']:
        raise RuntimeError('Weights do not match the completion checksum')
    return record
