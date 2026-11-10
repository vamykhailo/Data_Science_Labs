"""Stream the official archive, retaining only variant 5 and metadata.

No full archive is stored. Re-running retains existing identical files.
"""
from pathlib import Path
import hashlib
import json
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'datasets' / 'food-101'
URL = 'https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
NUMBER = 5
meta_path = DEST / 'meta' / 'classes.txt'
EXPECTED_CLASSES = {'beef_tartare', 'eggs_benedict', 'miso_soup'}
if meta_path.is_file():
    all_classes = sorted(meta_path.read_text().splitlines())
    CLASSES = {all_classes[i] for i in [NUMBER-1, NUMBER+29, NUMBER+59]}
    assert CLASSES == EXPECTED_CLASSES
else:
    # Bootstrap; verify against the official metadata after downloading.
    CLASSES = EXPECTED_CLASSES


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    started = time.time()
    count = 0
    with urllib.request.urlopen(URL, timeout=120) as response:
        print('Official archive bytes:', response.headers.get('Content-Length'), flush=True)
        with tarfile.open(fileobj=response, mode='r|gz') as archive:
            for member in archive:
                parts = Path(member.name).parts
                if not member.isfile() or len(parts) < 3 or parts[0] != 'food-101':
                    continue
                keep = parts[1] == 'meta' or (parts[1] == 'images' and parts[2] in CLASSES)
                if not keep:
                    continue
                target = DEST.joinpath(*parts[1:])
                if not target.resolve().is_relative_to(DEST.resolve()):
                    raise ValueError(member.name)
                data = archive.extractfile(member).read()
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(data).digest():
                        raise ValueError(f'Existing file differs: {target}')
                else:
                    target.write_bytes(data)
                count += 1
                if count % 250 == 0:
                    print(f'Retained {count} files, elapsed {time.time()-started:.0f}s', flush=True)
    all_classes = sorted(meta_path.read_text().splitlines())
    assert {all_classes[i] for i in [NUMBER-1, NUMBER+29, NUMBER+59]} == CLASSES
    info = {'url': URL, 'retained_classes': sorted(CLASSES), 'files': count,
            'download_seconds': time.time()-started, 'method': 'official tar.gz streaming; selected files only'}
    (ROOT / 'download_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(info, flush=True)


if __name__ == '__main__':
    main()
