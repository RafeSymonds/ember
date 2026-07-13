#!/usr/bin/env python3
"""Download datasets into data/. Infra, not learning material.

Usage: bin/ember data [name]     (default: shakespeare)

A1 adds a larger corpus for BPE round-trip testing; A2's FineWeb-Edu shards
are prepared on the training box, not here.
"""

import sys
import urllib.request
from pathlib import Path

DATASETS = {
    "shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "data/shakespeare/input.txt",
    ),
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "shakespeare"
    if name not in DATASETS:
        sys.exit(f"unknown dataset: {name} (available: {', '.join(DATASETS)})")
    url, dest = DATASETS[name]
    dest = Path(dest)
    if dest.exists():
        print(f"{dest} already present ({dest.stat().st_size:,} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"wrote {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
