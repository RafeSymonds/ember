#!/usr/bin/env python3
"""A0 scorekeeping: compare your best val loss against the nanoGPT baseline.

Usage:
    bin/ember compare --ours 1.49 --baseline 1.47
    bin/ember compare --ours runs/a0.log --baseline baselines/nanogpt_shakespeare_char.log

Each side accepts either a number (best val loss, nats/char) or a log file,
from which the minimum "val loss <x>" is extracted — so if your train loop
prints lines like "step 1250: train loss 1.61, val loss 1.58", logs work
directly. Pass bar: within 2% of the baseline (PLAN.md, milestone A0).
"""

import argparse
import math
import re
import sys
from pathlib import Path

VAL_RE = re.compile(r"val loss:?\s*([0-9]+\.[0-9]+)")


def resolve(value: str, label: str) -> float:
    try:
        return float(value)
    except ValueError:
        pass
    path = Path(value)
    if not path.exists():
        sys.exit(f"error: {label} is neither a number nor a file: {value}")
    losses = [float(m.group(1)) for m in VAL_RE.finditer(path.read_text())]
    if not losses:
        sys.exit(f"error: no 'val loss <x>' lines found in {path}")
    return min(losses)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ours", required=True, help="best val loss (nats) or log file")
    ap.add_argument("--baseline", required=True, help="best val loss (nats) or log file")
    args = ap.parse_args()

    ours = resolve(args.ours, "--ours")
    base = resolve(args.baseline, "--baseline")
    ln2 = math.log(2)
    delta = (ours - base) / base

    print(f"{'':10}{'val loss (nats)':>18}{'bits-per-byte':>16}")
    print(f"{'ours':10}{ours:>18.4f}{ours / ln2:>16.4f}")
    print(f"{'baseline':10}{base:>18.4f}{base / ln2:>16.4f}")
    print(f"\ndelta vs baseline: {delta:+.2%}   (pass bar: within +2.00%)")

    if delta <= -0.02:
        print("result: BETTER than baseline by >2% — don't celebrate yet.")
        print("        Suspect leakage or an eval bug; investigate with the same")
        print("        energy you'd bring to being worse (A0 brief, step 5).")
    elif delta <= 0.02:
        print("result: PASS — A0 metric hit. Record both numbers in NOTES.md.")
    else:
        print("result: FAIL — over the 2% bar. Silent-bug checklist first (A0 brief).")
        sys.exit(1)


if __name__ == "__main__":
    main()
