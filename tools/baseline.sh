#!/usr/bin/env bash
# A0 baseline: nanoGPT shakespeare_char — the known-good number to match.
#
# Usage:
#   bin/ember baseline --prepare-only     # clone + pin + prepare data, no training
#   bin/ember baseline                    # full run (CUDA box, e.g. the 2080S)
#   bin/ember baseline --device=mps --compile=False   # extra args pass through
#
# FAIRNESS RULE: run this on the SAME machine, same iters, as your own metric
# run. The pinned commit is recorded in baselines/nanoGPT.commit.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p baselines
if [ ! -d baselines/nanoGPT ]; then
  git clone --depth 1 https://github.com/karpathy/nanoGPT baselines/nanoGPT
  git -C baselines/nanoGPT rev-parse HEAD > baselines/nanoGPT.commit
fi
echo "nanoGPT commit: $(cat baselines/nanoGPT.commit)"

python3 -c "import requests" 2>/dev/null || pip install -q requests

cd baselines/nanoGPT
if [ ! -f data/shakespeare_char/train.bin ]; then
  python3 data/shakespeare_char/prepare.py
fi

if [ "${1:-}" = "--prepare-only" ]; then
  echo "prepared. run 'bin/ember baseline' on the metric machine to train."
  exit 0
fi

log=../nanogpt_shakespeare_char.log
echo "training (log: baselines/nanogpt_shakespeare_char.log)"
python3 train.py config/train_shakespeare_char.py "$@" 2>&1 | tee "$log"

echo
echo "best val loss: $(grep -o 'val loss [0-9.]*' "$log" | sort -t' ' -k3 -n | head -1)"
echo "score it: bin/ember compare --ours <num-or-log> --baseline baselines/nanogpt_shakespeare_char.log"
