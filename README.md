# ember

A from-scratch, hackable GPT training library. Pretraining, finetuning, evals —
and the models trained with it. Built milestone by milestone against
[the plan](../PLAN.md); each milestone has a number that says whether it shipped.

## Status

| Milestone | State |
|---|---|
| **A0 — correctness spine** | **in progress → [docs/milestones/A0.md](docs/milestones/A0.md)** |
| A1 — tokenizer + data pipeline | not started |
| A2 — GPT-2 124M reproduction | not started |
| A3 — finetuning stack | not started |
| A4 — experiments + leaderboard | not started |

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bin/ember test          # runs the suite (the A0 gate test skips until you write it)
```

## Entry points

```
bin/ember train    [args]   # training (python -m ember.train)
bin/ember test     [args]   # test suite (pytest tests/)
bin/ember data     [name]   # fetch datasets into data/ (default: shakespeare)
bin/ember baseline [args]   # nanoGPT A0 baseline (--prepare-only to stage it)
bin/ember compare  [args]   # score your run vs the baseline (2% bar)
bin/ember env               # device/precision/speed report for this machine
bin/ember eval     [args]   # arrives in A2
```

`bin/ember` automatically uses `.venv/` if present — no activate needed.

## Layout

```
ember/            the library: model.py, data.py, train.py (stubs carry the spec)
tests/            starts with the A0 overfit gate test — written FIRST
tools/            infra, not learning material: data fetch, baseline, scoring
docs/milestones/  one brief per milestone: spec, learning kit, definition of done
experiments/      leaderboard (empty until A4)
NOTES.md          lab notebook: hypotheses, dead ends, numbers, open questions
```

## The rules (short version — full version in ../PLAN.md)

1. References are consulted, never transcribed. No coding along with videos.
2. Stuck = 45-min timebox → write the one-sentence question → get the *idea*
   from the reference → close it → implement from your note.
3. The metric is the exit gate. No metric, no next milestone.
