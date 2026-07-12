"""A0 gate test — WRITE THIS FIRST, before the model is finished.

Contract (from PLAN.md and docs/milestones/A0.md):

    A tiny GPT (2 layers, 2 heads, d_model 64, block size 64) trained on ONE
    fixed batch for a few hundred Adam steps must drive training loss to ~0.
    Assert loss <= 0.05 to start; tighten to 0.01 once it passes.

    If it can't memorize one batch, the model is wrong. No exceptions, and
    no "it probably just needs more steps."

Why this works: memorizing a single batch exercises the entire gradient path.
Shape, masking, and broadcasting bugs that still "train" on real data — just
worse — usually fail this test loudly.

Cheap extra asserts worth adding while you're here:
    - initial loss ~= ln(vocab_size)   (uniform predictions before training)
    - loss decreases over the first ~50 steps

Delete the skip and implement. This is the first thing CI runs, forever.
"""

import pytest


def test_overfit_single_batch():
    pytest.skip("A0 gate test: write me first — see module docstring")
