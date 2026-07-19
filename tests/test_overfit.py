"""A0 gate tests. Written by Claude under learning-mode override (2026-07-18).

These pin the model API — change it if you want, but update here consciously:

    GPT(vocab_size, block_size, n_layer, n_head, n_embd)
    forward(idx) -> (logits, None)          logits: (B, T, vocab_size)
    forward(idx, targets) -> (logits, loss) loss: scalar

Three guards, in the order they'll catch bugs:
  1. forward contract — shapes and the optional-targets convention
  2. causality — black-box: perturbing a future token must not change past
     logits. Catches mask off-by-ones without needing attention internals.
  3. overfit one batch — the gate. Memorizing 4x64 positions exercises the
     entire gradient path; a model that can't is wrong, full stop.

These are RED until the model ladder is climbed. That's the design — they are
the rung-checker, not a formality. Never skip or loosen them to green CI.
"""

import math

import torch

from ember.model import GPT

VOCAB_SIZE = 65
BLOCK_SIZE = 64


def make_model() -> GPT:
    torch.manual_seed(1337)
    return GPT(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=2,
        n_head=2,
        n_embd=64,
    )


def make_batch(batch_size: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(42)
    x = torch.randint(VOCAB_SIZE, (batch_size, BLOCK_SIZE), generator=g)
    y = torch.randint(VOCAB_SIZE, (batch_size, BLOCK_SIZE), generator=g)
    return x, y


def test_forward_contract():
    model = make_model()
    x, y = make_batch()

    logits, loss = model(x, y)
    assert logits.shape == (4, BLOCK_SIZE, VOCAB_SIZE)
    assert loss.ndim == 0 and loss.item() > 0

    logits, loss = model(x)
    assert logits.shape == (4, BLOCK_SIZE, VOCAB_SIZE)
    assert loss is None


def test_no_future_leakage():
    model = make_model()
    model.eval()  # dropout off — this test needs determinism
    x, _ = make_batch(batch_size=1)

    x_perturbed = x.clone()
    x_perturbed[0, -1] = (x_perturbed[0, -1] + 1) % VOCAB_SIZE

    with torch.no_grad():
        logits, _ = model(x)
        logits_perturbed, _ = model(x_perturbed)

    # changing the LAST token must not move any EARLIER position's logits...
    assert torch.allclose(logits[0, :-1], logits_perturbed[0, :-1], atol=1e-5), (
        "future token influenced past logits — causal mask is wrong"
    )
    # ...but must move its own position's logits (model isn't ignoring input)
    assert not torch.allclose(logits[0, -1], logits_perturbed[0, -1])


def test_overfit_single_batch():
    model = make_model()
    x, y = make_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(1000):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if loss.item() < 0.01:
            break

    # untrained model ~ uniform predictions: loss near ln(vocab) (~4.17).
    # far off means the loss/logits wiring is broken before training begins.
    assert abs(losses[0] - math.log(VOCAB_SIZE)) < 1.0, (
        f"initial loss {losses[0]:.3f}, expected ~{math.log(VOCAB_SIZE):.3f}"
    )
    assert losses[50] < losses[0], "loss not decreasing — gradient path broken"
    assert losses[-1] <= 0.05, (
        f"final loss {losses[-1]:.4f} after {len(losses)} steps — "
        "cannot memorize one batch, the model is wrong (A0 brief, step 2)"
    )
