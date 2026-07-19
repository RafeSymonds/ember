"""ember/model.py — your Transformer. Milestone: A0.

Ship (the contract you're building toward — see docs/milestones/A0.md):

    class GPT(nn.Module):
        - token embeddings + learned positional embeddings
        - N blocks: causal multi-head self-attention, MLP (4x expansion),
          residual connections, layer norm (pick pre-norm; know why)
        - forward(idx, targets=None) -> (logits, loss)   # logits: (B, T, vocab)

Build it as a ladder, overfit-test green at every rung:
    (a) embeddings -> LM head only (a bigram model in disguise)
    (b) one causal self-attention head
    (c) multi-head attention + output projection
    (d) the full block: pre-norm, MLP, residuals
    (e) the stack: N blocks + positional embeddings + final norm

Reference rule: Karpathy's video is for ideas when stuck (after the 45-min
timebox). nanoGPT source is for diffing AFTER yours trains, never before.
"""

# Your code here.

from torch import nn


class GPT(nn.Module):
    pass
