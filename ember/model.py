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

import torch
from torch import nn

from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size) -> None:
        super().__init__()

        self.head_size = head_size

        self.query = nn.Linear(n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x: (B, T, C)   B sequences, T positions, C = n_embd channels each
        B, T, C = x.shape
        Q = self.query(x)  # (B, T, head_size)  each position: "what am I looking for?"
        K = self.key(x)    # (B, T, head_size)  each position: "what do I contain?"

        # scores: every query position vs every key position.
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # wei[b, i, j] = how much position i wants to hear from position j.
        # scale by 1/sqrt(head_size): the dot product sums head_size terms, so its
        # variance grows with head_size — unscaled, softmax saturates and gradients die.
        wei = Q @ K.transpose(-2, -1) * self.head_size**-0.5

        # causality: kill every j > i (the upper triangle) BEFORE softmax.
        # -inf -> softmax weight of exactly 0: the future gets no vote.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -float("inf"))
        # normalize each row i over its allowed keys j <= i: rows now sum to 1
        wei = F.softmax(wei, dim=-1)

        V = self.value(x)  # (B, T, head_size)  what each position actually broadcasts
        # each position's output = weighted average of the values it attends to:
        # (B, T, T) @ (B, T, hs) -> (B, T, head_size)
        return wei @ V


class GPT(nn.Module):
    """Decoder-only transformer, built rung by rung (see module docstring).

    Constructor args are ARCHITECTURE; batch size B and sequence length T are
    runtime choices — the model accepts any T <= block_size.

    vocab_size: how many distinct symbols exist. A fact about the data, not a
        tuning knob: sets the token table's rows and the logits' width.
        (65 for char-level shakespeare; ~50k once BPE arrives in A1.)
    block_size: the context window — the farthest back attention can EVER see.
        Sets the position table's rows and the causal mask's size. Attention
        builds a T x T score matrix, so this is the quadratically expensive knob.
    n_layer:    depth — how many blocks stack at rung (e). Each block re-mixes
        and refines what the previous one produced. (Unused before rung (e).)
    n_head:     how many parallel attention conversations per layer, from rung
        (c). Each gets head_size = n_embd // n_head dims; outputs concatenate
        back to exactly n_embd. Hard constraint: n_embd % n_head == 0.
    n_embd:     the working width (the C in every shape comment) — every token
        becomes a vector this wide at the embedding border and stays that wide
        through the trunk. The primary capacity knob.

    head_size is deliberately NOT an arg — it's always derived (n_embd // n_head;
    full n_embd for rung (b)'s single head). It's the width the Q.K dot product
    sums over, hence Head's 1/sqrt(head_size) scale.

    Reference points: GPT-2 124M is block_size 1024, n_layer 12, n_head 12,
    n_embd 768 — same skeleton, bigger numbers.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.token_embedding_layer = nn.Embedding(vocab_size, n_embd)
        # one row per position slot, so the table is block_size deep
        self.position_embedding_layer = nn.Embedding(block_size, n_embd)
        # rung (b): one full-width head; rung (c) splits into n_head of n_embd // n_head
        self.sa_head = Head(n_embd, head_size=n_embd, block_size=block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) integer token ids — 2-D outside the model, no channel axis
        targets: (B, T) integer next-token ids, or None at inference

        Returns:
            (logits, loss)  logits: (B, T, vocab_size); loss: scalar or None
        """

        B, T = idx.shape
        # the border crossing: ids become vectors. (B, T) -> (B, T, C)
        token_embedding = self.token_embedding_layer(idx)
        # (T, C): one learned vector per position slot 0..T-1
        pos_embedding = self.position_embedding_layer(
            torch.arange(T, device=idx.device)
        )

        # (B, T, C) + (T, C): broadcasting copies the position rows across B.
        # after this, a token's vector encodes WHAT it is and WHERE it sits.
        x = token_embedding + pos_embedding
        x = self.sa_head(x)  # (B, T, C) -> (B, T, C), contents now context-mixed

        # the only exit from model-space: (B, T, C) -> (B, T, vocab_size)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # NOTE: C here rebinds to vocab_size (logits' last dim), not n_embd
            B, T, C = logits.shape
            # cross_entropy wants (N, classes) vs (N,): fold batch and time together
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

            logits = logits.view(B, T, C)  # restore the contract shape

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T) — grows by one column per iteration
        for _ in range(max_new_tokens):
            # crop to the last block_size positions: the position table has no
            # row T for T >= block_size, and the mask buffer is block_size wide
            logits, _ = self(idx[:, -self.block_size :])

            logits = logits[:, -1, :]  # (B, vocab): only the final position predicts

            probs = F.softmax(logits, dim=-1)  # scores -> sampling distribution

            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) sampled ids

            idx = torch.cat((idx, idx_next), dim=1)  # append: (B, T) -> (B, T+1)

        return idx


if __name__ == "__main__":
    # Dev smoke harness — run as you build:  python -m ember.model
    # Deliberately the dumb loop (no schedule/clip/checkpoints); the real
    # training loop is ember/train.py and it's yours to write.
    #
    # Corpus is imported HERE, not at module top: the model class stays a pure
    # tokens->logits function; wiring to data happens only at the edges.
    from ember.data import Corpus

    corpus = Corpus()
    model = GPT(
        vocab_size=corpus.vocab_size,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
    )
    print(f"{sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(2001):
        x, y = corpus.get_batch("train", batch_size=32, block_size=64)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            xv, yv = corpus.get_batch("val", batch_size=32, block_size=64)
            with torch.no_grad():
                _, val_loss = model(xv, yv)
            print(
                f"step {step:4d}   train {loss.item():.3f}   val {val_loss.item():.3f}"
            )

    print("--- sample ---")
    prompt = torch.zeros((1, 1), dtype=torch.long)  # id 0 = '\n' in sorted vocab
    print(corpus.decode(model.generate(prompt, max_new_tokens=300)[0].tolist()))
