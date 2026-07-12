"""ember/train.py — the training loop. Milestone: A0.

Ship (the contract — see docs/milestones/A0.md):

    - AdamW, LR schedule (linear warmup -> cosine decay), gradient clipping
    - periodic val-loss estimation (model.eval() + torch.no_grad(), then back
      to train mode — forgetting either is a classic silent bug)
    - checkpointing: save {model, optimizer, iter, config}; resume must work
      (test it by killing a run on purpose)
    - mixed precision: fp16 + GradScaler on the 2080S (Turing has no bf16);
      bf16 autocast later on the 4070/H100; plain fp32 on MPS/CPU
    - device configurable: "cpu" | "cuda" | "mps"

Run via `bin/ember train` once implemented.
"""

# Your code here.


if __name__ == "__main__":
    raise SystemExit(
        "ember train: not implemented yet — that's milestone A0.\n"
        "Start at docs/milestones/A0.md."
    )
