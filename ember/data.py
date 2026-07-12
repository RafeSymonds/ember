"""ember/data.py — corpus loading and batching. Milestone: A0.

Ship (the contract — see docs/milestones/A0.md):

    - load the corpus (tiny shakespeare for A0), build the char-level vocab
    - encode(str) -> list[int] / decode(list[int]) -> str
    - a contiguous 90/10 train/val split (contiguous, not shuffled-by-token —
      know why shuffling tokens before splitting is leakage)
    - get_batch(split, batch_size, block_size, device) -> (x, y)
      where y is x shifted by one position (the next-token targets)

A real BPE tokenizer and a streaming loader arrive in A1. Keep this simple.
"""

# Your code here.
