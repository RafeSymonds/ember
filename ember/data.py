"""Char-level corpus loading and batching. Milestone: A0.

The whole corpus in memory, a char vocab, a contiguous train/val split, and
random windows as (x, y) tensor pairs. A real BPE tokenizer and a streaming
loader arrive in A1 — this stays deliberately simple.
"""

from pathlib import Path
from typing import Literal

import torch

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "shakespeare" / "input.txt"
TRAIN_FRACTION = 0.9


class Corpus:
    def __init__(self, path: Path = DATA_PATH, train_fraction: float = TRAIN_FRACTION):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run: bin/ember data")
        text = path.read_text()

        # sorted() makes the vocab order canonical. set order changes per
        # process, and a checkpoint's embedding rows are married to these ids.
        self.vocab = sorted(set(text))
        self.vocab_size = len(self.vocab)
        self._char_to_id = {c: i for i, c in enumerate(self.vocab)}

        data = torch.tensor(self.encode(text), dtype=torch.long)
        split = int(train_fraction * len(data))
        # contiguous split — shuffling tokens before splitting leaks val into train
        self._splits = {"train": data[:split], "val": data[split:]}

    def encode(self, s: str) -> list[int]:
        return [self._char_to_id[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)

    def get_batch(
        self,
        split: Literal["train", "val"],
        batch_size: int,
        block_size: int,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._splits[split]
        starts = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[s : s + block_size] for s in starts])
        y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
        return x.to(device), y.to(device)
