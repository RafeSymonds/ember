"""Data-layer guards — these catch the silent bugs the A0 brief warns about."""

import pytest
import torch

from ember.data import DATA_PATH, Corpus


@pytest.fixture(scope="module")
def corpus():
    if not DATA_PATH.exists():
        pytest.skip("corpus missing — run: bin/ember data")
    return Corpus()


def test_round_trip_is_exact(corpus):
    text = DATA_PATH.read_text()
    assert corpus.decode(corpus.encode(text)) == text


def test_vocab_is_canonical(corpus):
    # a re-derived vocab must map identically, or checkpoints decode as garbage
    assert corpus.vocab == sorted(set(DATA_PATH.read_text()))


def test_batch_shapes_and_shift(corpus):
    x, y = corpus.get_batch("train", batch_size=4, block_size=8)
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == y.dtype == torch.long
    # y is x shifted by one position within the same window
    assert torch.equal(x[:, 1:], y[:, :-1])
