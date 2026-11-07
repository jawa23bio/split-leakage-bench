"""Sequence helpers: one-hot encoding, reverse complement, GC content."""

import numpy as np

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def gc_fraction(seq: str) -> float:
    s = seq.upper()
    return (s.count("G") + s.count("C")) / max(len(s), 1)


def repeat_fraction(seq: str) -> float:
    """Fraction of soft-masked (lowercase) bases. UCSC hg38 marks repeats in lowercase."""
    return sum(c.islower() for c in seq) / max(len(seq), 1)


def one_hot(seq: str) -> np.ndarray:
    """(L, 4) float32 array in A, C, G, T order. N becomes all zeros."""
    out = np.zeros((len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        j = _IDX.get(c)
        if j is not None:
            out[i, j] = 1.0
    return out


def one_hot_batch(seqs) -> np.ndarray:
    """(N, L, 4) array for equal-length sequences."""
    return np.stack([one_hot(s) for s in seqs])
