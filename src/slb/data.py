"""PyTorch dataset over dataset.tsv + one column of splits.tsv."""

import numpy as np
import pandas as pd

from .seq import one_hot


def load_split(dataset_tsv, splits_tsv, scheme, part):
    """Return (sequences, labels, ids) for one part ('train'/'val'/'test') of one scheme."""
    df = pd.read_csv(dataset_tsv, sep="\t")
    sp = pd.read_csv(splits_tsv, sep="\t")
    df = df.merge(sp[["id", scheme]], on="id")
    df = df[df[scheme] == part]
    return df["seq"].tolist(), df["label"].to_numpy(np.float32), df["id"].to_numpy()


def make_torch_dataset(seqs, labels, rc_augment=False):
    """Torch Dataset yielding (x, y) with x shaped (4, L) for Conv1d.

    rc_augment: with probability 0.5, return the reverse complement instead.
    Use it for training only.
    """
    import torch
    from torch.utils.data import Dataset

    X = np.stack([one_hot(s).T for s in seqs])  # (N, 4, L)

    class _DS(Dataset):
        def __len__(self):
            return len(X)

        def __getitem__(self, i):
            x = X[i]
            if rc_augment and np.random.rand() < 0.5:
                x = x[::-1, ::-1].copy()  # A<->T, C<->G and reverse positions
            return torch.from_numpy(x), torch.tensor(labels[i])

    return _DS()
