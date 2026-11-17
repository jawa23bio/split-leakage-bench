"""A small Basset-style CNN for 200 bp one-hot DNA.

    input                                         (batch, 4, 200)
    Conv1d 4->64,  kernel 19, 'same' -> BN -> ReLU (or exp) -> MaxPool 4   (batch, 64, 50)
    Conv1d 64->64, kernel 7,  'same' -> BN -> ReLU          -> MaxPool 4   (batch, 64, 12)
    Flatten -> Linear 768->64 -> ReLU -> Dropout 0.3 -> Linear 64->1       (batch,)

About 83k parameters with the defaults.

Why it looks like this:
  - Kernel 19 in layer 1. Each first-layer filter acts as a motif scanner. The ATF/CREB
    site TGACGTCA is 8 bp, so 19 leaves room for flanks and for the filter to learn the
    motif at any offset inside its window.
  - 'same' padding keeps length 200 through each conv, so the pooling arithmetic is
    simple (200 -> 50 -> 12) and motifs at the window edges are still scanned.
  - Max pooling keeps the strongest match in each 4 bp stretch and discards where exactly
    it was, which is what you want for "is the motif here" rather than "at which base".
  - Layer 2 (kernel 7 on the pooled track, so about 28 bp of sequence) can combine nearby
    motif hits, e.g. two half sites or a motif plus a GC-rich flank.
  - Batch norm after each conv makes training less sensitive to learning rate and init,
    which matters when 5 seeds are compared. Conv biases are dropped because batch norm
    has its own shift.
  - Dropout sits only in the dense head, where most of the parameters are (768*64).
  - The output is a logit, not a probability. BCEWithLogitsLoss applies the sigmoid
    inside the loss in a numerically stable way; apply torch.sigmoid only at prediction.
  - The model is not strand-symmetric by construction. Strand is handled outside it:
    reverse-complement augmentation in training, forward/RC averaging at test.

first_activation="exp" swaps the first ReLU for an exponential (Koo & Ploenzke 2021),
which tends to give cleaner first-layer motifs at similar accuracy. Use it for the
optional motif check; keep "relu" for the main comparison.
"""

import torch
import torch.nn as nn


class ATF3CNN(nn.Module):
    def __init__(self, seq_len: int = 200, n_filters: int = 64, first_kernel: int = 19,
                 hidden: int = 64, dropout: float = 0.3, first_activation: str = "relu"):
        super().__init__()
        if first_activation not in ("relu", "exp"):
            raise ValueError("first_activation must be 'relu' or 'exp'")
        self.first_activation = first_activation

        self.conv1 = nn.Conv1d(4, n_filters, first_kernel, padding="same", bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.conv2 = nn.Conv1d(n_filters, n_filters, 7, padding="same", bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.pool = nn.MaxPool1d(4)

        pooled_len = seq_len // 4 // 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_filters * pooled_len, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        """x: (batch, 4, seq_len) one-hot. Returns one logit per sequence, shape (batch,)."""
        h = self.bn1(self.conv1(x))
        h = torch.exp(h) if self.first_activation == "exp" else torch.relu(h)
        h = self.pool(h)
        h = self.pool(torch.relu(self.bn2(self.conv2(h))))
        return self.head(h).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
