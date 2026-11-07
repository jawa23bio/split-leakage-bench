"""YOUR PART: a small Basset-style CNN for 200 bp one-hot DNA.

Write this yourself; it is the part interviewers will ask about.

Suggested architecture (start small, it trains in minutes on a Colab GPU):
    input            (batch, 4, 200)
    Conv1d 4->64,  kernel 19, padding 'same'  -> ReLU -> MaxPool 4     # first layer = motif scanners
    Conv1d 64->64, kernel 7,  padding 'same'  -> ReLU -> MaxPool 4
    Flatten -> Linear -> ReLU -> Dropout 0.3 -> Linear -> 1 logit

Things to decide and be able to explain:
  - why kernel 19 in the first layer (ATF3's motif, TGACGTCA, is 8 bp; leave room)
  - why the output is a logit and the loss is BCEWithLogitsLoss
  - batch norm or not; dropout where; how many parameters you end up with
  - optional: exponential activation in layer 1 (Koo & Ploenzke 2021) to get cleaner motifs

Quick check once written:
    m = ATF3CNN(); x = torch.zeros(8, 4, 200); assert m(x).shape == (8,)
"""

import torch.nn as nn


class ATF3CNN(nn.Module):
    def __init__(self, seq_len: int = 200):
        super().__init__()
        raise NotImplementedError("Define the layers here.")

    def forward(self, x):
        raise NotImplementedError("Return one logit per sequence, shape (batch,).")
