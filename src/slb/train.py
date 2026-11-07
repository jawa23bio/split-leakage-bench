"""YOUR PART: train one model on one split scheme and save test predictions.

Data loading and metrics are done; the training loop is yours to write.

Run (after writing it):
    python -m slb.train --scheme chrom --seed 0
    for s in random chrom homology; do for seed in 0 1 2 3 4; do
        python -m slb.train --scheme $s --seed $seed; done; done

Each run should write results/<scheme>/seed<seed>/test_predictions.tsv with
columns id, label, prob, so the ensemble analysis can combine the 5 seeds.
"""

import argparse
from pathlib import Path

from .data import load_split, make_torch_dataset
from .metrics import calibration, discrimination  # noqa: F401  (use these at the end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset.tsv")
    ap.add_argument("--splits", default="data/splits.tsv")
    ap.add_argument("--scheme", choices=["random", "chrom", "homology"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    parts = {p: load_split(args.dataset, args.splits, args.scheme, p) for p in ("train", "val", "test")}
    train_ds = make_torch_dataset(*parts["train"][:2], rc_augment=True)
    val_ds = make_torch_dataset(*parts["val"][:2])
    test_ds = make_torch_dataset(*parts["test"][:2])
    out_dir = Path(args.out) / args.scheme / f"seed{args.seed}"

    # TODO (yours):
    # 1. set seeds (torch, numpy, random) from args.seed
    # 2. DataLoaders; pick device (cuda if available)
    # 3. model = ATF3CNN(); optimizer Adam(lr); loss BCEWithLogitsLoss
    # 4. loop over epochs: train, then val AUROC; keep the best state; stop after `patience` epochs without gain
    # 5. reload best state; predict test probabilities (sigmoid); average with the reverse-complement prediction
    # 6. print discrimination(...) and calibration(...); write out_dir/test_predictions.tsv (id, label, prob)
    raise NotImplementedError("Write the training loop.")


if __name__ == "__main__":
    main()
