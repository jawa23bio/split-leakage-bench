"""Train one model on one split scheme and save its test predictions.

    PYTHONPATH=src python -m slb.train --scheme chrom --seed 0
    for s in random chrom homology; do for seed in 0 1 2 3 4; do
        PYTHONPATH=src python -m slb.train --scheme $s --seed $seed; done; done

Each run writes results/<scheme>/seed<seed>/:
    test_predictions.tsv   id, label, prob   (what scripts/04_analyze.py reads)
    history.tsv            epoch, train_loss, val_loss, val_auroc
    metrics.json           best epoch, val AUROC, test metrics, settings
    model.pt               weights of the best epoch (for the motif check)

The split is fixed by splits.tsv; the seed only changes weight initialisation, batch
order and which examples get reverse-complemented. Five seeds on the same split are
therefore a deep ensemble, and their disagreement is the uncertainty estimate.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import load_split, make_torch_dataset
from .metrics import calibration, discrimination
from .model import ATF3CNN, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)  # data.py draws the reverse-complement coin from numpy
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    """Probabilities averaged over the forward strand and its reverse complement."""
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device)
        p_fwd = torch.sigmoid(model(x))
        p_rc = torch.sigmoid(model(x.flip(1, 2)))  # flip channels (A<->T, C<->G) and positions
        out.append(((p_fwd + p_rc) / 2).cpu().numpy())
    return np.concatenate(out)


def log_loss(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


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
    ap.add_argument("--first-activation", choices=["relu", "exp"], default="relu")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parts = {p: load_split(args.dataset, args.splits, args.scheme, p) for p in ("train", "val", "test")}
    train_ds = make_torch_dataset(*parts["train"][:2], rc_augment=True)
    val_ds = make_torch_dataset(*parts["val"][:2])
    test_ds = make_torch_dataset(*parts["test"][:2])
    out_dir = Path(args.out) / args.scheme / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    shuffle_rng = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=shuffle_rng)
    val_dl = DataLoader(val_ds, batch_size=512)
    test_dl = DataLoader(test_ds, batch_size=512)

    seq_len = len(parts["train"][0][0])
    model = ATF3CNN(seq_len=seq_len, first_activation=args.first_activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    print(f"{args.scheme} seed {args.seed} on {device}: "
          f"{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test, "
          f"{count_parameters(model)} parameters")

    # Keep the epoch with the best val AUROC. Ties go to the lower val loss, so that when
    # AUROC saturates (it does on the synthetic test) the choice is not simply the earliest,
    # still underconfident epoch.
    y_val = parts["val"][1]
    best, best_epoch, best_state, history = (-1.0, float("inf")), 0, None, []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer, loss_fn, device)
        p_val = predict(model, val_dl, device)
        val_auroc, val_loss = discrimination(y_val, p_val)["auroc"], log_loss(y_val, p_val)
        history.append((epoch, train_loss, val_loss, val_auroc))
        improved = (val_auroc, -val_loss) > (best[0], -best[1])
        if improved:
            best, best_epoch = (val_auroc, val_loss), epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"  epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"val AUROC {val_auroc:.4f}{' *' if improved else ''}")
        if epoch - best_epoch >= args.patience:
            print(f"  no val gain for {args.patience} epochs, stopping")
            break
    best_auroc = best[0]

    model.load_state_dict(best_state)
    _, y_test, ids = parts["test"]
    prob = predict(model, test_dl, device)
    metrics = {**discrimination(y_test, prob), **calibration(y_test, prob)}
    metrics = {k: float(v) for k, v in metrics.items()}
    print(f"test (best epoch {best_epoch}, val AUROC {best_auroc:.4f}): "
          + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

    pd.DataFrame({"id": ids, "label": y_test.astype(int), "prob": prob}).to_csv(
        out_dir / "test_predictions.tsv", sep="\t", index=False, float_format="%.6f")
    pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "val_auroc"]).to_csv(
        out_dir / "history.tsv", sep="\t", index=False, float_format="%.6f")
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "metrics.json").write_text(json.dumps({
        "scheme": args.scheme, "seed": args.seed, "best_epoch": best_epoch,
        "val_auroc": float(best_auroc), "val_loss": float(best[1]), "test": metrics,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "parameters": count_parameters(model), "settings": vars(args),
    }, indent=2))
    print(f"wrote {out_dir}/")


if __name__ == "__main__":
    main()
