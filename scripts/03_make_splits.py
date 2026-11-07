#!/usr/bin/env python
"""Make three train/val/test splits of the same dataset and measure leakage in each.

Schemes (all aim for about the same test fraction so results are comparable):
  random    shuffle windows, stratified by label
  chrom     hold out whole chromosomes (default val chr7, test chr8 + chr9)
  homology  group windows that share >= --threshold of their canonical k-mers,
            then keep each group inside one split, so no test window has a
            close relative in train

Leakage = for each test window, its highest k-mer similarity to any train window.
The leakage summary is a result in itself: it shows how much a chromosome
holdout still lets near-copies of test windows into training.

Usage:
    python scripts/03_make_splits.py --dataset data/dataset.tsv --out data
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slb.similarity import KmerIndex, similarity_components  # noqa: E402

LEVELS = (0.3, 0.5, 0.8)


def random_split(df, frac_val, frac_test, rng):
    out = np.array(["train"] * len(df), dtype=object)
    for _, idx in df.groupby("label").groups.items():
        idx = rng.permutation(np.asarray(idx))
        n_test, n_val = int(len(idx) * frac_test), int(len(idx) * frac_val)
        out[idx[:n_test]] = "test"
        out[idx[n_test:n_test + n_val]] = "val"
    return out


def chrom_split(df, val_chroms, test_chroms):
    out = np.array(["train"] * len(df), dtype=object)
    out[df["chrom"].isin(val_chroms).to_numpy()] = "val"
    out[df["chrom"].isin(test_chroms).to_numpy()] = "test"
    return out


def homology_split(components, frac_val, frac_test, rng):
    n = len(components)
    groups = defaultdict(list)
    for i, c in enumerate(components):
        groups[c].append(i)
    order = rng.permutation(list(groups.keys()))
    out = np.array(["train"] * n, dtype=object)
    target = {"test": frac_test * n, "val": frac_val * n}
    filled = {"test": 0, "val": 0}
    for g in order:
        members = groups[g]
        for part in ("test", "val"):
            if len(members) <= 0.5 * target[part] and filled[part] + len(members) <= target[part]:
                out[members] = part
                filled[part] += len(members)
                break
    sizes = sorted((len(m) for m in groups.values()), reverse=True)
    return out, sizes


def leakage(index, split, label):
    train = set(np.nonzero(split == "train")[0].tolist())
    rows = []
    for i in np.nonzero(split == "test")[0]:
        rows.append((int(i), int(label[i]), index.max_similarity(int(i), train)))
    return pd.DataFrame(rows, columns=["id", "label", "max_sim_to_train"])


def summarize(name, split, label, leak):
    lines = [f"### {name}"]
    for part in ("train", "val", "test"):
        m = split == part
        lines.append(f"- {part}: {m.sum()} windows, {label[m].mean():.1%} positive")
    s = leak["max_sim_to_train"]
    lines.append(f"- test windows' highest similarity to train: median {s.median():.2f}, "
                 f"90th pct {s.quantile(0.9):.2f}, max {s.max():.2f}")
    for t in LEVELS:
        pos = leak.loc[leak.label == 1, "max_sim_to_train"]
        neg = leak.loc[leak.label == 0, "max_sim_to_train"]
        lines.append(f"- share of test windows with similarity >= {t}: {(s >= t).mean():.1%} "
                     f"(positives {(pos >= t).mean():.1%}, negatives {(neg >= t).mean():.1%})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--val-chroms", default="chr7")
    ap.add_argument("--test-chroms", default="chr8,chr9")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--max-df", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.dataset, sep="\t")
    label = df["label"].to_numpy()

    print(f"indexing {len(df)} windows with canonical {args.k}-mers ...")
    index = KmerIndex(df["seq"].tolist(), k=args.k, max_df=args.max_df)
    print(f"  skipped {index.skipped} k-mers seen in more than {args.max_df} windows (mostly repeats)")

    comps = similarity_components(index, args.threshold)
    hom, sizes = homology_split(comps, args.val_frac, args.test_frac, rng)
    splits = {
        "random": random_split(df, args.val_frac, args.test_frac, rng),
        "chrom": chrom_split(df, args.val_chroms.split(","), args.test_chroms.split(",")),
        "homology": hom,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df["id"], **splits}).to_csv(out / "splits.tsv", sep="\t", index=False)

    report = [
        "# Leakage summary",
        "",
        f"{len(df)} windows. Similarity = share of a window's canonical {args.k}-mers found in another window "
        f"(both strands). Homology groups link windows with similarity >= {args.threshold}; "
        f"largest groups: {sizes[:5]}; {sum(s > 1 for s in sizes)} groups have more than one window.",
        "",
    ]
    leaks = []
    for name, split in splits.items():
        leak = leakage(index, split, label)
        leak.insert(0, "scheme", name)
        leaks.append(leak)
        report += [summarize(name, split, label, leak), ""]
    pd.concat(leaks).to_csv(out / "leakage.tsv", sep="\t", index=False)
    (out / "leakage_summary.md").write_text("\n".join(report))
    print("\n".join(report))
    print(f"wrote {out/'splits.tsv'}, {out/'leakage.tsv'}, {out/'leakage_summary.md'}")


if __name__ == "__main__":
    main()
