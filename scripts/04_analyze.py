#!/usr/bin/env python
"""Combine the per-seed test predictions and answer questions 2 to 4.

For each split scheme found under --results:
  Q2  test AUROC/AUPRC of single models (mean and SD over seeds) and of the ensemble,
      plus the ensemble's Brier score and calibration error
  Q3  the same test set divided into leaked windows (similarity to train >= --threshold)
      and clean ones, to see whether performance is carried by the leaked windows
  Q4  whether ensemble spread (SD of the seeds' probabilities) is higher where the
      ensemble mean is wrong

Reading Q4: Spearman(spread, |error|) is reported because it is the headline number,
but it flatters any model. For a correct call, |error| is just the distance of the
prediction from 0 or 1, and predictions far from 0 and 1 also have the most room to
disagree, so the two correlate by construction. The stricter test is whether spread
picks out the wrong calls, and whether it does so better than an uncertainty that
needs no ensemble at all: the mean prediction's closeness to 0.5, 1 - |2p - 1|.
Both are reported as AUROCs for detecting wrong calls (0.5 = chance).

Reading Q3: leaked and clean windows can differ in other ways (repeat content, peak
strength), so a gap between them is suggestive, not causal. The three schemes also
have different test sets; `clean` rows are the closest like-for-like comparison.

Usage:
    python scripts/04_analyze.py --results results --leakage data/leakage.tsv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slb.metrics import calibration, discrimination, ensemble_uncertainty  # noqa: E402

SCHEMES = ("random", "chrom", "homology")
MIN_PER_CLASS = 10  # do not report AUROC on a subset with fewer positives or negatives than this


def load_scheme(results: Path, scheme: str):
    """Return (ids, labels, member_probs (n_seeds, n_test), seed names), or None if no runs."""
    runs = sorted((results / scheme).glob("seed*/test_predictions.tsv"),
                  key=lambda p: int(p.parent.name[4:]))
    if not runs:
        return None
    frames = [pd.read_csv(p, sep="\t").sort_values("id").reset_index(drop=True) for p in runs]
    for p, f in zip(runs, frames):
        if not (f["id"].equals(frames[0]["id"]) and f["label"].equals(frames[0]["label"])):
            sys.exit(f"{p} does not cover the same test windows as {runs[0]}; "
                     "were the splits regenerated between runs?")
    probs = np.stack([f["prob"].to_numpy() for f in frames])
    return frames[0]["id"].to_numpy(), frames[0]["label"].to_numpy(), probs, [p.parent.name for p in runs]


def fmt(x, digits=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def subset_row(name, y, p, mask):
    y, p = y[mask], p[mask]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    enough = min(n_pos, n_neg) >= MIN_PER_CLASS
    return {
        "subset": name, "n": int(mask.sum()), "positives": n_pos, "negatives": n_neg,
        "auroc": roc_auc_score(y, p) if enough else float("nan"),
        # mean |label - prob| per class: comparable across subsets whatever their class balance
        "err_pos": float(np.mean(1 - p[y == 1])) if n_pos else float("nan"),
        "err_neg": float(np.mean(p[y == 0])) if n_neg else float("nan"),
    }


def analyze_scheme(scheme, ids, y, probs, seeds, leak, threshold):
    single = pd.DataFrame([discrimination(y, p) for p in probs])
    ens = ensemble_uncertainty(y, probs)
    mean, spread = ens["mean"], ens["spread"]
    row = {
        "scheme": scheme, "seeds": len(seeds), "n_test": len(y), "positive_frac": float(y.mean()),
        "single_auroc_mean": single["auroc"].mean(), "single_auroc_sd": single["auroc"].std(ddof=1) if len(seeds) > 1 else float("nan"),
        "single_auprc_mean": single["auprc"].mean(), "single_auprc_sd": single["auprc"].std(ddof=1) if len(seeds) > 1 else float("nan"),
        **{f"ens_{k}": v for k, v in discrimination(y, mean).items()},
        **{f"ens_{k}": v for k, v in calibration(y, mean).items()},
    }

    # Q3: leaked vs clean
    subsets = None
    if leak is not None:
        sim = leak.set_index("id")["max_sim_to_train"].reindex(ids)
        if sim.isna().any():
            sys.exit(f"{scheme}: {int(sim.isna().sum())} test windows are missing from the leakage table; "
                     "it must come from the same 03_make_splits.py run as the splits used for training.")
        leaked = sim.to_numpy() >= threshold
        row["leaked_frac"] = float(leaked.mean())
        subsets = pd.DataFrame([
            subset_row("all", y, mean, np.ones(len(y), bool)),
            subset_row(f"clean (< {threshold})", y, mean, ~leaked),
            subset_row(f"leaked (>= {threshold})", y, mean, leaked),
        ])
        row["ens_auroc_clean"] = subsets.loc[1, "auroc"]

    # Q4: does spread track error?
    wrong = (mean >= 0.5) != (y == 1)
    several = len(seeds) > 1
    row["spread_vs_error_spearman"] = ens["spread_vs_error_spearman"] if several else float("nan")
    row["error_rate"] = float(wrong.mean())
    row["spread_when_right"] = float(spread[~wrong].mean()) if several and (~wrong).any() else float("nan")
    row["spread_when_wrong"] = float(spread[wrong].mean()) if several and wrong.any() else float("nan")
    # can an uncertainty score pick out the wrong calls? 0.5 = no better than chance
    enough = min(wrong.sum(), (~wrong).sum()) >= MIN_PER_CLASS
    row["wrong_call_auroc_spread"] = roc_auc_score(wrong, spread) if several and enough else float("nan")
    row["wrong_call_auroc_confidence"] = roc_auc_score(wrong, 1 - np.abs(2 * mean - 1)) if enough else float("nan")
    return row, subsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--leakage", default="data/leakage.tsv", help="leakage.tsv from 03_make_splits.py")
    ap.add_argument("--threshold", type=float, default=0.3, help="similarity at or above which a test window counts as leaked")
    ap.add_argument("--out", help="where to write summary.md / summary.tsv (default: --results)")
    args = ap.parse_args()
    results, out = Path(args.results), Path(args.out or args.results)

    leak_all = None
    if Path(args.leakage).exists():
        leak_all = pd.read_csv(args.leakage, sep="\t")
    else:
        print(f"note: {args.leakage} not found, skipping the leaked/clean comparison")

    rows, sections = [], {}
    for scheme in SCHEMES:
        loaded = load_scheme(results, scheme)
        if loaded is None:
            continue
        leak = None if leak_all is None else leak_all[leak_all["scheme"] == scheme]
        row, subsets = analyze_scheme(scheme, *loaded, leak, args.threshold)
        rows.append(row)
        sections[scheme] = subsets
    if not rows:
        sys.exit(f"no results/<scheme>/seed*/test_predictions.tsv found under {results}")
    summary = pd.DataFrame(rows)

    md = ["# Results summary", ""]
    md += ["## Q2. Test performance by split", "",
           "| split | seeds | test windows | single-model AUROC | single-model AUPRC | ensemble AUROC | ensemble AUPRC | Brier | ECE |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['scheme']} | {r['seeds']} | {r['n_test']} ({r['positive_frac']:.0%} positive) "
                  f"| {fmt(r['single_auroc_mean'])} ± {fmt(r['single_auroc_sd'])} "
                  f"| {fmt(r['single_auprc_mean'])} ± {fmt(r['single_auprc_sd'])} "
                  f"| {fmt(r['ens_auroc'])} | {fmt(r['ens_auprc'])} | {fmt(r['ens_brier'])} | {fmt(r['ens_ece'])} |")
    md += ["", "Single-model columns are mean ± SD over seeds. AUPRC depends on the share of positives, "
               "so compare it across splits only where that share is similar.", ""]

    if leak_all is not None:
        md += [f"## Q3. Leaked vs clean test windows (ensemble, similarity threshold {args.threshold})", "",
               "| split | subset | windows | positives | negatives | AUROC | mean error on positives | mean error on negatives |",
               "|---|---|---|---|---|---|---|---|"]
        for scheme, sub in sections.items():
            for s in sub.to_dict("records"):
                md.append(f"| {scheme} | {s['subset']} | {s['n']} | {s['positives']} | {s['negatives']} "
                          f"| {fmt(s['auroc'])} | {fmt(s['err_pos'])} | {fmt(s['err_neg'])} |")
        md += ["", f"Mean error is |label - probability|. AUROC is n/a when a subset has fewer than {MIN_PER_CLASS} "
                   "positives or negatives.", ""]

    md += ["## Q4. Does ensemble spread track error?", "",
           "| split | wrong calls | Spearman(spread, error) | mean spread, right calls | mean spread, wrong calls "
           "| wrong-call AUROC, spread | wrong-call AUROC, mean prediction only |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['scheme']} | {r['error_rate']:.1%} | {fmt(r['spread_vs_error_spearman'])} "
                  f"| {fmt(r['spread_when_right'], 4)} | {fmt(r['spread_when_wrong'], 4)} "
                  f"| {fmt(r['wrong_call_auroc_spread'])} | {fmt(r['wrong_call_auroc_confidence'])} |")
    md += ["", "Spread is the SD of the seeds' probabilities; a call is wrong when the ensemble mean falls on the "
               "wrong side of 0.5. The last two columns ask how well each uncertainty score ranks wrong calls above "
               "right ones (0.5 = chance). The final column uses only the mean prediction's closeness to 0.5, so it "
               f"is the bar the ensemble has to clear. n/a means fewer than {MIN_PER_CLASS} wrong (or right) calls.", ""]

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(md))
    summary.to_csv(out / "summary.tsv", sep="\t", index=False, float_format="%.5f")
    print("\n".join(md))
    print(f"wrote {out/'summary.md'}, {out/'summary.tsv'}")


if __name__ == "__main__":
    main()
