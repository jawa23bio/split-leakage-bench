#!/usr/bin/env python
"""End-to-end check on a synthetic genome (no downloads, about a minute on a laptop).

Builds 9 random chromosomes, plants the ATF3 motif at 'peaks', and copies 40
peak regions from the test chromosomes (chr8, chr9) onto a training chromosome
(chr1). A correct pipeline should:
  - GC-match negatives to positives
  - report those copies as leakage under the chrom split
  - keep near-copies out of test under the homology split

With --train it also trains two seeds on the chrom split (CPU, under a minute) and runs
the analysis, to check the model, the training loop and 04_analyze.py end to end. The
planted motif makes the task trivial, so this checks that the code runs and learns,
not that the benchmark's conclusions hold.
"""

import argparse
import os

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tests" / "_synthetic"
MOTIF = "TGACGTCA"


def rand_seq(rng, n, gc=0.42):
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return "".join(rng.choice(list("ACGT"), size=n, p=p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="also train two seeds and run the analysis")
    args = ap.parse_args()

    rng = np.random.default_rng(1)
    WORK.mkdir(parents=True, exist_ok=True)
    chroms = {f"chr{i}": list(rand_seq(rng, 600_000)) for i in range(1, 10)}

    peaks = []
    for c, seq in chroms.items():
        for _ in range(60):
            ctr = int(rng.integers(2_000, len(seq) - 2_000))
            for j in (-30, 0, 25):  # three motif copies around the summit
                seq[ctr + j:ctr + j + len(MOTIF)] = list(MOTIF)
            peaks.append((c, ctr - 150, ctr + 150, float(rng.uniform(5, 50)), 150))

    # leakage: copy 40 test-chromosome peak regions (+-150 bp) onto chr1
    test_peaks = [p for p in peaks if p[0] in ("chr8", "chr9")][:40]
    for k, (c, s, e, sig, summit) in enumerate(test_peaks):
        block = chroms[c][s - 50:e + 50]
        dest = 50_000 + k * 1_000
        chroms["chr1"][dest:dest + len(block)] = block
        peaks.append(("chr1", dest + 50, dest + 50 + (e - s), sig, summit))

    with open(WORK / "genome.fa", "w") as f:
        for c, seq in chroms.items():
            s = "".join(seq)
            f.write(f">{c}\n" + "\n".join(s[i:i + 80] for i in range(0, len(s), 80)) + "\n")
    with open(WORK / "peaks.narrowPeak", "w") as f:
        for c, s, e, sig, summit in peaks:
            f.write(f"{c}\t{s}\t{e}\t.\t0\t.\t{sig:.2f}\t-1\t-1\t{summit}\n")

    py = sys.executable
    subprocess.run([py, ROOT / "scripts/02_make_dataset.py", "--peaks", WORK / "peaks.narrowPeak",
                    "--genome", WORK / "genome.fa", "--out", WORK / "dataset.tsv",
                    "--chroms", ",".join(chroms)], check=True)
    subprocess.run([py, ROOT / "scripts/03_make_splits.py", "--dataset", WORK / "dataset.tsv",
                    "--out", WORK], check=True)

    ds = pd.read_csv(WORK / "dataset.tsv", sep="\t")
    leak = pd.read_csv(WORK / "leakage.tsv", sep="\t")
    sp = pd.read_csv(WORK / "splits.tsv", sep="\t")
    g = ds.groupby("label")["gc"].mean()
    assert abs(g[1] - g[0]) < 0.01, f"GC not matched: {g.to_dict()}"
    chrom_hi = (leak.query("scheme == 'chrom'")["max_sim_to_train"] >= 0.5).sum()
    hom_hi = (leak.query("scheme == 'homology'")["max_sim_to_train"] >= 0.3).sum()
    assert chrom_hi >= 30, f"expected the planted copies to show as chrom-split leakage, got {chrom_hi}"
    assert hom_hi == 0, f"homology split should have no test window >= 0.3 similar to train, got {hom_hi}"
    for s in ("random", "chrom", "homology"):
        assert set(sp[s]) == {"train", "val", "test"}, s
    print(f"\nSMOKE TEST PASSED: chrom split leaks {chrom_hi} test windows (>=0.5 similar to train); "
          f"homology split leaks {hom_hi}.")

    if args.train:
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        res = WORK / "results"
        for seed in (0, 1):
            subprocess.run([py, "-m", "slb.train", "--dataset", WORK / "dataset.tsv", "--splits", WORK / "splits.tsv",
                            "--scheme", "chrom", "--seed", str(seed), "--epochs", "8", "--out", res],
                           check=True, env=env)
        subprocess.run([py, ROOT / "scripts/04_analyze.py", "--results", res, "--leakage", WORK / "leakage.tsv"],
                       check=True)
        summ = pd.read_csv(res / "summary.tsv", sep="\t").set_index("scheme").loc["chrom"]
        pred = pd.read_csv(res / "chrom" / "seed0" / "test_predictions.tsv", sep="\t")
        n_test = int((sp["chrom"] == "test").sum())
        assert len(pred) == n_test and pred["prob"].between(0, 1).all(), "bad test_predictions.tsv"
        assert summ["seeds"] == 2 and summ["ens_auroc"] > 0.95, f"model did not learn the planted motif: {summ['ens_auroc']}"
        assert abs(summ["leaked_frac"] - chrom_hi / n_test) < 0.02, "leaked share in the analysis disagrees with leakage.tsv"
        print(f"\nTRAINING SMOKE TEST PASSED: chrom ensemble AUROC {summ['ens_auroc']:.3f} on {n_test} test windows.")


if __name__ == "__main__":
    main()
