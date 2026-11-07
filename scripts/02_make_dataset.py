#!/usr/bin/env python
"""Build a binary classification dataset: ATF3-bound windows vs GC-matched background.

Positives: a window of --width bp centered on each peak summit, keeping the
strongest peak when windows overlap. Negatives: random genomic windows that stay
--flank bp away from every peak, contain no N, and match the positives' GC
distribution bin by bin (otherwise a model can score well by learning GC content).

Output: one TSV with id, chrom, start, end, label, gc, repeat_frac, seq.
Sequences keep UCSC soft-masking (lowercase = repeat); the model upper-cases them.

Usage:
    python scripts/02_make_dataset.py --peaks data/peaks.narrowPeak.gz --genome data/hg38.fa \
        --out data/dataset.tsv
"""

import argparse
import bisect
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slb.seq import gc_fraction, repeat_fraction  # noqa: E402

DEFAULT_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def read_bed(path, ncols=None):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith(("#", "track", "browser")) or not line.strip():
                continue
            rows.append(line.rstrip("\n").split("\t"))
    df = pd.DataFrame(rows)
    df[1] = df[1].astype(int)
    df[2] = df[2].astype(int)
    return df


class IntervalSet:
    """Merged intervals per chromosome with a fast overlap test."""

    def __init__(self, intervals):
        by_chrom = {}
        for c, s, e in intervals:
            by_chrom.setdefault(c, []).append((s, e))
        self.starts, self.ends = {}, {}
        for c, iv in by_chrom.items():
            iv.sort()
            merged = [list(iv[0])]
            for s, e in iv[1:]:
                if s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            self.starts[c] = [m[0] for m in merged]
            self.ends[c] = [m[1] for m in merged]

    def overlaps(self, c, s, e):
        starts = self.starts.get(c)
        if not starts:
            return False
        i = bisect.bisect_right(starts, e) - 1
        while i >= 0 and self.ends[c][i] > s:
            if starts[i] < e:
                return True
            i -= 1
        return False


def build_positives(peaks, genome, chroms, width):
    half = width // 2
    df = peaks[peaks[0].isin(chroms)].copy()
    summit = df[9].astype(int) if df.shape[1] > 9 else pd.Series(-1, index=df.index)
    center = np.where(summit >= 0, df[1] + summit, (df[1] + df[2]) // 2)
    df["center"] = center
    df["signal"] = df[6].astype(float) if df.shape[1] > 6 else 0.0
    df = df.sort_values("signal", ascending=False)

    accepted = {}
    rows = []
    for c, ctr in zip(df[0], df["center"]):
        s, e = int(ctr) - half, int(ctr) - half + width
        if s < 0 or e > len(genome[c]):
            continue
        taken = accepted.setdefault(c, [])
        i = bisect.bisect_left(taken, s)
        if (i < len(taken) and taken[i] < e) or (i > 0 and taken[i - 1] + width > s):
            continue  # overlaps a stronger peak's window
        seq = genome[c][s:e].seq
        if "N" in seq.upper():
            continue
        taken.insert(i, s)
        rows.append((c, s, e, 1, seq))
    return rows


def build_negatives(positives, genome, chroms, width, forbidden, ratio, bins, rng, max_tries_factor=400):
    edges = np.linspace(0, 1, bins + 1)
    pos_gc = np.array([gc_fraction(r[4]) for r in positives])
    want = np.histogram(pos_gc, bins=edges)[0] * ratio
    have = np.zeros_like(want)
    lengths = np.array([len(genome[c]) for c in chroms], dtype=float)
    probs = lengths / lengths.sum()

    rows, tries = [], 0
    max_tries = int(max_tries_factor * want.sum()) + 1000
    while (have < want).any() and tries < max_tries:
        tries += 1
        c = chroms[rng.choice(len(chroms), p=probs)]
        s = int(rng.integers(0, len(genome[c]) - width))
        e = s + width
        if forbidden.overlaps(c, s, e):
            continue
        seq = genome[c][s:e].seq
        if "N" in seq.upper():
            continue
        b = min(np.searchsorted(edges, gc_fraction(seq), side="right") - 1, bins - 1)
        if have[b] >= want[b]:
            continue
        have[b] += 1
        rows.append((c, s, e, 0, seq))
    short = int((want - have).sum())
    if short:
        print(f"warning: {short} negatives could not be GC-matched after {tries} tries "
              f"(bins short: {np.nonzero(want - have)[0].tolist()})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--out", default="data/dataset.tsv")
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--flank", type=int, default=500, help="keep negatives this far from any peak")
    ap.add_argument("--neg-ratio", type=int, default=1)
    ap.add_argument("--gc-bins", type=int, default=20)
    ap.add_argument("--exclude", help="extra BED of regions to avoid (e.g. data/blacklist.bed.gz)")
    ap.add_argument("--chroms", default=",".join(DEFAULT_CHROMS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    genome = Fasta(args.genome, sequence_always_upper=False, as_raw=False)
    chroms = [c for c in args.chroms.split(",") if c in genome.keys()]
    peaks = read_bed(args.peaks)

    pos = build_positives(peaks, genome, chroms, args.width)
    print(f"positives: {len(pos)} windows from {len(peaks)} peaks")

    forbidden = [(r[0], max(0, r[1] - args.flank), r[2] + args.flank) for r in peaks.itertuples(index=False, name=None)]
    if args.exclude:
        ex = read_bed(args.exclude)
        forbidden += list(zip(ex[0], ex[1], ex[2]))
    neg = build_negatives(pos, genome, chroms, args.width, IntervalSet(forbidden),
                          args.neg_ratio, args.gc_bins, rng)
    print(f"negatives: {len(neg)} GC-matched windows")

    df = pd.DataFrame(pos + neg, columns=["chrom", "start", "end", "label", "seq"])
    df["gc"] = df["seq"].map(gc_fraction).round(4)
    df["repeat_frac"] = df["seq"].map(repeat_fraction).round(4)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df.insert(0, "id", np.arange(len(df)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df[["id", "chrom", "start", "end", "label", "gc", "repeat_frac", "seq"]].to_csv(args.out, sep="\t", index=False)

    g = df.groupby("label")
    print(f"mean GC   pos {g['gc'].mean().get(1, float('nan')):.3f}  neg {g['gc'].mean().get(0, float('nan')):.3f}")
    print(f"mean rep  pos {g['repeat_frac'].mean().get(1, float('nan')):.3f}  neg {g['repeat_frac'].mean().get(0, float('nan')):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
