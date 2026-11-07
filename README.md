# split-leakage-bench

**Does a chromosome holdout overstate how well a DNA sequence model generalizes?**

Sequence-to-function models are usually tested on held-out chromosomes. Nagai, Murphy, Rizzo and Koo
([2026 review](https://arxiv.org/abs/2602.01230)) point out that this split does not prevent
information leakage, because similar sequences can sit on different chromosomes. This project
measures that directly on one task, ATF3 binding in K562, by training the same small CNN under
three splits and comparing them:

| Split | What is held out | Leakage controlled? |
|---|---|---|
| `random` | 10% of windows at random | no |
| `chrom` | chr8 + chr9 (val: chr7), the usual practice | partly |
| `homology` | groups of windows that share >= 30% of their 16-mers (either strand) | yes, by construction |

It also checks whether a 5-model ensemble's uncertainty tracks its errors, following up
[pert-prior-bench](https://github.com/jawa23bio/pert-prior-bench), where bootstrap intervals
were badly overconfident.

## Questions

1. How many chromosome-holdout test windows have a near-copy in training? (measured before any training)
2. How much does test AUROC/AUPRC drop from `random` to `chrom` to `homology`?
3. Is the drop concentrated in the leaked windows, or spread across all of them?
4. Does ensemble spread rise where predictions are wrong, and does that hold under the homology split?
5. Optional: does the trained model recover the ATF/CREB motif (TGACGTCA)? Check with [SQUID](https://github.com/evanseitz/squid-nn) or first-layer filters.

## Status

| Piece | Status |
|---|---|
| `scripts/01_download.py` ENCODE peaks + hg38 | written; run on Colab (ENCODE is not reachable from where it was written) |
| `scripts/02_make_dataset.py` positives + GC-matched negatives | written, tested on synthetic data |
| `scripts/03_make_splits.py` three splits + leakage report | written, tested on synthetic data |
| `src/slb/data.py`, `src/slb/metrics.py` | written |
| `src/slb/model.py` the CNN | **yours to write** |
| `src/slb/train.py` the training loop | **yours to write** (steps listed in the file) |
| ensemble + results analysis | next, after training works |

## Milestones

| By | Milestone |
|---|---|
| Nov 15 | data built on Colab; leakage summary read; baseline CNN trains on the `chrom` split |
| Nov 22 | all three splits trained (1 seed each) and compared |
| Nov 29 | 5 seeds per split; ensemble calibration; repo public with results in this README |
| Dec 5 | one sentence in the CSHL personal statement, one CV entry |

## Run it on Colab

```bash
!git clone https://github.com/jawa23bio/split-leakage-bench.git   # after you push it
%cd split-leakage-bench
!pip install -q -r requirements.txt

# 1. data (about 1 GB download, 3 GB unpacked; a few minutes)
!python scripts/01_download.py --out data --blacklist

# 2. dataset: 200 bp summit windows + GC-matched background
!python scripts/02_make_dataset.py --peaks data/peaks.narrowPeak.gz --genome data/hg38.fa \
    --exclude data/blacklist.bed.gz --out data/dataset.tsv

# 3. splits + leakage summary (read data/leakage_summary.md first: it is result 1)
!python scripts/03_make_splits.py --dataset data/dataset.tsv --out data

# 4. after you write model.py and train.py
!PYTHONPATH=src python -m slb.train --scheme chrom --seed 0
```

Check the pipeline without downloading anything: `python tests/smoke_test.py` builds a synthetic
genome with planted motifs and 40 test-chromosome peaks copied onto chr1, and confirms the chrom
split reports them as leakage while the homology split does not.

## Design notes

- **Negatives** are random genomic windows at least 500 bp from any peak, matched to the positives'
  GC content in 20 bins. Without GC matching, a model can score well by learning GC alone.
- **Similarity** is the share of a window's canonical 16-mers found in another window. It is a fast
  stand-in for alignment. k-mers in more than 200 windows (mostly repeats such as Alu) are skipped
  to keep it fast, so repeat-driven similarity is undercounted. `dataset.tsv` keeps each window's
  repeat fraction (UCSC soft-masking) so you can check whether results change when repeat-rich
  windows are dropped.
- **Threshold** 0.3 is a choice, not a law. Report how the leakage numbers move at 0.3 / 0.5 / 0.8
  (the summary prints all three).
- **Reverse complement**: train with RC augmentation and average forward and RC predictions at test.

## Layout

```
scripts/01_download.py      ENCODE ATF3 K562 peaks (ENCSR028UIU) + UCSC hg38 (+ optional exclusion list)
scripts/02_make_dataset.py  summit windows, GC-matched negatives -> data/dataset.tsv
scripts/03_make_splits.py   random / chrom / homology splits + leakage report -> data/splits.tsv
src/slb/seq.py              one-hot, reverse complement, GC
src/slb/similarity.py       canonical k-mer index, similarity groups
src/slb/data.py             loads one part of one split as a torch Dataset
src/slb/metrics.py          AUROC, AUPRC, Brier, ECE, ensemble spread vs error
src/slb/model.py            the CNN (to write)
src/slb/train.py            the training loop (to write)
tests/smoke_test.py         synthetic end-to-end check
```

Data: ENCODE experiment [ENCSR028UIU](https://www.encodeproject.org/experiments/ENCSR028UIU/)
(ATF3 ChIP-seq, K562), UCSC hg38.
