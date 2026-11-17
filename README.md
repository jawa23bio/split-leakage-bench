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

## Run it on Colab

```bash
!git clone https://github.com/jawa23bio/split-leakage-bench.git
%cd split-leakage-bench
!pip install -q -r requirements.txt

# 1. data (about 1 GB download, 3 GB unpacked; a few minutes)
!python scripts/01_download.py --out data --blacklist

# 2. dataset: 200 bp summit windows + GC-matched background
!python scripts/02_make_dataset.py --peaks data/peaks.narrowPeak.gz --genome data/hg38.fa \
    --exclude data/blacklist.bed.gz --out data/dataset.tsv

# 3. splits + leakage summary (read data/leakage_summary.md first: it is result 1)
!python scripts/03_make_splits.py --dataset data/dataset.tsv --out data

# 4. train 5 seeds per split (15 runs; switch the runtime to GPU first)
!for s in random chrom homology; do for seed in 0 1 2 3 4; do PYTHONPATH=src python -m slb.train --scheme $s --seed $seed; done; done

# 5. ensemble the seeds and answer questions 2-4 -> results/summary.md
!python scripts/04_analyze.py --results results --leakage data/leakage.tsv
```

Check the pipeline without downloading anything: `python tests/smoke_test.py` builds a synthetic
genome with planted motifs and 40 test-chromosome peaks copied onto chr1, and confirms the chrom
split reports them as leakage while the homology split does not. Add `--train` to also train two
seeds on it and run the analysis (under a minute on CPU).

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
- **Model selection**: the epoch with the best validation AUROC is kept (ties go to the lower
  validation loss), with early stopping after 5 epochs without gain. Validation follows the same
  scheme as test, so under `homology` the model is also selected on windows unlike its training set.
- **Ensemble**: the 5 seeds share one split and differ in initialisation, batch order and RC
  augmentation. Spread is the SD of their probabilities. Because spread and error both shrink as a
  prediction nears 0 or 1, the analysis also reports how well spread picks out wrong calls compared
  with the mean prediction's closeness to 0.5, which needs no ensemble.

## Layout

```
scripts/01_download.py      ENCODE ATF3 K562 peaks (ENCSR028UIU) + UCSC hg38 (+ optional exclusion list)
scripts/02_make_dataset.py  summit windows, GC-matched negatives -> data/dataset.tsv
scripts/03_make_splits.py   random / chrom / homology splits + leakage report -> data/splits.tsv
src/slb/seq.py              one-hot, reverse complement, GC
src/slb/similarity.py       canonical k-mer index, similarity groups
src/slb/data.py             loads one part of one split as a torch Dataset
src/slb/metrics.py          AUROC, AUPRC, Brier, ECE, ensemble spread vs error
src/slb/model.py            the CNN (2 conv layers, about 83k parameters)
src/slb/train.py            trains one model on one split -> results/<scheme>/seed<n>/
scripts/04_analyze.py       ensembles the seeds, leaked vs clean, spread vs error -> results/summary.md
tests/smoke_test.py         synthetic end-to-end check (--train to include training and analysis)
```

Data: ENCODE experiment [ENCSR028UIU](https://www.encodeproject.org/experiments/ENCSR028UIU/)
(ATF3 ChIP-seq, K562), UCSC hg38.
