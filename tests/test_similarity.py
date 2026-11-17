#!/usr/bin/env python
"""Check the packed-integer k-mer index against the plain string definition.

The reference below is the obvious implementation: Python sets of k-mer strings and a
dict of posting lists. It is too memory-hungry for the real dataset (about 50 MB per
1000 windows) but easy to trust, so the fast index must reproduce it exactly.

    python tests/test_similarity.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slb.seq import revcomp  # noqa: E402
from slb.similarity import KmerIndex, canonical_kmers, similarity_components  # noqa: E402


class ReferenceIndex:
    def __init__(self, seqs, k, max_df):
        self.kmers = [canonical_kmers(s, k) for s in seqs]
        postings = defaultdict(list)
        for i, ks in enumerate(self.kmers):
            for km in ks:
                postings[km].append(i)
        self.postings = {km: ids for km, ids in postings.items() if len(ids) <= max_df}
        self.skipped = len(postings) - len(self.postings)

    def shared_counts(self, i, allowed=None):
        c = Counter()
        for km in self.kmers[i]:
            for j in self.postings.get(km, ()):
                if j != i and (allowed is None or j in allowed):
                    c[j] += 1
        return dict(c)


def reference_components(ref, threshold):
    """Component labels as a partition (set of frozensets), by plain graph search."""
    n = len(ref.kmers)
    adj = defaultdict(set)
    for i in range(n):
        ni = len(ref.kmers[i]) or 1
        for j, shared in ref.shared_counts(i).items():
            if shared / ni >= threshold:
                adj[i].add(j)
                adj[j].add(i)
    seen, parts = set(), set()
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adj[x] - comp)
        seen |= comp
        parts.add(frozenset(comp))
    return parts


def make_sequences(rng, n=600, length=200):
    def rand(m):
        return "".join(rng.choice(list("ACGT"), size=m))

    seqs = [rand(length) for _ in range(n)]
    repeat = rand(120)
    for i in range(0, 80):                       # a repeat family: 120 bp shared by 80 windows
        at = int(rng.integers(0, length - 120))
        seqs[i] = seqs[i][:at] + repeat + seqs[i][at + 120:]
    for i in range(100, 140):                    # exact copies and reverse-complement copies
        seqs[i] = seqs[i + 100] if i % 2 else revcomp(seqs[i + 100])
    for i in range(300, 340):                    # partial overlaps (shifted windows), some on the other strand
        shift = int(rng.integers(20, 180))
        s = seqs[i + 100][shift:] + rand(shift)
        seqs[i] = s if i % 3 else revcomp(s)
    for i in range(450, 480):                    # N runs and soft-masked (lowercase) stretches
        s = list(seqs[i])
        a = int(rng.integers(0, length - 30))
        s[a:a + int(rng.integers(1, 30))] = "N" * 1
        seqs[i] = "".join(s)[:length].ljust(length, "N")
        seqs[i] = seqs[i][:50].lower() + seqs[i][50:]
    for lo, hi in ((520, 550), (550, 581)):      # k-mers in exactly 30 and 31 windows: the max_df=30 boundary
        block = rand(40)
        for i in range(lo, hi):
            seqs[i] = seqs[i][:80] + block + seqs[i][120:]
    seqs[500] = "A" * length                     # low complexity: one distinct k-mer
    seqs[501] = "AT" * (length // 2)
    seqs[502] = "N" * length                     # no usable k-mer at all
    seqs += [rand(150), rand(150), rand(10)]     # other lengths, one shorter than k
    seqs.append(seqs[-3][:100] + rand(100))      # a 200 bp window sharing 100 bp with a 150 bp one
    return seqs


def main():
    rng = np.random.default_rng(7)
    seqs = make_sequences(rng)
    n = len(seqs)
    checked = 0
    for k, max_df in ((16, 200), (16, 30), (8, 50), (21, 200)):
        ref = ReferenceIndex(seqs, k, max_df)
        idx = KmerIndex(seqs, k=k, max_df=max_df, block=97)
        assert idx.skipped == ref.skipped, (k, max_df, idx.skipped, ref.skipped)
        assert idx.n_kmers.tolist() == [len(s) for s in ref.kmers], "distinct k-mer counts differ"
        allowed = set(rng.choice(n, size=n // 2, replace=False).tolist())
        mask = idx.as_mask(allowed)
        for i in range(n):
            want = ref.shared_counts(i)
            assert idx.shared_counts(i) == want, (k, max_df, i)
            want_allowed = {j: c for j, c in want.items() if j in allowed}
            assert idx.shared_counts(i, allowed) == want_allowed, (k, max_df, i)
            nk = len(ref.kmers[i])
            want_max = max(want_allowed.values()) / nk if want_allowed and nk else 0.0
            assert idx.max_similarity(i, mask) == want_max == idx.max_similarity(i, allowed), (k, max_df, i)
            checked += 1
        for threshold in (0.3, 0.8):
            comps = similarity_components(idx, threshold)
            groups = defaultdict(set)
            for i, c in enumerate(comps):
                groups[c].add(i)
            assert {frozenset(g) for g in groups.values()} == reference_components(ref, threshold), (k, max_df, threshold)
    print(f"SIMILARITY TEST PASSED: fast index matches the string reference on {n} sequences "
          f"({checked} query checks across 4 settings).")


if __name__ == "__main__":
    main()
