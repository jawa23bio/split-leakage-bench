"""Sequence similarity by shared canonical k-mers (both strands).

Similarity of query q to target t = (shared canonical k-mers) / (k-mers in q).
It is a cheap stand-in for alignment: two 200 bp windows that share 30% of their
16-mers almost always share a long aligned stretch. Strand is ignored because a
model that sees a sequence on one strand has effectively seen its reverse complement.

Very common k-mers (repeats such as Alu) are skipped once they appear in more than
`max_df` sequences, which keeps the index fast. Those repeats are still a possible
source of leakage: the dataset table keeps each window's repeat fraction so you can
check whether results change when repeat-rich windows are dropped.

Implementation: k-mers are packed into integers (2 bits per base) and held in numpy
arrays, with an inverted index stored as one sorted array. That is about 20 bytes per
k-mer, so 120k windows of 200 bp fit in well under 1 GB. `canonical_kmers` is the plain
string definition of the same thing; tests/test_similarity.py checks the two agree.
"""

import numpy as np

from .seq import revcomp

# A, C, G, T -> 0..3 (either case); anything else (N) -> 4
_CODE = np.full(256, 4, dtype=np.uint8)
for _i, _c in enumerate("ACGT"):
    _CODE[ord(_c)] = _CODE[ord(_c.lower())] = _i
_INVALID = np.iinfo(np.uint64).max


def canonical_kmers(seq: str, k: int) -> set:
    """Reference definition: the set of k-mers of seq, each replaced by the smaller of
    itself and its reverse complement. k-mers containing N are left out."""
    s = seq.upper()
    rc = revcomp(s)
    n = len(s)
    out = set()
    for i in range(n - k + 1):
        a = s[i:i + k]
        if "N" in a:
            continue
        b = rc[n - i - k:n - i]
        out.add(a if a < b else b)
    return out


def _encode_block(seqs, k):
    """Canonical k-mer codes for equal-length sequences: (len(seqs), L-k+1) uint64, sorted
    along each row, with N-containing and repeated k-mers replaced by _INVALID."""
    length = len(seqs[0])
    if length < k:
        return np.full((len(seqs), 0), _INVALID, dtype=np.uint64)
    base = _CODE[np.frombuffer("".join(seqs).encode("ascii", "replace"), dtype=np.uint8)].reshape(len(seqs), length)
    win = np.lib.stride_tricks.sliding_window_view(base, k, axis=1)  # (n, L-k+1, k), no copy
    shifts = (2 * np.arange(k - 1, -1, -1)).astype(np.uint64)
    fwd = np.zeros(win.shape[:2], dtype=np.uint64)
    rev = np.zeros(win.shape[:2], dtype=np.uint64)
    bad = np.zeros(win.shape[:2], dtype=bool)
    for j in range(k):
        b = win[:, :, j]
        bad |= b == 4
        b = (b & 3).astype(np.uint64)
        fwd |= b << shifts[j]                               # base j is the j-th most significant
        rev |= (np.uint64(3) - b) << shifts[k - 1 - j]      # complement, read from the other end
    codes = np.minimum(fwd, rev)
    codes[bad] = _INVALID
    codes.sort(axis=1)
    codes[:, 1:][codes[:, 1:] == codes[:, :-1]] = _INVALID  # keep one copy of each k-mer per sequence
    return codes


class KmerIndex:
    def __init__(self, seqs, k=16, max_df=200, block=5000):
        if k > 32:
            raise ValueError("k must be <= 32 (k-mers are packed into 64 bits)")
        self.k, self.n = k, len(seqs)
        self.n_kmers = np.zeros(self.n, dtype=np.int64)  # distinct canonical k-mers per sequence

        # all (k-mer, sequence) pairs; sequences are grouped by length so each block is a rectangle
        by_len = {}
        for i, s in enumerate(seqs):
            by_len.setdefault(len(s), []).append(i)
        code_parts, id_parts = [], []
        for ids in by_len.values():
            for a in range(0, len(ids), block):
                chunk = np.asarray(ids[a:a + block])
                codes = _encode_block([seqs[i] for i in chunk], k)
                keep = codes != _INVALID
                self.n_kmers[chunk] = keep.sum(axis=1)
                code_parts.append(codes[keep])
                id_parts.append(np.repeat(chunk, keep.sum(axis=1)).astype(np.int32))
        codes = np.concatenate(code_parts) if code_parts else np.zeros(0, np.uint64)
        ids = np.concatenate(id_parts) if id_parts else np.zeros(0, np.int32)
        del code_parts, id_parts

        order = np.argsort(codes, kind="stable")
        codes, ids = codes[order], ids[order]
        del order
        # runs of equal k-mers = posting lists; df = how many sequences hold that k-mer
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]]) if len(codes) else np.zeros(0, np.int64)
        df = np.diff(np.r_[starts, len(codes)])
        self.skipped = int((df > max_df).sum())
        # only k-mers shared by 2..max_df sequences can link two sequences
        run_keep = (df >= 2) & (df <= max_df)
        entry_keep = np.repeat(run_keep, df)
        run_of_entry = np.repeat(np.cumsum(run_keep) - 1, df)[entry_keep]
        self._post_ids = ids[entry_keep]                       # sequence ids, grouped by k-mer
        self._post_ptr = np.r_[0, np.cumsum(df[run_keep])]     # run r is _post_ids[ptr[r]:ptr[r+1]]
        # for each sequence, the runs it takes part in
        by_seq = np.argsort(self._post_ids, kind="stable")
        self._seq_runs = run_of_entry[by_seq]
        self._seq_ptr = np.r_[0, np.cumsum(np.bincount(self._post_ids, minlength=self.n))]

    def _shared(self, i, allowed=None):
        """(target ids, shared k-mer counts) for sequence i, itself excluded."""
        runs = self._seq_runs[self._seq_ptr[i]:self._seq_ptr[i + 1]]
        if len(runs) == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.int64)
        lo = self._post_ptr[runs]
        lens = self._post_ptr[runs + 1] - lo
        # gather every posting list at once: positions lo[r], lo[r]+1, ... for each run r
        pos = np.repeat(lo - (np.cumsum(lens) - lens), lens) + np.arange(lens.sum())
        targets = self._post_ids[pos]
        targets = targets[targets != i]
        if allowed is not None:
            targets = targets[allowed[targets]]
        return np.unique(targets, return_counts=True)

    def as_mask(self, ids):
        """Boolean mask over sequences from a collection of ids (or a mask, returned as is)."""
        if isinstance(ids, np.ndarray) and ids.dtype == bool:
            return ids
        mask = np.zeros(self.n, dtype=bool)
        mask[np.fromiter(ids, dtype=np.int64)] = True
        return mask

    def shared_counts(self, i, allowed=None):
        """dict of target id -> shared k-mers with sequence i (i itself excluded)."""
        t, c = self._shared(i, None if allowed is None else self.as_mask(allowed))
        return dict(zip(t.tolist(), c.tolist()))

    def max_similarity(self, i, allowed):
        """Highest similarity of sequence i to any sequence in `allowed`.

        `allowed` is a boolean mask over sequences (fastest; build it once with as_mask)
        or any collection of ids."""
        n = self.n_kmers[i]
        if n == 0:
            return 0.0
        _, c = self._shared(i, self.as_mask(allowed))
        return float(c.max() / n) if len(c) else 0.0


def similarity_components(index: KmerIndex, threshold=0.3):
    """Group sequences linked by similarity >= threshold (in either direction).

    Returns a list where entry i is the component id of sequence i.
    """
    n = index.n
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        ni = index.n_kmers[i] or 1
        targets, shared = index._shared(i)
        for j in targets[shared / ni >= threshold].tolist():
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    return [find(i) for i in range(n)]
