"""Sequence similarity by shared canonical k-mers (both strands).

Similarity of query q to target t = (shared canonical k-mers) / (k-mers in q).
It is a cheap stand-in for alignment: two 200 bp windows that share 30% of their
16-mers almost always share a long aligned stretch. Strand is ignored because a
model that sees a sequence on one strand has effectively seen its reverse complement.

Very common k-mers (repeats such as Alu) are skipped once they appear in more than
`max_df` sequences, which keeps the index fast. Those repeats are still a possible
source of leakage: the dataset table keeps each window's repeat fraction so you can
check whether results change when repeat-rich windows are dropped.
"""

from collections import Counter, defaultdict

from .seq import revcomp


def canonical_kmers(seq: str, k: int) -> set:
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


class KmerIndex:
    def __init__(self, seqs, k=16, max_df=200):
        self.k = k
        self.kmers = [canonical_kmers(s, k) for s in seqs]
        postings = defaultdict(list)
        for i, ks in enumerate(self.kmers):
            for km in ks:
                postings[km].append(i)
        self.postings = {km: ids for km, ids in postings.items() if len(ids) <= max_df}
        self.skipped = len(postings) - len(self.postings)

    def shared_counts(self, i, allowed=None):
        """Counter of target id -> shared k-mers with sequence i (i itself excluded)."""
        c = Counter()
        for km in self.kmers[i]:
            ids = self.postings.get(km)
            if not ids:
                continue
            for j in ids:
                if j != i and (allowed is None or j in allowed):
                    c[j] += 1
        return c

    def max_similarity(self, i, allowed):
        """Highest similarity of sequence i to any sequence in `allowed` (a set of ids)."""
        n = len(self.kmers[i])
        if n == 0:
            return 0.0
        c = self.shared_counts(i, allowed)
        return max(c.values()) / n if c else 0.0


def similarity_components(index: KmerIndex, threshold=0.3):
    """Group sequences linked by similarity >= threshold (in either direction).

    Returns a list where entry i is the component id of sequence i.
    """
    n = len(index.kmers)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        ni = len(index.kmers[i]) or 1
        for j, shared in index.shared_counts(i).items():
            if shared / ni >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return [find(i) for i in range(n)]
