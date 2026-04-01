"""Cosine-similarity novelty filter using character-trigram TF vectors.

A candidate policy is considered novel if its maximum cosine similarity to any
member of the archive is below `threshold`.  Requires no external NLP libraries.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List


def _trigram_vector(text: str) -> dict[str, int]:
    """Return character-trigram frequency counts for `text`."""
    counts: Counter[str] = Counter()
    for i in range(len(text) - 2):
        counts[text[i : i + 3]] += 1
    return dict(counts)


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    """Cosine similarity between two sparse frequency vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0) * v for k, v in b.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mean_pairwise_similarity(codes: list[str]) -> float:
    """Mean cosine similarity across all distinct pairs of policy strings.

    Used by the main loop to detect diversity collapse and trigger island spawning.
    Returns 0.0 for fewer than two codes.
    """
    if len(codes) < 2:
        return 0.0
    vecs = [_trigram_vector(c) for c in codes]
    total, count = 0.0, 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            total += _cosine(vecs[i], vecs[j])
            count += 1
    return total / count if count else 0.0


class NoveltyFilter:
    """Archive-backed cosine-similarity novelty gate.

    Usage::

        nf = NoveltyFilter(threshold=0.95)
        if nf.is_novel(candidate_code):
            mse = score(candidate_code, data)
            nf.register(candidate_code)
        else:
            # rejected — count as stale, skip evaluation
            pass
    """

    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold
        self._archive: List[dict[str, int]] = []
        self._seen_this_gen: int = 0
        self._rejected_this_gen: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def is_novel(self, code: str) -> bool:
        """Return True if the candidate passes the novelty gate.

        A candidate is novel if its maximum cosine similarity to any archived
        member is strictly below `self.threshold`.  Always returns True when
        the archive is empty (first candidate is inherently novel).
        """
        self._seen_this_gen += 1
        vec = _trigram_vector(code)
        for archived_vec in self._archive:
            if _cosine(vec, archived_vec) >= self.threshold:
                self._rejected_this_gen += 1
                return False
        return True

    def register(self, code: str) -> None:
        """Add a code string to the archive (call after evaluation, not before)."""
        self._archive.append(_trigram_vector(code))

    @property
    def rejection_rate(self) -> float:
        """Fraction of candidates seen this generation that were rejected."""
        if self._seen_this_gen == 0:
            return 0.0
        return self._rejected_this_gen / self._seen_this_gen

    def reset_gen_counters(self) -> None:
        """Reset per-generation seen/rejected counters (call at start of each gen)."""
        self._seen_this_gen = 0
        self._rejected_this_gen = 0

    @property
    def archive_size(self) -> int:
        return len(self._archive)
