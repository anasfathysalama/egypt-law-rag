"""Cosine similarity of a query embedding against the index baseline."""

from __future__ import annotations

import numpy as np


def baseline_vector(vectors: np.ndarray) -> np.ndarray:
    """Mean of the stored document vectors. This is the drift baseline."""
    if len(vectors) == 0:
        raise ValueError("the index has no vectors")
    return np.asarray(vectors, dtype=np.float32).mean(axis=0)


def cosine_similarity(query: np.ndarray, baseline: np.ndarray) -> float:
    left = np.asarray(query, dtype=np.float32).reshape(-1)
    right = np.asarray(baseline, dtype=np.float32).reshape(-1)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
