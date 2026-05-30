"""Batch prediction using MLX."""

import mlx.core as mx
from typing import List, Tuple


def batch_predict(
    contexts: List[List[int]],
    candidate_items: List[List[int]],
    scores: mx.array
) -> List[List[Tuple[int, float]]]:
    """Perform batch prediction.
    
    Args:
        contexts: List of context sequences.
        candidate_items: List of candidate items per context.
        scores: Score array from model.
        
    Returns:
        List of prediction lists.
    """
    results = []
    for i, candidates in enumerate(candidate_items):
        ctx_scores = scores[i, candidates]
        sorted_idx = mx.argsort(ctx_scores)[::-1]
        top_candidates = [candidates[idx] for idx in sorted_idx[:10]]
        top_scores = [float(ctx_scores[idx]) for idx in sorted_idx[:10]]
        results.append(list(zip(top_candidates, top_scores)))
    return results