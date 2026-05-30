"""MLX ranking kernel."""

import mlx.core as mx


def ranking_kernel(scores: mx.array, top_k: int = 10) -> mx.array:
    """Rank items by scores using MLX.
    
    Args:
        scores: Score array of shape (num_items,).
        top_k: Number of top items to return.
        
    Returns:
        Top-k scores and indices.
    """
    sorted_scores, sorted_indices = mx.sort(scores)
    return sorted_scores[:top_k], sorted_indices[:top_k]


def normalize_scores_kernel(scores: mx.array) -> mx.array:
    """Normalize scores to [0, 1].
    
    Args:
        scores: Score array.
        
    Returns:
        Normalized scores.
    """
    min_s = mx.min(scores)
    max_s = mx.max(scores)
    return (scores - min_s) / (max_s - min_s + 1e-8)