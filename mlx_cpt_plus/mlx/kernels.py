"""MLX kernel utilities."""

import mlx.core as mx


def kernels():
    """Return available MLX kernels."""
    return {
        "similarity": similarity_kernel,
        "vote": vote_kernel,
        "ranking": ranking_kernel,
    }


def similarity_kernel(a: mx.array, b: mx.array) -> mx.array:
    """Compute similarity between arrays.
    
    Args:
        a: First array.
        b: Second array.
        
    Returns:
        Similarity scores.
    """
    # Cosine similarity
    a_norm = mx.normalize(a, axis=-1)
    b_norm = mx.normalize(b, axis=-1)
    return mx.matmul(a_norm, b_norm.T)


def vote_kernel(votes: mx.array, weights: mx.array = None) -> mx.array:
    """Aggregate votes.
    
    Args:
        votes: Vote array.
        weights: Optional weights.
        
    Returns:
        Aggregated scores.
    """
    if weights is None:
        return mx.sum(votes, axis=0)
    return mx.sum(votes * weights, axis=0)


def ranking_kernel(scores: mx.array, context: mx.array) -> mx.array:
    """Rank items by scores.
    
    Args:
        scores: Item scores.
        context: Context mask.
        
    Returns:
        Ranked scores.
    """
    # Apply context mask
    masked = scores * context
    # Sort
    return mx.sort(masked)[0]