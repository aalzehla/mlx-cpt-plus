"""MLX voting kernel."""

import mlx.core as mx


def vote_kernel(votes: mx.array, weights: mx.array = None) -> mx.array:
    """Aggregate votes using MLX.
    
    Args:
        votes: Vote array of shape (n, num_items).
        weights: Optional weights of shape (n,).
        
    Returns:
        Aggregated scores of shape (num_items,).
    """
    if weights is None:
        return mx.sum(votes, axis=0)
    
    # Weighted sum
    weighted = mx.expand_dims(weights, -1) * votes
    return mx.sum(weighted, axis=0)


def ranked_vote_kernel(rankings: mx.array, weights: mx.array) -> mx.array:
    """Compute ranked voting (Borda count).
    
    Args:
        rankings: Ranking matrix of shape (n, num_items).
        weights: Weights of shape (n,).
        
    Returns:
        Borda scores of shape (num_items,).
    """
    n_items = rankings.shape[1]
    borda_scores = n_items - rankings
    weighted = mx.expand_dims(weights, -1) * borda_scores
    return mx.sum(weighted, axis=0)