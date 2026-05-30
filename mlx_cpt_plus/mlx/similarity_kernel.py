"""MLX similarity kernel."""

import mlx.core as mx


def similarity_kernel(a: mx.array, b: mx.array, method: str = "cosine") -> mx.array:
    """Compute similarity between arrays using MLX.
    
    Args:
        a: First array of shape (n, d).
        b: Second array of shape (m, d).
        method: Similarity method ('cosine', 'dot', 'jaccard').
        
    Returns:
        Similarity matrix of shape (n, m).
    """
    if method == "cosine":
        a_norm = mx.normalize(a, axis=-1)
        b_norm = mx.normalize(b, axis=-1)
        return mx.matmul(a_norm, b_norm.T)
    elif method == "dot":
        return mx.matmul(a, b.T)
    elif method == "jaccard":
        # Approximate Jaccard using element-wise operations
        intersection = mx.sum(mx.minimum(mx.expand_dims(a, 1), mx.expand_dims(b, 0)), axis=-1)
        union = mx.sum(mx.maximum(mx.expand_dims(a, 1), mx.expand_dims(b, 0)), axis=-1)
        return intersection / (union + 1e-8)
    else:
        raise ValueError(f"Unknown similarity method: {method}")