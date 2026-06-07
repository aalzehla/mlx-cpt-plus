"""Basic usage example for MLX CPT+."""

from mlx_cpt_plus import CPTPlus, Config


def main():
    """Demonstrate basic usage."""
    # Create configuration
    config = Config(
        max_sequence_length=100,
        min_frequency=1,
        num_leaves=100,
    )
    
    # Initialize model
    model = CPTPlus(config=config)
    
    # Training data
    sequences = [
        [1, 2, 3, 4, 5],
        [1, 2, 4, 5, 6],
        [2, 3, 4, 7, 8],
        [1, 3, 5, 7, 9],
    ]
    
    # Fit the model
    model.fit(sequences)
    
    # Make predictions
    context = [1, 2, 3]
    predictions = model.predict(context, k=5)
    
    print(f"Context: {context}")
    print(f"Predictions: {predictions}")
    
    # Online learning
    model.partial_fit([4, 5, 6, 7])
    
    print("Model updated with new data")


if __name__ == "__main__":
    main()