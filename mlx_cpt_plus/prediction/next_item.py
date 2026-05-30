"""Next item prediction."""

from typing import List, Tuple, Optional


class NextItem:
    """Predicts the next item in a sequence.
    
    Simple wrapper for single-item prediction.
    """
    
    def __init__(self, predictor):
        """Initialize next item predictor.
        
        Args:
            predictor: Predictor instance to use.
        """
        self.predictor = predictor
    
    def predict(self, context: List[int]) -> Optional[int]:
        """Predict the next item.
        
        Args:
            context: Context items.
            
        Returns:
            Predicted item or None.
        """
        predictions = self.predictor.predict(context, k=1)
        return predictions[0][0] if predictions else None
    
    def predict_with_score(self, context: List[int]) -> Tuple[Optional[int], float]:
        """Predict next item with score.
        
        Args:
            context: Context items.
            
        Returns:
            Tuple of (item, score) or (None, 0.0).
        """
        predictions = self.predictor.predict(context, k=1)
        if predictions:
            return predictions[0]
        return None, 0.0
    
    def __repr__(self) -> str:
        return f"NextItem(predictor={self.predictor})"