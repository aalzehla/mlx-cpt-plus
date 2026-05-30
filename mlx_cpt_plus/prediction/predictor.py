"""Main predictor class."""

from typing import List, Tuple, Optional
from mlx_cpt_plus.prediction.candidate_retrieval import CandidateRetrieval
from mlx_cpt_plus.prediction.similarity import Similarity
from mlx_cpt_plus.prediction.voting import Voting
from mlx_cpt_plus.prediction.ranking import Ranking
from mlx_cpt_plus.exceptions import PredictionError


class Predictor:
    """Main prediction engine.
    
    Combines candidate retrieval, similarity, voting, and ranking
    to provide next-item predictions.
    """
    
    def __init__(
        self,
        candidate_retrieval: Optional[CandidateRetrieval] = None,
        similarity: Optional[Similarity] = None,
        voting: Optional[Voting] = None,
        ranking: Optional[Ranking] = None,
    ):
        """Initialize predictor.
        
        Args:
            candidate_retrieval: Candidate retrieval component.
            similarity: Similarity computation component.
            voting: Voting mechanism.
            ranking: Ranking algorithm.
        """
        self.candidate_retrieval = candidate_retrieval or CandidateRetrieval()
        self.similarity = similarity or Similarity()
        self.voting = voting or Voting()
        self.ranking = ranking or Ranking()
    
    def predict(
        self,
        context: List[int],
        k: int = 10,
        **kwargs
    ) -> List[Tuple[int, float]]:
        """Predict next items given context.
        
        Args:
            context: Context items.
            k: Number of predictions to return.
            **kwargs: Additional arguments.
            
        Returns:
            List of (item, score) tuples.
        """
        if not context:
            raise PredictionError("Context cannot be empty")
        
        # Retrieve candidates
        candidates = self.candidate_retrieval.retrieve(context, k * 10)
        
        # Compute scores (simplified)
        scored = [(item, float(self.candidate_retrieval.index.get_frequency(item))) 
                  for item in candidates]
        
        # Vote
        voted = self.voting.vote(scored)
        
        # Rank
        ranked = self.ranking.rank(voted, context, **kwargs)
        
        return ranked[:k]
    
    def predict_batch(
        self,
        contexts: List[List[int]],
        k: int = 10,
        **kwargs
    ) -> List[List[Tuple[int, float]]]:
        """Predict for multiple contexts.
        
        Args:
            contexts: List of context sequences.
            k: Number of predictions per context.
            **kwargs: Additional arguments.
            
        Returns:
            List of prediction lists.
        """
        return [self.predict(ctx, k, **kwargs) for ctx in contexts]
    
    def __repr__(self) -> str:
        return f"Predictor(method={self.ranking.method!r})"