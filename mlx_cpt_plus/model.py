"""Main CPT+ model class."""

from typing import List, Tuple, Optional
from mlx_cpt_plus.config import Config
from mlx_cpt_plus.core.vocabulary import Vocabulary
from mlx_cpt_plus.core.sequence_store import SequenceStore
from mlx_cpt_plus.core.inverted_index import InvertedIndex
from mlx_cpt_plus.tree.prediction_tree import PredictionTree
from mlx_cpt_plus.tree.tree_builder import TreeBuilder
from mlx_cpt_plus.prediction.predictor import Predictor
from mlx_cpt_plus.prediction.pnr import PNR
from mlx_cpt_plus.online.updater import Updater
from mlx_cpt_plus.storage.msgpack_store import MsgPackStore
from mlx_cpt_plus.exceptions import CPTPlusError


class CPTPlus:
    """Main CPT+ model class.
    
    A production-grade CPT+ implementation for next-item prediction
    on Apple Silicon using Python and MLX.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize CPT+ model.
        
        Args:
            config: Model configuration.
        """
        self.config = config or Config()
        
        # Core components
        self.vocabulary = Vocabulary(
            min_frequency=self.config.min_frequency,
            max_size=self.config.max_vocabulary_size
        )
        self.sequence_store = SequenceStore()
        self.inverted_index = InvertedIndex()
        self.prediction_tree = PredictionTree(
            max_depth=self.config.max_depth,
            branching_factor=self.config.branching_factor
        )
        self.tree_builder = TreeBuilder(
            max_depth=self.config.max_depth,
            branching_factor=self.config.branching_factor
        )
        
        # Prediction components
        self.predictor = Predictor()
        self.pnr = PNR()
        
        # Online learning
        self.updater = Updater(self.config)
        
        # Storage
        self.storage = MsgPackStore()
    
    def fit(self, sequences: List[List[int]]) -> "CPTPlus":
        """Fit the model on sequences.
        
        Args:
            sequences: Training sequences.
            
        Returns:
            self
        """
        # Build vocabulary
        for seq in sequences:
            for item in seq:
                self.vocabulary.add_token(str(item))
        
# Store sequences and update inverted index
        for seq_id, seq in enumerate(sequences):
            self.sequence_store.add(seq)
            self.inverted_index.add_batch(seq, seq_id)

        # Build prediction tree
        self.prediction_tree = self.tree_builder.build(sequences)

        # Set predictor index
        self.predictor.candidate_retrieval.index = self.inverted_index
        
        # Learn patterns
        self.pnr.learn_patterns(sequences)
        
        return self
    
    def predict(self, context: List[int], k: int = 10) -> List[Tuple[int, float]]:
        """Predict next items.
        
        Args:
            context: Context items.
            k: Number of predictions.
            
        Returns:
            List of (item, score) tuples.
        """
        return self.predictor.predict(context, k)
    
    def partial_fit(self, sequence: List[int]) -> None:
        """Incrementally update the model.
        
        Args:
            sequence: New sequence to learn.
        """
        self.updater.partial_fit(sequence)
        self.sequence_store.add(sequence)
    
    def save(self, path: str) -> None:
        """Save model to file.
        
        Args:
            path: File path.
        """
        self.storage.save(path)
    
    def load(self, path: str) -> None:
        """Load model from file.
        
        Args:
            path: File path.
        """
        self.storage.load(path)
    
    def __repr__(self) -> str:
        return f"CPTPlus(vocab={len(self.vocabulary)}, tree={len(self.prediction_tree)})"