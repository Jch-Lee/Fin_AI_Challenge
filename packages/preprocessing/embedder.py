"""
Backward compatibility layer for embedder module
The embedder has been moved to packages.rag.embeddings

This file provides import redirects to maintain compatibility
"""
import warnings

# Always issue deprecation warning when this module is imported
def _issue_warning():
    warnings.warn(
        "Importing from packages.preprocessing.embedder is deprecated. "
        "Please use packages.rag.embeddings instead.",
        DeprecationWarning,
        stacklevel=3
    )

_issue_warning()

# Import from new location
from ..rag.embeddings import KUREEmbedder as _KUREEmbedder
from ..rag.embeddings import E5Embedder

# Create backward-compatible wrapper
class EmbeddingGenerator(_KUREEmbedder):
    """Backward compatibility wrapper for KUREEmbedder"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def generate_embedding(self, text):
        """Old method name for backward compatibility"""
        return self.embed(text, is_query=False)

# Aliases for compatibility
KUREEmbedder = EmbeddingGenerator
TextEmbedder = EmbeddingGenerator

# For backward compatibility with HybridEmbedding
class HybridEmbedding:
    """
    Backward compatibility wrapper for HybridEmbedding
    Redirects to the new hybrid retriever system
    """
    def __init__(self, dense_model_name: str = "nlpai-lab/KURE-v1", use_bm25: bool = True):
        warnings.warn(
            "HybridEmbedding is deprecated. Use packages.rag.retrieval.HybridRetriever instead.",
            DeprecationWarning
        )
        self.dense_generator = KUREEmbedder(model_name=dense_model_name)
        self.use_bm25 = use_bm25

# Export all for compatibility
__all__ = [
    'EmbeddingGenerator',
    'TextEmbedder', 
    'KUREEmbedder',
    'E5Embedder',
    'HybridEmbedding'
]