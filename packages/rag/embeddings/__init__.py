"""
RAG System Embedding Module
Provides various embedding models for text vectorization
"""

from .base_embedder import BaseEmbedder
from .kure_embedder import KUREEmbedder, EmbeddingGenerator, TextEmbedder
from .e5_embedder import E5Embedder

# Default embedder
DefaultEmbedder = KUREEmbedder

__all__ = [
    'BaseEmbedder',
    'KUREEmbedder', 
    'E5Embedder',
    'EmbeddingGenerator',  # Backward compatibility
    'TextEmbedder',  # Backward compatibility
    'DefaultEmbedder'
]