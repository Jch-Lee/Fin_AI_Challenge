"""
RAG (Retrieval-Augmented Generation) System
Unified module for document retrieval and augmented generation
"""

# Core components
from .knowledge_base import KnowledgeBase
from .rag_pipeline import RAGPipeline, create_rag_pipeline

# Embeddings
from .embeddings import (
    BaseEmbedder,
    KUREEmbedder,
    # E5Embedder,  # 더 이상 사용하지 않음
    EmbeddingGenerator,  # Backward compatibility
    TextEmbedder  # Backward compatibility
)

# Retrieval
from .retrieval import (
    VectorRetriever,
    BM25Retriever,
    HybridRetriever
)

__all__ = [
    # Core
    'KnowledgeBase',
    'RAGPipeline',
    'create_rag_pipeline',
    
    # Embeddings
    'BaseEmbedder',
    'KUREEmbedder',
    # 'E5Embedder',  # 더 이상 사용하지 않음
    'EmbeddingGenerator',
    'TextEmbedder',
    
    # Retrieval
    'VectorRetriever',
    'BM25Retriever',
    'HybridRetriever'
]