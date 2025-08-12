"""
RAG System Retrieval Module
Provides various retrieval methods for document search
"""

import logging

logger = logging.getLogger(__name__)

# Always available
from .simple_vector_retriever import VectorRetriever

# Optional imports with graceful handling
try:
    from .bm25_retriever import BM25Retriever
    BM25_AVAILABLE = True
except ImportError as e:
    logger.warning(f"BM25Retriever not available: {e}")
    BM25Retriever = None
    BM25_AVAILABLE = False

try:
    from .hybrid_retriever import HybridRetriever
    HYBRID_AVAILABLE = True
except ImportError as e:
    logger.warning(f"HybridRetriever not available: {e}")
    HybridRetriever = None
    HYBRID_AVAILABLE = False

# Default retriever (use vector if hybrid not available)
if HYBRID_AVAILABLE:
    DefaultRetriever = HybridRetriever
else:
    DefaultRetriever = VectorRetriever

# Export what's available
__all__ = ['VectorRetriever']

if BM25_AVAILABLE:
    __all__.append('BM25Retriever')
    
if HYBRID_AVAILABLE:
    __all__.append('HybridRetriever')

__all__.append('DefaultRetriever')