"""
RAG System Retrieval Module
Provides various retrieval methods for document search
"""

import logging

logger = logging.getLogger(__name__)

# Always available
from .simple_vector_retriever import VectorRetriever

# Kiwi 사용 가능 여부 확인
KIWI_AVAILABLE = False
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    pass

# BM25 Retriever 선택 (Kiwi 사용 가능하면 KiwiBM25Retriever 사용)
BM25_AVAILABLE = False
if KIWI_AVAILABLE:
    try:
        from .kiwi_bm25_retriever import KiwiBM25Retriever as BM25Retriever
        BM25_AVAILABLE = True
        logger.info("Using KiwiBM25Retriever for enhanced Korean BM25 search")
    except ImportError as e:
        logger.warning(f"KiwiBM25Retriever not available: {e}")
        # Fallback to basic BM25
        try:
            from .bm25_retriever import BM25Retriever
            BM25_AVAILABLE = True
            logger.info("Using basic BM25Retriever")
        except ImportError as e:
            logger.warning(f"BM25Retriever not available: {e}")
            BM25Retriever = None
else:
    # Kiwi 없으면 기본 BM25 사용
    try:
        from .bm25_retriever import BM25Retriever
        BM25_AVAILABLE = True
        logger.info("Using basic BM25Retriever (Kiwi not available)")
    except ImportError as e:
        logger.warning(f"BM25Retriever not available: {e}")
        BM25Retriever = None

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