"""
Base reranker abstract class for the RAG pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """
    Abstract base class for all reranker implementations.
    
    This class defines the interface that all rerankers must implement
    for integration with the RAG pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reranker with optional configuration.
        
        Args:
            config: Configuration dictionary for the reranker
        """
        self.config = config or {}
        self.device = self.config.get("device", "cuda")
        self.batch_size = self.config.get("batch_size", 8)
        self.max_length = self.config.get("max_length", 512)
        
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to rerank. Each document should have:
                - 'content': The text content
                - 'metadata': Optional metadata dictionary
                - 'score': Optional original retrieval score
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents with updated scores
        """
        pass
    
    @abstractmethod
    def batch_rerank(
        self,
        queries: List[str],
        documents_batch: List[List[Dict[str, Any]]],
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch reranking for multiple queries.
        
        Args:
            queries: List of queries
            documents_batch: List of document lists, one for each query
            top_k: Number of top documents to return for each query
            
        Returns:
            List of reranked document lists
        """
        pass
    
    def compute_scores(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Compute relevance scores for query-document pairs.
        
        Args:
            query: The search query
            documents: List of document texts
            
        Returns:
            List of relevance scores
        """
        raise NotImplementedError("Subclasses must implement compute_scores")
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the query before reranking.
        
        Args:
            query: Raw query text
            
        Returns:
            Preprocessed query text
        """
        # Default implementation - can be overridden
        return query.strip()
    
    def preprocess_document(self, document: str) -> str:
        """
        Preprocess a document before reranking.
        
        Args:
            document: Raw document text
            
        Returns:
            Preprocessed document text
        """
        # Default implementation - can be overridden
        if len(document) > self.max_length * 5:  # Rough character estimate
            # Truncate long documents
            return document[:self.max_length * 5]
        return document
    
    def combine_scores(
        self,
        original_scores: List[float],
        rerank_scores: List[float],
        alpha: float = 0.7
    ) -> List[float]:
        """
        Combine original retrieval scores with reranking scores.
        
        Args:
            original_scores: Original retrieval scores
            rerank_scores: Reranking scores
            alpha: Weight for reranking scores (1-alpha for original scores)
            
        Returns:
            Combined scores
        """
        if not original_scores:
            return rerank_scores
            
        # Normalize scores to [0, 1] range
        def normalize(scores):
            if not scores:
                return []
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return [0.5] * len(scores)
            return [(s - min_score) / (max_score - min_score) for s in scores]
        
        norm_original = normalize(original_scores)
        norm_rerank = normalize(rerank_scores)
        
        # Combine scores
        combined = [
            alpha * r + (1 - alpha) * o
            for o, r in zip(norm_original, norm_rerank)
        ]
        
        return combined
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get reranker statistics for monitoring.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }
    
    def __repr__(self) -> str:
        """String representation of the reranker."""
        return f"{self.__class__.__name__}(device={self.device}, batch_size={self.batch_size})"