"""
Base retriever abstract class for the RAG pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """
    Abstract base class for all retriever implementations.
    
    This class defines the interface that all retrievers must implement
    for integration with the RAG pipeline.
    """
    
    def __init__(self):
        """Initialize the base retriever."""
        pass
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on a query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            filters: Optional filters to apply
            **kwargs: Additional retriever-specific arguments
            
        Returns:
            List of retrieved documents with scores and metadata
        """
        pass
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Alias for retrieve method.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of retrieved documents
        """
        return self.retrieve(query, top_k, **kwargs)
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch retrieval for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of documents per query
            filters: Optional filters to apply
            **kwargs: Additional retriever-specific arguments
            
        Returns:
            List of document lists, one for each query
        """
        results = []
        for query in queries:
            docs = self.retrieve(query, top_k, filters, **kwargs)
            results.append(docs)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {}
    
    def __repr__(self) -> str:
        """String representation of the retriever."""
        return f"{self.__class__.__name__}()"