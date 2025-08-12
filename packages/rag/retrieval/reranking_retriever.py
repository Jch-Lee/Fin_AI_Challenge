"""
Reranking retriever that integrates reranking with existing retrievers.
"""

from typing import List, Dict, Any, Optional, Union
import logging
from ..reranking import (
    BaseReranker,
    Qwen3Reranker,
    RerankerConfig,
    get_default_config,
    QueryCache,
)
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class RerankingRetriever(BaseRetriever):
    """
    Retriever that adds reranking capability to any base retriever.
    
    This class wraps an existing retriever and adds a reranking step
    to improve retrieval quality for Korean financial questions.
    """
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: Optional[BaseReranker] = None,
        reranker_config: Optional[Union[RerankerConfig, Dict[str, Any]]] = None,
        expand_factor: int = 3,
        use_cache: bool = True
    ):
        """
        Initialize reranking retriever.
        
        Args:
            base_retriever: The base retriever to wrap
            reranker: Optional reranker instance
            reranker_config: Configuration for reranker
            expand_factor: Factor to expand initial retrieval (e.g., retrieve 3x then rerank to x)
            use_cache: Whether to use query-level caching
        """
        super().__init__()
        
        self.base_retriever = base_retriever
        self.expand_factor = expand_factor
        self.use_cache = use_cache
        
        # Initialize reranker if not provided
        if reranker is None:
            if reranker_config is None:
                reranker_config = get_default_config("qwen3")
            elif isinstance(reranker_config, dict):
                reranker_config = RerankerConfig.from_dict(reranker_config)
            
            self.reranker = Qwen3Reranker(
                model_name=reranker_config.model_name,
                device=reranker_config.device,
                precision=reranker_config.precision,
                batch_size=reranker_config.batch_size,
                max_length=reranker_config.max_length,
                cache_enabled=reranker_config.cache_enabled,
                config=reranker_config.to_dict()
            )
            self.reranker_config = reranker_config
        else:
            self.reranker = reranker
            self.reranker_config = reranker_config or RerankerConfig()
        
        # Initialize query cache if enabled
        self.query_cache = QueryCache() if use_cache else None
        
        # Statistics
        self.stats = {
            "queries_processed": 0,
            "cache_hits": 0,
            "rerank_failures": 0,
        }
        
        logger.info(f"Initialized RerankingRetriever with {base_retriever.__class__.__name__}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank documents.
        
        Args:
            query: Search query
            top_k: Number of documents to return after reranking
            filters: Optional filters for base retriever
            **kwargs: Additional arguments for base retriever
            
        Returns:
            List of reranked documents
        """
        self.stats["queries_processed"] += 1
        
        # Check query cache first
        if self.query_cache is not None:
            # Need to get initial documents first for cache key
            expanded_k = min(top_k * self.expand_factor, 100)
            initial_docs = self.base_retriever.retrieve(
                query, top_k=expanded_k, filters=filters, **kwargs
            )
            
            cached_results = self.query_cache.get(query, initial_docs, top_k)
            if cached_results is not None:
                self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_results
        else:
            # No cache, get initial documents
            expanded_k = min(top_k * self.expand_factor, 100)
            initial_docs = self.base_retriever.retrieve(
                query, top_k=expanded_k, filters=filters, **kwargs
            )
        
        # If no documents retrieved, return empty
        if not initial_docs:
            logger.warning(f"No documents retrieved for query: {query}")
            return []
        
        try:
            # Perform reranking
            reranked_docs = self.reranker.rerank(
                query=query,
                documents=initial_docs,
                top_k=top_k
            )
            
            # Store in cache if enabled
            if self.query_cache is not None:
                self.query_cache.put(query, initial_docs, reranked_docs, top_k)
            
            logger.debug(f"Reranked {len(initial_docs)} -> {len(reranked_docs)} documents")
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            self.stats["rerank_failures"] += 1
            
            # Fallback to initial retrieval
            if self.reranker_config.fallback_enabled:
                logger.info("Using fallback: returning initial retrieval results")
                return initial_docs[:top_k]
            else:
                raise
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch retrieval with reranking.
        
        Args:
            queries: List of queries
            top_k: Number of documents per query
            filters: Optional filters
            **kwargs: Additional arguments
            
        Returns:
            List of document lists
        """
        results = []
        
        # Process each query
        for query in queries:
            docs = self.retrieve(query, top_k, filters, **kwargs)
            results.append(docs)
        
        return results
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Alias for retrieve method.
        
        Args:
            query: Search query
            top_k: Number of results
            **kwargs: Additional arguments
            
        Returns:
            List of reranked documents
        """
        return self.retrieve(query, top_k, **kwargs)
    
    def update_index(self, documents: List[Dict[str, Any]]):
        """
        Update the base retriever's index.
        
        Args:
            documents: Documents to add to index
        """
        if hasattr(self.base_retriever, "update_index"):
            self.base_retriever.update_index(documents)
        else:
            logger.warning(f"{self.base_retriever.__class__.__name__} does not support index updates")
    
    def clear_cache(self):
        """Clear all caches (reranker cache and query cache)."""
        # Clear reranker cache
        if hasattr(self.reranker, "clear_cache"):
            self.reranker.clear_cache()
        
        # Clear query cache
        if self.query_cache is not None:
            self.query_cache = QueryCache()
        
        logger.info("All caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()
        
        # Add reranker stats
        if hasattr(self.reranker, "get_stats"):
            stats["reranker"] = self.reranker.get_stats()
        
        # Add base retriever stats
        if hasattr(self.base_retriever, "get_stats"):
            stats["base_retriever"] = self.base_retriever.get_stats()
        
        # Add cache stats
        if self.query_cache is not None:
            stats["query_cache"] = {
                "hits": self.query_cache.stats["hits"],
                "misses": self.query_cache.stats["misses"],
                "hit_rate": (
                    self.query_cache.stats["hits"] /
                    (self.query_cache.stats["hits"] + self.query_cache.stats["misses"])
                    if (self.query_cache.stats["hits"] + self.query_cache.stats["misses"]) > 0
                    else 0
                ),
            }
        
        return stats
    
    def set_reranker(self, reranker: BaseReranker):
        """
        Set a new reranker.
        
        Args:
            reranker: New reranker instance
        """
        self.reranker = reranker
        logger.info(f"Reranker updated to {reranker.__class__.__name__}")
    
    def enable_reranking(self):
        """Enable reranking."""
        self.reranker_config.fallback_enabled = False
        logger.info("Reranking enabled")
    
    def disable_reranking(self):
        """Disable reranking (pass-through mode)."""
        self.reranker_config.fallback_enabled = True
        logger.info("Reranking disabled (pass-through mode)")
    
    def warmup(self):
        """Warm up the reranker and base retriever."""
        # Warm up reranker
        if hasattr(self.reranker, "warmup"):
            self.reranker.warmup()
        
        # Warm up base retriever
        if hasattr(self.base_retriever, "warmup"):
            self.base_retriever.warmup()
        
        logger.info("Warmup completed")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RerankingRetriever("
            f"base={self.base_retriever.__class__.__name__}, "
            f"reranker={self.reranker.__class__.__name__}, "
            f"expand_factor={self.expand_factor})"
        )