"""
Caching layer for reranker performance optimization.
"""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import logging
import pickle

logger = logging.getLogger(__name__)


class RerankerCache:
    """
    LRU cache with TTL for reranker results.
    
    This cache stores query-document pairs and their scores to avoid
    redundant computations, especially important for Korean financial
    questions that may be repeated or similar.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl: int = 3600,
        enable_stats: bool = True
    ):
        """
        Initialize the reranker cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl: Time-to-live for cache entries in seconds
            enable_stats: Whether to track cache statistics
        """
        self.max_size = max_size
        self.ttl = ttl
        self.enable_stats = enable_stats
        
        # LRU cache using OrderedDict
        self.cache: OrderedDict = OrderedDict()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
        }
        
        logger.info(f"Initialized cache with max_size={max_size}, ttl={ttl}s")
    
    def _generate_key(self, query: str, document: str) -> str:
        """
        Generate a cache key for query-document pair.
        
        Args:
            query: The search query
            document: The document text
            
        Returns:
            Cache key string
        """
        # Normalize texts for better cache hits
        normalized_query = self._normalize_text(query)
        normalized_doc = self._normalize_text(document[:1000])  # Use first 1000 chars
        
        # Create hash
        content = f"{normalized_query}||{normalized_doc}"
        key = hashlib.md5(content.encode("utf-8")).hexdigest()
        
        return key
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for cache key generation.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        # Convert to lowercase for Korean and English
        text = text.lower()
        # Remove punctuation at the end
        text = text.rstrip(".,!?;:")
        
        return text
    
    def get(self, query: str, document: str) -> Optional[float]:
        """
        Get cached score for query-document pair.
        
        Args:
            query: The search query
            document: The document text
            
        Returns:
            Cached score or None if not found/expired
        """
        key = self._generate_key(query, document)
        
        if key not in self.cache:
            if self.enable_stats:
                self.stats["misses"] += 1
            return None
        
        # Check if entry is expired
        entry = self.cache[key]
        if time.time() - entry["timestamp"] > self.ttl:
            # Remove expired entry
            del self.cache[key]
            if self.enable_stats:
                self.stats["expired"] += 1
                self.stats["misses"] += 1
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        
        if self.enable_stats:
            self.stats["hits"] += 1
        
        return entry["score"]
    
    def put(self, query: str, document: str, score: float):
        """
        Store score in cache.
        
        Args:
            query: The search query
            document: The document text
            score: The relevance score
        """
        key = self._generate_key(query, document)
        
        # Check if we need to evict
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            self.cache.popitem(last=False)
            if self.enable_stats:
                self.stats["evictions"] += 1
        
        # Add new entry
        self.cache[key] = {
            "score": score,
            "timestamp": time.time(),
            "query_preview": query[:50],  # For debugging
        }
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
    
    def get_batch(
        self,
        query: str,
        documents: List[str]
    ) -> Tuple[List[Optional[float]], List[int]]:
        """
        Get cached scores for multiple documents.
        
        Args:
            query: The search query
            documents: List of document texts
            
        Returns:
            Tuple of (scores, miss_indices)
            scores: List of scores (None for cache misses)
            miss_indices: Indices of documents that were cache misses
        """
        scores = []
        miss_indices = []
        
        for i, doc in enumerate(documents):
            score = self.get(query, doc)
            scores.append(score)
            if score is None:
                miss_indices.append(i)
        
        return scores, miss_indices
    
    def put_batch(
        self,
        query: str,
        documents: List[str],
        scores: List[float],
        indices: Optional[List[int]] = None
    ):
        """
        Store multiple scores in cache.
        
        Args:
            query: The search query
            documents: List of document texts
            scores: List of relevance scores
            indices: Optional indices to specify which documents to cache
        """
        if indices is None:
            indices = range(len(documents))
        
        for idx, score in zip(indices, scores):
            if idx < len(documents):
                self.put(query, documents[idx], score)
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self.stats["evictions"],
            "expired": self.stats["expired"],
            "ttl": self.ttl,
        }
    
    def reset_stats(self):
        """Reset cache statistics."""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
        }
    
    def save(self, filepath: str):
        """
        Save cache to disk.
        
        Args:
            filepath: Path to save cache file
        """
        try:
            data = {
                "cache": dict(self.cache),
                "stats": self.stats,
                "config": {
                    "max_size": self.max_size,
                    "ttl": self.ttl,
                }
            }
            
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            
            logger.info(f"Cache saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load(self, filepath: str):
        """
        Load cache from disk.
        
        Args:
            filepath: Path to cache file
        """
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            
            # Restore cache
            self.cache = OrderedDict(data["cache"])
            
            # Restore stats if available
            if "stats" in data:
                self.stats = data["stats"]
            
            # Remove expired entries
            current_time = time.time()
            expired_keys = []
            for key, entry in self.cache.items():
                if current_time - entry["timestamp"] > self.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                if self.enable_stats:
                    self.stats["expired"] += 1
            
            logger.info(f"Cache loaded from {filepath} ({len(self.cache)} entries)")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def __len__(self) -> int:
        """Get number of entries in cache."""
        return len(self.cache)
    
    def __repr__(self) -> str:
        """String representation of cache."""
        hit_rate = 0
        if self.enable_stats:
            total = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total if total > 0 else 0
        
        return (
            f"RerankerCache(size={len(self.cache)}/{self.max_size}, "
            f"hit_rate={hit_rate:.2%}, ttl={self.ttl}s)"
        )


class QueryCache:
    """
    Cache for complete query results (multiple documents).
    
    This is useful for caching entire reranking results for a query,
    not just individual query-document pairs.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600
    ):
        """
        Initialize query cache.
        
        Args:
            max_size: Maximum number of queries to cache
            ttl: Time-to-live in seconds
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        
        self.stats = {
            "hits": 0,
            "misses": 0,
        }
    
    def _generate_key(
        self,
        query: str,
        doc_hashes: List[str],
        top_k: int
    ) -> str:
        """Generate cache key for query and document set."""
        # Include document hashes to ensure cache validity
        content = {
            "query": query.lower().strip(),
            "docs": sorted(doc_hashes),  # Sort for consistency
            "top_k": top_k,
        }
        
        key = hashlib.md5(
            json.dumps(content, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        
        return key
    
    def get(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached reranking result.
        
        Args:
            query: The search query
            documents: Original documents
            top_k: Number of top documents
            
        Returns:
            Cached reranked documents or None
        """
        # Generate document hashes
        doc_hashes = [
            hashlib.md5(
                doc.get("content", doc.get("text", ""))[:1000].encode("utf-8")
            ).hexdigest()
            for doc in documents
        ]
        
        key = self._generate_key(query, doc_hashes, top_k)
        
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if time.time() - entry["timestamp"] > self.ttl:
            del self.cache[key]
            self.stats["misses"] += 1
            return None
        
        # Move to end
        self.cache.move_to_end(key)
        self.stats["hits"] += 1
        
        return entry["results"]
    
    def put(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        top_k: int
    ):
        """Store reranking result in cache."""
        # Generate document hashes
        doc_hashes = [
            hashlib.md5(
                doc.get("content", doc.get("text", ""))[:1000].encode("utf-8")
            ).hexdigest()
            for doc in documents
        ]
        
        key = self._generate_key(query, doc_hashes, top_k)
        
        # Evict if necessary
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        # Store result
        self.cache[key] = {
            "results": results,
            "timestamp": time.time(),
        }
        
        self.cache.move_to_end(key)