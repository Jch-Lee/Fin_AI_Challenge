"""
Qwen3-Reranker-4B implementation for Korean financial security questions.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional, Tuple
import logging
from functools import lru_cache
import warnings
from .base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class Qwen3Reranker(BaseReranker):
    """
    Qwen3-Reranker-4B implementation using transformers library.
    
    This reranker is optimized for Korean financial security domain questions
    and uses FP16 precision for memory efficiency.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        device: Optional[str] = None,
        precision: str = "fp16",
        batch_size: int = 8,
        max_length: int = 512,
        cache_enabled: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Qwen3-Reranker-4B.
        
        Args:
            model_name: Hugging Face model name or path
            device: Device to use ('cuda', 'cpu', or None for auto)
            precision: Model precision ('fp16', 'fp32', 'bf16')
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            cache_enabled: Whether to enable result caching
            config: Additional configuration
        """
        super().__init__(config)
        
        self.model_name = model_name
        self.precision = precision
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_enabled = cache_enabled
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Set dtype based on precision
        self.dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        self.dtype = self.dtype_map.get(precision, torch.float16)
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize cache if enabled
        if self.cache_enabled:
            self._init_cache()
            
        logger.info(f"Initialized {self.__class__.__name__} on {self.device} with {precision} precision")
    
    def _load_model(self):
        """Load Qwen3-Reranker-4B model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate dtype
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            # Only use dtype for GPU
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = self.dtype
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if self.device != "cuda" or "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
            
            # Set to eval mode
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load Qwen3-Reranker-4B: {e}")
    
    def _init_cache(self):
        """Initialize LRU cache for query-document pairs."""
        # Create cached version of compute_scores
        self._cached_compute_scores = lru_cache(maxsize=10000)(self._compute_scores_impl)
    
    def _prepare_inputs(
        self,
        queries: List[str],
        documents: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model.
        
        Args:
            queries: List of queries
            documents: List of documents
            
        Returns:
            Dictionary of tokenized inputs
        """
        # Create query-document pairs
        pairs = [[q, d] for q, d in zip(queries, documents)]
        
        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
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
        if self.cache_enabled:
            # Use cached version
            scores = []
            for doc in documents:
                score = self._cached_compute_scores(query, doc)
                scores.append(score)
            return scores
        else:
            return self._compute_scores_batch(query, documents)
    
    def _compute_scores_impl(self, query: str, document: str) -> float:
        """
        Compute score for a single query-document pair (for caching).
        
        Args:
            query: The search query
            document: Document text
            
        Returns:
            Relevance score
        """
        scores = self._compute_scores_batch(query, [document])
        return scores[0] if scores else 0.0
    
    def _compute_scores_batch(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Compute scores for a batch of query-document pairs.
        
        Args:
            query: The search query
            documents: List of document texts
            
        Returns:
            List of relevance scores
        """
        if not documents:
            return []
        
        scores = []
        
        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            batch_queries = [query] * len(batch_docs)
            
            # Prepare inputs
            inputs = self._prepare_inputs(batch_queries, batch_docs)
            
            # Compute scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get logits and convert to scores
                logits = outputs.logits
                
                # For binary classification, use the positive class score
                if logits.shape[-1] == 2:
                    batch_scores = F.softmax(logits, dim=-1)[:, 1]
                else:
                    # For single score output
                    batch_scores = torch.sigmoid(logits.squeeze(-1))
                
                batch_scores = batch_scores.cpu().tolist()
                
                # Handle single score case
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                
                scores.extend(batch_scores)
        
        return scores
    
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
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents with updated scores
        """
        if not documents:
            return []
        
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Extract document texts and preprocess
        doc_texts = []
        for doc in documents:
            text = doc.get("content", doc.get("text", ""))
            processed_text = self.preprocess_document(text)
            doc_texts.append(processed_text)
        
        # Compute reranking scores
        rerank_scores = self.compute_scores(processed_query, doc_texts)
        
        # Get original scores if available
        original_scores = []
        for doc in documents:
            original_score = doc.get("score", 0.0)
            original_scores.append(original_score)
        
        # Combine scores if original scores exist
        if any(s > 0 for s in original_scores):
            final_scores = self.combine_scores(original_scores, rerank_scores)
        else:
            final_scores = rerank_scores
        
        # Create result documents with updated scores
        results = []
        for doc, score, rerank_score in zip(documents, final_scores, rerank_scores):
            result = doc.copy()
            result["score"] = score
            result["rerank_score"] = rerank_score
            if "original_score" not in result and doc.get("score"):
                result["original_score"] = doc.get("score")
            results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-k
        return results[:top_k]
    
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
        results = []
        for query, documents in zip(queries, documents_batch):
            reranked = self.rerank(query, documents, top_k)
            results.append(reranked)
        return results
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess Korean financial query.
        
        Args:
            query: Raw query text
            
        Returns:
            Preprocessed query text
        """
        # Remove excessive whitespace
        query = " ".join(query.split())
        
        # Korean financial domain specific preprocessing can be added here
        # For now, just basic cleaning
        query = query.strip()
        
        return query
    
    def preprocess_document(self, document: str) -> str:
        """
        Preprocess Korean financial document.
        
        Args:
            document: Raw document text
            
        Returns:
            Preprocessed document text
        """
        # Remove excessive whitespace
        document = " ".join(document.split())
        
        # Truncate if too long
        if len(document) > self.max_length * 5:
            # Keep beginning and end for context
            mid_point = self.max_length * 5 // 2
            document = document[:mid_point] + " ... " + document[-mid_point:]
        
        return document.strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get reranker statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = super().get_stats()
        stats.update({
            "model_name": self.model_name,
            "precision": self.precision,
            "cache_enabled": self.cache_enabled,
        })
        
        if self.cache_enabled and hasattr(self, "_cached_compute_scores"):
            cache_info = self._cached_compute_scores.cache_info()
            stats["cache_stats"] = {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "size": cache_info.currsize,
                "maxsize": cache_info.maxsize,
            }
        
        return stats
    
    def clear_cache(self):
        """Clear the score cache if enabled."""
        if self.cache_enabled and hasattr(self, "_cached_compute_scores"):
            self._cached_compute_scores.cache_clear()
            logger.info("Cache cleared")
    
    def warmup(self):
        """Warm up the model with a dummy input."""
        try:
            dummy_query = "테스트 질문"
            dummy_doc = "테스트 문서"
            _ = self.compute_scores(dummy_query, [dummy_doc])
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def __repr__(self) -> str:
        """String representation of the reranker."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"precision={self.precision}, "
            f"batch_size={self.batch_size})"
        )