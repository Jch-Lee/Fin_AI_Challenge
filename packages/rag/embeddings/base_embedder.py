"""
Base Embedder Interface for RAG System
Provides a consistent interface for all embedding models
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for all embedding models"""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the embedder
        
        Args:
            model_name: Name or path of the embedding model
            device: Device to use for computation ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device
        self.embedding_dim = None
    
    @abstractmethod
    def embed(self, text: Union[str, List[str]], is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for text
        
        Args:
            text: Single text or list of texts to embed
            is_query: Whether the text is a query (for models that distinguish)
        
        Returns:
            Embedding array (single vector or batch of vectors)
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32, is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            is_query: Whether the texts are queries
        
        Returns:
            Array of embeddings
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings"""
        pass
    
    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Document embedding vectors
        
        Returns:
            Similarity scores
        """
        # Assuming normalized embeddings, simple dot product gives cosine similarity
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
        return similarities