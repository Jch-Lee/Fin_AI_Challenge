"""
KURE-v1 Embedder for RAG System
Korean specialized embedding model implementation
"""
import torch
import numpy as np
from typing import List, Optional, Union, Dict
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm

from .base_embedder import BaseEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KUREEmbedder(BaseEmbedder):
    """KURE-v1 Korean embedding model implementation"""
    
    def __init__(self, 
                 model_name: str = "nlpai-lab/KURE-v1",
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 show_progress: bool = True):
        """
        Initialize KURE embedder
        
        Args:
            model_name: KURE model name (default: nlpai-lab/KURE-v1)
            device: Computation device ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
        """
        super().__init__(model_name, device)
        
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        # Set device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"KURE Embedder using device: {self.device}")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the KURE model with fallback options"""
        try:
            logger.info(f"Loading KURE model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"KURE model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load KURE model: {e}")
            # Fallback to Korean alternative models
            logger.info("Falling back to Korean multilingual model")
            self.model_name = "jhgan/ko-sroberta-multitask"
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            except Exception:
                # Final fallback
                logger.info("Falling back to base multilingual model")
                self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed(self, text: Union[str, List[str]], is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for text
        
        Args:
            text: Single text or list of texts to embed
            is_query: Not used for KURE (kept for interface compatibility)
        
        Returns:
            Embedding array
        """
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(
            text,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        return embeddings if len(text) > 1 else embeddings[0]
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            is_query: Not used for KURE (kept for interface compatibility)
        
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Update batch size if specified
        original_batch_size = self.batch_size
        self.batch_size = batch_size
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Restore original batch size
        self.batch_size = original_batch_size
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings"""
        return self.embedding_dim
    
    def generate_chunk_embeddings(self, 
                                 chunks: List[Dict],
                                 content_key: str = 'content') -> List[Dict]:
        """
        Generate embeddings for chunks and add them to the chunk dictionaries
        
        Args:
            chunks: List of chunk dictionaries
            content_key: Key for the text content in chunks
        
        Returns:
            Chunks with added embeddings
        """
        texts = [chunk[content_key] for chunk in chunks]
        embeddings = self.embed_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        return chunks
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to file"""
        np.save(filepath, embeddings)
        logger.info(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file"""
        embeddings = np.load(filepath)
        logger.info(f"Loaded embeddings from {filepath}")
        return embeddings
    
    def save_model_cache(self, cache_dir: str):
        """Save model cache for offline use"""
        self.model.save(cache_dir)
        logger.info(f"Saved model cache to {cache_dir}")
    
    @staticmethod
    def load_from_cache(cache_dir: str, device: Optional[str] = None):
        """Load model from cache"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = SentenceTransformer(cache_dir, device=device)
        embedder = KUREEmbedder.__new__(KUREEmbedder)
        embedder.model = model
        embedder.device = device
        embedder.embedding_dim = model.get_sentence_embedding_dimension()
        embedder.batch_size = 32
        embedder.show_progress = True
        
        return embedder


# Backward compatibility aliases
EmbeddingGenerator = KUREEmbedder
TextEmbedder = KUREEmbedder