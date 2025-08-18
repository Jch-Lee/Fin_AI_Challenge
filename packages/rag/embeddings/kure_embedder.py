"""
KURE-v1 Embedder for RAG System
Korean specialized embedding model implementation
"""
import torch
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
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
    
    def compute_similarity_matrix(self, 
                                 embeddings1: Union[np.ndarray, torch.Tensor],
                                 embeddings2: Optional[Union[np.ndarray, torch.Tensor]] = None) -> np.ndarray:
        """
        Compute similarity matrix using KURE-v1's native similarity method
        Optimized for GPU acceleration and large-scale processing
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings (if None, computes self-similarity)
        
        Returns:
            Similarity matrix as numpy array
        """
        # Convert to torch tensors if needed
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1).to(self.device)
        else:
            embeddings1 = embeddings1.to(self.device)
            
        if embeddings2 is None:
            embeddings2 = embeddings1
        elif isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2).to(self.device)
        else:
            embeddings2 = embeddings2.to(self.device)
        
        # Check if model has similarity method (newer versions)
        if hasattr(self.model, 'similarity'):
            # Use KURE's optimized similarity computation
            with torch.no_grad():
                similarities = self.model.similarity(embeddings1, embeddings2)
            return similarities.cpu().numpy()
        else:
            # Fallback to manual computation for older versions
            # Normalize embeddings if not already normalized
            embeddings1_norm = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
            embeddings2_norm = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
            
            # Compute cosine similarity using matrix multiplication
            with torch.no_grad():
                similarities = torch.mm(embeddings1_norm, embeddings2_norm.t())
            
            return similarities.cpu().numpy()
    
    def batch_similarity_search(self,
                               query_embeddings: Union[np.ndarray, torch.Tensor],
                               document_embeddings: Union[np.ndarray, torch.Tensor],
                               top_k: int = 5,
                               batch_size: int = 100) -> List[List[Tuple[int, float]]]:
        """
        Perform batch similarity search for multiple queries
        Optimized for large-scale retrieval
        
        Args:
            query_embeddings: Query embeddings (shape: [n_queries, dim])
            document_embeddings: Document embeddings (shape: [n_docs, dim])
            top_k: Number of top documents to retrieve per query
            batch_size: Batch size for processing large document sets
        
        Returns:
            List of top-k results for each query [(doc_idx, score), ...]
        """
        results = []
        n_queries = query_embeddings.shape[0]
        n_docs = document_embeddings.shape[0]
        
        # Convert to tensors
        if isinstance(query_embeddings, np.ndarray):
            query_embeddings = torch.from_numpy(query_embeddings)
        if isinstance(document_embeddings, np.ndarray):
            document_embeddings = torch.from_numpy(document_embeddings)
        
        # Process in batches to manage memory
        for q_start in range(0, n_queries, batch_size):
            q_end = min(q_start + batch_size, n_queries)
            query_batch = query_embeddings[q_start:q_end].to(self.device)
            
            batch_results = []
            
            # Process documents in chunks if needed
            if n_docs > 10000:  # Large document set
                chunk_size = 5000
                all_scores = []
                
                for d_start in range(0, n_docs, chunk_size):
                    d_end = min(d_start + chunk_size, n_docs)
                    doc_chunk = document_embeddings[d_start:d_end].to(self.device)
                    
                    with torch.no_grad():
                        if hasattr(self.model, 'similarity'):
                            chunk_scores = self.model.similarity(query_batch, doc_chunk)
                        else:
                            # Fallback to manual computation
                            query_norm = torch.nn.functional.normalize(query_batch, p=2, dim=1)
                            doc_norm = torch.nn.functional.normalize(doc_chunk, p=2, dim=1)
                            chunk_scores = torch.mm(query_norm, doc_norm.t())
                    
                    # Adjust indices for chunk offset
                    chunk_scores_with_offset = torch.zeros_like(chunk_scores)
                    chunk_scores_with_offset[:, :] = chunk_scores
                    all_scores.append((chunk_scores.cpu(), d_start))
                
                # Combine scores from all chunks
                for q_idx in range(query_batch.shape[0]):
                    query_scores = []
                    for chunk_scores, offset in all_scores:
                        for d_idx, score in enumerate(chunk_scores[q_idx]):
                            query_scores.append((offset + d_idx, score.item()))
                    
                    # Sort and get top-k
                    query_scores.sort(key=lambda x: x[1], reverse=True)
                    batch_results.append(query_scores[:top_k])
            else:
                # Process all documents at once for smaller sets
                doc_tensor = document_embeddings.to(self.device)
                
                with torch.no_grad():
                    if hasattr(self.model, 'similarity'):
                        similarities = self.model.similarity(query_batch, doc_tensor)
                    else:
                        # Fallback to manual computation
                        query_norm = torch.nn.functional.normalize(query_batch, p=2, dim=1)
                        doc_norm = torch.nn.functional.normalize(doc_tensor, p=2, dim=1)
                        similarities = torch.mm(query_norm, doc_norm.t())
                
                # Get top-k for each query
                topk_scores, topk_indices = torch.topk(similarities, min(top_k, n_docs), dim=1)
                
                for q_idx in range(query_batch.shape[0]):
                    query_results = []
                    for k in range(min(top_k, n_docs)):
                        doc_idx = topk_indices[q_idx, k].item()
                        score = topk_scores[q_idx, k].item()
                        query_results.append((doc_idx, score))
                    batch_results.append(query_results)
            
            results.extend(batch_results)
        
        return results
    
    def compute_similarity_scores(self,
                                 query_embedding: Union[np.ndarray, torch.Tensor],
                                 doc_embeddings: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Compute similarity scores between a single query and multiple documents
        
        Args:
            query_embedding: Single query embedding (1D or 2D with shape [1, dim])
            doc_embeddings: Document embeddings (shape: [n_docs, dim])
        
        Returns:
            Similarity scores as numpy array
        """
        # Reshape query if needed
        if isinstance(query_embedding, np.ndarray):
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_tensor = torch.from_numpy(query_embedding).to(self.device)
        else:
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.unsqueeze(0)
            query_tensor = query_embedding.to(self.device)
        
        # Convert documents to tensor
        if isinstance(doc_embeddings, np.ndarray):
            doc_tensor = torch.from_numpy(doc_embeddings).to(self.device)
        else:
            doc_tensor = doc_embeddings.to(self.device)
        
        # Compute similarities
        with torch.no_grad():
            if hasattr(self.model, 'similarity'):
                similarities = self.model.similarity(query_tensor, doc_tensor)
            else:
                # Fallback to manual computation
                query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
                doc_norm = torch.nn.functional.normalize(doc_tensor, p=2, dim=1)
                similarities = torch.mm(query_norm, doc_norm.t())
        
        return similarities.squeeze().cpu().numpy()
    
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