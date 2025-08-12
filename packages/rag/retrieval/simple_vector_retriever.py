"""
Simple Vector Retriever
Basic vector search without BM25 dependencies
"""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Search result data class"""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str = "dense"


class VectorRetriever:
    """
    Simple vector-based retriever using dense embeddings
    """
    
    def __init__(self, knowledge_base=None):
        """
        Initialize the vector retriever
        
        Args:
            knowledge_base: Knowledge base instance with FAISS index
        """
        self.knowledge_base = knowledge_base
        logger.info("VectorRetriever initialized")
    
    def retrieve(self, 
                 query: str,
                 query_embedding: np.ndarray,
                 top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents using vector similarity
        
        Args:
            query: Query text (for compatibility)
            query_embedding: Query embedding vector
            top_k: Number of documents to retrieve
        
        Returns:
            List of retrieved documents with scores
        """
        if self.knowledge_base is None:
            logger.warning("No knowledge base available")
            return []
        
        # Search using FAISS index with top_k parameter
        results = self.knowledge_base.search(query_embedding, top_k=top_k)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'doc_id': result.get('id', ''),
                'content': result.get('content', ''),
                'score': result.get('score', 0.0),
                'metadata': result.get('metadata', {}),
                'retrieval_method': 'dense'
            })
        
        return formatted_results
    
    def update_index(self, texts: List[str], documents: List[Dict]):
        """
        Update the retriever index (no-op for vector retriever)
        
        Args:
            texts: List of document texts
            documents: List of document dictionaries
        """
        # Vector retriever uses pre-computed embeddings in knowledge base
        pass