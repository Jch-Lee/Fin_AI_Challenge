"""
Unified RAG Pipeline
Central orchestrator for the complete RAG system
"""
import logging
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from .embeddings import KUREEmbedder, BaseEmbedder
from .retrieval import HybridRetriever, VectorRetriever, BM25Retriever
from .retrieval.reranking_retriever import RerankingRetriever
from .knowledge_base import KnowledgeBase
from .reranking import (
    BaseReranker,
    Qwen3Reranker,
    RerankerConfig,
    get_default_config,
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG Pipeline that orchestrates:
    1. Document embedding
    2. Knowledge base management
    3. Query processing
    4. Document retrieval
    5. Context preparation for generation
    """
    
    def __init__(self,
                 embedder: Optional[BaseEmbedder] = None,
                 retriever_type: str = "hybrid",
                 knowledge_base_path: Optional[str] = None,
                 device: Optional[str] = None,
                 enable_reranking: bool = False,
                 reranker_config: Optional[Union[RerankerConfig, Dict]] = None):
        """
        Initialize the RAG pipeline
        
        Args:
            embedder: Embedding model to use (default: KUREEmbedder)
            retriever_type: Type of retriever ("vector", "bm25", "hybrid")
            knowledge_base_path: Path to existing knowledge base
            device: Device for computation
            enable_reranking: Whether to enable reranking with Qwen3-Reranker-4B
            reranker_config: Configuration for the reranker
        """
        # Initialize embedder
        if embedder is None:
            logger.info("Using default KURE-v1 embedder")
            self.embedder = KUREEmbedder(device=device)
        else:
            self.embedder = embedder
        
        # Set retriever type first (needed for load_knowledge_base)
        self.retriever_type = retriever_type
        self.enable_reranking = enable_reranking
        self.reranker_config = reranker_config
        self.reranker = None
        
        # Initialize knowledge base
        if knowledge_base_path:
            # Load existing knowledge base
            self.load_knowledge_base(knowledge_base_path)
        else:
            # Create new knowledge base
            self.knowledge_base = KnowledgeBase(
                embedding_dim=self.embedder.get_embedding_dim()
            )
            # Initialize retriever with new knowledge base
            self._init_retriever()
        
        # Initialize reranker if enabled
        if self.enable_reranking:
            self._init_reranker()
        
        logger.info(f"RAG Pipeline initialized with {self.embedder.model_name} embedder and {retriever_type} retriever" + 
                   (" with reranking" if enable_reranking else ""))
    
    def _init_retriever(self):
        """Initialize the retriever based on type"""
        if self.retriever_type == "vector":
            base_retriever = VectorRetriever(self.knowledge_base)
        elif self.retriever_type == "bm25":
            if BM25Retriever is not None:
                base_retriever = BM25Retriever()
            else:
                logger.warning("BM25 not available, falling back to vector retriever")
                base_retriever = VectorRetriever(self.knowledge_base)
        elif self.retriever_type == "hybrid":
            if HybridRetriever is not None:
                # Try to create hybrid retriever with proper arguments
                try:
                    base_retriever = HybridRetriever()
                except Exception as e:
                    logger.warning(f"Failed to create hybrid retriever: {e}, falling back to vector")
                    base_retriever = VectorRetriever(self.knowledge_base)
            else:
                logger.warning("Hybrid retriever not available, falling back to vector retriever")
                base_retriever = VectorRetriever(self.knowledge_base)
        else:
            raise ValueError(f"Unknown retriever type: {self.retriever_type}")
        
        # Wrap with reranking retriever if enabled
        if self.enable_reranking and self.reranker:
            self.retriever = RerankingRetriever(
                base_retriever=base_retriever,
                reranker=self.reranker,
                reranker_config=self.reranker_config
            )
        else:
            self.retriever = base_retriever
    
    def _init_reranker(self):
        """Initialize the reranker"""
        try:
            # Use provided config or get default
            if self.reranker_config is None:
                self.reranker_config = get_default_config("qwen3")
            elif isinstance(self.reranker_config, dict):
                self.reranker_config = RerankerConfig.from_dict(self.reranker_config)
            
            # Create Qwen3 reranker
            self.reranker = Qwen3Reranker(
                model_name=self.reranker_config.model_name,
                device=self.reranker_config.device,
                precision=self.reranker_config.precision,
                batch_size=self.reranker_config.batch_size,
                max_length=self.reranker_config.max_length,
                cache_enabled=self.reranker_config.cache_enabled,
                config=self.reranker_config.to_dict()
            )
            
            logger.info(f"Initialized {self.reranker_config.model_name} reranker")
            
            # Update retriever if already initialized
            if hasattr(self, 'retriever'):
                self._init_retriever()
                
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            self.enable_reranking = False
            self.reranker = None
    
    def _update_retriever_kb(self):
        """Update retriever's knowledge base reference"""
        if hasattr(self.retriever, 'knowledge_base'):
            self.retriever.knowledge_base = self.knowledge_base
    
    def add_documents(self, 
                      texts: List[str],
                      metadata: Optional[List[Dict]] = None,
                      batch_size: int = 32) -> int:
        """
        Add documents to the knowledge base
        
        Args:
            texts: List of document texts
            metadata: Optional metadata for each document
            batch_size: Batch size for embedding generation
        
        Returns:
            Number of documents added
        """
        logger.info(f"Adding {len(texts)} documents to knowledge base")
        
        # Generate embeddings
        embeddings = self.embedder.embed_batch(texts, batch_size=batch_size)
        
        # Prepare documents
        documents = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            doc = {
                "id": i,
                "content": text,
                "embedding": embedding
            }
            if metadata and i < len(metadata):
                doc["metadata"] = metadata[i]
            documents.append(doc)
        
        # Add to knowledge base
        self.knowledge_base.add_documents(embeddings, documents)
        
        # Update retriever if needed
        if hasattr(self.retriever, 'update_index'):
            self.retriever.update_index(texts, documents)
        
        logger.info(f"Successfully added {len(documents)} documents")
        return len(documents)
    
    def retrieve(self,
                 query: str,
                 top_k: int = 5,
                 threshold: float = 0.0) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            threshold: Minimum similarity threshold
        
        Returns:
            List of retrieved documents with scores
        """
        logger.info(f"Retrieving documents for query: {query[:100]}...")
        
        # Generate query embedding
        query_embedding = self.embedder.embed(query, is_query=True)
        
        # Retrieve documents
        if self.retriever_type == "vector":
            results = self.knowledge_base.search(query_embedding, top_k=top_k)
        else:
            # HybridRetriever 또는 다른 retriever 사용
            try:
                # HybridRetriever의 search 메서드 사용
                if hasattr(self.retriever, 'search'):
                    # HybridRetriever는 search 메서드를 가지고 있음
                    hybrid_results = self.retriever.search(query, k=top_k)
                    # HybridResult를 일반 딕셔너리로 변환
                    results = []
                    for hr in hybrid_results:
                        results.append({
                            'content': hr.content if hasattr(hr, 'content') else '',
                            'score': hr.hybrid_score if hasattr(hr, 'hybrid_score') else 0.0,
                            'metadata': hr.metadata if hasattr(hr, 'metadata') else {}
                        })
                # 일반 retrieve 메서드 사용
                elif hasattr(self.retriever, 'retrieve'):
                    results = self.retriever.retrieve(query, query_embedding, top_k)
                else:
                    # 폴백: 벡터 검색 사용
                    logger.warning(f"Retriever {type(self.retriever)} has no retrieve/search method, using vector search")
                    results = self.knowledge_base.search(query_embedding, top_k=top_k)
            except Exception as e:
                logger.error(f"Retriever failed: {e}, falling back to vector search")
                results = self.knowledge_base.search(query_embedding, top_k=top_k)
        
        # Filter by threshold
        if threshold > 0:
            results = [r for r in results if r.get('score', 0) >= threshold]
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def generate_context(self,
                        query: str,
                        top_k: int = 5,
                        max_length: int = 2000) -> str:
        """
        Generate context for LLM generation
        
        Args:
            query: Query text
            top_k: Number of documents to include
            max_length: Maximum context length
        
        Returns:
            Formatted context string
        """
        # Retrieve relevant documents
        documents = self.retrieve(query, top_k)
        
        # Format context
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            content = doc.get('content', '')
            doc_text = f"[문서 {i+1}]\n{content}\n"
            
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        context = "\n".join(context_parts)
        return context
    
    def save_knowledge_base(self, path: str):
        """Save the knowledge base to disk"""
        self.knowledge_base.save(path)
        logger.info(f"Knowledge base saved to {path}")
    
    def load_knowledge_base(self, path: str):
        """Load knowledge base from disk"""
        # Load returns a new instance
        self.knowledge_base = KnowledgeBase.load(path)
        
        # Initialize or update retriever
        if hasattr(self, 'retriever'):
            # Update existing retriever's knowledge base reference
            self._update_retriever_kb()
        else:
            # Initialize retriever for the first time
            self._init_retriever()
        
        logger.info(f"Knowledge base loaded from {path}")
    
    def update_embedder(self, embedder: BaseEmbedder):
        """
        Update the embedding model
        
        Args:
            embedder: New embedding model
        """
        logger.info(f"Updating embedder from {self.embedder.model_name} to {embedder.model_name}")
        self.embedder = embedder
        
        # Re-initialize retriever if needed
        if hasattr(self.retriever, 'embedder'):
            self.retriever.embedder = embedder
    
    def enable_reranking(self, reranker_config: Optional[Union[RerankerConfig, Dict]] = None):
        """
        Enable reranking in the pipeline
        
        Args:
            reranker_config: Optional configuration for the reranker
        """
        self.enable_reranking = True
        if reranker_config:
            self.reranker_config = reranker_config
        
        # Initialize reranker if not already done
        if self.reranker is None:
            self._init_reranker()
        
        # Update retriever
        self._init_retriever()
        logger.info("Reranking enabled")
    
    def disable_reranking(self):
        """Disable reranking in the pipeline"""
        self.enable_reranking = False
        
        # Re-initialize retriever without reranking
        self._init_retriever()
        logger.info("Reranking disabled")
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        stats = {
            "embedder_model": self.embedder.model_name,
            "embedding_dim": self.embedder.get_embedding_dim(),
            "retriever_type": self.retriever_type,
            "reranking_enabled": self.enable_reranking,
            "num_documents": len(self.knowledge_base.documents) if hasattr(self.knowledge_base, 'documents') else 0,
            "index_size": self.knowledge_base.index.ntotal if hasattr(self.knowledge_base.index, 'ntotal') else 0
        }
        
        # Add reranker stats if enabled
        if self.enable_reranking and self.reranker:
            stats["reranker_model"] = self.reranker_config.model_name
            if hasattr(self.reranker, "get_stats"):
                stats["reranker_stats"] = self.reranker.get_stats()
        
        return stats


# Factory function for easy pipeline creation
def create_rag_pipeline(embedder_type: str = "kure",
                       retriever_type: str = "hybrid",
                       knowledge_base_path: Optional[str] = None,
                       device: Optional[str] = None,
                       enable_reranking: bool = False,
                       reranker_config: Optional[Union[RerankerConfig, Dict]] = None) -> RAGPipeline:
    """
    Factory function to create RAG pipeline with specified configuration
    
    Args:
        embedder_type: Type of embedder ("kure" or "e5")
        retriever_type: Type of retriever ("vector", "bm25", "hybrid")
        knowledge_base_path: Path to existing knowledge base
        device: Device for computation
        enable_reranking: Whether to enable reranking with Qwen3-Reranker-4B
        reranker_config: Configuration for the reranker
    
    Returns:
        Configured RAG pipeline
    """
    # Select embedder
    if embedder_type == "kure":
        embedder = KUREEmbedder(device=device)
    # E5Embedder는 더 이상 사용하지 않음 - KURE로 대체
    elif embedder_type == "e5":
        logger.warning("E5 embedder is deprecated, using KURE instead")
        embedder = KUREEmbedder(device=device)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")
    
    # Create pipeline
    pipeline = RAGPipeline(
        embedder=embedder,
        retriever_type=retriever_type,
        knowledge_base_path=knowledge_base_path,
        device=device,
        enable_reranking=enable_reranking,
        reranker_config=reranker_config
    )
    
    return pipeline