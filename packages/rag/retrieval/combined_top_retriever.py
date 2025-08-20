"""
BM25+Vector Combined Top-K Retriever
각 검색 방법에서 독립적으로 상위 K개를 선택하여 결합하는 방식
리더보드 점수 0.55 달성한 검증된 접근법
"""
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import pickle

from sentence_transformers import SentenceTransformer
from kiwipiepy import Kiwi
import faiss
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class CombinedTopRetriever:
    """
    BM25와 Vector 검색에서 각각 독립적으로 상위 K개를 선택하여 결합하는 Retriever
    
    검증된 성능:
    - 리더보드 점수: 0.55 (기존 0.46 대비 19.6% 향상)
    - 하이브리드 가중치 방식보다 우수한 성능 입증
    - 각 검색 방법의 강점을 희석시키지 않고 보존
    """
    
    def __init__(self,
                 embedder: Optional[SentenceTransformer] = None,
                 chunks: Optional[List[Dict]] = None,
                 faiss_index: Optional[faiss.Index] = None,
                 bm25_index: Optional[BM25Okapi] = None,
                 bm25_k: int = 3,
                 vector_k: int = 3,
                 data_dir: Optional[Path] = None):
        """
        Initialize Combined Top-K Retriever
        
        Args:
            embedder: Sentence transformer for encoding queries
            chunks: List of document chunks
            faiss_index: Pre-built FAISS index
            bm25_index: Pre-built BM25 index
            bm25_k: Number of top documents from BM25 (default: 3)
            vector_k: Number of top documents from Vector (default: 3)
            data_dir: Directory containing saved indices
        """
        self.embedder = embedder
        self.chunks = chunks
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.bm25_k = bm25_k
        self.vector_k = vector_k
        self.kiwi = Kiwi()
        self.data_dir = data_dir
        
        # Auto-load if data_dir is provided
        if data_dir and not all([chunks, faiss_index, bm25_index]):
            self.load_indices(data_dir)
    
    def load_indices(self, data_dir: Path):
        """Load pre-built indices from directory"""
        data_dir = Path(data_dir)
        
        # Load chunks
        if not self.chunks:
            chunks_path = data_dir / "chunks_2300.json"
            if chunks_path.exists():
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                logger.info(f"Loaded {len(self.chunks)} chunks")
        
        # Load FAISS index
        if not self.faiss_index:
            faiss_path = data_dir / "faiss_index_2300.index"
            if faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
                logger.info("Loaded FAISS index")
        
        # Load BM25 index
        if not self.bm25_index:
            bm25_path = data_dir / "bm25_index_2300.pkl"
            if bm25_path.exists():
                with open(bm25_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)
                logger.info("Loaded BM25 index")
        
        # Initialize embedder if not provided
        if not self.embedder:
            self.embedder = SentenceTransformer("nlpai-lab/KURE-v1", device="cuda")
            logger.info("Initialized KURE-v1 embedder")
    
    def search_bm25(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        BM25 검색
        
        Args:
            query: Search query
            k: Number of documents to retrieve
        
        Returns:
            List of documents with scores
        """
        if k is None:
            k = self.bm25_k
        
        try:
            # Tokenize query using Kiwi
            tokens = []
            for token in self.kiwi.tokenize(query):
                tokens.append(token.form)
            
            # Create corpus for BM25
            corpus = []
            for chunk in self.chunks:
                if isinstance(chunk, dict):
                    corpus.append(chunk.get('content', ''))
                else:
                    corpus.append(chunk)
            
            # Get top-k documents
            top_docs = self.bm25_index.get_top_n(tokens, corpus, n=k)
            
            # Convert to standardized format
            results = []
            for doc_content in top_docs:
                # Find original chunk
                for chunk in self.chunks:
                    chunk_content = chunk.get('content', '') if isinstance(chunk, dict) else chunk
                    if chunk_content == doc_content:
                        results.append({
                            'content': doc_content,
                            'score': 1.0,  # BM25 doesn't provide normalized scores
                            'metadata': chunk.get('metadata', {}) if isinstance(chunk, dict) else {},
                            'source': 'bm25'
                        })
                        break
            
            logger.debug(f"BM25 retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def search_vector(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        Vector 검색
        
        Args:
            query: Search query
            k: Number of documents to retrieve
        
        Returns:
            List of documents with scores
        """
        if k is None:
            k = self.vector_k
        
        try:
            # Encode query
            query_embedding = self.embedder.encode([query])
            
            # FAISS search
            scores, indices = self.faiss_index.search(query_embedding, k)
            
            # Extract documents
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    content = chunk.get('content', '') if isinstance(chunk, dict) else chunk
                    results.append({
                        'content': content,
                        'score': float(score),
                        'metadata': chunk.get('metadata', {}) if isinstance(chunk, dict) else {},
                        'source': 'vector'
                    })
            
            logger.debug(f"Vector retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def search(self, query: str, 
              bm25_k: Optional[int] = None,
              vector_k: Optional[int] = None,
              deduplicate: bool = False) -> List[Dict]:
        """
        Combined search: BM25 top-k + Vector top-k
        
        Args:
            query: Search query
            bm25_k: Number of documents from BM25
            vector_k: Number of documents from Vector
            deduplicate: Whether to remove duplicate documents
        
        Returns:
            Combined list of documents
        """
        # Use default k values if not specified
        if bm25_k is None:
            bm25_k = self.bm25_k
        if vector_k is None:
            vector_k = self.vector_k
        
        # Get documents from both methods
        bm25_results = self.search_bm25(query, k=bm25_k)
        vector_results = self.search_vector(query, k=vector_k)
        
        # Combine results
        if deduplicate:
            # Remove duplicates based on content
            seen_contents = set()
            combined = []
            
            # Add all documents, avoiding duplicates
            for doc in bm25_results + vector_results:
                content_hash = hash(doc['content'][:500])  # Use first 500 chars for hash
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    combined.append(doc)
        else:
            # Keep all documents (may have duplicates)
            combined = bm25_results + vector_results
        
        logger.info(f"Combined search: {len(bm25_results)} BM25 + {len(vector_results)} Vector = {len(combined)} total")
        
        return combined
    
    def retrieve(self, query: str, k: int = 6) -> List[Dict]:
        """
        Retrieve documents using combined approach
        
        Default behavior: BM25 top-3 + Vector top-3 = 6 documents
        
        Args:
            query: Search query
            k: Total number of documents (will be split evenly)
        
        Returns:
            List of retrieved documents
        """
        # Split k evenly between methods
        bm25_k = k // 2
        vector_k = k - bm25_k
        
        return self.search(query, bm25_k=bm25_k, vector_k=vector_k, deduplicate=False)