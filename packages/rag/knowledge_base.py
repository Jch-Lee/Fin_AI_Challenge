"""
지식 베이스 구축 및 관리 모듈
FAISS 인덱스를 사용한 벡터 검색
"""
import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str
    doc_id: str


class KnowledgeBase:
    """FAISS 기반 지식 베이스"""
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 index_type: str = "Flat",  # Default to Flat for small datasets
                 nlist: int = 100,
                 nprobe: int = 10):
        """
        Args:
            embedding_dim: 임베딩 차원
            index_type: 인덱스 타입 ("Flat", "IVF", "HNSW")
            nlist: IVF 인덱스의 클러스터 수
            nprobe: 검색 시 탐색할 클러스터 수
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        
        self.index = None
        self.documents = []  # 원본 문서 저장
        self.metadata = []   # 메타데이터 저장
        self.doc_count = 0
        
        self._build_index()
    
    def _build_index(self):
        """FAISS 인덱스 생성"""
        if self.index_type == "Flat":
            # 정확한 검색 (작은 데이터셋용)
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (코사인 유사도)
            
        elif self.index_type == "IVF":
            # Inverted File Index (중간 크기 데이터셋용)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.embedding_dim, 
                self.nlist, 
                faiss.METRIC_INNER_PRODUCT
            )
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World (큰 데이터셋용)
            self.index = faiss.IndexHNSWFlat(
                self.embedding_dim, 
                32,  # M parameter
                faiss.METRIC_INNER_PRODUCT
            )
            self.index.hnsw.efConstruction = 40
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        logger.info(f"Created {self.index_type} index with dimension {self.embedding_dim}")
    
    def add_documents(self, 
                     embeddings: np.ndarray,
                     documents: List[Any],
                     metadata: Optional[List[Dict]] = None):
        """문서 추가"""
        # Handle both string documents and dict documents
        if documents and isinstance(documents[0], dict):
            # Extract content from dict documents
            actual_documents = [doc.get('content', str(doc)) for doc in documents]
            if metadata is None:
                metadata = [doc.get('metadata', {}) for doc in documents]
        else:
            actual_documents = documents
        
        if len(embeddings) != len(actual_documents):
            raise ValueError("Number of embeddings and documents must match")
        
        if metadata and len(metadata) != len(documents):
            raise ValueError("Number of metadata and documents must match")
        
        # 임베딩 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(embeddings)
        
        # IVF 인덱스는 학습이 필요하며, 충분한 데이터가 필요함
        if self.index_type == "IVF" and not self.index.is_trained:
            if len(embeddings) < self.nlist:
                logger.warning(f"Not enough documents ({len(embeddings)}) for IVF with nlist={self.nlist}. Switching to Flat index.")
                self.index_type = "Flat"
                self._build_index()
            else:
                logger.info("Training IVF index...")
                self.index.train(embeddings)
        
        # 인덱스에 추가
        self.index.add(embeddings)
        
        # 문서와 메타데이터 저장
        self.documents.extend(actual_documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in actual_documents])
        
        self.doc_count += len(actual_documents)
        logger.info(f"Added {len(actual_documents)} documents. Total: {self.doc_count}")
    
    def search(self, 
              query_embedding: np.ndarray,
              k: int = 5,
              top_k: Optional[int] = None,  # Support both k and top_k
              threshold: Optional[float] = None) -> List[Union[SearchResult, Dict]]:
        """벡터 검색"""
        if self.doc_count == 0:
            logger.warning("Knowledge base is empty")
            return []
        
        # 쿼리 임베딩 정규화
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # IVF 인덱스의 경우 nprobe 설정
        if self.index_type == "IVF":
            self.index.nprobe = self.nprobe
        
        # Support both k and top_k parameters
        k_value = top_k if top_k is not None else k
        
        # 검색 수행
        scores, indices = self.index.search(query_embedding, min(k_value, self.doc_count))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # 유효하지 않은 인덱스
                continue
                
            if threshold and score < threshold:
                continue
            
            metadata = self.metadata[idx] if idx < len(self.metadata) else {}
            
            # Return dict format for compatibility
            result = {
                'content': self.documents[idx],
                'score': float(score),
                'metadata': metadata,
                'chunk_id': metadata.get('chunk_id', str(idx)),
                'doc_id': metadata.get('doc_id', 'unknown'),
                'id': idx
            }
            results.append(result)
        
        return results
    
    def batch_search(self, 
                    query_embeddings: np.ndarray,
                    k: int = 5,
                    use_kure_similarity: bool = False,
                    embedder=None) -> List[List[SearchResult]]:
        """
        Batch search for multiple queries simultaneously
        Optimized for large-scale retrieval
        
        Args:
            query_embeddings: Multiple query embeddings (shape: [n_queries, dim])
            k: Number of results per query
            use_kure_similarity: Whether to use KURE's native similarity
            embedder: KUREEmbedder instance (required if use_kure_similarity=True)
        
        Returns:
            List of search results for each query
        """
        if self.doc_count == 0:
            return [[] for _ in range(len(query_embeddings))]
        
        all_results = []
        
        if use_kure_similarity and embedder is not None:
            # Use KURE's optimized batch similarity search
            # Get all document embeddings from FAISS index
            doc_embeddings = self.index.reconstruct_n(0, self.doc_count)
            
            # Perform batch similarity search
            top_k_results = embedder.batch_similarity_search(
                query_embeddings,
                doc_embeddings,
                top_k=k
            )
            
            # Convert to SearchResult objects
            for query_results in top_k_results:
                results = []
                for doc_idx, score in query_results:
                    if doc_idx < len(self.documents):
                        # Handle both string and dict documents
                        if isinstance(self.documents[doc_idx], dict):
                            content = self.documents[doc_idx].get('content', '')
                            doc_metadata = self.documents[doc_idx].get('metadata', {})
                        else:
                            content = self.documents[doc_idx]
                            doc_metadata = self.metadata[doc_idx] if doc_idx < len(self.metadata) else {}
                        
                        result = SearchResult(
                            content=content,
                            score=score,
                            metadata=doc_metadata,
                            chunk_id=doc_metadata.get('chunk_id', str(doc_idx)),
                            doc_id=doc_metadata.get('doc_id', 'unknown')
                        )
                        results.append(result)
                all_results.append(results)
        else:
            # Use standard FAISS batch search
            # 쿼리 임베딩 정규화
            query_embeddings = query_embeddings.astype('float32')
            faiss.normalize_L2(query_embeddings)
            
            # IVF 인덱스의 경우 nprobe 설정
            if self.index_type == "IVF":
                self.index.nprobe = self.nprobe
            
            # 배치 검색
            scores, indices = self.index.search(query_embeddings, min(k, self.doc_count))
            
            for query_scores, query_indices in zip(scores, indices):
                results = []
                for score, idx in zip(query_scores, query_indices):
                    if idx == -1:
                        continue
                    
                    metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                    
                    result = SearchResult(
                        content=self.documents[idx],
                        score=float(score),
                        metadata=metadata,
                        chunk_id=metadata.get('chunk_id', str(idx)),
                        doc_id=metadata.get('doc_id', 'unknown')
                    )
                    results.append(result)
                
                all_results.append(results)
        
        return all_results
    
    def similarity_search_with_kure(self,
                                   query_embedding: np.ndarray,
                                   k: int = 5,
                                   embedder=None) -> List[SearchResult]:
        """
        Search using KURE's native similarity computation
        
        Args:
            query_embedding: Query embedding
            k: Number of results
            embedder: KUREEmbedder instance
        
        Returns:
            List of search results
        """
        if embedder is None:
            logger.warning("KUREEmbedder not provided, falling back to standard search")
            return self.search(query_embedding, k)
        
        # Get all document embeddings
        doc_embeddings = self.index.reconstruct_n(0, self.doc_count)
        
        # Compute similarities using KURE
        similarities = embedder.compute_similarity_scores(query_embedding, doc_embeddings)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            if idx < len(self.documents):
                # Handle both string and dict documents
                if isinstance(self.documents[idx], dict):
                    content = self.documents[idx].get('content', '')
                    doc_metadata = self.documents[idx].get('metadata', {})
                else:
                    content = self.documents[idx]
                    doc_metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                
                result = SearchResult(
                    content=content,
                    score=float(similarities[idx]),
                    metadata=doc_metadata,
                    chunk_id=doc_metadata.get('chunk_id', str(idx)),
                    doc_id=doc_metadata.get('doc_id', 'unknown')
                )
                results.append(result)
        
        return results
    
    def save(self, save_dir: str):
        """지식 베이스 저장"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS 인덱스 저장
        if self.index is not None:
            index_path = save_path / "faiss.index"
            faiss.write_index(self.index, str(index_path))
        else:
            logger.warning("No index to save")
        
        # 문서와 메타데이터 저장
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'doc_count': self.doc_count,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'nprobe': self.nprobe
        }
        
        data_path = save_path / "knowledge_base.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved knowledge base to {save_dir}")
    
    @classmethod
    def load(cls, save_dir: str) -> 'KnowledgeBase':
        """지식 베이스 로드 (새 형식 및 레거시 형식 지원)"""
        save_path = Path(save_dir)
        
        # Check for new format first
        data_path = save_path / "knowledge_base.pkl"
        
        if data_path.exists():
            # New format with pickle file
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        else:
            # Legacy format - only FAISS index exists
            logger.warning("Loading legacy format (FAISS index only)")
            # Create minimal data structure for legacy support
            data = {
                'documents': [],
                'metadata': [],
                'doc_count': 0,
                'embedding_dim': 1024,  # Assume KURE default
                'index_type': 'Flat',  # Assume Flat for legacy
                'nlist': 100,
                'nprobe': 10
            }
        
        # 인스턴스 생성
        kb = cls(
            embedding_dim=data['embedding_dim'],
            index_type=data['index_type'],
            nlist=data.get('nlist', 100),
            nprobe=data.get('nprobe', 10)
        )
        
        # FAISS 인덱스 로드
        index_path = save_path / "faiss.index"
        if index_path.exists():
            kb.index = faiss.read_index(str(index_path))
            # For legacy format, update doc_count from index
            if not data_path.exists() and hasattr(kb.index, 'ntotal'):
                data['doc_count'] = kb.index.ntotal
        else:
            logger.warning(f"No FAISS index found at {index_path}")
        
        # 데이터 복원
        kb.documents = data.get('documents', [])
        kb.metadata = data.get('metadata', [])
        kb.doc_count = data.get('doc_count', 0)
        
        # For legacy format with empty documents, create placeholders
        if kb.doc_count > 0 and len(kb.documents) == 0:
            logger.info(f"Creating placeholder documents for {kb.doc_count} legacy entries")
            kb.documents = [f"Legacy document {i}" for i in range(kb.doc_count)]
            kb.metadata = [{'legacy': True, 'index': i} for i in range(kb.doc_count)]
        
        logger.info(f"Loaded knowledge base from {save_dir}")
        return kb
    
    def get_stats(self) -> Dict:
        """통계 정보 반환"""
        return {
            'total_documents': self.doc_count,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'index_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            'index_size': self.index.ntotal if hasattr(self.index, 'ntotal') else self.doc_count
        }
    
    def clear(self):
        """지식 베이스 초기화"""
        self._build_index()
        self.documents = []
        self.metadata = []
        self.doc_count = 0
        logger.info("Knowledge base cleared")


if __name__ == "__main__":
    # 테스트
    kb = KnowledgeBase(embedding_dim=1024, index_type="Flat")
    
    # 더미 데이터 생성
    n_docs = 100
    embeddings = np.random.randn(n_docs, 1024).astype('float32')
    documents = [f"Document {i}: This is test content." for i in range(n_docs)]
    metadata = [{'doc_id': f'doc_{i}', 'source': 'test'} for i in range(n_docs)]
    
    # 문서 추가
    kb.add_documents(embeddings, documents, metadata)
    
    # 검색
    query_embedding = np.random.randn(768).astype('float32')
    results = kb.search(query_embedding, k=5)
    
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Content: {result.content}")
        print(f"  Score: {result.score:.4f}")
        print(f"  Doc ID: {result.doc_id}")
    
    # 통계
    print("\nKnowledge Base Stats:")
    print(kb.get_stats())