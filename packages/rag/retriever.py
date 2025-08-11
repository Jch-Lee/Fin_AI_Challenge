"""
Multi-Stage Retriever Component
Architecture.md의 MultiStageRetriever 인터페이스 구현
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging
import bm25s
import re
from collections import Counter
import Stemmer  # for bm25s

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """검색 결과 데이터 클래스"""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str  # "dense", "sparse", "hybrid"


class MultiStageRetriever:
    """
    다단계 검색 컴포넌트
    BM25 + FAISS 하이브리드 검색 구현
    Pipeline.md 3.2.3 요구사항 준수
    """
    
    def __init__(self, knowledge_base=None, embedder=None):
        self.knowledge_base = knowledge_base
        self.embedder = embedder
        self.cache = {}
        
        # BM25 인덱스
        self.bm25_index = None
        self.documents = []
        self.document_ids = []
        
    def _tokenize_korean(self, text: str) -> List[str]:
        """한국어 텍스트 토큰화"""
        # 간단한 토큰화: 공백 및 특수문자로 분리
        # 실제로는 KoNLPy 사용 권장
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def build_bm25_index(self, documents: List[str], doc_ids: List[str]):
        """BM25s 인덱스 구축"""
        self.documents = documents
        self.document_ids = doc_ids
        
        # 문서 토큰화 (bm25s는 텍스트를 직접 받음)
        tokenized_docs = [self._tokenize_korean(doc) for doc in documents]
        
        # BM25s 인덱스 생성
        self.bm25_index = bm25s.BM25()
        corpus_tokens = tokenized_docs
        
        # 인덱스 구축
        self.bm25_index.index(corpus_tokens)
        
        logger.info(f"BM25s index built with {len(documents)} documents")
    
    def sparse_search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """BM25s 기반 sparse 검색"""
        if not self.bm25_index:
            logger.warning("BM25s index not built, returning empty results")
            return []
        
        # 쿼리 토큰화
        query_tokens = [self._tokenize_korean(query)]
        
        # BM25s 검색 (상위 k개)
        results_tuple = self.bm25_index.retrieve(query_tokens, k=k)
        doc_indices = results_tuple[1][0]  # 첫 번째 쿼리의 결과
        scores = results_tuple[0][0]  # 점수
        
        results = []
        for i, idx in enumerate(doc_indices):
            idx_int = int(idx)  # Convert to int
            if i < len(scores) and scores[i] > 0:  # 점수가 0보다 큰 경우만
                if idx_int < len(self.documents):  # Bounds check
                    results.append(RetrievalResult(
                        doc_id=self.document_ids[idx_int],
                        content=self.documents[idx_int],
                        score=float(scores[i]),
                        metadata={"method": "bm25s"},
                        retrieval_method="sparse"
                    ))
        
        return results
    
    def dense_search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """FAISS 기반 dense 검색"""
        if not self.knowledge_base or not self.embedder:
            logger.warning("Knowledge base or embedder not available")
            return []
        
        # 쿼리 임베딩
        query_embedding = self.embedder.embed(query)
        
        # FAISS 검색
        results = self.knowledge_base.search(query_embedding, k)
        
        retrieval_results = []
        for r in results:
            retrieval_results.append(RetrievalResult(
                doc_id=r.metadata.get('chunk_id', ''),
                content=r.content,
                score=r.score,
                metadata=r.metadata,
                retrieval_method="dense"
            ))
        
        return retrieval_results
    
    def hybrid_search(self, 
                     query: str,
                     dense_weight: float = 0.7,
                     sparse_weight: float = 0.3,
                     k: int = 5) -> List[RetrievalResult]:
        """
        하이브리드 검색 (Dense + Sparse)
        Pipeline.md 3.2.3 요구사항
        
        Args:
            query: 검색 쿼리
            dense_weight: Dense 검색 가중치
            sparse_weight: Sparse 검색 가중치  
            k: 반환할 문서 개수
        
        Returns:
            통합된 검색 결과
        """
        # Dense 검색
        dense_results = self.dense_search(query, k * 2)
        
        # Sparse 검색
        sparse_results = self.sparse_search(query, k * 2)
        
        # 결과 통합
        doc_scores = {}
        doc_contents = {}
        doc_metadata = {}
        
        # Dense 결과 처리
        for result in dense_results:
            doc_id = result.doc_id
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + result.score * dense_weight
            doc_contents[doc_id] = result.content
            doc_metadata[doc_id] = result.metadata
        
        # Sparse 결과 처리
        for result in sparse_results:
            doc_id = result.doc_id
            # BM25 점수 정규화 (0-1 범위로)
            normalized_score = min(result.score / 10.0, 1.0)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + normalized_score * sparse_weight
            if doc_id not in doc_contents:
                doc_contents[doc_id] = result.content
                doc_metadata[doc_id] = result.metadata
        
        # 점수순 정렬
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # 최종 결과 생성
        final_results = []
        for doc_id, score in sorted_docs:
            final_results.append(RetrievalResult(
                doc_id=doc_id,
                content=doc_contents[doc_id],
                score=score,
                metadata=doc_metadata[doc_id],
                retrieval_method="hybrid"
            ))
        
        return final_results
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """기본 검색 메서드 (하이브리드 검색 사용)"""
        results = self.hybrid_search(query, k=k)
        
        # Dict 형태로 변환
        return [
            {
                'doc_id': r.doc_id,
                'content': r.content,
                'score': r.score,
                'metadata': r.metadata
            }
            for r in results
        ]