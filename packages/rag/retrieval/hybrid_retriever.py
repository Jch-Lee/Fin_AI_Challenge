"""
Hybrid Retriever: BM25 + Vector 검색 결합
BM25 sparse 검색과 FAISS dense 검색을 결합한 하이브리드 검색 시스템
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

from .bm25_retriever import BM25Retriever, BM25Result

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """하이브리드 검색 결과"""
    doc_id: str
    content: str
    hybrid_score: float
    bm25_score: float
    vector_score: float
    rank: int
    metadata: Dict[str, Any] = None
    retrieval_methods: List[str] = None  # ["bm25", "vector"] or ["bm25"] or ["vector"]


class ScoreNormalizer:
    """점수 정규화 클래스"""
    
    @staticmethod
    def min_max_normalize(scores: List[float], 
                         min_val: float = 0.0, 
                         max_val: float = 1.0) -> List[float]:
        """Min-Max 정규화"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = []
        for score in scores:
            norm_score = (score - min_score) / (max_score - min_score)
            norm_score = min_val + norm_score * (max_val - min_val)
            normalized.append(norm_score)
        
        return normalized
    
    @staticmethod
    def rank_based_normalize(scores: List[float]) -> List[float]:
        """순위 기반 정규화 (Rank-based normalization)"""
        if not scores:
            return []
        
        # 점수와 원래 인덱스를 함께 저장
        indexed_scores = [(score, i) for i, score in enumerate(scores)]
        # 점수 기준 내림차순 정렬
        indexed_scores.sort(key=lambda x: x[0], reverse=True)
        
        # 정규화된 점수 배열 초기화
        normalized = [0.0] * len(scores)
        
        # 순위 기반 점수 할당
        for rank, (_, original_idx) in enumerate(indexed_scores):
            normalized[original_idx] = 1.0 / (rank + 1)
        
        return normalized
    
    @staticmethod
    def softmax_normalize(scores: List[float], temperature: float = 1.0) -> List[float]:
        """소프트맥스 정규화"""
        if not scores:
            return []
        
        # 점수를 temperature로 나누어 조정
        scaled_scores = [s / temperature for s in scores]
        
        # 수치 안정성을 위해 최대값을 빼기
        max_score = max(scaled_scores)
        exp_scores = [np.exp(s - max_score) for s in scaled_scores]
        
        # 소프트맥스 계산
        sum_exp = sum(exp_scores)
        if sum_exp == 0:
            return [1.0 / len(scores)] * len(scores)
        
        return [exp_s / sum_exp for exp_s in exp_scores]


class HybridRetriever:
    """BM25 + Vector 하이브리드 검색"""
    
    def __init__(self,
                 bm25_retriever: Optional[BM25Retriever] = None,
                 vector_retriever = None,  # packages.rag.knowledge_base.KnowledgeBase
                 embedder = None,  # packages.preprocessing.embedder_e5.E5Embedder
                 bm25_weight: float = 0.5,
                 vector_weight: float = 0.5,
                 normalization_method: str = "min_max"):
        """
        Args:
            bm25_retriever: BM25 검색기
            vector_retriever: 벡터 검색기 (KnowledgeBase)
            embedder: 임베딩 생성기 (E5Embedder)
            bm25_weight: BM25 검색 가중치 (alpha)
            vector_weight: Vector 검색 가중치 (beta)
            normalization_method: 점수 정규화 방법 ("min_max", "rank", "softmax")
        """
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.embedder = embedder
        
        # 가중치 검증 및 정규화
        total_weight = bm25_weight + vector_weight
        if total_weight <= 0:
            raise ValueError("Sum of weights must be positive")
        
        self.bm25_weight = bm25_weight / total_weight
        self.vector_weight = vector_weight / total_weight
        
        # 정규화 방법
        if normalization_method not in ["min_max", "rank", "softmax"]:
            raise ValueError("normalization_method must be one of: min_max, rank, softmax")
        self.normalization_method = normalization_method
        
        self.normalizer = ScoreNormalizer()
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """점수 정규화"""
        if self.normalization_method == "min_max":
            return self.normalizer.min_max_normalize(scores)
        elif self.normalization_method == "rank":
            return self.normalizer.rank_based_normalize(scores)
        elif self.normalization_method == "softmax":
            return self.normalizer.softmax_normalize(scores)
        else:
            return scores
    
    def _vector_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """벡터 검색 (기존 KnowledgeBase 사용)"""
        if not self.vector_retriever or not self.embedder:
            return []
        
        try:
            # 쿼리 임베딩
            query_embedding = self.embedder.embed(query, is_query=True)
            if query_embedding is None:
                return []
            
            # FAISS 검색
            search_results = self.vector_retriever.search(query_embedding, k)
            
            # 결과 변환 - dict 형태로 반환되는 경우와 SearchResult 객체로 반환되는 경우 모두 처리
            results = []
            for result in search_results:
                if isinstance(result, dict):
                    # 이미 dict 형태인 경우
                    results.append({
                        'doc_id': result.get('chunk_id', result.get('id', '')),
                        'content': result.get('content', ''),
                        'score': result.get('score', 0.0),
                        'metadata': result.get('metadata', {})
                    })
                else:
                    # SearchResult 객체인 경우 - dict처럼 처리
                    try:
                        # __dict__ 속성이 있는 경우
                        if hasattr(result, '__dict__'):
                            result_dict = result.__dict__
                            results.append({
                                'doc_id': result_dict.get('metadata', {}).get('chunk_id', ''),
                                'content': result_dict.get('content', ''),
                                'score': result_dict.get('score', 0.0),
                                'metadata': result_dict.get('metadata', {})
                            })
                        else:
                            # 속성으로 직접 접근
                            doc_id = ''
                            if hasattr(result, 'metadata') and result.metadata:
                                doc_id = result.metadata.get('chunk_id', '')
                            
                            results.append({
                                'doc_id': doc_id,
                                'content': getattr(result, 'content', ''),
                                'score': getattr(result, 'score', 0.0),
                                'metadata': getattr(result, 'metadata', {})
                            })
                    except Exception as e:
                        logger.warning(f"Failed to process result: {e}, result type: {type(result)}")
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def retrieve(self,
                query: str,
                top_k: int = 5,
                filters: Optional[Dict[str, Any]] = None,
                **kwargs) -> List[Dict[str, Any]]:
        """
        Retriever interface for compatibility with RerankingRetriever
        
        Args:
            query: Search query
            top_k: Number of documents to return
            filters: Optional filters (not used in current implementation)
            **kwargs: Additional arguments
            
        Returns:
            List of documents with scores
        """
        # Call search method and convert results to dict format
        results = self.search(
            query=query,
            k=top_k,
            bm25_k=kwargs.get('bm25_k'),
            vector_k=kwargs.get('vector_k'),
            min_bm25_score=kwargs.get('min_bm25_score', 0.0),
            min_vector_score=kwargs.get('min_vector_score', 0.0),
            use_parallel=kwargs.get('use_parallel', True)
        )
        
        # Convert HybridResult to dict format
        documents = []
        for result in results:
            doc = {
                'id': result.doc_id,
                'content': result.content,
                'score': result.hybrid_score,
                'bm25_score': result.bm25_score,
                'vector_score': result.vector_score,
                'metadata': result.metadata or {}
            }
            documents.append(doc)
        
        return documents
    
    def search(self,
               query: str,
               k: int = 5,
               bm25_k: Optional[int] = None,
               vector_k: Optional[int] = None,
               min_bm25_score: float = 0.0,
               min_vector_score: float = 0.0,
               use_parallel: bool = True) -> List[HybridResult]:
        """
        하이브리드 검색
        
        Args:
            query: 검색 쿼리
            k: 최종 반환할 결과 개수
            bm25_k: BM25 검색에서 가져올 결과 개수 (None이면 k*2)
            vector_k: Vector 검색에서 가져올 결과 개수 (None이면 k*2)
            min_bm25_score: BM25 최소 점수 threshold
            min_vector_score: Vector 최소 점수 threshold
            use_parallel: 병렬 검색 사용 여부
        """
        if not query.strip():
            return []
        
        # 검색할 결과 수 결정 (더 많이 가져와서 다양성 확보)
        bm25_k = bm25_k or min(k * 3, 50)
        vector_k = vector_k or min(k * 3, 50)
        
        # 검색 실행
        if use_parallel and self.bm25_retriever and (self.vector_retriever and self.embedder):
            # 병렬 검색
            bm25_results, vector_results = self._parallel_search(
                query, bm25_k, vector_k, min_bm25_score, min_vector_score
            )
        else:
            # 순차 검색
            bm25_results = self._bm25_search(query, bm25_k, min_bm25_score)
            vector_results = self._vector_search(query, vector_k)
            # Vector 결과 필터링
            vector_results = [r for r in vector_results if r['score'] >= min_vector_score]
        
        # 결과 통합 및 점수 계산
        hybrid_results = self._combine_results(bm25_results, vector_results, k)
        
        logger.debug(f"Hybrid search: BM25={len(bm25_results)}, Vector={len(vector_results)}, Final={len(hybrid_results)}")
        return hybrid_results
    
    def _parallel_search(self, 
                        query: str, 
                        bm25_k: int, 
                        vector_k: int,
                        min_bm25_score: float,
                        min_vector_score: float) -> Tuple[List[BM25Result], List[Dict[str, Any]]]:
        """병렬 검색 실행"""
        bm25_results = []
        vector_results = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            # BM25 검색 제출
            if self.bm25_retriever:
                futures['bm25'] = executor.submit(
                    self._bm25_search, query, bm25_k, min_bm25_score
                )
            
            # Vector 검색 제출
            if self.vector_retriever and self.embedder:
                futures['vector'] = executor.submit(
                    self._vector_search, query, vector_k
                )
            
            # 결과 수집
            for name, future in futures.items():
                try:
                    result = future.result(timeout=30)  # 30초 타임아웃
                    if name == 'bm25':
                        bm25_results = result
                    elif name == 'vector':
                        vector_results = [r for r in result if r['score'] >= min_vector_score]
                except Exception as e:
                    logger.error(f"{name} search failed: {e}")
        
        return bm25_results, vector_results
    
    def _bm25_search(self, query: str, k: int, min_score: float) -> List[BM25Result]:
        """BM25 검색"""
        if not self.bm25_retriever:
            return []
        
        try:
            return self.bm25_retriever.search(query, k=k, min_score=min_score)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _combine_results(self,
                        bm25_results: List[BM25Result],
                        vector_results: List[Dict[str, Any]],
                        k: int) -> List[HybridResult]:
        """검색 결과 통합 및 점수 계산"""
        # 문서별 점수 수집
        doc_scores = {}  # doc_id -> {'bm25': score, 'vector': score, 'content': str, 'metadata': dict}
        
        # BM25 결과 처리
        bm25_scores = [r.score for r in bm25_results] if bm25_results else []
        normalized_bm25_scores = self._normalize_scores(bm25_scores)
        
        for result, norm_score in zip(bm25_results, normalized_bm25_scores):
            doc_id = result.doc_id
            doc_scores[doc_id] = {
                'bm25': norm_score,
                'vector': 0.0,
                'content': result.content,
                'metadata': result.metadata or {},
                'methods': ['bm25']
            }
        
        # Vector 결과 처리
        vector_scores = [r['score'] for r in vector_results] if vector_results else []
        normalized_vector_scores = self._normalize_scores(vector_scores)
        
        for result, norm_score in zip(vector_results, normalized_vector_scores):
            doc_id = result['doc_id']
            if doc_id in doc_scores:
                # 이미 BM25에서 찾은 문서
                doc_scores[doc_id]['vector'] = norm_score
                if 'vector' not in doc_scores[doc_id]['methods']:
                    doc_scores[doc_id]['methods'].append('vector')
            else:
                # Vector에서만 찾은 문서
                doc_scores[doc_id] = {
                    'bm25': 0.0,
                    'vector': norm_score,
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'methods': ['vector']
                }
        
        # 하이브리드 점수 계산
        hybrid_scored_docs = []
        for doc_id, scores in doc_scores.items():
            hybrid_score = (
                scores['bm25'] * self.bm25_weight + 
                scores['vector'] * self.vector_weight
            )
            
            hybrid_scored_docs.append({
                'doc_id': doc_id,
                'hybrid_score': hybrid_score,
                'bm25_score': scores['bm25'],
                'vector_score': scores['vector'],
                'content': scores['content'],
                'metadata': scores['metadata'],
                'methods': scores['methods']
            })
        
        # 하이브리드 점수로 정렬
        hybrid_scored_docs.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # 상위 k개 선택 및 HybridResult 생성
        final_results = []
        for rank, doc in enumerate(hybrid_scored_docs[:k]):
            result = HybridResult(
                doc_id=doc['doc_id'],
                content=doc['content'],
                hybrid_score=doc['hybrid_score'],
                bm25_score=doc['bm25_score'],
                vector_score=doc['vector_score'],
                rank=rank + 1,
                metadata=doc['metadata'],
                retrieval_methods=doc['methods']
            )
            final_results.append(result)
        
        return final_results
    
    def batch_search(self, 
                    queries: List[str], 
                    k: int = 5) -> List[List[HybridResult]]:
        """배치 하이브리드 검색"""
        results = []
        for query in queries:
            query_results = self.search(query, k=k)
            results.append(query_results)
        return results
    
    def explain_search(self, 
                      query: str, 
                      k: int = 5) -> Dict[str, Any]:
        """검색 결과 설명 (디버깅용)"""
        # 개별 검색 결과 수집
        bm25_results = self._bm25_search(query, k*2, 0.0)
        vector_results = self._vector_search(query, k*2)
        
        # 하이브리드 결과
        hybrid_results = self.search(query, k)
        
        explanation = {
            'query': query,
            'weights': {
                'bm25_weight': self.bm25_weight,
                'vector_weight': self.vector_weight
            },
            'normalization_method': self.normalization_method,
            'individual_results': {
                'bm25': [
                    {
                        'doc_id': r.doc_id,
                        'score': r.score,
                        'content_preview': r.content[:100] + "..."
                    } for r in bm25_results
                ],
                'vector': [
                    {
                        'doc_id': r['doc_id'],
                        'score': r['score'],
                        'content_preview': r['content'][:100] + "..."
                    } for r in vector_results
                ]
            },
            'hybrid_results': [
                {
                    'doc_id': r.doc_id,
                    'hybrid_score': r.hybrid_score,
                    'bm25_score': r.bm25_score,
                    'vector_score': r.vector_score,
                    'methods': r.retrieval_methods,
                    'content_preview': r.content[:100] + "..."
                } for r in hybrid_results
            ]
        }
        
        return explanation
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보"""
        stats = {
            'weights': {
                'bm25_weight': self.bm25_weight,
                'vector_weight': self.vector_weight
            },
            'normalization_method': self.normalization_method,
            'components': {
                'bm25_available': self.bm25_retriever is not None,
                'vector_available': self.vector_retriever is not None and self.embedder is not None
            }
        }
        
        # BM25 통계
        if self.bm25_retriever:
            stats['bm25_stats'] = self.bm25_retriever.get_stats()
        
        # Vector 통계
        if self.vector_retriever:
            try:
                stats['vector_stats'] = self.vector_retriever.get_stats()
            except AttributeError:
                stats['vector_stats'] = {'status': 'available'}
        
        return stats
    
    def save(self, save_path: str):
        """하이브리드 검색기 설정 저장"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 설정만 저장 (실제 검색기들은 별도 저장)
        config = {
            'bm25_weight': self.bm25_weight,
            'vector_weight': self.vector_weight,
            'normalization_method': self.normalization_method
        }
        
        config_path = save_path / "hybrid_config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Hybrid retriever config saved to {config_path}")


if __name__ == "__main__":
    # 테스트용 코드
    print("HybridRetriever implementation completed!")
    print("Available normalization methods:", ["min_max", "rank", "softmax"])
    
    # 점수 정규화 테스트
    normalizer = ScoreNormalizer()
    test_scores = [10.5, 8.2, 5.1, 2.3, 0.8]
    
    print(f"Original scores: {test_scores}")
    print(f"Min-Max normalized: {normalizer.min_max_normalize(test_scores)}")
    print(f"Rank normalized: {normalizer.rank_based_normalize(test_scores)}")
    print(f"Softmax normalized: {normalizer.softmax_normalize(test_scores)}")