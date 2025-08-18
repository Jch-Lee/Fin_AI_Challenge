"""
BM25 기반 Sparse Retriever
한국어 토크나이저를 사용한 BM25 검색 구현
"""

import logging
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import re

try:
    import bm25s
    BMSLIB_AVAILABLE = True
except ImportError:
    BMSLIB_AVAILABLE = False

try:
    from konlpy.tag import Okt, Mecab, Hannanum
    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """BM25 검색 결과"""
    doc_id: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any] = None


class KoreanTokenizer:
    """한국어 토크나이저"""
    
    def __init__(self, method: str = "simple"):
        """
        Args:
            method: 토크나이제이션 방법
                - "simple": 정규식 기반 간단한 분리
                - "okt": KoNLPy Okt 형태소 분석기
                - "mecab": KoNLPy Mecab 형태소 분석기 (가장 빠름)
        """
        self.method = method
        self.tokenizer = None
        
        if method in ["okt", "mecab", "hannanum"] and not KONLPY_AVAILABLE:
            logger.warning("KoNLPy not available, falling back to simple tokenizer")
            self.method = "simple"
        
        if self.method == "okt" and KONLPY_AVAILABLE:
            self.tokenizer = Okt()
        elif self.method == "mecab" and KONLPY_AVAILABLE:
            try:
                self.tokenizer = Mecab()
            except Exception as e:
                logger.warning(f"Mecab initialization failed: {e}, using Okt")
                self.tokenizer = Okt()
                self.method = "okt"
        elif self.method == "hannanum" and KONLPY_AVAILABLE:
            self.tokenizer = Hannanum()
    
    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분리"""
        if not text or not text.strip():
            return []
        
        text = text.strip().lower()
        
        if self.method == "simple":
            # 한글, 영문, 숫자만 추출하고 공백으로 분리
            tokens = re.findall(r'[가-힣a-zA-Z0-9]+', text)
            # 2글자 이상만 유지 (불용어 제거 효과)
            tokens = [token for token in tokens if len(token) >= 2]
            return tokens
            
        elif self.method in ["okt", "mecab", "hannanum"] and self.tokenizer:
            try:
                # 형태소 분석 (명사, 동사, 형용사만 추출)
                morphs = self.tokenizer.morphs(text)
                # 2글자 이상 형태소만 유지
                morphs = [morph for morph in morphs if len(morph) >= 2]
                return morphs
            except Exception as e:
                logger.warning(f"Tokenization failed: {e}, using simple fallback")
                return self.tokenize_simple(text)
        
        return []
    
    def tokenize_simple(self, text: str) -> List[str]:
        """간단한 정규식 기반 토크나이저"""
        tokens = re.findall(r'[가-힣a-zA-Z0-9]+', text.lower())
        return [token for token in tokens if len(token) >= 2]


class BM25Retriever:
    """BM25 기반 Sparse Retriever"""
    
    def __init__(self, 
                 tokenizer_method: str = "simple",
                 k1: float = 1.2,
                 b: float = 0.75):
        """
        Args:
            tokenizer_method: 토크나이저 방법 ("simple", "okt", "mecab")
            k1: BM25 파라미터 k1 (term frequency saturation point)
            b: BM25 파라미터 b (field length normalization)
        """
        self.tokenizer = KoreanTokenizer(tokenizer_method)
        self.k1 = k1
        self.b = b
        
        # BM25s 인덱스
        self.bm25_index = None
        self.documents = []
        self.document_ids = []
        self.tokenized_documents = []
        
        if not BMSLIB_AVAILABLE:
            raise ImportError("bm25s library is required. Install with: pip install bm25s")
    
    def build_index(self, 
                   documents: List[str], 
                   doc_ids: List[str],
                   metadata: Optional[List[Dict[str, Any]]] = None):
        """BM25 인덱스 구축"""
        if len(documents) != len(doc_ids):
            raise ValueError("Number of documents and doc_ids must match")
        
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        
        # 문서 저장
        self.documents = documents.copy()
        self.document_ids = doc_ids.copy()
        # 메타데이터 저장
        self.metadata = metadata.copy() if metadata else [{} for _ in documents]
        
        # 문서 토크나이제이션
        self.tokenized_documents = []
        for i, doc in enumerate(documents):
            if i % 100 == 0 and i > 0:
                logger.info(f"Tokenized {i}/{len(documents)} documents")
            
            tokens = self.tokenizer.tokenize(doc)
            self.tokenized_documents.append(tokens)
        
        # BM25s 인덱스 생성 및 훈련
        self.bm25_index = bm25s.BM25()
        self.bm25_index.index(self.tokenized_documents)
        
        logger.info(f"BM25 index built successfully with {len(documents)} documents")
        return self
    
    def search(self, 
               query: str, 
               k: int = 5,
               min_score: float = 0.0) -> List[BM25Result]:
        """BM25 검색"""
        if not self.bm25_index:
            logger.warning("BM25 index not built")
            return []
        
        if not query.strip():
            return []
        
        # 쿼리 토크나이제이션
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            logger.warning(f"No valid tokens found in query: {query}")
            return []
        
        try:
            # BM25s 검색
            results_tuple = self.bm25_index.retrieve(
                [query_tokens], 
                k=min(k, len(self.documents))
            )
            
            # 결과 파싱
            scores = results_tuple[0][0]  # 첫 번째 쿼리의 점수들
            doc_indices = results_tuple[1][0]  # 첫 번째 쿼리의 문서 인덱스들
            
            # 결과 생성
            results = []
            for rank, (score, idx) in enumerate(zip(scores, doc_indices)):
                idx = int(idx)
                score = float(score)
                
                # 최소 점수 필터링
                if score < min_score:
                    continue
                
                # 유효한 인덱스 확인
                if 0 <= idx < len(self.documents):
                    # 메타데이터 병합
                    result_metadata = self.metadata[idx].copy() if hasattr(self, 'metadata') and idx < len(self.metadata) else {}
                    result_metadata.update({
                        "method": "bm25s",
                        "tokenizer": self.tokenizer.method,
                        "query_tokens": query_tokens
                    })
                    
                    result = BM25Result(
                        doc_id=self.document_ids[idx],
                        content=self.documents[idx],
                        score=score,
                        rank=rank + 1,
                        metadata=result_metadata
                    )
                    results.append(result)
            
            logger.debug(f"BM25 search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def batch_search(self, 
                    queries: List[str], 
                    k: int = 5) -> List[List[BM25Result]]:
        """배치 검색"""
        if not self.bm25_index:
            return [[] for _ in queries]
        
        # 쿼리들을 토크나이제이션
        tokenized_queries = []
        for query in queries:
            tokens = self.tokenizer.tokenize(query)
            tokenized_queries.append(tokens)
        
        # 빈 쿼리 확인
        valid_queries = [q for q in tokenized_queries if q]
        if not valid_queries:
            return [[] for _ in queries]
        
        try:
            # 배치 검색
            results_tuple = self.bm25_index.retrieve(
                valid_queries, 
                k=min(k, len(self.documents))
            )
            
            all_scores = results_tuple[0]
            all_indices = results_tuple[1]
            
            # 결과 생성
            all_results = []
            valid_idx = 0
            
            for i, (query, tokens) in enumerate(zip(queries, tokenized_queries)):
                if not tokens:  # 빈 쿼리
                    all_results.append([])
                    continue
                
                scores = all_scores[valid_idx]
                indices = all_indices[valid_idx]
                valid_idx += 1
                
                results = []
                for rank, (score, idx) in enumerate(zip(scores, indices)):
                    idx = int(idx)
                    if 0 <= idx < len(self.documents):
                        result = BM25Result(
                            doc_id=self.document_ids[idx],
                            content=self.documents[idx],
                            score=float(score),
                            rank=rank + 1,
                            metadata={
                                "method": "bm25s",
                                "tokenizer": self.tokenizer.method,
                                "query_tokens": tokens
                            }
                        )
                        results.append(result)
                
                all_results.append(results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Batch BM25 search failed: {e}")
            return [[] for _ in queries]
    
    def save(self, save_path: str):
        """BM25 인덱스 저장"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 인덱스 저장
        index_path = save_path / "bm25_index.pkl"
        with open(index_path, 'wb') as f:
            pickle.dump({
                'bm25_index': self.bm25_index,
                'documents': self.documents,
                'document_ids': self.document_ids,
                'tokenized_documents': self.tokenized_documents,
                'tokenizer_method': self.tokenizer.method,
                'k1': self.k1,
                'b': self.b
            }, f)
        
        logger.info(f"BM25 index saved to {index_path}")
    
    @classmethod
    def load(cls, save_path: str) -> 'BM25Retriever':
        """BM25 인덱스 로드"""
        save_path = Path(save_path)
        index_path = save_path / "bm25_index.pkl"
        
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        
        # 인스턴스 생성
        retriever = cls(
            tokenizer_method=data.get('tokenizer_method', 'simple'),
            k1=data.get('k1', 1.2),
            b=data.get('b', 0.75)
        )
        
        # 데이터 복원
        retriever.bm25_index = data['bm25_index']
        retriever.documents = data['documents']
        retriever.document_ids = data['document_ids']
        retriever.tokenized_documents = data['tokenized_documents']
        
        logger.info(f"BM25 index loaded from {index_path}")
        return retriever
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보"""
        if not self.bm25_index:
            return {"status": "not_built"}
        
        return {
            "status": "ready",
            "num_documents": len(self.documents),
            "tokenizer_method": self.tokenizer.method,
            "k1": self.k1,
            "b": self.b,
            "avg_doc_length": np.mean([len(tokens) for tokens in self.tokenized_documents]) if self.tokenized_documents else 0
        }


if __name__ == "__main__":
    # 테스트
    retriever = BM25Retriever(tokenizer_method="simple")
    
    # 테스트 문서
    documents = [
        "금융보안은 개인정보 보호의 핵심입니다",
        "피싱 공격을 방어하기 위한 보안 조치가 필요합니다",
        "암호화폐 거래 시 보안 주의사항을 확인하세요",
        "온라인 뱅킹 서비스 이용 시 보안 인증이 중요합니다"
    ]
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    
    # 인덱스 구축
    retriever.build_index(documents, doc_ids)
    
    # 검색 테스트
    results = retriever.search("보안 인증", k=3)
    
    print("BM25 Search Results:")
    for result in results:
        print(f"Doc ID: {result.doc_id}")
        print(f"Score: {result.score:.4f}")
        print(f"Content: {result.content}")
        print("-" * 50)
    
    # 통계 출력
    print("\nBM25 Stats:")
    stats = retriever.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")