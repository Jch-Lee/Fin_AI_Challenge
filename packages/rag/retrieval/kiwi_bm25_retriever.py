"""
Kiwi 기반 고성능 BM25 검색기
- KoNLPy 대비 10-30배 빠른 토크나이징
- 사용자 사전 없이 순수 Kiwi 기능만 사용 (대회 규정 준수)
"""

from typing import List, Optional, Dict, Any
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Kiwi 사용 가능 여부 확인
KIWI_AVAILABLE = False
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    logger.warning("Kiwi not available for BM25 tokenization")

# BM25 라이브러리 확인
BM25_AVAILABLE = False
try:
    import bm25s
    BM25_AVAILABLE = True
except ImportError:
    logger.warning("bm25s not available")


@dataclass
class BM25Result:
    """BM25 검색 결과"""
    doc_id: str
    content: str
    score: float
    rank: int


class KiwiBM25Tokenizer:
    """
    Kiwi 기반 토크나이저 (BM25 전용)
    - 사용자 사전 없음
    - 기본 형태소 분석만 사용
    - 고속 처리 최적화
    """
    
    def __init__(self, 
                 min_token_length: int = 2,
                 extract_pos: List[str] = None,
                 lowercase: bool = True):
        """
        Args:
            min_token_length: 최소 토큰 길이
            extract_pos: 추출할 품사 태그 (None이면 기본값 사용)
            lowercase: 소문자 변환 여부
        """
        if not KIWI_AVAILABLE:
            raise ImportError("Kiwi is required for KiwiBM25Tokenizer")
        
        self.kiwi = Kiwi()
        self.min_token_length = min_token_length
        self.lowercase = lowercase
        
        # 추출할 품사 설정 (의미 있는 품사만)
        if extract_pos is None:
            self.extract_pos = [
                'N',   # 명사 (NNG, NNP, NNB 등)
                'V',   # 동사 (VV, VA 등)
                'VA',  # 형용사
                'SL',  # 외국어
            ]
        else:
            self.extract_pos = extract_pos
        
        # 토크나이징 통계
        self.stats = {
            'total_docs': 0,
            'total_tokens': 0,
            'failed_docs': 0
        }
    
    def tokenize(self, text: str) -> List[str]:
        """
        BM25용 토큰 추출
        - 명사, 동사, 형용사, 외국어만 추출
        - 띄어쓰기 교정 포함
        - 길이 필터링 적용
        """
        if not text or not text.strip():
            return []
        
        try:
            # 띄어쓰기 교정 (검색 정확도 향상)
            text = self.kiwi.space(text)
            
            # 형태소 분석
            tokens = []
            for token in self.kiwi.tokenize(text):
                # 품사 필터링
                pos_match = False
                for pos_prefix in self.extract_pos:
                    if token.tag.startswith(pos_prefix):
                        pos_match = True
                        break
                
                if not pos_match:
                    continue
                
                # 길이 필터링 (외국어는 예외)
                if len(token.form) < self.min_token_length and not token.tag.startswith('SL'):
                    continue
                
                # 토큰 추가
                token_form = token.form.lower() if self.lowercase else token.form
                tokens.append(token_form)
            
            # 통계 업데이트
            self.stats['total_tokens'] += len(tokens)
            
            return tokens
            
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            self.stats['failed_docs'] += 1
            
            # Fallback: 단순 공백 분리
            if self.lowercase:
                return text.lower().split()
            return text.split()
    
    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """
        배치 토크나이징
        """
        self.stats['total_docs'] += len(texts)
        return [self.tokenize(text) for text in texts]
    
    def get_stats(self) -> Dict:
        """토크나이징 통계 반환"""
        return self.stats.copy()


class KiwiBM25Retriever:
    """
    Kiwi 토크나이저를 사용하는 고성능 BM25 검색기
    """
    
    def __init__(self, 
                 k1: float = 1.2,
                 b: float = 0.75,
                 epsilon: float = 0.25,
                 min_token_length: int = 2):
        """
        Args:
            k1: BM25 term frequency saturation parameter
            b: BM25 length normalization parameter
            epsilon: BM25 floor value for IDF
            min_token_length: 최소 토큰 길이
        """
        if not KIWI_AVAILABLE:
            raise ImportError("Kiwi is required for KiwiBM25Retriever")
        
        if not BM25_AVAILABLE:
            raise ImportError("bm25s is required for KiwiBM25Retriever")
        
        # Kiwi 토크나이저 초기화
        self.tokenizer = KiwiBM25Tokenizer(
            min_token_length=min_token_length,
            lowercase=True
        )
        
        # BM25 파라미터
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # 인덱스 및 문서 저장
        self.bm25_index = None
        self.documents = []
        self.document_ids = []
        self.tokenized_documents = []
        
        logger.info(f"KiwiBM25Retriever initialized (k1={k1}, b={b})")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Kiwi 기반 고속 토크나이징
        """
        return self.tokenizer.tokenize(text)
    
    def build_index(self, 
                   documents: List[str],
                   document_ids: Optional[List[str]] = None,
                   show_progress: bool = True) -> None:
        """
        BM25 인덱스 구축
        
        Args:
            documents: 문서 리스트
            document_ids: 문서 ID 리스트 (없으면 자동 생성)
            show_progress: 진행 상황 표시
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        # 문서 ID 생성
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(documents))]
        
        if len(documents) != len(document_ids):
            raise ValueError("Documents and IDs must have the same length")
        
        # 문서 저장
        self.documents = documents
        self.document_ids = document_ids
        
        # 토크나이징
        logger.info(f"Tokenizing {len(documents)} documents with Kiwi...")
        
        if show_progress:
            from tqdm import tqdm
            self.tokenized_documents = [
                self.tokenizer.tokenize(doc) 
                for doc in tqdm(documents, desc="Tokenizing")
            ]
        else:
            self.tokenized_documents = self.tokenizer.batch_tokenize(documents)
        
        # BM25 인덱스 생성
        logger.info("Building BM25 index...")
        self.bm25_index = bm25s.BM25(
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon
        )
        
        # 인덱스 피팅
        self.bm25_index.index(self.tokenized_documents)
        
        # 통계 출력
        stats = self.tokenizer.get_stats()
        avg_tokens = stats['total_tokens'] / max(stats['total_docs'], 1)
        logger.info(f"Index built: {len(documents)} docs, {stats['total_tokens']} tokens, "
                   f"avg {avg_tokens:.1f} tokens/doc")
    
    def search(self, 
              query: str,
              k: int = 10,
              min_score: float = 0.0) -> List[BM25Result]:
        """
        BM25 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            min_score: 최소 점수 임계값
        """
        if self.bm25_index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # 쿼리 토크나이징
        query_tokens = self.tokenizer.tokenize(query)
        
        if not query_tokens:
            logger.warning(f"No tokens extracted from query: {query}")
            return []
        
        # BM25 검색
        scores, indices = self.bm25_index.retrieve(
            query_tokens, 
            k=min(k, len(self.documents))
        )
        
        # 결과 생성
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if score < min_score:
                continue
                
            results.append(BM25Result(
                doc_id=self.document_ids[idx],
                content=self.documents[idx],
                score=float(score),
                rank=rank + 1
            ))
        
        return results
    
    def batch_search(self, 
                    queries: List[str],
                    k: int = 10,
                    min_score: float = 0.0) -> List[List[BM25Result]]:
        """
        배치 검색
        """
        return [self.search(query, k, min_score) for query in queries]
    
    def update_index(self, 
                    new_documents: List[str],
                    new_document_ids: Optional[List[str]] = None) -> None:
        """
        인덱스 업데이트 (새 문서 추가)
        """
        if not new_documents:
            return
        
        # 새 문서 ID 생성
        if new_document_ids is None:
            start_idx = len(self.documents)
            new_document_ids = [f"doc_{i}" for i in range(start_idx, start_idx + len(new_documents))]
        
        # 기존 문서와 합치기
        all_documents = self.documents + new_documents
        all_document_ids = self.document_ids + new_document_ids
        
        # 인덱스 재구축
        self.build_index(all_documents, all_document_ids, show_progress=False)
    
    def save_index(self, path: str) -> None:
        """인덱스 저장"""
        import pickle
        
        data = {
            'bm25_index': self.bm25_index,
            'documents': self.documents,
            'document_ids': self.document_ids,
            'tokenized_documents': self.tokenized_documents,
            'params': {
                'k1': self.k1,
                'b': self.b,
                'epsilon': self.epsilon
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str) -> None:
        """인덱스 로드"""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25_index = data['bm25_index']
        self.documents = data['documents']
        self.document_ids = data['document_ids']
        self.tokenized_documents = data['tokenized_documents']
        self.k1 = data['params']['k1']
        self.b = data['params']['b']
        self.epsilon = data['params']['epsilon']
        
        logger.info(f"Index loaded from {path}: {len(self.documents)} documents")


# 테스트 코드
if __name__ == "__main__":
    # 기본 동작 테스트
    retriever = KiwiBM25Retriever()
    
    # 테스트 문서
    test_docs = [
        "한국은행이 기준금리를 인상했습니다. 금융시장이 크게 반응했습니다.",
        "금융위원회가 새로운 가상자산 규제를 발표했습니다.",
        "인터넷뱅킹 보안이 강화되었습니다. 2단계 인증이 의무화됩니다.",
        "디지털 자산 거래소에서 해킹 사고가 발생했습니다.",
        "중앙은행 디지털화폐(CBDC) 도입이 논의되고 있습니다."
    ]
    
    doc_ids = [f"doc_{i}" for i in range(len(test_docs))]
    
    # 인덱스 구축
    print("Building index...")
    retriever.build_index(test_docs, doc_ids)
    
    # 검색 테스트
    queries = [
        "기준금리 인상",
        "가상자산 규제",
        "인터넷뱅킹 보안"
    ]
    
    print("\n=== 검색 테스트 ===")
    for query in queries:
        print(f"\n쿼리: {query}")
        results = retriever.search(query, k=3)
        
        for result in results:
            print(f"  [{result.rank}] Score: {result.score:.3f}")
            print(f"      {result.content[:100]}...")
    
    # 통계 출력
    stats = retriever.tokenizer.get_stats()
    print(f"\n토크나이징 통계: {stats}")