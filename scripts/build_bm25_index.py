#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BM25 인덱스 구축 스크립트
한국어 형태소 분석 기반 BM25 검색 인덱스 생성
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# BM25 관련 임포트
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    logger.warning("rank_bm25 not available, using simple BM25 implementation")
    BM25_AVAILABLE = False

# Kiwi 토크나이저 임포트
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    logger.warning("Kiwipiepy not available, using basic tokenization")
    KIWI_AVAILABLE = False


class SimpleBM25:
    """간단한 BM25 구현 (rank_bm25 없을 때 사용)"""
    
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self._initialize()
    
    def _initialize(self):
        """초기화"""
        nd = len(self.corpus)
        num_doc = 0
        for document in self.corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)
            
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)
            
            for word, freq in frequencies.items():
                if word not in self.idf:
                    self.idf[word] = 0
                self.idf[word] += 1
        
        self.avgdl = num_doc / nd
        
        # IDF 계산
        for word, freq in self.idf.items():
            self.idf[word] = np.log((nd - freq + 0.5) / (freq + 0.5))
    
    def get_scores(self, query):
        """BM25 스코어 계산"""
        scores = []
        for idx, doc in enumerate(self.corpus):
            score = 0
            doc_freqs = self.doc_freqs[idx]
            dl = self.doc_len[idx]
            
            for word in query:
                if word not in doc_freqs:
                    continue
                
                freq = doc_freqs[word]
                idf = self.idf.get(word, 0)
                numerator = idf * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += numerator / denominator
            
            scores.append(score)
        
        return scores


class KoreanTokenizer:
    """한국어 토크나이저"""
    
    def __init__(self, use_kiwi: bool = True):
        """
        Args:
            use_kiwi: Kiwi 사용 여부
        """
        self.use_kiwi = use_kiwi and KIWI_AVAILABLE
        
        if self.use_kiwi:
            self.kiwi = Kiwi()
            self.kiwi.prepare()
            logger.info("Using Kiwi tokenizer")
        else:
            logger.info("Using basic tokenizer")
    
    def tokenize(self, text: str) -> List[str]:
        """
        텍스트 토크나이징
        
        Args:
            text: 입력 텍스트
            
        Returns:
            토큰 리스트
        """
        if self.use_kiwi:
            # Kiwi 형태소 분석
            result = self.kiwi.tokenize(text)
            
            # 명사, 동사, 형용사, 외국어만 추출
            tokens = []
            for token in result:
                if token.tag[0] in ['N', 'V'] or token.tag in ['VA', 'SL']:
                    tokens.append(token.form)
            
            return tokens
        else:
            # 기본 토크나이징 (공백 기준)
            import re
            # 특수문자 제거하고 공백으로 분리
            text = re.sub(r'[^\w\s]', ' ', text)
            tokens = text.split()
            # 2글자 이상만 필터링
            return [t for t in tokens if len(t) >= 2]


class BM25IndexBuilder:
    """BM25 인덱스 구축기"""
    
    def __init__(self,
                 tokenizer: str = "kiwi",
                 k1: float = 1.5,
                 b: float = 0.75):
        """
        Args:
            tokenizer: 토크나이저 종류
            k1: BM25 k1 파라미터
            b: BM25 b 파라미터
        """
        self.k1 = k1
        self.b = b
        
        # 토크나이저 초기화
        if tokenizer == "kiwi":
            self.tokenizer = KoreanTokenizer(use_kiwi=True)
        else:
            self.tokenizer = KoreanTokenizer(use_kiwi=False)
        
        logger.info(f"BM25IndexBuilder initialized with {tokenizer} tokenizer")
    
    def load_documents(self, input_path: str) -> List[Dict]:
        """
        문서 로드
        
        Args:
            input_path: 입력 경로 (디렉토리 또는 JSON 파일)
            
        Returns:
            문서 리스트
        """
        input_path = Path(input_path)
        
        if input_path.is_dir():
            # 디렉토리에서 텍스트 파일 로드
            documents = []
            txt_files = list(input_path.glob("*.txt"))
            
            logger.info(f"Loading {len(txt_files)} text files from directory")
            
            for file_path in tqdm(txt_files, desc="Loading documents"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc = {
                        'id': file_path.stem,
                        'content': content,
                        'source': str(file_path)
                    }
                    documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
            
        elif input_path.suffix == '.json':
            # JSON 파일에서 로드 (chunks.json)
            logger.info(f"Loading documents from JSON: {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
        
        else:
            raise ValueError(f"Invalid input path: {input_path}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def tokenize_documents(self, documents: List[Dict]) -> List[List[str]]:
        """
        문서들을 토크나이징
        
        Args:
            documents: 문서 리스트
            
        Returns:
            토크나이즈된 문서 리스트
        """
        tokenized_docs = []
        
        for doc in tqdm(documents, desc="Tokenizing documents"):
            content = doc.get('content', '')
            tokens = self.tokenizer.tokenize(content)
            tokenized_docs.append(tokens)
        
        # 통계
        token_counts = [len(tokens) for tokens in tokenized_docs]
        logger.info(f"Tokenization complete:")
        logger.info(f"  Average tokens per doc: {np.mean(token_counts):.1f}")
        logger.info(f"  Min tokens: {np.min(token_counts)}")
        logger.info(f"  Max tokens: {np.max(token_counts)}")
        
        return tokenized_docs
    
    def build_bm25_index(self, tokenized_docs: List[List[str]]):
        """
        BM25 인덱스 생성
        
        Args:
            tokenized_docs: 토크나이즈된 문서들
            
        Returns:
            BM25 인덱스
        """
        logger.info("Building BM25 index...")
        
        if BM25_AVAILABLE:
            # rank_bm25 사용
            bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        else:
            # 간단한 구현 사용
            bm25 = SimpleBM25(tokenized_docs, k1=self.k1, b=self.b)
        
        logger.info("BM25 index built successfully")
        return bm25
    
    def save_index(self, 
                   bm25_index,
                   documents: List[Dict],
                   tokenized_docs: List[List[str]],
                   output_path: str):
        """
        인덱스 저장
        
        Args:
            bm25_index: BM25 인덱스
            documents: 원본 문서들
            tokenized_docs: 토크나이즈된 문서들
            output_path: 출력 경로
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 저장할 데이터
        index_data = {
            'bm25': bm25_index,
            'documents': documents,
            'tokenized_docs': tokenized_docs,
            'metadata': {
                'num_docs': len(documents),
                'tokenizer': 'kiwi' if self.tokenizer.use_kiwi else 'basic',
                'k1': self.k1,
                'b': self.b
            }
        }
        
        # Pickle로 저장
        logger.info(f"Saving BM25 index to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        # 파일 크기 확인
        file_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Index saved: {file_size:.2f} MB")
    
    def build(self, input_path: str, output_path: str):
        """
        전체 BM25 인덱스 구축 프로세스
        
        Args:
            input_path: 입력 경로
            output_path: 출력 경로
        """
        # 1. 문서 로드
        documents = self.load_documents(input_path)
        
        if not documents:
            logger.error("No documents found!")
            return
        
        # 2. 토크나이징
        tokenized_docs = self.tokenize_documents(documents)
        
        # 3. BM25 인덱스 구축
        bm25_index = self.build_bm25_index(tokenized_docs)
        
        # 4. 저장
        self.save_index(bm25_index, documents, tokenized_docs, output_path)
        
        # 5. 통계 출력
        self._print_statistics(documents, tokenized_docs)
    
    def _print_statistics(self, documents: List[Dict], tokenized_docs: List[List[str]]):
        """통계 출력"""
        print("\n" + "="*60)
        print(" BM25 인덱스 구축 완료")
        print("="*60)
        print(f"총 문서 수: {len(documents)}")
        print(f"토크나이저: {'Kiwi' if self.tokenizer.use_kiwi else 'Basic'}")
        print(f"BM25 파라미터: k1={self.k1}, b={self.b}")
        
        # 토큰 통계
        token_counts = [len(tokens) for tokens in tokenized_docs]
        unique_tokens = set()
        for tokens in tokenized_docs:
            unique_tokens.update(tokens)
        
        print(f"\n토큰 통계:")
        print(f"  총 고유 토큰: {len(unique_tokens)}")
        print(f"  문서당 평균 토큰: {np.mean(token_counts):.1f}")
        print(f"  문서당 최소 토큰: {np.min(token_counts)}")
        print(f"  문서당 최대 토큰: {np.max(token_counts)}")
        print("="*60)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BM25 인덱스 구축")
    parser.add_argument("--input-path", type=str, required=True,
                       help="입력 경로 (텍스트 디렉토리 또는 chunks.json)")
    parser.add_argument("--output-path", type=str, required=True,
                       help="출력 인덱스 경로 (.pkl)")
    parser.add_argument("--tokenizer", type=str, default="kiwi",
                       choices=["kiwi", "basic"],
                       help="토크나이저 종류")
    parser.add_argument("--k1", type=float, default=1.5,
                       help="BM25 k1 파라미터")
    parser.add_argument("--b", type=float, default=0.75,
                       help="BM25 b 파라미터")
    
    args = parser.parse_args()
    
    # 빌더 생성
    builder = BM25IndexBuilder(
        tokenizer=args.tokenizer,
        k1=args.k1,
        b=args.b
    )
    
    # 인덱스 구축
    builder.build(
        input_path=args.input_path,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()