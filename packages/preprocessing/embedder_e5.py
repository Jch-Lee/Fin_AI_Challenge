"""
E5 기반 한국어 임베딩 모듈
dragonkue/multilingual-e5-small-ko 사용 (384차원)
한국어 RAG 시스템 최적화
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional
from transformers import AutoTokenizer, AutoModel
import logging
import os

logger = logging.getLogger(__name__)


class E5Embedder:
    """
    E5 한국어 임베딩 모델
    PyTorch 2.1.0 호환, 384차원 출력
    """
    
    def __init__(self, 
                 model_name: str = "dragonkue/multilingual-e5-small-ko",
                 device: Optional[str] = None,
                 cache_dir: str = "./models"):
        """
        Args:
            model_name: 사용할 E5 모델 (기본: dragonkue/multilingual-e5-small-ko)
                - dragonkue/multilingual-e5-small-ko (384차원, 한국어 특화)
            device: 연산 디바이스 (None이면 자동 선택)
            cache_dir: 모델 캐시 디렉토리
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.dimension = 384  # dragonkue e5-small-ko 차원
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Loading E5 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        
        # 임베딩 차원 확인 및 설정
        self.embedding_dim = self.dimension  # 384로 설정
        
        # 실제 차원 검증
        with torch.no_grad():
            test_input = self.tokenizer(["test"], return_tensors="pt").to(self.device)
            test_output = self.model(**test_input).last_hidden_state
            actual_dim = test_output.shape[-1]
            if actual_dim != self.dimension:
                logger.warning(f"Expected dimension {self.dimension}, got {actual_dim}")
                self.embedding_dim = actual_dim
        
        logger.info(f"E5 model loaded. Embedding dimension: {self.embedding_dim}")
    
    def mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling for sentence embeddings"""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    
    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: int = 32,
        max_length: int = 512
    ) -> np.ndarray:
        """
        텍스트를 임베딩으로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            is_query: 쿼리인지 문서인지 구분 (E5는 prefix 사용)
            batch_size: 배치 크기
            max_length: 최대 토큰 길이
        
        Returns:
            임베딩 배열 (N, 384)
        """
        # E5 모델은 query/passage prefix 사용
        prefix = "query: " if is_query else "passage: "
        prefixed_texts = [prefix + text for text in texts]
        
        all_embeddings = []
        
        for i in range(0, len(prefixed_texts), batch_size):
            batch_texts = prefixed_texts[i:i + batch_size]
            
            # 토크나이징
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # 모델 추론
            outputs = self.model(**batch)
            embeddings = self.mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            
            # L2 정규화
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def embed(self, text: str, is_query: bool = False) -> np.ndarray:
        """단일 텍스트 임베딩"""
        return self.encode([text], is_query=is_query)[0]
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        is_query: bool = False
    ) -> np.ndarray:
        """배치 임베딩 (TextEmbedder 호환 인터페이스)"""
        return self.encode(texts, is_query=is_query, batch_size=batch_size)
    
    def get_embedding_dim(self) -> int:
        """임베딩 차원 반환"""
        return self.embedding_dim


# 기존 TextEmbedder와 호환되는 래퍼
class TextEmbedder:
    """
    기존 코드와의 호환성을 위한 래퍼 클래스
    E5 모델을 기본으로 사용
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Args:
            model_name: 모델명 (무시되고 E5 사용)
        """
        # E5 한국어 모델 우선, 실패 시 다국어 모델
        try:
            self.embedder = E5Embedder("dragonkue/multilingual-e5-small-ko")
            logger.info("Using Korean-optimized E5 model")
        except Exception as e:
            logger.warning(f"Korean E5 failed: {e}, falling back to multilingual")
            self.embedder = E5Embedder("intfloat/multilingual-e5-small")
        
        self.embedding_dim = self.embedder.embedding_dim
    
    def embed(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩"""
        return self.embedder.embed(text, is_query=False)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """배치 임베딩"""
        return self.embedder.embed_batch(texts, batch_size, is_query=False)
    
    def embed_query(self, query: str) -> np.ndarray:
        """쿼리 임베딩 (E5 prefix 적용)"""
        return self.embedder.embed(query, is_query=True)