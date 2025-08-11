"""
임베딩 생성 모듈
Sentence Transformers를 사용한 텍스트 임베딩
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """텍스트 임베딩 생성 클래스"""
    
    def __init__(self, 
                 model_name: str = "nlpai-lab/KURE-v1",
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 show_progress: bool = True):
        """
        Args:
            model_name: Sentence Transformer 모델명
            device: 연산 디바이스 ('cuda', 'cpu', None=자동)
            batch_size: 배치 크기
            show_progress: 진행 상황 표시 여부
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        # 디바이스 설정
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Embedding generator using device: {self.device}")
        
        # 모델 로드
        self._load_model()
        
    def _load_model(self):
        """모델 로드"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # 대체 모델 사용 (한국어 특화 모델 우선)
            logger.info("Falling back to Korean multilingual model")
            self.model_name = "jhgan/ko-sroberta-multitask"
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            except Exception:
                # 최후 대체 모델
                logger.info("Falling back to base multilingual model")
                self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def generate_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """단일 텍스트 또는 텍스트 리스트의 임베딩 생성"""
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(
            text,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 정규화
        )
        
        return embeddings if len(text) > 1 else embeddings[0]
    
    def generate_embeddings_batch(self, 
                                 texts: List[str],
                                 normalize: bool = True) -> np.ndarray:
        """배치 단위로 임베딩 생성"""
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def generate_chunk_embeddings(self, 
                                 chunks: List[Dict],
                                 content_key: str = 'content') -> List[Dict]:
        """청크 리스트에 임베딩 추가"""
        texts = [chunk[content_key] for chunk in chunks]
        embeddings = self.generate_embeddings_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        return chunks
    
    def compute_similarity(self, 
                         query_embedding: np.ndarray,
                         doc_embeddings: np.ndarray) -> np.ndarray:
        """코사인 유사도 계산"""
        # 이미 정규화된 경우 단순 내적이 코사인 유사도
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
        return similarities
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """임베딩을 파일로 저장"""
        np.save(filepath, embeddings)
        logger.info(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """파일에서 임베딩 로드"""
        embeddings = np.load(filepath)
        logger.info(f"Loaded embeddings from {filepath}")
        return embeddings
    
    def save_model_cache(self, cache_dir: str):
        """모델 캐시 저장 (오프라인 사용)"""
        self.model.save(cache_dir)
        logger.info(f"Saved model cache to {cache_dir}")
    
    @staticmethod
    def load_from_cache(cache_dir: str, device: Optional[str] = None):
        """캐시에서 모델 로드"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = SentenceTransformer(cache_dir, device=device)
        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        generator.model = model
        generator.device = device
        generator.embedding_dim = model.get_sentence_embedding_dimension()
        generator.batch_size = 32
        generator.show_progress = True
        
        return generator


class HybridEmbedding:
    """하이브리드 임베딩 (Dense + Sparse)"""
    
    def __init__(self, 
                 dense_model_name: str = "nlpai-lab/KURE-v1",
                 use_bm25: bool = True):
        """
        Args:
            dense_model_name: Dense 임베딩 모델
            use_bm25: BM25 스파스 임베딩 사용 여부
        """
        self.dense_generator = EmbeddingGenerator(dense_model_name)
        self.use_bm25 = use_bm25
        
        if use_bm25:
            try:
                import bm25s
                self.bm25_enabled = True
            except ImportError:
                logger.warning("BM25S not installed. Using dense embeddings only.")
                self.bm25_enabled = False
                self.use_bm25 = False
    
    def generate_hybrid_embedding(self, 
                                 text: str,
                                 alpha: float = 0.7) -> Dict:
        """하이브리드 임베딩 생성
        
        Args:
            text: 입력 텍스트
            alpha: Dense 임베딩 가중치 (1-alpha는 Sparse 가중치)
        """
        result = {
            'text': text,
            'dense_embedding': None,
            'sparse_features': None,
            'alpha': alpha
        }
        
        # Dense 임베딩
        result['dense_embedding'] = self.dense_generator.generate_embedding(text)
        
        # Sparse 특징 (BM25용 토큰화된 텍스트)
        if self.use_bm25 and self.bm25_enabled:
            # 간단한 토큰화 (실제로는 형태소 분석기 사용 권장)
            tokens = text.lower().split()
            result['sparse_features'] = tokens
        
        return result


# 별칭 추가 (호환성 유지)
TextEmbedder = EmbeddingGenerator

# embedder.py 단순화된 인터페이스
class TextEmbedder:
    """간단한 텍스트 임베딩 인터페이스"""
    
    def __init__(self, model_name: str = "nlpai-lab/KURE-v1"):
        self.generator = EmbeddingGenerator(model_name=model_name)
        self.embedding_dim = self.generator.embedding_dim
    
    def embed(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩"""
        return self.generator.generate_embedding(text)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """배치 임베딩"""
        self.generator.batch_size = batch_size
        return self.generator.generate_embeddings_batch(texts)


if __name__ == "__main__":
    # 테스트
    generator = EmbeddingGenerator()
    
    test_texts = [
        "금융보안은 매우 중요합니다.",
        "Financial security is very important.",
        "디지털 자산을 안전하게 보호해야 합니다."
    ]
    
    # 단일 텍스트 임베딩
    embedding = generator.generate_embedding(test_texts[0])
    print(f"Single embedding shape: {embedding.shape}")
    
    # 배치 임베딩
    embeddings = generator.generate_embeddings_batch(test_texts)
    print(f"Batch embeddings shape: {embeddings.shape}")
    
    # 유사도 계산
    query_emb = embeddings[0]
    doc_embs = embeddings[1:]
    similarities = generator.compute_similarity(query_emb, doc_embs)
    print(f"Similarities: {similarities}")