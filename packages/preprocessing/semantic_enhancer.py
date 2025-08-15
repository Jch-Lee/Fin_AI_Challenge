"""
의미 기반 청킹 개선 모듈
LlamaIndex SemanticSplitterNodeParser를 활용한 선택적 청킹 개선
"""

import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import numpy as np

from .chunker import DocumentChunk

logger = logging.getLogger(__name__)

# LlamaIndex 가용성 확인
LLAMAINDEX_AVAILABLE = False
try:
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Document
    LLAMAINDEX_AVAILABLE = True
    logger.info("LlamaIndex available - semantic chunking enabled")
except ImportError:
    logger.warning("LlamaIndex not available - semantic enhancement disabled")
    SemanticSplitterNodeParser = None
    HuggingFaceEmbedding = None
    Document = None


@dataclass
class SemanticConfig:
    """의미 기반 청킹 설정"""
    embedding_model: str = "nlpai-lab/KURE-v1"
    buffer_size: int = 1
    breakpoint_percentile_threshold: int = 95
    max_chunk_size: int = 512
    min_chunk_size: int = 100
    cache_folder: str = "./cache"
    device: str = "auto"


class SemanticEnhancer:
    """
    기존 청크를 의미 기반으로 개선하는 클래스
    LlamaIndex SemanticSplitterNodeParser 활용
    """
    
    def __init__(self, config: Optional[SemanticConfig] = None):
        """
        Args:
            config: 의미 기반 청킹 설정
        """
        self.config = config or SemanticConfig()
        self.semantic_parser = None
        self.embed_model = None
        self._initialized = False
        
        if not LLAMAINDEX_AVAILABLE:
            logger.warning("LlamaIndex not available - semantic enhancement disabled")
            return
        
        logger.info("SemanticEnhancer initialized (lazy loading)")
    
    def _initialize_semantic_parser(self):
        """LlamaIndex 의미 파서 초기화 (지연 초기화)"""
        
        if self._initialized or not LLAMAINDEX_AVAILABLE:
            return
        
        try:
            # KURE 임베딩 모델 로드 (기존 시스템과 동일)
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.config.embedding_model,
                cache_folder=self.config.cache_folder,
                device=self.config.device,
                trust_remote_code=True
            )
            
            # SemanticSplitter 설정
            self.semantic_parser = SemanticSplitterNodeParser(
                embed_model=self.embed_model,
                buffer_size=self.config.buffer_size,
                breakpoint_percentile_threshold=self.config.breakpoint_percentile_threshold,
                max_chunk_size=self.config.max_chunk_size,
                show_progress=False  # 로그 출력 제어
            )
            
            self._initialized = True
            logger.info(f"Semantic parser initialized with {self.config.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic parser: {e}")
            self._initialized = False
    
    def is_available(self) -> bool:
        """의미 기반 청킹 사용 가능 여부"""
        return LLAMAINDEX_AVAILABLE
    
    def _should_enhance_chunk(self, chunk: DocumentChunk) -> bool:
        """청크가 의미 기반 개선 대상인지 판단"""
        
        # 너무 짧은 청크는 스킵
        if len(chunk.content) < self.config.min_chunk_size:
            return False
        
        # 너무 긴 청크만 처리 (성능 고려)
        if len(chunk.content) < 400:
            return False
        
        # 이미 의미 기반으로 처리된 청크는 스킵
        if chunk.metadata.get('semantic_enhanced'):
            return False
        
        # 테이블이나 리스트는 구조 보존을 위해 스킵
        content_lower = chunk.content.lower()
        if ('|' in chunk.content and '---' in chunk.content) or \
           chunk.content.count('\n-') > 3 or \
           chunk.content.count('\n1.') > 2:
            return False
        
        return True
    
    def _convert_chunks_to_documents(self, chunks: List[DocumentChunk]) -> List[Document]:
        """DocumentChunk를 LlamaIndex Document로 변환"""
        
        documents = []
        for chunk in chunks:
            # 메타데이터를 LlamaIndex 형식으로 변환
            metadata = chunk.metadata.copy()
            metadata.update({
                'chunk_id': chunk.chunk_id,
                'doc_id': chunk.doc_id,
                'chunk_index': chunk.chunk_index,
                'original_length': len(chunk.content)
            })
            
            doc = Document(
                text=chunk.content,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def _convert_nodes_to_chunks(self, nodes, original_chunk: DocumentChunk) -> List[DocumentChunk]:
        """LlamaIndex Node를 DocumentChunk로 변환"""
        
        chunks = []
        
        for i, node in enumerate(nodes):
            # 원본 청크의 메타데이터 복사
            new_metadata = original_chunk.metadata.copy()
            new_metadata.update({
                'semantic_enhanced': True,
                'semantic_split_index': i,
                'semantic_split_total': len(nodes),
                'semantic_method': 'llamaindex_semantic_splitter',
                'original_chunk_id': original_chunk.chunk_id,
                'enhanced_length': len(node.text)
            })
            
            # Node의 메타데이터도 병합
            if hasattr(node, 'metadata') and node.metadata:
                new_metadata.update(node.metadata)
            
            # 새로운 청크 ID 생성 (원본 + 서브 인덱스)
            new_chunk_id = f"{original_chunk.chunk_id}_sem_{i}"
            
            enhanced_chunk = DocumentChunk(
                content=node.text,
                metadata=new_metadata,
                chunk_id=new_chunk_id,
                doc_id=original_chunk.doc_id,
                chunk_index=original_chunk.chunk_index  # 원본 인덱스 유지
            )
            
            chunks.append(enhanced_chunk)
        
        return chunks
    
    def enhance_chunks(self, 
                      chunks: List[DocumentChunk],
                      selective: bool = True) -> List[DocumentChunk]:
        """
        청크 리스트를 의미 기반으로 개선
        
        Args:
            chunks: 개선할 청크 리스트
            selective: True이면 긴 청크만 선택적 처리
            
        Returns:
            개선된 청크 리스트
        """
        
        if not self.is_available():
            logger.warning("Semantic enhancement not available - returning original chunks")
            return chunks
        
        if not chunks:
            return chunks
        
        # 지연 초기화
        self._initialize_semantic_parser()
        
        if not self._initialized:
            logger.warning("Semantic parser initialization failed - returning original chunks")
            return chunks
        
        enhanced_chunks = []
        processed_count = 0
        
        for chunk in chunks:
            try:
                # 선택적 처리
                if selective and not self._should_enhance_chunk(chunk):
                    enhanced_chunks.append(chunk)
                    continue
                
                # LlamaIndex Document로 변환
                documents = [self._convert_chunks_to_documents([chunk])[0]]
                
                # 의미 기반 분할 수행
                nodes = self.semantic_parser.get_nodes_from_documents(documents)
                
                # 분할 결과 확인
                if len(nodes) <= 1:
                    # 분할되지 않은 경우 원본 유지
                    enhanced_chunks.append(chunk)
                else:
                    # 분할된 청크들로 교체
                    semantic_chunks = self._convert_nodes_to_chunks(nodes, chunk)
                    enhanced_chunks.extend(semantic_chunks)
                    processed_count += 1
                    
                    logger.debug(f"Chunk {chunk.chunk_id} split into {len(semantic_chunks)} semantic chunks")
                
            except Exception as e:
                logger.warning(f"Failed to enhance chunk {chunk.chunk_id}: {e}")
                # 실패 시 원본 청크 유지
                enhanced_chunks.append(chunk)
        
        logger.info(f"Semantic enhancement completed: {processed_count}/{len(chunks)} chunks processed")
        
        return enhanced_chunks
    
    def enhance_single_chunk(self, chunk: DocumentChunk) -> List[DocumentChunk]:
        """단일 청크 의미 기반 개선"""
        
        enhanced = self.enhance_chunks([chunk], selective=False)
        return enhanced
    
    def get_enhancement_stats(self, 
                            original_chunks: List[DocumentChunk],
                            enhanced_chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """개선 전후 통계 비교"""
        
        original_count = len(original_chunks)
        enhanced_count = len(enhanced_chunks)
        
        # 의미 기반으로 처리된 청크 수
        semantic_processed = sum(1 for chunk in enhanced_chunks 
                               if chunk.metadata.get('semantic_enhanced', False))
        
        # 평균 청크 크기
        original_avg_size = np.mean([len(chunk.content) for chunk in original_chunks]) if original_chunks else 0
        enhanced_avg_size = np.mean([len(chunk.content) for chunk in enhanced_chunks]) if enhanced_chunks else 0
        
        return {
            'original_count': original_count,
            'enhanced_count': enhanced_count,
            'semantic_processed': semantic_processed,
            'count_change_ratio': enhanced_count / original_count if original_count > 0 else 0,
            'original_avg_size': original_avg_size,
            'enhanced_avg_size': enhanced_avg_size,
            'size_change_ratio': enhanced_avg_size / original_avg_size if original_avg_size > 0 else 0,
            'enhancement_rate': semantic_processed / original_count if original_count > 0 else 0
        }


def enhance_chunks_semantic(chunks: List[DocumentChunk], 
                          config: Optional[SemanticConfig] = None,
                          selective: bool = True) -> List[DocumentChunk]:
    """
    편의 함수: 청크 리스트를 의미 기반으로 개선
    
    Args:
        chunks: 개선할 청크 리스트
        config: 의미 기반 청킹 설정
        selective: 선택적 처리 여부
        
    Returns:
        개선된 청크 리스트
    """
    
    enhancer = SemanticEnhancer(config)
    return enhancer.enhance_chunks(chunks, selective=selective)


if __name__ == "__main__":
    # 간단한 테스트
    print("=== SemanticEnhancer 테스트 ===")
    
    enhancer = SemanticEnhancer()
    print(f"LlamaIndex 사용 가능: {enhancer.is_available()}")
    
    if enhancer.is_available():
        # 테스트 청크 생성
        test_chunk = DocumentChunk(
            content="""
금융기관의 사이버 보안은 매우 중요한 문제입니다. 해커들의 공격이 점점 정교해지고 있어 새로운 대응 방안이 필요합니다.

다요소 인증은 기본적인 보안 조치 중 하나입니다. 사용자의 신원을 확인하기 위해 두 가지 이상의 인증 방법을 사용합니다.

최소 권한 원칙을 적용해야 합니다. 각 사용자에게는 업무 수행에 필요한 최소한의 권한만 부여해야 합니다. 이를 통해 보안 위험을 크게 줄일 수 있습니다.

모니터링 시스템은 실시간으로 모든 접근을 감시해야 합니다. 비정상적인 활동이 감지되면 즉시 차단하고 관리자에게 알려야 합니다.
            """.strip(),
            metadata={'test': True, 'source': 'manual'},
            chunk_id="test_001",
            doc_id="test_doc",
            chunk_index=0
        )
        
        print(f"원본 청크 길이: {len(test_chunk.content)}자")
        
        try:
            enhanced = enhancer.enhance_single_chunk(test_chunk)
            print(f"개선 후 청크 수: {len(enhanced)}")
            
            for i, chunk in enumerate(enhanced):
                print(f"  청크 {i+1}: {len(chunk.content)}자, 의미 개선: {chunk.metadata.get('semantic_enhanced', False)}")
        
        except Exception as e:
            print(f"테스트 실패 (정상 - LlamaIndex 미설치): {e}")
    
    else:
        print("LlamaIndex가 설치되지 않아 의미 기반 청킹을 사용할 수 없습니다.")
        print("설치하려면: pip install llama-index-core llama-index-embeddings-huggingface")