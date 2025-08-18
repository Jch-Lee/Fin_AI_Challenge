"""
하이브리드 청킹 모듈 (2300자 청크 크기)
섹션 경계 오버랩을 포함한 개선된 청킹 방식
"""

import re
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

from .chunker import DocumentChunk
from .text_cleaner import ChunkCleaner

logger = logging.getLogger(__name__)


@dataclass
class HybridChunkConfig:
    """하이브리드 청킹 설정"""
    chunk_size: int = 2300
    chunk_overlap: int = 200
    boundary_overlap: int = 300
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", "。", "! ", "? ", " "]


class HybridChunker2300:
    """
    2300자 크기의 하이브리드 청킹
    - 큰 청크 크기로 더 많은 문맥 보존
    - 섹션 경계에 특별한 오버랩 청크 생성
    - 메타데이터 풍부화
    """
    
    def __init__(self, 
                 config: Optional[HybridChunkConfig] = None,
                 use_chunk_cleaner: bool = True):
        """
        Args:
            config: 청킹 설정 (None이면 기본값 사용)
            use_chunk_cleaner: ChunkCleaner 사용 여부
        """
        self.config = config or HybridChunkConfig()
        
        # 마크다운 헤더 파서 (주요 섹션만 구분)
        self.markdown_parser = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2")
            ],
            return_each_line=False,
            strip_headers=False
        )
        
        self.chunk_cleaner = ChunkCleaner() if use_chunk_cleaner else None
        
        logger.info(f"HybridChunker2300 초기화: "
                   f"chunk_size={self.config.chunk_size}, "
                   f"overlap={self.config.chunk_overlap}, "
                   f"boundary_overlap={self.config.boundary_overlap}")
    
    def _merge_page_boundaries(self, text: str) -> str:
        """페이지 연속성 마커 처리"""
        # [다음 페이지에 계속] ... [이전 페이지에서 계속] 패턴 병합
        pattern = r'\[다음 페이지에 계속\]\s*\n+.*?\[이전 페이지에서 계속\]\s*'
        text = re.sub(pattern, ' ', text, flags=re.DOTALL)
        
        # 남은 마커 제거
        text = re.sub(r'\[(다음|이전) 페이지(에서?|에) 계속\]', '', text)
        
        # 연속된 공백 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _generate_doc_id(self, document: str) -> str:
        """문서 고유 ID 생성"""
        return hashlib.md5(document[:500].encode()).hexdigest()[:16]
    
    def _generate_chunk_id(self, content: str, doc_id: str, index: int) -> str:
        """청크 고유 ID 생성"""
        hash_input = f"{doc_id}_{index}_{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _create_boundary_chunk(self, 
                              previous_content: str, 
                              current_content: str,
                              prev_section_idx: int,
                              curr_section_idx: int,
                              doc_id: str,
                              chunk_index: int) -> DocumentChunk:
        """섹션 경계 청크 생성"""
        
        # 경계 청크 내용 구성
        boundary_content = (
            f"{previous_content[-self.config.boundary_overlap:]}\n"
            f"{'=' * 60}\n"
            f"[섹션 전환: {prev_section_idx} → {curr_section_idx}]\n"
            f"{'=' * 60}\n"
            f"{current_content[:self.config.boundary_overlap]}"
        )
        
        # 메타데이터 구성
        metadata = {
            'is_boundary_chunk': True,
            'from_section': prev_section_idx,
            'to_section': curr_section_idx,
            'chunk_size': self.config.chunk_size,
            'boundary_overlap': self.config.boundary_overlap,
            'chunking_method': 'hybrid_2300',
            'content_length': len(boundary_content)
        }
        
        return DocumentChunk(
            content=boundary_content,
            metadata=metadata,
            chunk_id=self._generate_chunk_id(boundary_content, doc_id, chunk_index),
            doc_id=doc_id,
            chunk_index=chunk_index
        )
    
    def chunk_document(self, 
                      document: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        문서를 하이브리드 방식으로 청킹
        
        Args:
            document: 처리할 문서 텍스트
            metadata: 문서 메타데이터
            
        Returns:
            DocumentChunk 리스트
        """
        if not document or not document.strip():
            return []
        
        if metadata is None:
            metadata = {}
        
        # 1. 페이지 경계 처리
        processed_text = self._merge_page_boundaries(document)
        
        # 2. 마크다운 구조 파싱 (주요 섹션으로 분리)
        try:
            sections = self.markdown_parser.split_text(processed_text)
            logger.debug(f"문서를 {len(sections)}개 섹션으로 분리")
        except Exception as e:
            logger.warning(f"마크다운 파싱 실패: {e}, 전체를 단일 섹션으로 처리")
            # Fallback: 전체를 단일 섹션으로
            sections = [type('MockSection', (), {
                'page_content': processed_text,
                'metadata': {}
            })()]
        
        doc_id = self._generate_doc_id(document)
        all_chunks = []
        chunk_index = 0
        previous_section_content = ""
        
        # 3. 섹션별 처리
        for section_idx, section in enumerate(sections):
            section_content = section.page_content
            
            # 섹션 정보 로깅
            header_info = (section.metadata.get('h1', '') or 
                          section.metadata.get('h2', ''))[:50]
            if header_info:
                logger.debug(f"섹션 {section_idx}: {header_info}... "
                           f"({len(section_content)}자)")
            
            # 4. 섹션 경계 청크 생성 (첫 섹션 제외)
            if previous_section_content and section_idx > 0:
                boundary_chunk = self._create_boundary_chunk(
                    previous_section_content,
                    section_content,
                    section_idx - 1,
                    section_idx,
                    doc_id,
                    chunk_index
                )
                all_chunks.append(boundary_chunk)
                chunk_index += 1
            
            # 5. 섹션 내용 청킹
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=self.config.separators,
                length_function=len,
                is_separator_regex=False
            )
            
            chunk_texts = splitter.split_text(section_content)
            logger.debug(f"  → {len(chunk_texts)}개 청크 생성")
            
            # 6. 각 청크를 DocumentChunk로 변환
            for i, chunk_text in enumerate(chunk_texts):
                # 청크 정제 (옵션)
                if self.chunk_cleaner:
                    cleaned_text = self.chunk_cleaner.clean_chunk(chunk_text)
                    if not cleaned_text or len(cleaned_text) < 50:
                        continue
                    chunk_text = cleaned_text
                
                # 메타데이터 구성
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    # 섹션 정보
                    'section_idx': section_idx,
                    'chunk_in_section': i,
                    'total_chunks_in_section': len(chunk_texts),
                    
                    # 헤더 정보
                    'section_headers': section.metadata,
                    
                    # 청킹 설정
                    'chunk_size': self.config.chunk_size,
                    'chunk_overlap': self.config.chunk_overlap,
                    'chunking_method': 'hybrid_2300',
                    
                    # 기본 정보
                    'content_length': len(chunk_text),
                    'is_boundary_chunk': False
                })
                
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=self._generate_chunk_id(chunk_text, doc_id, chunk_index),
                    doc_id=doc_id,
                    chunk_index=chunk_index
                )
                
                all_chunks.append(chunk)
                chunk_index += 1
            
            # 다음 섹션을 위해 현재 섹션 내용 저장
            previous_section_content = section_content
        
        # 7. 총 청크 수 업데이트
        for chunk in all_chunks:
            chunk.metadata['total_chunks'] = len(all_chunks)
        
        logger.info(f"문서 청킹 완료: {len(all_chunks)}개 청크 생성 "
                   f"(일반: {sum(1 for c in all_chunks if not c.metadata.get('is_boundary_chunk'))}, "
                   f"경계: {sum(1 for c in all_chunks if c.metadata.get('is_boundary_chunk'))})")
        
        return all_chunks
    
    def chunk_documents(self, 
                       documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """여러 문서 배치 처리"""
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # 문서별 메타데이터 추가
            metadata['doc_index'] = doc_idx
            if 'filename' in doc:
                metadata['filename'] = doc['filename']
            
            chunks = self.chunk_document(content, metadata)
            all_chunks.extend(chunks)
            
            logger.info(f"문서 {doc_idx+1}/{len(documents)} 처리 완료")
        
        return all_chunks
    
    def get_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """청킹 통계 생성"""
        if not chunks:
            return {}
        
        regular_chunks = [c for c in chunks if not c.metadata.get('is_boundary_chunk')]
        boundary_chunks = [c for c in chunks if c.metadata.get('is_boundary_chunk')]
        
        sizes = [len(c.content) for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'regular_chunks': len(regular_chunks),
            'boundary_chunks': len(boundary_chunks),
            'avg_chunk_size': sum(sizes) / len(sizes),
            'min_chunk_size': min(sizes),
            'max_chunk_size': max(sizes),
            'total_characters': sum(sizes),
            'unique_docs': len(set(c.doc_id for c in chunks))
        }