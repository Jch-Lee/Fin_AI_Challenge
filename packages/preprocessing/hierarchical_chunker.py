"""
계층적 마크다운 청킹 모듈
LangChain 기반으로 마크다운 구조 인식, 계층적 청킹, 오버랩 처리
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
class HierarchyConfig:
    """계층별 청킹 설정"""
    level: int
    chunk_size: int
    chunk_overlap: int
    separators: List[str]


class HierarchicalMarkdownChunker:
    """
    Vision V2 마크다운 출력 전용 계층적 청킹
    - 마크다운 헤더 구조 인식
    - 계층별 다른 청크 크기와 오버랩
    - 페이지 경계 마커 처리
    """
    
    def __init__(self, 
                 use_chunk_cleaner: bool = True,
                 enable_semantic: bool = False):
        """
        Args:
            use_chunk_cleaner: ChunkCleaner 사용 여부
            enable_semantic: 의미 기반 청킹 사용 여부 (향후 확장)
        """
        # 마크다운 헤더 파서
        self.markdown_parser = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4")
            ],
            return_each_line=False,
            strip_headers=False
        )
        
        # 계층별 설정
        self.hierarchy_configs = {
            1: HierarchyConfig(
                level=1,
                chunk_size=1024,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", "。", "! ", "? ", " "]
            ),
            2: HierarchyConfig(
                level=2,
                chunk_size=512,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", "。", "! ", "? ", " "]
            ),
            3: HierarchyConfig(
                level=3,
                chunk_size=256,
                chunk_overlap=30,
                separators=["\n\n", "\n", ". ", "。", "! ", "? ", " "]
            ),
            4: HierarchyConfig(
                level=4,
                chunk_size=256,
                chunk_overlap=20,
                separators=["\n\n", "\n", ". ", "。", "! ", "? ", " "]
            )
        }
        
        # 기본 설정 (헤더가 없는 경우)
        self.default_config = self.hierarchy_configs[2]
        
        self.chunk_cleaner = ChunkCleaner() if use_chunk_cleaner else None
        self.enable_semantic = enable_semantic
        
        logger.info("HierarchicalMarkdownChunker initialized")
    
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
    
    def _detect_header_level(self, metadata: Dict[str, Any]) -> int:
        """섹션 메타데이터에서 헤더 레벨 추출"""
        
        # LangChain이 제공하는 헤더 정보 확인
        for level in range(1, 5):
            header_key = f"h{level}"
            if header_key in metadata and metadata[header_key]:
                return level
        
        # 헤더가 없는 경우 기본 레벨
        return 2
    
    def _build_hierarchy_path(self, metadata: Dict[str, Any]) -> str:
        """계층 경로 생성 (예: "Chapter 1 > Section 2.1 > Subsection 2.1.1")"""
        
        path_parts = []
        for level in range(1, 5):
            header_key = f"h{level}"
            if header_key in metadata and metadata[header_key]:
                path_parts.append(metadata[header_key])
        
        return " > ".join(path_parts) if path_parts else "Root"
    
    def _create_chunks_with_overlap(self, 
                                   text: str, 
                                   config: HierarchyConfig) -> List[str]:
        """설정에 따른 청킹 with 오버랩"""
        
        if not text.strip():
            return []
        
        # RecursiveCharacterTextSplitter 사용
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = splitter.split_text(text)
        
        # 빈 청크 제거
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _generate_chunk_id(self, content: str, doc_id: str, index: int) -> str:
        """청크 고유 ID 생성"""
        hash_input = f"{doc_id}_{index}_{content[:50]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _generate_doc_id(self, document: str) -> str:
        """문서 고유 ID 생성"""
        return hashlib.md5(document[:200].encode()).hexdigest()[:16]
    
    def chunk_document(self, 
                      document: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        마크다운 문서를 계층적으로 청킹
        
        Args:
            document: Vision V2가 추출한 마크다운 텍스트
            metadata: 문서 메타데이터
            
        Returns:
            계층 정보가 포함된 DocumentChunk 리스트
        """
        if not document or not document.strip():
            return []
        
        if metadata is None:
            metadata = {}
        
        # 1. 페이지 경계 처리
        processed_text = self._merge_page_boundaries(document)
        
        # 2. 마크다운 구조 파싱
        try:
            sections = self.markdown_parser.split_text(processed_text)
            logger.debug(f"Found {len(sections)} markdown sections")
        except Exception as e:
            logger.warning(f"Markdown parsing failed: {e}, using fallback")
            # Fallback: 전체 텍스트를 단일 섹션으로 처리
            sections = [type('MockSection', (), {
                'page_content': processed_text,
                'metadata': {}
            })()]
        
        doc_id = self._generate_doc_id(document)
        all_chunks = []
        chunk_index = 0
        
        # 3. 섹션별 계층적 청킹
        for section_idx, section in enumerate(sections):
            # 헤더 레벨 감지
            level = self._detect_header_level(section.metadata)
            config = self.hierarchy_configs.get(level, self.default_config)
            
            # 계층 정보
            hierarchy_path = self._build_hierarchy_path(section.metadata)
            parent_header = section.metadata.get(f'h{level}', '')
            
            # 청킹 수행
            chunk_texts = self._create_chunks_with_overlap(
                section.page_content, 
                config
            )
            
            logger.debug(f"Section {section_idx} (level {level}): {len(chunk_texts)} chunks")
            
            # DocumentChunk 객체 생성
            for chunk_text in chunk_texts:
                # 청크 정제
                if self.chunk_cleaner:
                    cleaned_chunk = self.chunk_cleaner.clean_chunk(chunk_text)
                    if not cleaned_chunk or len(cleaned_chunk) < 30:
                        continue
                    chunk_text = cleaned_chunk
                
                # 메타데이터 구성
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    # 계층 정보
                    'hierarchy_level': level,
                    'hierarchy_path': hierarchy_path,
                    'parent_header': parent_header,
                    'section_index': section_idx,
                    
                    # 청킹 설정
                    'chunk_size': config.chunk_size,
                    'chunk_overlap': config.chunk_overlap,
                    'chunking_method': 'hierarchical_markdown',
                    
                    # 기본 정보
                    'total_chunks': 0,  # 나중에 업데이트
                    'content_length': len(chunk_text),
                    'has_header': bool(parent_header),
                    
                    # 처리 정보
                    'processed_boundaries': True
                })
                
                # Vision V2 연계 정보 (기본값, 사용자 메타데이터가 없을 때만)
                if 'source' not in chunk_metadata:
                    chunk_metadata['source'] = 'vision_v2_markdown'
                
                # 헤더별 메타데이터 추가
                for h_level in range(1, 5):
                    h_key = f'h{h_level}'
                    if h_key in section.metadata:
                        chunk_metadata[f'header_{h_level}'] = section.metadata[h_key]
                
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=self._generate_chunk_id(chunk_text, doc_id, chunk_index),
                    doc_id=doc_id,
                    chunk_index=chunk_index
                )
                
                all_chunks.append(chunk)
                chunk_index += 1
        
        # 4. 총 청크 수 업데이트
        for chunk in all_chunks:
            chunk.metadata['total_chunks'] = len(all_chunks)
        
        logger.info(f"Created {len(all_chunks)} hierarchical chunks from markdown document")
        
        return all_chunks
    
    def chunk_documents(self, 
                       documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """여러 문서 배치 처리"""
        all_chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            chunks = self.chunk_document(content, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_hierarchy_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """계층별 통계 정보"""
        
        if not chunks:
            return {}
        
        stats = {
            'total_chunks': len(chunks),
            'by_level': {},
            'avg_chunk_size': 0,
            'hierarchy_paths': set()
        }
        
        total_size = 0
        
        for chunk in chunks:
            level = chunk.metadata.get('hierarchy_level', 0)
            path = chunk.metadata.get('hierarchy_path', 'Unknown')
            size = chunk.metadata.get('content_length', len(chunk.content))
            
            # 레벨별 통계
            if level not in stats['by_level']:
                stats['by_level'][level] = {
                    'count': 0,
                    'avg_size': 0,
                    'total_size': 0
                }
            
            stats['by_level'][level]['count'] += 1
            stats['by_level'][level]['total_size'] += size
            stats['hierarchy_paths'].add(path)
            total_size += size
        
        # 평균 계산
        stats['avg_chunk_size'] = total_size / len(chunks)
        
        for level_stats in stats['by_level'].values():
            if level_stats['count'] > 0:
                level_stats['avg_size'] = level_stats['total_size'] / level_stats['count']
        
        stats['hierarchy_paths'] = list(stats['hierarchy_paths'])
        
        return stats


# 호환성을 위한 alias
HierarchicalChunker = HierarchicalMarkdownChunker


if __name__ == "__main__":
    # 간단한 테스트
    chunker = HierarchicalMarkdownChunker()
    
    test_markdown = """
# 금융보안 가이드라인

## 제1장 총칙

### 제1조 (목적)
이 가이드라인은 금융기관의 정보보호 및 사이버보안 강화를 위한 기본 원칙과 세부 실행방안을 제시함을 목적으로 한다.

### 제2조 (정의)
1. "금융기관"이란 은행법에 따른 은행, 금융지주회사법에 따른 금융지주회사를 말한다.
2. "사이버 위협"이란 정보통신망을 통하여 금융기관의 정보자산에 피해를 줄 수 있는 모든 행위를 말한다.

[다음 페이지에 계속]

[이전 페이지에서 계속]

## 제2장 보안 요구사항

### 제3조 (접근통제)

| 구분 | 요구사항 | 중요도 |
|------|----------|--------|
| 인증 | 다요소 인증 필수 | 높음 |
| 권한 | 최소 권한 원칙 | 높음 |
| 모니터링 | 실시간 감시 | 중간 |

모든 시스템 접근은 로그를 남겨야 한다.
"""
    
    chunks = chunker.chunk_document(test_markdown)
    
    print(f"생성된 청크 수: {len(chunks)}")
    print("\n=== 계층별 통계 ===")
    stats = chunker.get_hierarchy_stats(chunks)
    
    for level, level_stats in stats['by_level'].items():
        print(f"Level {level}: {level_stats['count']}개 청크, 평균 {level_stats['avg_size']:.0f}자")
    
    print(f"\n전체 평균 크기: {stats['avg_chunk_size']:.0f}자")
    print(f"계층 경로 수: {len(stats['hierarchy_paths'])}")
    
    print("\n=== 첫 번째 청크 ===")
    if chunks:
        first_chunk = chunks[0]
        print(f"레벨: {first_chunk.metadata.get('hierarchy_level')}")
        print(f"경로: {first_chunk.metadata.get('hierarchy_path')}")
        print(f"헤더: {first_chunk.metadata.get('parent_header')}")
        print(f"내용: {first_chunk.content[:100]}...")