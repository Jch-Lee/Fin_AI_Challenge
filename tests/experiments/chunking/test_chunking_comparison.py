"""
청킹 방식 비교 테스트
- 기존 방식: 섹션별 독립 청킹
- 하이브리드 방식: 섹션 내 전체 청킹 with 경계 오버랩
"""

import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import hashlib

from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)


@dataclass
class ChunkResult:
    """청크 결과 저장"""
    content: str
    chunk_index: int
    metadata: Dict[str, Any]
    

class OriginalChunker:
    """기존 방식: 섹션별 독립 청킹"""
    
    def __init__(self):
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
    
    def chunk_document(self, document: str) -> List[ChunkResult]:
        """기존 방식으로 청킹"""
        # 마크다운 섹션 분리
        sections = self.markdown_parser.split_text(document)
        
        all_chunks = []
        chunk_index = 0
        
        for section_idx, section in enumerate(sections):
            # 각 섹션 독립적으로 청킹
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # 테스트용 작은 크기
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " "],
                length_function=len
            )
            
            chunk_texts = splitter.split_text(section.page_content)
            
            for chunk_text in chunk_texts:
                chunk = ChunkResult(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    metadata={
                        'section_idx': section_idx,
                        'method': 'original',
                        'declared_overlap': 50,
                        'headers': section.metadata
                    }
                )
                all_chunks.append(chunk)
                chunk_index += 1
        
        return all_chunks


class HybridChunker:
    """하이브리드 방식: 섹션 내 전체 청킹 + 경계 오버랩"""
    
    def __init__(self):
        self.markdown_parser = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2")  # 주요 섹션만
            ],
            return_each_line=False,
            strip_headers=False
        )
    
    def chunk_document(self, document: str) -> List[ChunkResult]:
        """하이브리드 방식으로 청킹"""
        # 주요 섹션으로 분리
        major_sections = self.markdown_parser.split_text(document)
        
        all_chunks = []
        chunk_index = 0
        previous_tail = ""  # 이전 섹션 끝부분
        
        for section_idx, section in enumerate(major_sections):
            # 섹션 경계 오버랩 추가
            if previous_tail:
                section_text = previous_tail + "\n[섹션 연결]\n" + section.page_content
            else:
                section_text = section.page_content
            
            # 전체 섹션을 한번에 청킹 (서브섹션 무시)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,  # 청크 간 오버랩
                separators=["\n\n", "\n", ". ", " "],
                length_function=len
            )
            
            chunk_texts = splitter.split_text(section_text)
            
            # 현재 섹션 끝부분 저장 (다음 섹션과 연결용)
            if section.page_content:
                previous_tail = section.page_content[-100:]  # 100자 저장
            
            for i, chunk_text in enumerate(chunk_texts):
                # 첫 청크는 섹션 경계 오버랩 포함
                has_boundary_overlap = (i == 0 and section_idx > 0)
                
                chunk = ChunkResult(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    metadata={
                        'section_idx': section_idx,
                        'method': 'hybrid',
                        'declared_overlap': 50,
                        'boundary_overlap': 100 if has_boundary_overlap else 0,
                        'has_boundary_overlap': has_boundary_overlap,
                        'headers': section.metadata
                    }
                )
                all_chunks.append(chunk)
                chunk_index += 1
        
        return all_chunks


def calculate_actual_overlap(chunks: List[ChunkResult]) -> List[Dict]:
    """실제 오버랩 계산"""
    overlap_info = []
    
    for i in range(len(chunks) - 1):
        current = chunks[i]
        next_chunk = chunks[i + 1]
        
        # 실제 오버랩 찾기
        actual_overlap = 0
        for size in range(min(len(current.content), len(next_chunk.content)), 0, -1):
            if current.content[-size:] == next_chunk.content[:size]:
                actual_overlap = size
                break
        
        overlap_info.append({
            'chunk_pair': f"{i} → {i+1}",
            'actual_overlap': actual_overlap,
            'declared_overlap': current.metadata['declared_overlap'],
            'is_section_boundary': (
                current.metadata.get('section_idx') != 
                next_chunk.metadata.get('section_idx')
            )
        })
    
    return overlap_info


def compare_methods(document_path: str):
    """두 방식 비교"""
    # 문서 읽기
    with open(document_path, 'r', encoding='utf-8') as f:
        document = f.read()
    
    # 테스트용으로 문서 일부만 사용 (처음 10000자)
    document = document[:10000]
    
    print(f"문서 크기: {len(document)}자\n")
    print("=" * 80)
    
    # 기존 방식 청킹
    print("\n[기존 방식 - 섹션별 독립 청킹]")
    original_chunker = OriginalChunker()
    original_chunks = original_chunker.chunk_document(document)
    print(f"생성된 청크 수: {len(original_chunks)}")
    
    # 오버랩 분석
    original_overlaps = calculate_actual_overlap(original_chunks)
    
    # 섹션 경계 오버랩 분석
    boundary_overlaps = [o for o in original_overlaps if o['is_section_boundary']]
    internal_overlaps = [o for o in original_overlaps if not o['is_section_boundary']]
    
    if internal_overlaps:
        avg_internal = sum(o['actual_overlap'] for o in internal_overlaps) / len(internal_overlaps)
        print(f"섹션 내부 평균 오버랩: {avg_internal:.1f}자")
    
    if boundary_overlaps:
        avg_boundary = sum(o['actual_overlap'] for o in boundary_overlaps) / len(boundary_overlaps)
        print(f"섹션 경계 평균 오버랩: {avg_boundary:.1f}자")
    else:
        print("섹션 경계 평균 오버랩: 0자 (섹션 간 오버랩 없음)")
    
    # 샘플 출력
    print("\n첫 3개 청크 간 오버랩:")
    for overlap in original_overlaps[:3]:
        boundary = " [섹션 경계]" if overlap['is_section_boundary'] else ""
        print(f"  청크 {overlap['chunk_pair']}: {overlap['actual_overlap']}자{boundary}")
    
    print("\n" + "=" * 80)
    
    # 하이브리드 방식 청킹
    print("\n[하이브리드 방식 - 섹션 내 전체 청킹 + 경계 오버랩]")
    hybrid_chunker = HybridChunker()
    hybrid_chunks = hybrid_chunker.chunk_document(document)
    print(f"생성된 청크 수: {len(hybrid_chunks)}")
    
    # 오버랩 분석
    hybrid_overlaps = calculate_actual_overlap(hybrid_chunks)
    
    # 섹션 경계 오버랩 분석
    boundary_overlaps = [o for o in hybrid_overlaps if o['is_section_boundary']]
    internal_overlaps = [o for o in hybrid_overlaps if not o['is_section_boundary']]
    
    if internal_overlaps:
        avg_internal = sum(o['actual_overlap'] for o in internal_overlaps) / len(internal_overlaps)
        print(f"섹션 내부 평균 오버랩: {avg_internal:.1f}자")
    
    if boundary_overlaps:
        avg_boundary = sum(o['actual_overlap'] for o in boundary_overlaps) / len(boundary_overlaps)
        print(f"섹션 경계 평균 오버랩: {avg_boundary:.1f}자")
    
    # 샘플 출력
    print("\n첫 3개 청크 간 오버랩:")
    for overlap in hybrid_overlaps[:3]:
        boundary = " [섹션 경계]" if overlap['is_section_boundary'] else ""
        print(f"  청크 {overlap['chunk_pair']}: {overlap['actual_overlap']}자{boundary}")
    
    # 경계 오버랩 청크 샘플
    boundary_chunks = [c for c in hybrid_chunks if c.metadata.get('has_boundary_overlap')]
    if boundary_chunks:
        print(f"\n섹션 경계 오버랩이 있는 청크: {len(boundary_chunks)}개")
        sample = boundary_chunks[0]
        print(f"예시 (청크 {sample.chunk_index}):")
        print(f"  내용 앞부분: {sample.content[:100]}...")
    
    print("\n" + "=" * 80)
    print("\n[비교 결과]")
    print(f"기존 방식: {len(original_chunks)}개 청크")
    print(f"하이브리드 방식: {len(hybrid_chunks)}개 청크")
    
    # 전체 오버랩 통계
    original_total_overlap = sum(o['actual_overlap'] for o in original_overlaps)
    hybrid_total_overlap = sum(o['actual_overlap'] for o in hybrid_overlaps)
    
    print(f"\n총 오버랩 문자 수:")
    print(f"  기존 방식: {original_total_overlap}자")
    print(f"  하이브리드 방식: {hybrid_total_overlap}자")
    print(f"  개선율: {(hybrid_total_overlap - original_total_overlap) / max(original_total_overlap, 1) * 100:.1f}%")
    
    # 결과 저장
    results = {
        'original': {
            'chunks': len(original_chunks),
            'total_overlap': original_total_overlap,
            'overlaps': original_overlaps[:10]  # 처음 10개만
        },
        'hybrid': {
            'chunks': len(hybrid_chunks),
            'total_overlap': hybrid_total_overlap,
            'overlaps': hybrid_overlaps[:10],
            'boundary_chunks': len(boundary_chunks)
        }
    }
    
    with open('chunking_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n결과가 'chunking_comparison_results.json'에 저장되었습니다.")


if __name__ == "__main__":
    # 테스트할 문서 선택 (작은 문서로 시작)
    test_document = "data/processed/개인정보 유출 등 사고 대응 매뉴얼(2024.3).txt"
    
    print(f"테스트 문서: {test_document}")
    compare_methods(test_document)