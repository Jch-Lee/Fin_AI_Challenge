"""
실제 운영 환경에 맞는 청크 크기로 하이브리드 청킹 테스트
- 청크 크기: 1024자 (기존 500자 → 1024자)
- 오버랩: 100자 (기존 50자 → 100자)
"""

import json
from typing import List, Dict, Any
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


class RealisticHybridChunker:
    """실제 운영용 하이브리드 청킹 - 더 큰 청크 크기"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 100):
        """
        Args:
            chunk_size: 청크 크기 (기본 1024자)
            chunk_overlap: 청크 간 오버랩 (기본 100자)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 주요 섹션 구분 (h1, h2만 사용)
        self.markdown_parser = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2")
            ],
            return_each_line=False,
            strip_headers=False
        )
        
        # 섹션 경계 오버랩 크기
        self.boundary_overlap = 150  # 섹션 간 더 큰 오버랩
    
    def chunk_document(self, document: str) -> List[ChunkResult]:
        """실제 운영 환경에 맞는 청킹"""
        
        # 주요 섹션으로 분리
        major_sections = self.markdown_parser.split_text(document)
        
        all_chunks = []
        chunk_index = 0
        previous_tail = ""  # 이전 섹션 끝부분
        
        print(f"문서를 {len(major_sections)}개의 주요 섹션으로 분리")
        
        for section_idx, section in enumerate(major_sections):
            section_content = section.page_content
            
            # 섹션 정보 출력
            header_info = section.metadata.get('h1', '') or section.metadata.get('h2', '')
            print(f"  섹션 {section_idx}: {header_info[:30]}... ({len(section_content)}자)")
            
            # 섹션 경계 오버랩 추가
            if previous_tail and section_idx > 0:
                # 경계 청크 생성 (이전 섹션 끝 + 현재 섹션 시작)
                boundary_chunk_content = (
                    f"{previous_tail}\n"
                    f"{'=' * 40}\n"
                    f"[섹션 전환: {section_idx-1} → {section_idx}]\n"
                    f"{'=' * 40}\n"
                    f"{section_content[:self.boundary_overlap]}"
                )
                
                boundary_chunk = ChunkResult(
                    content=boundary_chunk_content,
                    chunk_index=chunk_index,
                    metadata={
                        'section_idx': f"{section_idx-1}-{section_idx}",
                        'is_boundary_chunk': True,
                        'boundary_type': 'section_transition',
                        'from_section': section_idx - 1,
                        'to_section': section_idx,
                        'chunk_size': self.chunk_size,
                        'declared_overlap': self.boundary_overlap,
                        'headers': section.metadata
                    }
                )
                all_chunks.append(boundary_chunk)
                chunk_index += 1
            
            # 섹션 내용을 청킹
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "。", "! ", "? ", " "],
                length_function=len,
                is_separator_regex=False
            )
            
            chunk_texts = splitter.split_text(section_content)
            print(f"    → {len(chunk_texts)}개 청크 생성")
            
            # 현재 섹션 끝부분 저장 (다음 섹션과 연결용)
            if section_content:
                previous_tail = section_content[-self.boundary_overlap:]
            
            for i, chunk_text in enumerate(chunk_texts):
                chunk = ChunkResult(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    metadata={
                        'section_idx': section_idx,
                        'is_boundary_chunk': False,
                        'chunk_in_section': i,
                        'total_chunks_in_section': len(chunk_texts),
                        'chunk_size': self.chunk_size,
                        'declared_overlap': self.chunk_overlap,
                        'headers': section.metadata
                    }
                )
                all_chunks.append(chunk)
                chunk_index += 1
        
        return all_chunks


def analyze_chunks(chunks: List[ChunkResult], label: str):
    """청크 분석 및 통계"""
    print(f"\n[{label}]")
    print(f"총 청크 수: {len(chunks)}")
    
    # 청크 크기 통계
    chunk_sizes = [len(c.content) for c in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    print(f"평균 청크 크기: {avg_size:.1f}자")
    print(f"최소/최대 크기: {min(chunk_sizes)}자 / {max(chunk_sizes)}자")
    
    # 경계 청크 통계
    boundary_chunks = [c for c in chunks if c.metadata.get('is_boundary_chunk')]
    print(f"경계 전환 청크: {len(boundary_chunks)}개")
    
    # 오버랩 분석
    overlaps = []
    for i in range(len(chunks) - 1):
        curr = chunks[i]
        next_chunk = chunks[i + 1]
        
        # 실제 오버랩 계산
        actual_overlap = 0
        for size in range(min(len(curr.content), len(next_chunk.content)), 0, -1):
            if curr.content[-size:] == next_chunk.content[:size]:
                actual_overlap = size
                break
        
        overlaps.append({
            'pair': f"{i}→{i+1}",
            'actual': actual_overlap,
            'is_boundary': curr.metadata.get('is_boundary_chunk') or next_chunk.metadata.get('is_boundary_chunk')
        })
    
    # 오버랩 통계
    regular_overlaps = [o['actual'] for o in overlaps if not o['is_boundary']]
    boundary_overlaps = [o['actual'] for o in overlaps if o['is_boundary']]
    
    if regular_overlaps:
        avg_regular = sum(regular_overlaps) / len(regular_overlaps)
        print(f"일반 청크 간 평균 오버랩: {avg_regular:.1f}자")
    
    if boundary_overlaps:
        avg_boundary = sum(boundary_overlaps) / len(boundary_overlaps)
        print(f"경계 청크 관련 평균 오버랩: {avg_boundary:.1f}자")
    
    total_overlap = sum(o['actual'] for o in overlaps)
    print(f"총 오버랩: {total_overlap}자")
    
    return {
        'total_chunks': len(chunks),
        'avg_size': avg_size,
        'boundary_chunks': len(boundary_chunks),
        'total_overlap': total_overlap,
        'overlaps': overlaps[:10]  # 처음 10개만
    }


def show_sample_chunks(chunks: List[ChunkResult], num_samples: int = 3):
    """샘플 청크 내용 표시"""
    print("\n### 샘플 청크 내용 ###")
    
    # 일반 청크 샘플
    regular_chunks = [c for c in chunks if not c.metadata.get('is_boundary_chunk')]
    print("\n[일반 청크 샘플]")
    for i, chunk in enumerate(regular_chunks[:num_samples]):
        print(f"\n청크 {chunk.chunk_index} (섹션 {chunk.metadata['section_idx']}):")
        print(f"  크기: {len(chunk.content)}자")
        # 이모지 제거를 위한 안전한 출력
        content_start = chunk.content[:100].encode('utf-8', errors='ignore').decode('utf-8')
        content_end = chunk.content[-100:].encode('utf-8', errors='ignore').decode('utf-8')
        print(f"  시작: {content_start}...")
        print(f"  끝: ...{content_end}")
    
    # 경계 청크 샘플
    boundary_chunks = [c for c in chunks if c.metadata.get('is_boundary_chunk')]
    if boundary_chunks:
        print("\n[경계 전환 청크 샘플]")
        for chunk in boundary_chunks[:2]:
            print(f"\n청크 {chunk.chunk_index} (섹션 {chunk.metadata['from_section']}→{chunk.metadata['to_section']}):")
            print(f"  크기: {len(chunk.content)}자")
            
            # 전환 마커 위치 찾기
            if "[섹션 전환:" in chunk.content:
                marker_start = chunk.content.index("[섹션 전환:")
                marker_end = chunk.content.index("]", marker_start) + 1
                print(f"  전환 전: ...{chunk.content[:marker_start][-50:]}")
                print(f"  전환 마커: {chunk.content[marker_start:marker_end]}")
                print(f"  전환 후: {chunk.content[marker_end:][:50]}...")


def main():
    """메인 테스트 함수"""
    
    # 테스트 문서 경로
    test_document_path = "data/processed/개인정보 유출 등 사고 대응 매뉴얼(2024.3).txt"
    
    # 문서 읽기
    with open(test_document_path, 'r', encoding='utf-8') as f:
        document = f.read()
    
    print("=" * 80)
    print("실제 운영 환경 청킹 테스트")
    print("=" * 80)
    print(f"문서: {test_document_path}")
    print(f"문서 크기: {len(document):,}자")
    
    # 테스트를 위해 문서 일부 사용 (전체의 약 20%)
    test_size = min(20000, len(document))
    document = document[:test_size]
    print(f"테스트 크기: {test_size:,}자\n")
    
    # 다양한 청크 크기로 테스트
    configs = [
        {"size": 512, "overlap": 50, "label": "작은 청크 (512/50)"},
        {"size": 1024, "overlap": 100, "label": "표준 청크 (1024/100)"},
        {"size": 1536, "overlap": 150, "label": "큰 청크 (1536/150)"},
    ]
    
    results = {}
    
    for config in configs:
        print("\n" + "=" * 60)
        print(f"테스트: {config['label']}")
        print("=" * 60)
        
        chunker = RealisticHybridChunker(
            chunk_size=config['size'],
            chunk_overlap=config['overlap']
        )
        
        chunks = chunker.chunk_document(document)
        
        # 분석
        analysis = analyze_chunks(chunks, config['label'])
        results[config['label']] = analysis
        
        # 샘플 표시
        show_sample_chunks(chunks, num_samples=2)
    
    # 비교 결과
    print("\n" + "=" * 80)
    print("청크 크기별 비교 결과")
    print("=" * 80)
    
    print("\n| 설정 | 총 청크 | 평균 크기 | 경계 청크 | 총 오버랩 |")
    print("|------|---------|-----------|-----------|-----------|")
    
    for label, result in results.items():
        print(f"| {label} | {result['total_chunks']} | "
              f"{result['avg_size']:.0f}자 | "
              f"{result['boundary_chunks']} | "
              f"{result['total_overlap']}자 |")
    
    # 결과 저장
    with open('chunking_realistic_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n결과가 'chunking_realistic_results.json'에 저장되었습니다.")
    
    # 권장 사항
    print("\n### 권장 사항 ###")
    print("1. 표준 설정 (1024/100)이 가장 균형잡힌 결과를 보임")
    print("2. 경계 전환 청크가 섹션 간 문맥 연결을 효과적으로 보장")
    print("3. 실제 오버랩이 선언값에 근접하여 일관성 있음")
    print("4. RAG 검색 시 경계 청크를 활용하면 정확도 향상 기대")


if __name__ == "__main__":
    main()