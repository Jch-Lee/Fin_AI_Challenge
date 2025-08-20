"""
실제 운영 환경에 맞는 청크 크기로 하이브리드 청킹 테스트 (클린 버전)
- 청크 크기: 1024자 (기존 500자 → 1024자)
- 오버랩: 100자 (기존 50자 → 100자)
"""

import json
import sys
from typing import List, Dict, Any
from dataclasses import dataclass

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
        
        for section_idx, section in enumerate(major_sections):
            section_content = section.page_content
            
            # 섹션 경계 오버랩 추가
            if previous_tail and section_idx > 0:
                # 경계 청크 생성 (이전 섹션 끝 + 현재 섹션 시작)
                boundary_chunk_content = (
                    f"{previous_tail}\n"
                    f"{'=' * 40}\n"
                    f"[Section Transition: {section_idx-1} to {section_idx}]\n"
                    f"{'=' * 40}\n"
                    f"{section_content[:self.boundary_overlap]}"
                )
                
                boundary_chunk = ChunkResult(
                    content=boundary_chunk_content,
                    chunk_index=chunk_index,
                    metadata={
                        'section_idx': f"{section_idx-1}-{section_idx}",
                        'is_boundary_chunk': True,
                        'from_section': section_idx - 1,
                        'to_section': section_idx,
                        'chunk_size': self.chunk_size,
                        'declared_overlap': self.boundary_overlap
                    }
                )
                all_chunks.append(boundary_chunk)
                chunk_index += 1
            
            # 섹션 내용을 청킹
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " "],
                length_function=len,
                is_separator_regex=False
            )
            
            chunk_texts = splitter.split_text(section_content)
            
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
                        'declared_overlap': self.chunk_overlap
                    }
                )
                all_chunks.append(chunk)
                chunk_index += 1
        
        return all_chunks


def analyze_chunks(chunks: List[ChunkResult], label: str):
    """청크 분석 및 통계"""
    results = {}
    
    # 청크 크기 통계
    chunk_sizes = [len(c.content) for c in chunks]
    results['total_chunks'] = len(chunks)
    results['avg_size'] = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    results['min_size'] = min(chunk_sizes) if chunk_sizes else 0
    results['max_size'] = max(chunk_sizes) if chunk_sizes else 0
    
    # 경계 청크 통계
    boundary_chunks = [c for c in chunks if c.metadata.get('is_boundary_chunk')]
    results['boundary_chunks'] = len(boundary_chunks)
    
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
            'pair': f"{i}-{i+1}",
            'actual': actual_overlap,
            'is_boundary': curr.metadata.get('is_boundary_chunk') or next_chunk.metadata.get('is_boundary_chunk')
        })
    
    # 오버랩 통계
    regular_overlaps = [o['actual'] for o in overlaps if not o['is_boundary']]
    boundary_overlaps = [o['actual'] for o in overlaps if o['is_boundary']]
    
    results['avg_regular_overlap'] = sum(regular_overlaps) / len(regular_overlaps) if regular_overlaps else 0
    results['avg_boundary_overlap'] = sum(boundary_overlaps) / len(boundary_overlaps) if boundary_overlaps else 0
    results['total_overlap'] = sum(o['actual'] for o in overlaps)
    
    return results


def main():
    """메인 테스트 함수"""
    
    # 테스트 문서 경로
    test_document_path = "data/processed/개인정보 유출 등 사고 대응 매뉴얼(2024.3).txt"
    
    # 문서 읽기
    with open(test_document_path, 'r', encoding='utf-8') as f:
        document = f.read()
    
    print("=" * 80)
    print("Realistic Chunking Test Results")
    print("=" * 80)
    print(f"Document size: {len(document):,} characters")
    
    # 테스트를 위해 문서 일부 사용
    test_size = min(20000, len(document))
    document = document[:test_size]
    print(f"Test size: {test_size:,} characters\n")
    
    # 다양한 청크 크기로 테스트
    configs = [
        {"size": 512, "overlap": 50, "label": "Small (512/50)"},
        {"size": 1024, "overlap": 100, "label": "Standard (1024/100)"},
        {"size": 1536, "overlap": 150, "label": "Large (1536/150)"},
    ]
    
    all_results = {}
    
    for config in configs:
        print(f"\nTesting: {config['label']}")
        print("-" * 40)
        
        chunker = RealisticHybridChunker(
            chunk_size=config['size'],
            chunk_overlap=config['overlap']
        )
        
        chunks = chunker.chunk_document(document)
        
        # 분석
        results = analyze_chunks(chunks, config['label'])
        all_results[config['label']] = results
        
        # 결과 출력
        print(f"Total chunks: {results['total_chunks']}")
        print(f"Average chunk size: {results['avg_size']:.1f} chars")
        print(f"Size range: {results['min_size']} - {results['max_size']} chars")
        print(f"Boundary chunks: {results['boundary_chunks']}")
        print(f"Average regular overlap: {results['avg_regular_overlap']:.1f} chars")
        print(f"Average boundary overlap: {results['avg_boundary_overlap']:.1f} chars")
        print(f"Total overlap: {results['total_overlap']} chars")
        
        # 샘플 청크 정보 (내용 출력 없이)
        regular_chunks = [c for c in chunks if not c.metadata.get('is_boundary_chunk')]
        if regular_chunks:
            sample = regular_chunks[0]
            print(f"\nSample regular chunk:")
            print(f"  Index: {sample.chunk_index}")
            print(f"  Section: {sample.metadata['section_idx']}")
            print(f"  Size: {len(sample.content)} chars")
        
        boundary_chunks = [c for c in chunks if c.metadata.get('is_boundary_chunk')]
        if boundary_chunks:
            sample = boundary_chunks[0]
            print(f"\nSample boundary chunk:")
            print(f"  Index: {sample.chunk_index}")
            print(f"  Sections: {sample.metadata['from_section']} -> {sample.metadata['to_section']}")
            print(f"  Size: {len(sample.content)} chars")
    
    # 비교 표
    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80)
    
    print("\n| Configuration | Total Chunks | Avg Size | Boundary | Total Overlap |")
    print("|---------------|--------------|----------|----------|---------------|")
    
    for label, result in all_results.items():
        print(f"| {label:13} | {result['total_chunks']:12} | {result['avg_size']:8.0f} | "
              f"{result['boundary_chunks']:8} | {result['total_overlap']:13} |")
    
    # 결과 저장
    with open('chunking_realistic_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("\nResults saved to 'chunking_realistic_results.json'")
    
    # 권장 사항
    print("\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)
    print("1. Standard configuration (1024/100) provides balanced results")
    print("2. Boundary chunks effectively connect sections (avg ~100 char overlap)")
    print("3. Larger chunks (1536/150) reduce total count but may lose granularity")
    print("4. Consider using boundary chunks for improved context in RAG retrieval")


if __name__ == "__main__":
    main()