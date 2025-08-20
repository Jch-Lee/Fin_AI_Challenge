"""
실제 생성되는 청크 내용을 상세히 보여주는 스크립트
"""

import json
import re
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
    """실제 운영용 하이브리드 청킹"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.markdown_parser = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2")
            ],
            return_each_line=False,
            strip_headers=False
        )
        
        self.boundary_overlap = 150
    
    def chunk_document(self, document: str) -> List[ChunkResult]:
        major_sections = self.markdown_parser.split_text(document)
        
        all_chunks = []
        chunk_index = 0
        previous_tail = ""
        
        for section_idx, section in enumerate(major_sections):
            section_content = section.page_content
            
            # 섹션 경계 오버랩 추가
            if previous_tail and section_idx > 0:
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
                separators=["\n\n", "\n", ". ", "。", "! ", "? ", " "],
                length_function=len,
                is_separator_regex=False
            )
            
            chunk_texts = splitter.split_text(section_content)
            
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


def clean_text_for_display(text: str) -> str:
    """텍스트를 안전하게 출력할 수 있도록 정리"""
    # 특수 문자 제거/치환
    text = text.replace('📌', '[PIN]')
    text = text.replace('📍', '[MARKER]')
    text = text.replace('•', '*')
    text = text.replace('→', '->')
    text = text.replace('←', '<-')
    # 제어 문자 제거
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text


def show_chunk_details(chunks: List[ChunkResult], num_samples: int = 5):
    """청크의 실제 내용을 상세히 표시"""
    
    print("\n" + "=" * 80)
    print("실제 생성된 청크 내용 (표준 구성: 1024/100)")
    print("=" * 80)
    
    # 1. 일반 청크 샘플
    regular_chunks = [c for c in chunks if not c.metadata.get('is_boundary_chunk')]
    print(f"\n### 일반 청크 (총 {len(regular_chunks)}개) ###\n")
    
    for i, chunk in enumerate(regular_chunks[:num_samples]):
        print(f"\n{'='*60}")
        print(f"청크 #{chunk.chunk_index} (섹션 {chunk.metadata['section_idx']}, "
              f"섹션 내 {chunk.metadata['chunk_in_section']+1}/{chunk.metadata['total_chunks_in_section']})")
        print(f"크기: {len(chunk.content)}자")
        print("="*60)
        
        # 내용을 안전하게 출력
        content = clean_text_for_display(chunk.content)
        
        # 처음 300자
        print("\n[시작 부분]")
        print("-" * 40)
        print(content[:300])
        if len(content) > 300:
            print("...")
        
        # 마지막 200자
        if len(content) > 500:
            print("\n[끝 부분]")
            print("-" * 40)
            print("...")
            print(content[-200:])
        
        # 다음 청크와의 오버랩 확인
        if i < len(regular_chunks) - 1 and chunk.chunk_index < len(chunks) - 1:
            next_chunk = chunks[chunk.chunk_index + 1]
            
            # 실제 오버랩 계산
            overlap_size = 0
            for size in range(min(len(chunk.content), len(next_chunk.content)), 0, -1):
                if chunk.content[-size:] == next_chunk.content[:size]:
                    overlap_size = size
                    break
            
            if overlap_size > 0:
                print(f"\n[다음 청크와의 오버랩: {overlap_size}자]")
                print("-" * 40)
                overlap_text = clean_text_for_display(chunk.content[-overlap_size:])
                if overlap_size > 100:
                    print(overlap_text[:50] + "..." + overlap_text[-50:])
                else:
                    print(overlap_text)
    
    # 2. 경계 청크 샘플
    boundary_chunks = [c for c in chunks if c.metadata.get('is_boundary_chunk')]
    if boundary_chunks:
        print(f"\n\n### 경계 전환 청크 (총 {len(boundary_chunks)}개) ###\n")
        
        for chunk in boundary_chunks[:3]:  # 처음 3개만
            print(f"\n{'='*60}")
            print(f"경계 청크 #{chunk.chunk_index} "
                  f"(섹션 {chunk.metadata['from_section']} → {chunk.metadata['to_section']})")
            print(f"크기: {len(chunk.content)}자")
            print("="*60)
            
            content = clean_text_for_display(chunk.content)
            
            # 전환 마커 위치 찾기
            if "[섹션 전환:" in content:
                marker_start = content.index("[섹션 전환:")
                marker_end = content.index("]", marker_start) + 1
                
                # 전환 전 부분 (이전 섹션 끝)
                before_transition = content[:marker_start]
                if before_transition.strip():
                    print("\n[이전 섹션 끝부분]")
                    print("-" * 40)
                    if len(before_transition) > 150:
                        print("...")
                        print(before_transition[-150:].strip())
                    else:
                        print(before_transition.strip())
                
                # 전환 마커
                print(f"\n[전환 마커]")
                print("-" * 40)
                print(content[marker_start:marker_end])
                
                # 전환 후 부분 (새 섹션 시작)
                after_transition = content[marker_end:]
                if after_transition.strip():
                    print("\n[새 섹션 시작부분]")
                    print("-" * 40)
                    if len(after_transition) > 150:
                        print(after_transition[:150].strip())
                        print("...")
                    else:
                        print(after_transition.strip())


def analyze_overlap_patterns(chunks: List[ChunkResult]):
    """오버랩 패턴 분석"""
    print("\n\n### 오버랩 패턴 분석 ###\n")
    
    overlap_sizes = []
    for i in range(len(chunks) - 1):
        curr = chunks[i]
        next_chunk = chunks[i + 1]
        
        # 실제 오버랩 계산
        overlap = 0
        for size in range(min(len(curr.content), len(next_chunk.content)), 0, -1):
            if curr.content[-size:] == next_chunk.content[:size]:
                overlap = size
                break
        
        overlap_sizes.append({
            'pair': f"{i}->{i+1}",
            'size': overlap,
            'is_boundary': curr.metadata.get('is_boundary_chunk', False) or 
                          next_chunk.metadata.get('is_boundary_chunk', False),
            'curr_type': 'boundary' if curr.metadata.get('is_boundary_chunk') else 'regular',
            'next_type': 'boundary' if next_chunk.metadata.get('is_boundary_chunk') else 'regular'
        })
    
    # 패턴별 통계
    patterns = {}
    for overlap in overlap_sizes:
        pattern = f"{overlap['curr_type']}->{overlap['next_type']}"
        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append(overlap['size'])
    
    print("오버랩 패턴별 평균:")
    for pattern, sizes in patterns.items():
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        print(f"  {pattern}: 평균 {avg_size:.1f}자 (샘플 {len(sizes)}개)")


def main():
    """메인 함수"""
    
    # 테스트 문서
    test_document_path = "data/processed/개인정보 유출 등 사고 대응 매뉴얼(2024.3).txt"
    
    with open(test_document_path, 'r', encoding='utf-8') as f:
        document = f.read()
    
    # 테스트용으로 처음 15000자만 사용
    document = document[:15000]
    
    print(f"문서: {test_document_path}")
    print(f"테스트 크기: {len(document):,}자")
    
    # 표준 구성으로 청킹
    chunker = RealisticHybridChunker(chunk_size=1024, chunk_overlap=100)
    chunks = chunker.chunk_document(document)
    
    print(f"생성된 총 청크 수: {len(chunks)}개")
    
    # 청크 내용 표시
    show_chunk_details(chunks, num_samples=3)
    
    # 오버랩 패턴 분석
    analyze_overlap_patterns(chunks)
    
    # 청크 샘플 저장
    print("\n\n청크 샘플을 'actual_chunks_sample.json'에 저장 중...")
    
    samples = []
    for chunk in chunks[:10]:  # 처음 10개 저장
        samples.append({
            'index': chunk.chunk_index,
            'size': len(chunk.content),
            'is_boundary': chunk.metadata.get('is_boundary_chunk', False),
            'section': chunk.metadata.get('section_idx', ''),
            'content_preview': clean_text_for_display(chunk.content[:200]) + "..."
        })
    
    with open('actual_chunks_sample.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print("완료!")


if __name__ == "__main__":
    main()