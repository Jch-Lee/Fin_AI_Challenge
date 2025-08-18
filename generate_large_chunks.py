"""
더 큰 청크 크기(2300자)로 하이브리드 청킹 수행
"""

import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

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


class LargeChunkHybridChunker:
    """큰 청크 크기를 사용하는 하이브리드 청킹"""
    
    def __init__(self, chunk_size: int = 2300, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: 청크 크기 (기본 2300자)
            chunk_overlap: 청크 간 오버랩 (기본 200자)
        """
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
        
        # 섹션 경계 오버랩도 증가
        self.boundary_overlap = 300  # 섹션 간 더 큰 오버랩
    
    def chunk_document(self, document: str) -> List[ChunkResult]:
        major_sections = self.markdown_parser.split_text(document)
        
        all_chunks = []
        chunk_index = 0
        previous_tail = ""
        
        print(f"문서를 {len(major_sections)}개의 주요 섹션으로 분리")
        
        for section_idx, section in enumerate(major_sections):
            section_content = section.page_content
            
            # 섹션 정보 출력
            header_info = section.metadata.get('h1', '') or section.metadata.get('h2', '')
            if header_info:
                print(f"  섹션 {section_idx}: {header_info[:50]}... ({len(section_content)}자)")
            else:
                print(f"  섹션 {section_idx}: ({len(section_content)}자)")
            
            # 섹션 경계 오버랩 추가
            if previous_tail and section_idx > 0:
                # 경계 청크 생성 (이전 섹션 끝 + 현재 섹션 시작)
                boundary_chunk_content = (
                    f"{previous_tail}\n"
                    f"{'=' * 60}\n"
                    f"[섹션 전환: {section_idx-1} → {section_idx}]\n"
                    f"{'=' * 60}\n"
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
                        'declared_overlap': self.chunk_overlap
                    }
                )
                all_chunks.append(chunk)
                chunk_index += 1
        
        return all_chunks


def save_large_chunks_to_files(chunks: List[ChunkResult], output_dir: str = "generated_chunks_large"):
    """큰 청크를 파일로 저장"""
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 타임스탬프
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 전체 청크를 하나의 파일로 저장 (읽기 쉬운 형식)
    all_chunks_file = os.path.join(output_dir, f"all_chunks_2300_{timestamp}.txt")
    with open(all_chunks_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("하이브리드 청킹 결과 - 전체 청크 (청크 크기: 2300자)\n")
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"총 청크 수: {len(chunks)}\n")
        f.write("=" * 80 + "\n\n")
        
        for chunk in chunks:
            f.write(f"\n{'#' * 70}\n")
            
            if chunk.metadata.get('is_boundary_chunk'):
                f.write(f"### 경계 청크 #{chunk.chunk_index} ")
                f.write(f"(섹션 {chunk.metadata['from_section']} → {chunk.metadata['to_section']})\n")
            else:
                f.write(f"### 일반 청크 #{chunk.chunk_index} ")
                f.write(f"(섹션 {chunk.metadata['section_idx']}, ")
                f.write(f"섹션 내 {chunk.metadata['chunk_in_section']+1}/{chunk.metadata['total_chunks_in_section']})\n")
            
            f.write(f"크기: {len(chunk.content)}자\n")
            f.write("#" * 70 + "\n\n")
            
            f.write(chunk.content)
            f.write("\n\n")
            
            # 다음 청크와의 오버랩 표시
            if chunk.chunk_index < len(chunks) - 1:
                next_chunk = chunks[chunk.chunk_index + 1]
                overlap = 0
                for size in range(min(len(chunk.content), len(next_chunk.content)), 0, -1):
                    if chunk.content[-size:] == next_chunk.content[:size]:
                        overlap = size
                        break
                
                if overlap > 0:
                    f.write(f"\n>>> 다음 청크와 {overlap}자 오버랩 <<<\n")
            
            f.write("\n" + "-" * 70 + "\n")
    
    print(f"[완료] 전체 청크 파일 생성: {all_chunks_file}")
    
    # 2. 개별 청크 파일 생성 (선택적 - 처음 10개만)
    individual_dir = os.path.join(output_dir, f"individual_2300_{timestamp}")
    os.makedirs(individual_dir, exist_ok=True)
    
    for chunk in chunks[:min(10, len(chunks))]:  # 처음 10개 또는 전체 중 작은 수
        chunk_type = "boundary" if chunk.metadata.get('is_boundary_chunk') else "regular"
        filename = f"chunk_{chunk.chunk_index:03d}_{chunk_type}.txt"
        filepath = os.path.join(individual_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # 메타데이터 헤더
            f.write("--- METADATA ---\n")
            f.write(f"청크 번호: {chunk.chunk_index}\n")
            f.write(f"청크 타입: {chunk_type}\n")
            
            if chunk.metadata.get('is_boundary_chunk'):
                f.write(f"섹션 전환: {chunk.metadata['from_section']} → {chunk.metadata['to_section']}\n")
            else:
                f.write(f"섹션: {chunk.metadata['section_idx']}\n")
                f.write(f"섹션 내 위치: {chunk.metadata['chunk_in_section']+1}/{chunk.metadata['total_chunks_in_section']}\n")
            
            f.write(f"크기: {len(chunk.content)}자\n")
            f.write(f"선언된 오버랩: {chunk.metadata['declared_overlap']}자\n")
            f.write("\n--- CONTENT ---\n\n")
            f.write(chunk.content)
    
    print(f"[완료] 개별 청크 파일 생성 (처음 10개): {individual_dir}")
    
    # 3. JSON 형식으로도 저장 (분석용)
    json_file = os.path.join(output_dir, f"chunks_data_2300_{timestamp}.json")
    chunks_data = []
    
    for chunk in chunks:
        chunks_data.append({
            'index': chunk.chunk_index,
            'content': chunk.content,
            'metadata': chunk.metadata,
            'size': len(chunk.content)
        })
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    print(f"[완료] JSON 데이터 파일 생성: {json_file}")
    
    # 4. 요약 통계 파일
    summary_file = os.path.join(output_dir, f"summary_2300_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("청킹 요약 통계 (청크 크기: 2300자)\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"청크 크기 설정: {chunks[0].metadata['chunk_size']}자\n")
        f.write(f"오버랩 설정: {chunks[0].metadata['declared_overlap']}자\n")
        f.write(f"총 청크 수: {len(chunks)}\n")
        
        regular_chunks = [c for c in chunks if not c.metadata.get('is_boundary_chunk')]
        boundary_chunks = [c for c in chunks if c.metadata.get('is_boundary_chunk')]
        
        f.write(f"일반 청크: {len(regular_chunks)}개\n")
        f.write(f"경계 청크: {len(boundary_chunks)}개\n\n")
        
        # 크기 통계
        all_sizes = [len(c.content) for c in chunks]
        f.write(f"평균 크기: {sum(all_sizes)/len(all_sizes):.1f}자\n")
        f.write(f"최소 크기: {min(all_sizes)}자\n")
        f.write(f"최대 크기: {max(all_sizes)}자\n\n")
        
        # 오버랩 통계
        total_overlap = 0
        overlap_pairs = []
        for i in range(len(chunks) - 1):
            curr = chunks[i]
            next_chunk = chunks[i + 1]
            overlap = 0
            for size in range(min(len(curr.content), len(next_chunk.content)), 0, -1):
                if curr.content[-size:] == next_chunk.content[:size]:
                    overlap = size
                    break
            total_overlap += overlap
            overlap_pairs.append(overlap)
        
        f.write(f"총 오버랩: {total_overlap}자\n")
        f.write(f"평균 오버랩: {total_overlap/(len(chunks)-1) if len(chunks) > 1 else 0:.1f}자\n")
        
        # 오버랩 분포
        if overlap_pairs:
            f.write(f"최대 오버랩: {max(overlap_pairs)}자\n")
            f.write(f"최소 오버랩: {min(overlap_pairs)}자\n")
    
    print(f"[완료] 요약 통계 파일 생성: {summary_file}")
    
    # 5. 비교 분석 파일
    comparison_file = os.path.join(output_dir, f"comparison_1024_vs_2300_{timestamp}.txt")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("청크 크기 비교: 1024자 vs 2300자\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("### 2300자 청킹 결과\n")
        f.write(f"- 총 청크 수: {len(chunks)}\n")
        f.write(f"- 일반 청크: {len(regular_chunks)}\n")
        f.write(f"- 경계 청크: {len(boundary_chunks)}\n")
        f.write(f"- 평균 크기: {sum(all_sizes)/len(all_sizes):.1f}자\n")
        f.write(f"- 총 오버랩: {total_overlap}자\n\n")
        
        f.write("### 1024자 청킹 결과 (이전 테스트)\n")
        f.write("- 총 청크 수: 40\n")
        f.write("- 일반 청크: 31\n")
        f.write("- 경계 청크: 9\n")
        f.write("- 평균 크기: 602.5자\n")
        f.write("- 총 오버랩: 2875자\n\n")
        
        f.write("### 변화율\n")
        f.write(f"- 청크 수 감소: {40 - len(chunks)}개 ({(40 - len(chunks))/40*100:.1f}% 감소)\n")
        f.write(f"- 평균 크기 증가: {(sum(all_sizes)/len(all_sizes) - 602.5):.1f}자\n")
        f.write(f"- 오버랩 변화: {total_overlap - 2875}자\n")
    
    print(f"[완료] 비교 분석 파일 생성: {comparison_file}")
    
    return output_dir


def main():
    """메인 함수"""
    
    # 테스트 문서
    test_document_path = "data/processed/개인정보 유출 등 사고 대응 매뉴얼(2024.3).txt"
    
    print("=" * 60)
    print("큰 청크 크기(2300자) 테스트")
    print("=" * 60)
    print(f"문서: {test_document_path}\n")
    
    with open(test_document_path, 'r', encoding='utf-8') as f:
        document = f.read()
    
    # 테스트 크기 설정 (30000자로 증가)
    test_size = min(30000, len(document))
    document = document[:test_size]
    print(f"테스트 크기: {test_size:,}자\n")
    
    # 큰 청크 크기로 청킹
    print("청킹 진행 중...\n")
    chunker = LargeChunkHybridChunker(chunk_size=2300, chunk_overlap=200)
    chunks = chunker.chunk_document(document)
    print(f"\n총 생성된 청크 수: {len(chunks)}개\n")
    
    # 파일로 저장
    print("파일 저장 중...\n")
    output_dir = save_large_chunks_to_files(chunks)
    
    print("\n" + "=" * 60)
    print("[완료] 모든 파일 생성 완료!")
    print("=" * 60)
    print(f"\n생성된 파일들이 '{output_dir}' 폴더에 저장되었습니다.")
    print("\n다음 파일들을 확인하세요:")
    print("  1. all_chunks_2300_*.txt - 전체 청크를 읽기 쉽게 정리한 파일")
    print("  2. individual_2300_*/ - 개별 청크 파일들 (처음 10개)")
    print("  3. chunks_data_2300_*.json - JSON 형식 데이터")
    print("  4. summary_2300_*.txt - 요약 통계")
    print("  5. comparison_*.txt - 1024자 vs 2300자 비교 분석")


if __name__ == "__main__":
    main()