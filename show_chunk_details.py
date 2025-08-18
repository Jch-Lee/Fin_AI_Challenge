"""
청킹 결과의 실제 내용 확인
"""

import json
from test_chunking_comparison import OriginalChunker, HybridChunker


def show_chunk_details(document_path: str):
    """청크의 실제 내용을 상세히 보여줌"""
    
    # 문서 읽기
    with open(document_path, 'r', encoding='utf-8') as f:
        document = f.read()[:10000]  # 테스트용 10000자
    
    print("=" * 80)
    print("기존 방식 vs 하이브리드 방식 - 실제 청크 내용 비교")
    print("=" * 80)
    
    # 기존 방식 청킹
    original_chunker = OriginalChunker()
    original_chunks = original_chunker.chunk_document(document)
    
    # 하이브리드 방식 청킹
    hybrid_chunker = HybridChunker()
    hybrid_chunks = hybrid_chunker.chunk_document(document)
    
    # 1. 섹션 경계 부분 확인
    print("\n### 1. 섹션 경계 처리 비교 ###\n")
    
    # 기존 방식 - 첫 번째 섹션 경계 찾기
    for i in range(len(original_chunks) - 1):
        curr = original_chunks[i]
        next_chunk = original_chunks[i + 1]
        
        if curr.metadata['section_idx'] != next_chunk.metadata['section_idx']:
            print("[기존 방식 - 섹션 경계]")
            print(f"청크 {i} (섹션 {curr.metadata['section_idx']}) 끝부분:")
            print(f"  ...{curr.content[-150:]}")
            print(f"\n청크 {i+1} (섹션 {next_chunk.metadata['section_idx']}) 시작부분:")
            print(f"  {next_chunk.content[:150]}...")
            
            # 오버랩 체크
            overlap = 0
            for size in range(min(len(curr.content), len(next_chunk.content)), 0, -1):
                if curr.content[-size:] == next_chunk.content[:size]:
                    overlap = size
                    break
            print(f"\n실제 오버랩: {overlap}자")
            break
    
    print("\n" + "-" * 40 + "\n")
    
    # 하이브리드 방식 - 섹션 경계 청크 찾기
    for i, chunk in enumerate(hybrid_chunks):
        if chunk.metadata.get('has_boundary_overlap'):
            print("[하이브리드 방식 - 섹션 경계 오버랩 청크]")
            print(f"청크 {i} (섹션 경계 오버랩 포함):")
            
            # [섹션 연결] 마커 위치 찾기
            if "[섹션 연결]" in chunk.content:
                marker_idx = chunk.content.index("[섹션 연결]")
                print(f"  경계 전 (이전 섹션 끝): ...{chunk.content[:marker_idx][-100:]}")
                print(f"  [섹션 연결] 마커")
                print(f"  경계 후 (현재 섹션 시작): {chunk.content[marker_idx+7:][:100]}...")
            else:
                print(f"  시작 부분: {chunk.content[:200]}...")
            
            # 이전 청크와의 오버랩 확인
            if i > 0:
                prev = hybrid_chunks[i-1]
                overlap = 0
                for size in range(min(len(prev.content), len(chunk.content)), 0, -1):
                    if prev.content[-size:] == chunk.content[:size]:
                        overlap = size
                        break
                print(f"\n이전 청크와의 실제 오버랩: {overlap}자")
            break
    
    print("\n" + "=" * 80)
    
    # 2. 일반 청크 간 오버랩 비교
    print("\n### 2. 일반 청크 간 오버랩 비교 ###\n")
    
    # 기존 방식 - 같은 섹션 내 청크
    for i in range(len(original_chunks) - 1):
        curr = original_chunks[i]
        next_chunk = original_chunks[i + 1]
        
        if curr.metadata['section_idx'] == next_chunk.metadata['section_idx']:
            print("[기존 방식 - 섹션 내부 청크]")
            print(f"청크 {i} 끝부분 (마지막 100자):")
            print(f"  ...{curr.content[-100:]}")
            print(f"\n청크 {i+1} 시작부분 (처음 100자):")
            print(f"  {next_chunk.content[:100]}...")
            
            # 오버랩 체크
            overlap = 0
            for size in range(min(len(curr.content), len(next_chunk.content)), 0, -1):
                if curr.content[-size:] == next_chunk.content[:size]:
                    overlap = size
                    break
            print(f"\n실제 오버랩: {overlap}자")
            if overlap > 0:
                print(f"오버랩 내용: '{curr.content[-overlap:]}'")
            break
    
    print("\n" + "-" * 40 + "\n")
    
    # 하이브리드 방식 - 일반 청크
    for i in range(2, len(hybrid_chunks) - 1):  # 처음 몇 개는 경계 청크일 수 있으므로
        curr = hybrid_chunks[i]
        next_chunk = hybrid_chunks[i + 1]
        
        if not curr.metadata.get('has_boundary_overlap') and not next_chunk.metadata.get('has_boundary_overlap'):
            print("[하이브리드 방식 - 일반 청크]")
            print(f"청크 {i} 끝부분 (마지막 100자):")
            print(f"  ...{curr.content[-100:]}")
            print(f"\n청크 {i+1} 시작부분 (처음 100자):")
            print(f"  {next_chunk.content[:100]}...")
            
            # 오버랩 체크
            overlap = 0
            for size in range(min(len(curr.content), len(next_chunk.content)), 0, -1):
                if curr.content[-size:] == next_chunk.content[:size]:
                    overlap = size
                    break
            print(f"\n실제 오버랩: {overlap}자")
            if overlap > 0:
                print(f"오버랩 내용: '{curr.content[-overlap:]}'")
            break
    
    print("\n" + "=" * 80)
    
    # 3. 통계 요약
    print("\n### 3. 청크 크기 및 분포 ###\n")
    
    print("[기존 방식]")
    chunk_sizes = [len(c.content) for c in original_chunks]
    print(f"  청크 수: {len(original_chunks)}")
    print(f"  평균 크기: {sum(chunk_sizes)/len(chunk_sizes):.1f}자")
    print(f"  최소/최대: {min(chunk_sizes)}자 / {max(chunk_sizes)}자")
    
    # 섹션별 분포
    sections = {}
    for c in original_chunks:
        sid = c.metadata['section_idx']
        sections[sid] = sections.get(sid, 0) + 1
    print(f"  섹션별 분포: {dict(sorted(sections.items())[:5])}...")
    
    print("\n[하이브리드 방식]")
    chunk_sizes = [len(c.content) for c in hybrid_chunks]
    print(f"  청크 수: {len(hybrid_chunks)}")
    print(f"  평균 크기: {sum(chunk_sizes)/len(chunk_sizes):.1f}자")
    print(f"  최소/최대: {min(chunk_sizes)}자 / {max(chunk_sizes)}자")
    
    # 경계 청크 통계
    boundary_chunks = [c for c in hybrid_chunks if c.metadata.get('has_boundary_overlap')]
    print(f"  경계 오버랩 청크: {len(boundary_chunks)}개")
    
    # 섹션별 분포
    sections = {}
    for c in hybrid_chunks:
        sid = c.metadata['section_idx']
        sections[sid] = sections.get(sid, 0) + 1
    print(f"  섹션별 분포: {dict(sorted(sections.items())[:5])}...")
    
    # 4. 실제 청크 내용 샘플 저장
    print("\n결과를 'chunk_samples.json'에 저장 중...")
    
    samples = {
        'original': {
            'first_3_chunks': [
                {
                    'index': i,
                    'content': c.content[:200] + "..." if len(c.content) > 200 else c.content,
                    'size': len(c.content),
                    'section': c.metadata['section_idx']
                }
                for i, c in enumerate(original_chunks[:3])
            ]
        },
        'hybrid': {
            'first_3_chunks': [
                {
                    'index': i,
                    'content': c.content[:200] + "..." if len(c.content) > 200 else c.content,
                    'size': len(c.content),
                    'section': c.metadata['section_idx'],
                    'has_boundary_overlap': c.metadata.get('has_boundary_overlap', False)
                }
                for i, c in enumerate(hybrid_chunks[:3])
            ]
        }
    }
    
    with open('chunk_samples.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print("완료! 'chunk_samples.json'에서 전체 샘플을 확인할 수 있습니다.")


if __name__ == "__main__":
    test_document = "data/processed/개인정보 유출 등 사고 대응 매뉴얼(2024.3).txt"
    show_chunk_details(test_document)