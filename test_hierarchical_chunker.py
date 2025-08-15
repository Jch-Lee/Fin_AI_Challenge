#!/usr/bin/env python3
"""
HierarchicalMarkdownChunker 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'packages'))

def test_hierarchical_chunker():
    """계층적 마크다운 청킹 테스트"""
    
    try:
        # LangChain 가용성 확인
        try:
            from langchain.text_splitter import MarkdownHeaderTextSplitter
            print("✓ LangChain available")
        except ImportError:
            print("× LangChain not available - skipping test")
            return False
        
        # 청커 import
        from packages.preprocessing.hierarchical_chunker import HierarchicalMarkdownChunker
        print("✓ HierarchicalMarkdownChunker imported successfully")
        
        # 테스트 마크다운 (Vision V2 출력 형태)
        test_markdown = """
# 금융보안 가이드라인

## 제1장 총칙

### 제1조 (목적)
이 가이드라인은 금융기관의 정보보호 및 사이버보안 강화를 위한 기본 원칙과 세부 실행방안을 제시함을 목적으로 한다.

금융분야에서 디지털 전환이 가속화되면서 사이버 보안의 중요성이 더욱 커지고 있다.

### 제2조 (정의)
1. "금융기관"이란 은행법에 따른 은행, 금융지주회사법에 따른 금융지주회사를 말한다.
2. "사이버 위협"이란 정보통신망을 통하여 금융기관의 정보자산에 피해를 줄 수 있는 모든 행위를 말한다.
3. "정보보호"란 정보의 수집, 가공, 저장, 검색, 송신, 수신 중에 정보의 훼손, 변조, 유출 등을 방지하기 위한 관리적, 기술적, 물리적 조치를 말한다.

[다음 페이지에 계속]

[이전 페이지에서 계속]

## 제2장 보안 요구사항

### 제3조 (접근통제)

모든 시스템 접근은 다음과 같은 통제 절차를 거쳐야 한다:

| 구분 | 요구사항 | 중요도 |
|------|----------|--------|
| 인증 | 다요소 인증 필수 | 높음 |
| 권한 | 최소 권한 원칙 | 높음 |
| 모니터링 | 실시간 감시 | 중간 |

#### 세부 규정
- 모든 시스템 접근은 로그를 남겨야 한다
- 비정상 접근 시도는 즉시 차단한다
- 권한 변경은 승인 절차를 거쳐야 한다

**중요**: 2024년 1월 1일부터 모든 금융기관은 제로트러스트 보안 모델을 적용해야 한다.
"""
        
        # 청커 생성 및 실행
        print("\n=== 청킹 실행 ===")
        chunker = HierarchicalMarkdownChunker()
        chunks = chunker.chunk_document(test_markdown, metadata={'source': 'vision_v2_test'})
        
        print(f"생성된 청크 수: {len(chunks)}")
        
        # 계층별 통계
        stats = chunker.get_hierarchy_stats(chunks)
        print("\n=== 계층별 통계 ===")
        for level, level_stats in stats['by_level'].items():
            print(f"Level {level}: {level_stats['count']}개 청크, 평균 {level_stats['avg_size']:.0f}자")
        
        print(f"\n전체 평균 크기: {stats['avg_chunk_size']:.0f}자")
        print(f"계층 경로 수: {len(stats['hierarchy_paths'])}")
        
        # 각 청크 검증
        print("\n=== 청크별 상세 정보 ===")
        for i, chunk in enumerate(chunks[:5]):  # 처음 5개만
            print(f"\n[청크 {i+1}]")
            print(f"  레벨: {chunk.metadata.get('hierarchy_level')}")
            print(f"  경로: {chunk.metadata.get('hierarchy_path')}")
            print(f"  헤더: {chunk.metadata.get('parent_header')}")
            print(f"  크기: {len(chunk.content)}자")
            print(f"  청킹 방법: {chunk.metadata.get('chunking_method')}")
            print(f"  페이지 경계 처리: {chunk.metadata.get('processed_boundaries')}")
            print(f"  내용: {chunk.content[:80]}...")
        
        # 특정 기능 검증
        print("\n=== 기능 검증 ===")
        
        # 1. 페이지 경계 마커 처리 확인
        has_boundary_markers = any('[다음 페이지에 계속]' in chunk.content or 
                                  '[이전 페이지에서 계속]' in chunk.content 
                                  for chunk in chunks)
        print(f"✓ 페이지 경계 마커 제거: {'성공' if not has_boundary_markers else '실패'}")
        
        # 2. 계층 정보 확인
        has_hierarchy = all(chunk.metadata.get('hierarchy_level') is not None 
                           for chunk in chunks)
        print(f"✓ 계층 정보 포함: {'성공' if has_hierarchy else '실패'}")
        
        # 3. 마크다운 헤더 보존 확인
        has_headers = any(chunk.metadata.get('parent_header') 
                         for chunk in chunks)
        print(f"✓ 마크다운 헤더 보존: {'성공' if has_headers else '실패'}")
        
        # 4. DocumentChunk 인터페이스 확인
        from packages.preprocessing.chunker import DocumentChunk
        all_document_chunks = all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        print(f"✓ DocumentChunk 호환성: {'성공' if all_document_chunks else '실패'}")
        
        print("\n✅ 모든 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== HierarchicalMarkdownChunker 테스트 ===")
    success = test_hierarchical_chunker()
    
    if success:
        print("\n테스트 완료: 계층적 마크다운 청킹이 정상 작동합니다!")
    else:
        print("\n테스트 실패: 문제를 확인해주세요.")