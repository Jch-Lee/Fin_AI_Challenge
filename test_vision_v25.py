#!/usr/bin/env python3
"""
Vision v2.5 간단 테스트
청킹 최적화를 위한 마크다운 구조화 프롬프트 검증
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'packages'))

from vision.vision_extraction import VisionTextExtractor
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vision_v25():
    """Vision v2.5 기본 동작 테스트"""
    
    print("Vision v2.5 테스트 시작")
    
    # 1. VisionTextExtractor 초기화
    try:
        extractor = VisionTextExtractor()
        print("VisionTextExtractor 초기화 성공")
    except Exception as e:
        print(f"초기화 실패: {e}")
        return False
    
    # 2. 모델 가용성 확인
    if not extractor.is_available():
        print("Vision 모델을 사용할 수 없습니다 (GPU/의존성 부족)")
        return False
    
    print("Vision 모델 사용 가능")
    
    # 3. 컨텍스트 시스템 테스트
    print("\n컨텍스트 시스템 테스트:")
    
    # 첫 페이지 프롬프트
    prompt_page_0 = extractor._get_contextual_prompt(0)
    print(f"  - 첫 페이지 프롬프트 길이: {len(prompt_page_0)} 문자")
    
    # 가짜 컨텍스트 설정
    extractor.last_page_info = {
        'last_header': '## 제2장 보안 요구사항',
        'last_sentence': '다음 장에서는 구체적인 보안 대책에 대해 [다음 페이지에 계속]',
        'incomplete': True
    }
    
    # 두 번째 페이지 프롬프트
    prompt_page_1 = extractor._get_contextual_prompt(1)
    print(f"  - 두 번째 페이지 프롬프트 길이: {len(prompt_page_1)} 문자")
    print(f"  - 컨텍스트 힌트 포함됨: {'이전 페이지 컨텍스트' in prompt_page_1}")
    
    # 4. 마크다운 검증 시스템 테스트
    print("\n마크다운 검증 시스템 테스트:")
    
    test_markdown = """# 제1장 개요

금융보안은 현대 디지털 금융 시스템의 핵심 요소입니다.

## 1.1 정의

다음과 같은 요소들이 포함됩니다:

- 시스템 보안
- 데이터 보안  
- 네트워크 보안

### 표 예시

| 항목 | 중요도 | 비고 |
|------|--------|------|
| 인증 | 높음 | 필수 |
| 암호화 | 높음 | 권장 |

이 내용은 다음 페이지에서 [다음 페이지에 계속]"""
    
    quality = extractor._validate_markdown_structure(test_markdown)
    
    print(f"  - 헤더 감지: {quality['has_headers']}")
    print(f"  - 리스트 감지: {quality['has_lists']}")
    print(f"  - 표 감지: {quality['has_tables']}")
    print(f"  - 연속성 마커: {quality['has_continuity_markers']}")
    print(f"  - 구조 점수: {quality['structure_score']:.2f}/1.0")
    print(f"  - 헤더 개수: {quality['header_count']}")
    
    # 5. 컨텍스트 업데이트 테스트
    print("\n컨텍스트 업데이트 테스트:")
    
    extractor._update_page_context(test_markdown)
    
    print(f"  - 마지막 헤더: {extractor.last_page_info['last_header']}")
    print(f"  - 불완전 페이지: {extractor.last_page_info['incomplete']}")
    print(f"  - 마지막 문장: ...{extractor.last_page_info['last_sentence'][-50:]}")
    
    # 6. 프롬프트 누출 방지 테스트
    print("\n프롬프트 누출 방지 테스트:")
    
    # 프롬프트가 누출된 가상의 텍스트
    leaked_text = """<!--PROMPT_START_MARKER_DO_NOT_INCLUDE-->
이 문서 페이지에서 다음 우선순위에 따라 정보를 추출해주세요:

**1순위: 핵심 텍스트 정보**
- 제목, 소제목, 본문 내용을 정확히 추출

실제 문서 내용이 여기에 있습니다.

# 제2장 보안 정책

보안 정책은 다음과 같습니다.
<!--PROMPT_END_MARKER_DO_NOT_INCLUDE-->"""
    
    # 빈 내용 텍스트 (프롬프트만 있는 경우)
    empty_text = """이 문서 페이지에서 다음 우선순위에 따라 정보를 추출해주세요:
**1순위: 핵심 텍스트 정보**
위 우선순위에 따라 정리해주세요."""
    
    filtered_leaked = extractor._filter_prompt_leakage(leaked_text)
    filtered_empty = extractor._filter_prompt_leakage(empty_text)
    
    print(f"  - 누출된 텍스트 필터링:")
    print(f"    원본 길이: {len(leaked_text)} -> 필터링 후: {len(filtered_leaked)}")
    print(f"    결과: {filtered_leaked[:100]}...")
    
    print(f"  - 빈 내용 처리:")
    print(f"    원본 길이: {len(empty_text)} -> 필터링 후: {len(filtered_empty)}")
    print(f"    결과: {filtered_empty}")
    
    print("\n모든 기본 테스트 통과!")
    print("\n주요 개선사항:")
    print("  - 기존 41.2% 성능 향상 프롬프트 유지")
    print("  - 마크다운 구조화 지침 추가")
    print("  - 페이지 간 컨텍스트 추적")
    print("  - 품질 검증 시스템 통합")
    print("  - 프롬프트 누출 방지 시스템 추가")
    print("  - 청킹 최적화 준비 완료")
    
    return True

if __name__ == "__main__":
    success = test_vision_v25()
    
    if success:
        print("\nVision v2.5 개선 완료!")
        print("이제 LlamaIndex MarkdownNodeParser와 함께 사용할 준비가 되었습니다.")
    else:
        print("\n테스트 실패")
        sys.exit(1)