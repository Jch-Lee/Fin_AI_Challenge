#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
개선된 Vision 프롬프트 - 빈 페이지 감지 및 고품질 텍스트 추출
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def create_enhanced_vision_prompt_v3() -> str:
    """향상된 Vision V3 프롬프트 - 빈 페이지 감지 포함"""
    return """이미지를 정확히 분석하여 텍스트를 추출해주세요.

**1단계: 페이지 타입 판단**
먼저 이 이미지가 어떤 타입인지 확인하세요:
- 빈 페이지 (완전히 비어있거나 배경만 있음)
- 이미지만 있는 페이지 (차트, 그래프, 사진만 있고 텍스트 없음)
- 목차 페이지 (제목과 페이지 번호만 있음)
- 일반 텍스트 페이지 (문단, 설명, 본문이 있음)

**2단계: 텍스트 추출 기준**
다음 조건에 해당하면 "NO_TEXT_CONTENT"만 출력하세요:
- 읽을 수 있는 텍스트가 전혀 없는 경우
- 페이지 번호나 헤더/푸터만 있는 경우 (예: "12", "제2장", "- 12 -")
- 단순한 제목 1-2개만 있고 본문이 없는 경우
- 의미 있는 문장이나 설명이 없는 경우

**3단계: 텍스트가 있는 경우 추출 규칙**
의미 있는 텍스트가 있다면 다음 형식으로 추출하세요:
- 제목: ### 형태
- 소제목: #### 형태  
- 본문: 그대로 추출
- 목록: - 또는 번호 형태
- 표: 마크다운 테이블 형식

**중요**: 단순한 목차나 페이지 번호만 있는 페이지는 NO_TEXT_CONTENT로 처리하세요.

이제 이미지를 분석합니다:"""

def create_strict_text_detection_prompt() -> str:
    """엄격한 텍스트 감지 프롬프트"""
    return """이미지를 분석하여 의미 있는 텍스트가 있는지 판단해주세요.

**NO_TEXT_CONTENT 조건** (다음 중 하나라도 해당하면 NO_TEXT_CONTENT 출력):
1. 완전히 빈 페이지
2. 이미지, 차트, 그래프만 있고 텍스트가 없음
3. 페이지 번호만 있음 (예: "12", "- 12 -", "Page 12")
4. 헤더/푸터만 있음 (예: "금융보안원", "2023년")
5. 단순한 장/절 제목만 1-2개 있음 (예: "제2장", "제1절")
6. 의미 있는 본문이나 설명이 없음

**텍스트 추출 조건** (다음이 있어야 텍스트 추출):
- 완전한 문장이나 문단이 있음
- 목록이나 설명이 있음  
- 표나 구조화된 정보가 있음
- 3개 이상의 연결된 단어나 구문이 있음

판단 후 결과를 출력해주세요:"""

def create_two_stage_vision_prompt() -> str:
    """2단계 Vision 프롬프트 (판단 → 추출)"""
    return """
## Stage 1: 텍스트 존재 여부 판단

이미지를 보고 다음 중 하나를 선택하세요:
A) MEANINGFUL_TEXT - 의미 있는 텍스트가 있음 (문장, 문단, 목록, 표 등)
B) NO_TEXT_CONTENT - 텍스트가 없거나 의미 없음 (빈 페이지, 이미지만, 페이지 번호만)

## Stage 2: 텍스트 추출 (A를 선택한 경우만)

A를 선택했다면 모든 텍스트를 마크다운 형식으로 정확히 추출하세요:

### 제목은 이렇게
- 목록은 이렇게
- 번호 목록도 포함

| 표는 | 이렇게 |
|------|-------|
| 변환  | 하세요 |

일반 문단은 그대로 유지하세요.

B를 선택했다면 "NO_TEXT_CONTENT"만 출력하세요.

이제 분석을 시작합니다:
"""

def demo_improved_vision_extraction():
    """개선된 Vision 추출 데모"""
    
    print("="*70)
    print("개선된 Vision 프롬프트 데모")
    print("="*70)
    
    # 프롬프트 종류별 설명
    prompts = {
        "V3 Enhanced": create_enhanced_vision_prompt_v3(),
        "Strict Detection": create_strict_text_detection_prompt(),
        "Two Stage": create_two_stage_vision_prompt()
    }
    
    for name, prompt in prompts.items():
        print(f"\n[{name} 프롬프트]")
        print("-" * 50)
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
    
    print("\n" + "="*70)
    print("구현 예시:")
    print("="*70)
    
    implementation_example = '''
def extract_with_improved_vision(image_path: str, model_name: str = "qwen-vl") -> str:
    """개선된 Vision 추출"""
    
    # 프롬프트 선택
    prompt = create_two_stage_vision_prompt()
    
    # Vision 모델 호출
    response = vision_model.generate(
        image_path=image_path,
        prompt=prompt,
        max_tokens=2000,
        temperature=0.1  # 일관성을 위해 낮은 temperature
    )
    
    # 후처리
    if "NO_TEXT_CONTENT" in response:
        return ""  # 빈 페이지
    elif response.strip().startswith("B)"):
        return ""  # 명시적으로 텍스트 없음 선택
    else:
        # 실제 텍스트 추출된 경우
        return clean_vision_output(response)

def clean_vision_output(raw_output: str) -> str:
    """Vision 출력 정제"""
    # Stage 1 판단 부분 제거
    if "Stage 2:" in raw_output:
        parts = raw_output.split("Stage 2:", 1)
        if len(parts) > 1:
            raw_output = parts[1]
    
    # A), B) 선택 표시 제거
    raw_output = re.sub(r'^[AB]\)\s*', '', raw_output.strip())
    
    # 기타 정제
    return raw_output.strip()
    '''
    
    print(implementation_example)
    
    print("\n" + "="*70)
    print("기대 효과:")
    print("="*70)
    print("✅ 빈 페이지 자동 감지 및 빈 결과 출력")
    print("✅ 의미 없는 헤더/푸터만 있는 페이지 필터링") 
    print("✅ 이미지만 있는 페이지 건너뛰기")
    print("✅ 목차 페이지와 본문 페이지 구분")
    print("✅ 더 정확한 텍스트 추출")

def create_vision_config():
    """Vision 모델 설정 파일 생성"""
    
    config = {
        "vision_extraction_v3": {
            "prompt_version": "v3_enhanced",
            "enable_empty_page_detection": True,
            "min_text_length": 20,
            "min_word_count": 5,
            "skip_header_footer_only": True,
            "output_format": "markdown",
            "temperature": 0.1,
            "max_tokens": 2000
        },
        "empty_page_patterns": [
            "NO_TEXT_CONTENT",
            "^[\\s\\-\\*]*$",
            "^[0-9\\s\\-\\.]*$",
            "^제[0-9]+장$",
            "^제[0-9]+절$",
            "^- [0-9]+ -$"
        ],
        "meaningful_text_criteria": {
            "min_sentences": 1,
            "min_words": 5,
            "min_characters": 20,
            "require_complete_sentence": True
        }
    }
    
    import json
    with open("vision_extraction_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("Vision 설정 파일 생성됨: vision_extraction_config.json")

def main():
    print("개선된 Vision 텍스트 추출 프롬프트")
    print("="*70)
    
    # 데모 실행
    demo_improved_vision_extraction()
    
    # 설정 파일 생성
    print("\n설정 파일 생성:")
    create_vision_config()

if __name__ == "__main__":
    main()