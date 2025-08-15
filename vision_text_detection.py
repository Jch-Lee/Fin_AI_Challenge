#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vision 모델용 텍스트 존재 여부 판단 및 빈 페이지 처리
텍스트가 없는 페이지는 빈 결과 출력
"""

import sys
import io
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def has_meaningful_text(text: str, min_chars: int = 10, min_words: int = 3) -> bool:
    """
    텍스트에 의미 있는 내용이 있는지 판단
    
    Args:
        text: 검사할 텍스트
        min_chars: 최소 문자 수 (공백 제외)
        min_words: 최소 단어 수
    
    Returns:
        bool: 의미 있는 텍스트가 있으면 True
    """
    if not text or not text.strip():
        return False
    
    # 공백, 특수문자 제거한 순수 텍스트
    clean_text = re.sub(r'[^\w가-힣]', '', text)
    
    # 최소 문자 수 확인
    if len(clean_text) < min_chars:
        return False
    
    # 단어 분리 (한글, 영문, 숫자 단위)
    words = re.findall(r'[가-힣]+|[a-zA-Z]+|[0-9]+', text)
    
    # 최소 단어 수 확인
    if len(words) < min_words:
        return False
    
    # 의미 없는 패턴 확인
    meaningless_patterns = [
        r'^[\s\-\*\#\.]*$',  # 구분선, 점선 등만 있는 경우
        r'^[0-9\s\-\.]*$',   # 숫자와 구분자만 있는 경우
        r'^[a-zA-Z\s]*$' if len(words) <= 2 else None,  # 짧은 영문만 있는 경우
    ]
    
    for pattern in meaningless_patterns:
        if pattern and re.match(pattern, text.strip()):
            return False
    
    return True

def detect_page_type(image_path: str) -> str:
    """
    페이지 타입 감지 (실제 구현시 Vision 모델 사용)
    
    Args:
        image_path: 이미지 파일 경로
    
    Returns:
        str: 페이지 타입 ('text', 'image_only', 'empty', 'cover', 'toc')
    """
    # 현재는 파일명 기반 간단 판단 (실제로는 Vision 모델 사용)
    page_name = Path(image_path).parent.name
    
    # 경험적 규칙 (실제 Vision 모델로 대체 필요)
    if 'page_001' in page_name or 'page_056' in page_name:
        return 'cover'  # 표지 페이지
    elif any(x in page_name for x in ['page_011', 'page_025']):
        return 'toc'    # 목차 페이지
    else:
        return 'text'   # 일반 텍스트 페이지

def create_vision_prompt_with_detection() -> str:
    """텍스트 감지 기능이 포함된 Vision 프롬프트"""
    return """이미지를 분석하여 다음과 같이 처리해주세요:

1. 먼저 이미지에 읽을 수 있는 텍스트가 있는지 확인하세요.

2. 텍스트가 있는 경우:
   - 모든 텍스트를 정확히 추출하여 마크다운 형식으로 출력
   - 제목은 ### 형태로 표시
   - 목록은 - 또는 숫자로 표시
   - 표는 마크다운 테이블 형식으로 변환

3. 텍스트가 없거나 의미 있는 내용이 없는 경우:
   - "NO_TEXT_CONTENT" 만 출력

4. 판단 기준:
   - 단순한 페이지 번호, 헤더/푸터만 있는 경우: NO_TEXT_CONTENT
   - 이미지, 도표, 차트만 있고 텍스트가 없는 경우: NO_TEXT_CONTENT  
   - 빈 페이지이거나 구분선만 있는 경우: NO_TEXT_CONTENT
   - 의미 있는 문장이나 설명이 있는 경우: 텍스트 추출

텍스트 추출을 시작합니다:"""

def process_vision_result(vision_output: str) -> str:
    """
    Vision 모델 출력 후처리
    
    Args:
        vision_output: Vision 모델의 원본 출력
    
    Returns:
        str: 처리된 최종 결과 (빈 페이지면 빈 문자열)
    """
    if not vision_output or not vision_output.strip():
        return ""
    
    # NO_TEXT_CONTENT 체크
    if "NO_TEXT_CONTENT" in vision_output.upper():
        return ""
    
    # 의미 있는 텍스트 체크
    if not has_meaningful_text(vision_output):
        return ""
    
    # 정상적인 텍스트면 그대로 반환
    return vision_output.strip()

def update_vision_extraction_logic():
    """기존 Vision 추출 로직에 빈 페이지 감지 추가"""
    
    # 기존 Vision 추출 코드 수정 예시
    example_code = """
# 기존 코드 수정 예시

def extract_text_with_vision_v2(image_path: str) -> str:
    # Vision 모델 호출
    prompt = create_vision_prompt_with_detection()
    vision_output = vision_model.generate(image_path, prompt)
    
    # 후처리로 빈 페이지 필터링
    processed_result = process_vision_result(vision_output)
    
    return processed_result

# 또는 기존 결과 후처리
def filter_existing_results():
    base_path = Path("data/vision_extraction_benchmark/full_extraction_20250814_052410")
    
    for page_dir in base_path.glob("page_*"):
        vl_v2_file = page_dir / "vl_model_v2.txt"
        
        if vl_v2_file.exists():
            with open(vl_v2_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 텍스트 의미 검사
            if not has_meaningful_text(content):
                print(f"빈 페이지 감지: {page_dir.name}")
                # 빈 파일로 업데이트
                with open(vl_v2_file, 'w', encoding='utf-8') as f:
                    f.write("")
    """
    
    print("Vision 추출 로직 수정 가이드:")
    print("="*60)
    print(example_code)

def analyze_current_results():
    """현재 결과에서 빈 페이지 후보 찾기"""
    
    base_path = Path("data/vision_extraction_benchmark/full_extraction_20250814_052410")
    
    if not base_path.exists():
        print(f"경로를 찾을 수 없습니다: {base_path}")
        return
    
    empty_candidates = []
    short_text_pages = []
    
    print("현재 Vision 추출 결과 분석:")
    print("="*60)
    
    for page_dir in sorted(base_path.glob("page_*")):
        vl_v2_file = page_dir / "vl_model_v2.txt"
        
        if vl_v2_file.exists():
            with open(vl_v2_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 통계
            char_count = len(content.strip())
            word_count = len(re.findall(r'[가-힣]+|[a-zA-Z]+|[0-9]+', content))
            
            # 빈 페이지 후보 판단
            if char_count == 0:
                empty_candidates.append((page_dir.name, "완전히 빈 페이지"))
            elif char_count < 30 or word_count < 5:
                short_text_pages.append((page_dir.name, char_count, word_count, content[:50]))
            elif not has_meaningful_text(content):
                empty_candidates.append((page_dir.name, "의미 없는 텍스트"))
            
            # 간단 출력
            status = "✅" if has_meaningful_text(content) else "❌"
            print(f"{page_dir.name}: {status} 문자:{char_count:3d} 단어:{word_count:2d}")
    
    # 결과 요약
    print("\n" + "="*60)
    print("빈 페이지 후보:")
    for page, reason in empty_candidates:
        print(f"  {page}: {reason}")
    
    print("\n짧은 텍스트 페이지:")
    for page, chars, words, preview in short_text_pages:
        print(f"  {page}: {chars}자 {words}단어 - {preview}...")
    
    return empty_candidates, short_text_pages

def create_filtered_vision_results():
    """빈 페이지 필터링이 적용된 새 결과 생성"""
    
    base_path = Path("data/vision_extraction_benchmark/full_extraction_20250814_052410")
    
    updated_count = 0
    
    for page_dir in base_path.glob("page_*"):
        vl_v2_file = page_dir / "vl_model_v2.txt"
        vl_v2_filtered_file = page_dir / "vl_model_v2_filtered.txt"
        
        if vl_v2_file.exists():
            with open(vl_v2_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 필터링 적용
            filtered_content = process_vision_result(content)
            
            # 새 파일로 저장
            with open(vl_v2_filtered_file, 'w', encoding='utf-8') as f:
                f.write(filtered_content)
            
            if content != filtered_content:
                updated_count += 1
                print(f"필터링 적용: {page_dir.name}")
    
    print(f"\n총 {updated_count}개 페이지가 필터링되었습니다.")
    print("새 파일: vl_model_v2_filtered.txt")

def main():
    print("="*60)
    print("Vision 텍스트 감지 및 빈 페이지 필터링")
    print("="*60)
    
    # 1. 현재 결과 분석
    print("\n1. 현재 추출 결과 분석:")
    empty_candidates, short_text_pages = analyze_current_results()
    
    # 2. 필터링된 결과 생성
    print("\n2. 필터링된 결과 생성:")
    create_filtered_vision_results()
    
    # 3. Vision 추출 로직 수정 가이드
    print("\n3. 향후 Vision 추출 시 적용할 로직:")
    update_vision_extraction_logic()
    
    print("\n" + "="*60)
    print("완료!")

if __name__ == "__main__":
    main()