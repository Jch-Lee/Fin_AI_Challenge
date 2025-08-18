#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
프롬프트 누출 검사 및 제거 스크립트
처리된 텍스트 파일에서 반복되는 프롬프트 패턴 찾기
"""

import os
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

def find_prompt_patterns(directory: str) -> Dict[str, List[Tuple[str, int, int]]]:
    """
    디렉토리의 모든 텍스트 파일에서 프롬프트 패턴 찾기
    
    Returns:
        파일별 프롬프트 패턴 위치 (파일명: [(패턴, 시작줄, 끝줄)])
    """
    prompt_patterns = {}
    
    # 찾을 패턴들
    patterns_to_find = [
        r"### 구조.*?(?=\n\n|\Z)",  # ### 구조로 시작하는 부분
        r"### 분석.*?(?=\n\n|\Z)",  # ### 분석으로 시작하는 부분
        r"### 요약.*?(?=\n\n|\Z)",  # ### 요약으로 시작하는 부분
        r"위 페이지.*?분석해.*?(?=\n|\Z)",  # 분석 지시문
        r"다음.*?추출.*?(?=\n|\Z)",  # 추출 지시문
        r"페이지 \d+:.*?(?=\n\n|\Z)",  # 페이지 번호 패턴
    ]
    
    # 전체 프롬프트 블록 패턴 (더 넓은 범위)
    block_pattern = r"(### [구조|분석|요약].*?)(?=(?:### [구조|분석|요약])|(?:---)|(?:\n{3,})|\Z)"
    
    txt_files = list(Path(directory).glob("*.txt"))
    print(f"검사할 파일 수: {len(txt_files)}")
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            found_patterns = []
            
            # 블록 패턴 찾기
            for match in re.finditer(block_pattern, content, re.DOTALL | re.MULTILINE):
                pattern_text = match.group(0)
                start_pos = match.start()
                
                # 줄 번호 계산
                start_line = content[:start_pos].count('\n')
                end_line = start_line + pattern_text.count('\n')
                
                # 처음 100자만 표시
                preview = pattern_text[:100].replace('\n', '\\n')
                if len(pattern_text) > 100:
                    preview += "..."
                
                found_patterns.append((preview, start_line, end_line))
            
            if found_patterns:
                prompt_patterns[file_path.name] = found_patterns
                
        except Exception as e:
            print(f"오류 발생 ({file_path.name}): {e}")
    
    return prompt_patterns

def analyze_patterns(prompt_patterns: Dict[str, List[Tuple[str, int, int]]]):
    """패턴 분석 및 통계"""
    total_files_affected = len(prompt_patterns)
    total_patterns = sum(len(patterns) for patterns in prompt_patterns.values())
    
    print(f"\n=== 프롬프트 누출 분석 결과 ===")
    print(f"영향받은 파일 수: {total_files_affected}")
    print(f"발견된 패턴 총 개수: {total_patterns}")
    
    # 각 파일별 상세 정보
    print("\n=== 파일별 상세 정보 ===")
    for filename, patterns in list(prompt_patterns.items())[:5]:  # 처음 5개만 표시
        print(f"\n파일: {filename}")
        print(f"  발견된 패턴 수: {len(patterns)}")
        for i, (preview, start, end) in enumerate(patterns[:3], 1):  # 각 파일당 3개만 표시
            print(f"  패턴 {i} (줄 {start}-{end}): {preview}")
    
    if len(prompt_patterns) > 5:
        print(f"\n... 그리고 {len(prompt_patterns) - 5}개 파일 더")

def extract_common_prompts(directory: str) -> List[str]:
    """공통 프롬프트 패턴 추출"""
    all_blocks = []
    
    # 모든 파일에서 ### 로 시작하는 블록 수집
    txt_files = list(Path(directory).glob("*.txt"))
    
    for file_path in txt_files[:10]:  # 처음 10개 파일만 샘플링
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # ### 로 시작하는 블록 찾기
            blocks = re.findall(r'### .*?(?=###|\Z)', content, re.DOTALL)
            all_blocks.extend(blocks[:5])  # 각 파일당 최대 5개
            
        except Exception as e:
            continue
    
    # 블록 내용 분석
    print("\n=== 발견된 프롬프트 블록 샘플 ===")
    unique_starts = set()
    
    for block in all_blocks:
        lines = block.split('\n')
        if lines:
            first_line = lines[0].strip()
            if first_line.startswith('###'):
                unique_starts.add(first_line)
    
    for start in sorted(unique_starts)[:20]:  # 최대 20개 표시
        print(f"  {start}")
    
    return list(unique_starts)

def main():
    """메인 함수"""
    directory = "data/processed"
    
    print("=" * 60)
    print("프롬프트 누출 검사 시작")
    print("=" * 60)
    
    # 1. 프롬프트 패턴 찾기
    prompt_patterns = find_prompt_patterns(directory)
    
    # 2. 패턴 분석
    analyze_patterns(prompt_patterns)
    
    # 3. 공통 프롬프트 추출
    common_prompts = extract_common_prompts(directory)
    
    print("\n" + "=" * 60)
    print("검사 완료")
    print("=" * 60)
    
    # 제거 필요 여부 확인
    if prompt_patterns:
        print("\n[경고] 프롬프트 누출이 발견되었습니다!")
        print("clean_prompt_leakage.py 스크립트를 실행하여 제거하세요.")
        
        # 샘플 파일 하나 상세 분석
        sample_file = list(prompt_patterns.keys())[0]
        print(f"\n샘플 파일 상세 분석: {sample_file}")
        
        file_path = Path(directory) / sample_file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # ### 구조 패턴 찾기
        structure_blocks = re.findall(r'### 구조.*?(?=###|\n\n\n|\Z)', content, re.DOTALL)
        if structure_blocks:
            print(f"\n'### 구조' 블록 발견: {len(structure_blocks)}개")
            print("첫 번째 블록 내용:")
            print("-" * 40)
            print(structure_blocks[0][:500])
            print("-" * 40)

if __name__ == "__main__":
    main()