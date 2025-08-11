#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
프로젝트 전체 인코딩 검사 스크립트
UTF-8 인코딩이 누락된 파일 I/O를 찾아 보고
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def check_file_encoding(filepath: str) -> List[Tuple[int, str]]:
    """파일에서 인코딩이 누락된 open() 호출 찾기"""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines, 1):
            # open() 호출 패턴 찾기
            if 'open(' in line:
                # 바이너리 모드는 제외
                if "'rb'" in line or '"rb"' in line or "'wb'" in line or '"wb"' in line:
                    continue
                if "'ab'" in line or '"ab"' in line or "'r+b'" in line or '"r+b"' in line:
                    continue
                    
                # encoding 파라미터가 있는지 확인
                if 'encoding=' not in line:
                    # 텍스트 모드인 경우만 문제로 간주
                    if "'r'" in line or '"r"' in line or "'w'" in line or '"w"' in line:
                        issues.append((i, line.strip()))
                    elif "'a'" in line or '"a"' in line or "'r+'" in line or '"r+"' in line:
                        issues.append((i, line.strip()))
                    # 모드를 명시하지 않은 경우 (기본값 'r')
                    elif not any(mode in line for mode in ["'rb'", '"rb"', "'wb'", '"wb"']):
                        # open(filepath) 형태
                        if re.search(r'open\s*\([^,)]+\)', line):
                            issues.append((i, line.strip()))
                            
            # pd.read_csv() 호출 패턴 찾기
            if 'read_csv(' in line and 'encoding=' not in line:
                issues.append((i, line.strip()))
                
            # to_csv() 호출 패턴 찾기  
            if 'to_csv(' in line and 'encoding=' not in line:
                # index=False만 있는 경우도 체크
                issues.append((i, line.strip()))
                
    except Exception as e:
        print(f"  ❌ 파일 읽기 실패: {e}")
        
    return issues

def scan_project(root_dir: str = '.') -> dict:
    """프로젝트 전체 스캔"""
    results = {}
    python_files = []
    
    # Python 파일 찾기
    for root, dirs, files in os.walk(root_dir):
        # 가상환경 및 캐시 디렉토리 제외
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'env']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                python_files.append(filepath)
    
    print(f"총 {len(python_files)}개 Python 파일 검사 중...\n")
    
    total_issues = 0
    for filepath in python_files:
        issues = check_file_encoding(filepath)
        if issues:
            results[filepath] = issues
            total_issues += len(issues)
    
    return results, total_issues

def print_report(results: dict, total_issues: int):
    """검사 결과 보고서 출력"""
    print("\n" + "="*60)
    print("인코딩 검사 결과 보고서")
    print("="*60)
    
    if not results:
        print("\n[OK] 모든 파일이 올바른 인코딩 설정을 사용하고 있습니다!")
    else:
        print(f"\n[WARNING] 총 {len(results)}개 파일에서 {total_issues}개 이슈 발견:\n")
        
        for filepath, issues in results.items():
            rel_path = os.path.relpath(filepath)
            print(f"\n[FILE] {rel_path}")
            for line_num, line_content in issues:
                print(f"   Line {line_num}: {line_content[:80]}...")
    
    print("\n" + "="*60)
    print("권장사항:")
    print("="*60)
    print("1. 텍스트 파일 읽기: open(file, 'r', encoding='utf-8')")
    print("2. 텍스트 파일 쓰기: open(file, 'w', encoding='utf-8')")
    print("3. CSV 읽기: pd.read_csv(file, encoding='utf-8')")
    print("4. CSV 쓰기: df.to_csv(file, encoding='utf-8-sig', index=False)")
    print("5. 환경변수 설정: set PYTHONIOENCODING=utf-8")

if __name__ == "__main__":
    print("=== 프로젝트 인코딩 검사 시작 ===\n")
    
    # 현재 디렉토리에서 스캔
    results, total_issues = scan_project('.')
    
    # 결과 출력
    print_report(results, total_issues)
    
    # 수정이 필요한 경우 안내
    if results:
        print("\n[TIP] setup_utf8.bat 또는 set_encoding.py를 실행하여")
        print("      UTF-8 환경을 설정한 후 스크립트를 실행하세요.")