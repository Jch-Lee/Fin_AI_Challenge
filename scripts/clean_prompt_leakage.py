#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
프롬프트 누출 제거 스크립트
정확히 지정된 프롬프트 텍스트만 제거
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple

def remove_exact_prompt(content: str) -> Tuple[str, int]:
    """
    정확한 프롬프트 텍스트 제거
    
    Returns:
        (수정된 내용, 제거된 개수)
    """
    # 제거할 정확한 텍스트
    prompt_to_remove = """### 구조
- 문단: 빈 줄로 구분
- 번호 목록: 1. 2. 3.
- 글머리: - 또는 *
- 들여쓰기: 2칸 또는 4칸"""
    
    # 원본 길이 저장
    original_length = len(content)
    
    # 제거 횟수 카운트
    removal_count = content.count(prompt_to_remove)
    
    # 정확한 텍스트 제거
    cleaned_content = content.replace(prompt_to_remove, "")
    
    # 연속된 빈 줄 정리 (3개 이상의 연속 줄바꿈을 2개로)
    while "\n\n\n" in cleaned_content:
        cleaned_content = cleaned_content.replace("\n\n\n", "\n\n")
    
    return cleaned_content, removal_count

def process_files(input_dir: str, output_dir: str, backup: bool = True):
    """
    디렉토리의 모든 텍스트 파일 처리
    
    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
        backup: 백업 생성 여부
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 백업 디렉토리 생성
    if backup:
        backup_path = Path(input_dir + "_backup")
        if not backup_path.exists():
            print(f"백업 생성 중: {backup_path}")
            shutil.copytree(input_dir, backup_path)
            print("백업 완료!")
    
    txt_files = list(input_path.glob("*.txt"))
    print(f"처리할 파일 수: {len(txt_files)}")
    
    total_removed = 0
    affected_files = 0
    
    for file_path in txt_files:
        try:
            # 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 프롬프트 제거
            cleaned_content, removal_count = remove_exact_prompt(content)
            
            # 출력 파일 경로
            output_file = output_path / file_path.name
            
            # 파일 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            if removal_count > 0:
                affected_files += 1
                total_removed += removal_count
                print(f"  {file_path.name}: {removal_count}개 제거")
                
        except Exception as e:
            print(f"오류 발생 ({file_path.name}): {e}")
    
    print(f"\n=== 처리 완료 ===")
    print(f"총 처리 파일: {len(txt_files)}")
    print(f"수정된 파일: {affected_files}")
    print(f"제거된 프롬프트 수: {total_removed}")

def verify_removal(directory: str):
    """제거 확인"""
    prompt_to_check = """### 구조
- 문단: 빈 줄로 구분
- 번호 목록: 1. 2. 3.
- 글머리: - 또는 *
- 들여쓰기: 2칸 또는 4칸"""
    
    txt_files = list(Path(directory).glob("*.txt"))
    remaining_count = 0
    
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            count = content.count(prompt_to_check)
            if count > 0:
                remaining_count += count
                print(f"  {file_path.name}: 아직 {count}개 남음")
    
    if remaining_count == 0:
        print("[OK] 모든 프롬프트가 성공적으로 제거되었습니다!")
    else:
        print(f"[WARNING] 아직 {remaining_count}개의 프롬프트가 남아있습니다.")
    
    return remaining_count == 0

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="프롬프트 누출 제거")
    parser.add_argument("--input-dir", type=str, default="data/processed",
                       help="입력 디렉토리")
    parser.add_argument("--output-dir", type=str, default="data/processed_clean",
                       help="출력 디렉토리")
    parser.add_argument("--in-place", action="store_true",
                       help="원본 파일을 직접 수정 (백업 생성)")
    parser.add_argument("--no-backup", action="store_true",
                       help="백업 생성 안 함")
    parser.add_argument("--verify-only", action="store_true",
                       help="제거 확인만 수행")
    
    args = parser.parse_args()
    
    if args.verify_only:
        print("=== 프롬프트 제거 확인 ===")
        verify_removal(args.input_dir)
        return
    
    # in-place 모드면 입력/출력 디렉토리 동일
    if args.in_place:
        output_dir = args.input_dir
    else:
        output_dir = args.output_dir
    
    print("=" * 60)
    print("프롬프트 누출 제거 시작")
    print("=" * 60)
    print(f"입력 디렉토리: {args.input_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"백업 생성: {not args.no_backup}")
    print()
    
    # 처리 실행
    process_files(
        input_dir=args.input_dir,
        output_dir=output_dir,
        backup=(not args.no_backup and args.in_place)
    )
    
    # 제거 확인
    print("\n=== 제거 확인 ===")
    verify_removal(output_dir)

if __name__ == "__main__":
    main()