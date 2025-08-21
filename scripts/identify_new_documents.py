#!/usr/bin/env python3
"""
새로운 문서 식별 스크립트
raw 폴더의 PDF와 processed 폴더의 TXT를 비교하여 새로운 문서 파악
"""

from pathlib import Path
from typing import List, Set

def compare_directories() -> List[Path]:
    """
    raw와 processed 디렉토리를 비교하여 새로운 파일 목록 반환
    
    Returns:
        List[Path]: 처리되지 않은 새로운 PDF 파일 경로 리스트
    """
    # 프로젝트 루트 경로
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    # raw 폴더의 PDF 파일들
    raw_pdfs = list(raw_dir.glob("*.pdf"))
    raw_stems = {pdf.stem for pdf in raw_pdfs}
    
    # processed 폴더의 TXT 파일들
    processed_txts = list(processed_dir.glob("*.txt"))
    processed_stems = {txt.stem for txt in processed_txts}
    
    # 새로운 파일 찾기 (raw에는 있지만 processed에는 없는 파일)
    new_stems = raw_stems - processed_stems
    
    # 새로운 PDF 파일 경로 리스트 생성
    new_files = []
    for pdf in raw_pdfs:
        if pdf.stem in new_stems:
            new_files.append(pdf)
    
    # 파일명 기준 정렬
    new_files.sort(key=lambda x: x.name)
    
    return new_files

def main():
    """메인 실행 함수"""
    new_files = compare_directories()
    
    print("="*60)
    print("새로운 문서 분석 결과")
    print("="*60)
    
    if new_files:
        print(f"\n총 {len(new_files)}개의 새로운 문서 발견:")
        print("-"*60)
        for idx, file_path in enumerate(new_files, 1):
            print(f"{idx:3}. {file_path.name}")
    else:
        print("\n새로운 문서가 없습니다.")
    
    print("="*60)
    return new_files

if __name__ == "__main__":
    main()