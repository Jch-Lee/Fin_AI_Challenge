#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cleanup experimental files
실험 파일 정리
"""

import os
from pathlib import Path

# 삭제할 실험 파일 목록
files_to_delete = [
    # RAG 실험 파일들
    "test_bge_model_detailed.py",
    "test_complete_rag.py",
    "test_e5_with_real_pdf.py",
    "test_rag_final.py",
    "test_rag_simple.py",
    "test_rag_with_detailed_logs.py",
    "test_rag_with_e5.py",
    "test_recommended_models.py",
    "experiment_rag.py",
    
    # 통합 테스트는 유지 (주요 파일)
    # "test_full_pipeline_with_logs.py",  # 유지
    # "test_pdf_comparison.py",  # 유지  
    # "test_prompt_templates.py",  # 유지
    # "test_pymupdf4llm.py",  # 유지
    # "test_quality_checker.py",  # 유지
    # "test_quality_simple.py",  # 유지
    # "test_text_cleaner_integration.py",  # 유지
    
    # 기타 불필요한 파일
    "bge_test_output.log",
    "nul",  # 실수로 생성된 파일
    
    # 품질 보고서는 logs 폴더에 있으므로 루트의 것은 삭제
    "quality_report.json",
    "quality_report.txt",
]

# 현재 디렉토리
current_dir = Path(__file__).parent

print("=" * 60)
print(" 실험 파일 정리")
print("=" * 60)

deleted_count = 0
not_found_count = 0

for filename in files_to_delete:
    file_path = current_dir / filename
    if file_path.exists():
        try:
            file_path.unlink()
            print(f"[삭제] {filename}")
            deleted_count += 1
        except Exception as e:
            print(f"[오류] {filename}: {e}")
    else:
        print(f"[없음] {filename}")
        not_found_count += 1

print("\n" + "=" * 60)
print(f" 결과: {deleted_count}개 파일 삭제, {not_found_count}개 파일 없음")
print("=" * 60)

# 남은 테스트 파일 확인
print("\n[유지된 주요 테스트 파일]")
test_files_kept = [
    "test_full_pipeline_with_logs.py",
    "test_pdf_comparison.py",
    "test_prompt_templates.py", 
    "test_pymupdf4llm.py",
    "test_quality_checker.py",
    "test_quality_simple.py",
    "test_text_cleaner_integration.py",
    "view_pipeline_logs.py",
]

for filename in test_files_kept:
    file_path = current_dir / filename
    if file_path.exists():
        print(f"  - {filename}")

print("\n[유지된 설정 파일]")
config_files = [
    "CLAUDE.md",
    "README.md", 
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "activate_env.bat",
    "set_encoding.py",
    "setup_utf8.bat",
    "check_encoding.py",
]

for filename in config_files:
    file_path = current_dir / filename
    if file_path.exists():
        print(f"  - {filename}")

print("\n정리 완료!")