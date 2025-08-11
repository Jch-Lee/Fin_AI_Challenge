#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline Log Viewer
파이프라인 로그 뷰어 - 한눈에 보는 전체 과정
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

def view_latest_logs():
    """최신 로그 파일 보기"""
    
    log_dir = "logs"
    
    # 최신 로그 파일 찾기
    log_files = list(Path(log_dir).glob("pipeline_test_*.log"))
    if not log_files:
        print("로그 파일이 없습니다.")
        return
    
    latest_log = max(log_files, key=os.path.getctime)
    
    print("="*80)
    print(f" RAG 파이프라인 로그 분석")
    print(f" 로그 파일: {latest_log}")
    print("="*80)
    
    # 로그 파일 읽기
    with open(latest_log, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 단계별 분석
    stages = {
        "1단계: PDF → 텍스트": [],
        "2단계: 텍스트 → 청크": [],
        "3단계: 청크 → 임베딩": [],
        "4단계: 질문 → 검색": [],
        "5단계: 프롬프트 생성": [],
        "6단계: 답변 생성": [],
        "7단계: 품질 평가": []
    }
    
    current_stage = None
    
    for line in lines:
        # 단계 감지
        for stage_name in stages.keys():
            if stage_name in line:
                current_stage = stage_name
                break
        
        # 중요 정보 추출
        if current_stage and ("INFO" in line or "DEBUG" in line):
            # 시간 제거하고 내용만
            parts = line.split(" | ")
            if len(parts) >= 4:
                content = parts[3].strip()
                if content and not "====" in content:
                    stages[current_stage].append(content)
    
    # 결과 출력
    for stage_name, logs in stages.items():
        print(f"\n{'='*60}")
        print(f" {stage_name}")
        print(f"{'='*60}")
        
        if stage_name == "1단계: PDF → 텍스트":
            for log in logs:
                if "문자 추출" in log:
                    print(f"  [OK] {log}")
                elif "문제 패턴" in log and "0개" in log:
                    print(f"  [OK] {log}")
                elif "처리 시작" in log:
                    print(f"  [파일] {log}")
        
        elif stage_name == "2단계: 텍스트 → 청크":
            for log in logs:
                if "청크 생성" in log:
                    print(f"  [OK] {log}")
                elif "청크 정제" in log:
                    print(f"  [정제] {log}")
        
        elif stage_name == "3단계: 청크 → 임베딩":
            for log in logs:
                if "인덱싱 완료" in log:
                    print(f"  [OK] {log}")
        
        elif stage_name == "4단계: 질문 → 검색":
            for log in logs:
                if "검색 시작" in log:
                    print(f"  [검색] {log}")
                elif "검색 완료" in log:
                    print(f"  [OK] {log}")
                elif "검색 결과" in log and "점수" in log:
                    # 점수만 추출
                    if "점수:" in log:
                        score_part = log.split("점수:")[1].split(")")[0]
                        print(f"    - 점수: {score_part.strip()}")
        
        elif stage_name == "5단계: 프롬프트 생성":
            for log in logs:
                if "프롬프트 생성 완료" in log:
                    print(f"  [OK] {log}")
                elif "InferenceOrchestrator" in log:
                    print(f"  [모델] {log}")
        
        elif stage_name == "6단계: 답변 생성":
            for log in logs:
                if "답변 생성 완료" in log:
                    print(f"  [OK] {log}")
        
        elif stage_name == "7단계: 품질 평가":
            for log in logs:
                if "품질 평가 완료" in log:
                    print(f"  [OK] {log}")
    
    # 품질 보고서 읽기
    quality_report_path = Path(log_dir) / "quality_report_latest.json"
    if quality_report_path.exists():
        with open(quality_report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        print(f"\n{'='*60}")
        print(f" 품질 평가 결과")
        print(f"{'='*60}")
        
        print(f"\n전체 점수: {report['score']:.1f}/100")
        
        if 'details' in report and 'scores' in report['details']:
            scores = report['details']['scores']
            print("\n[세부 점수]")
            for component, score in scores.items():
                bar = "#" * int(score / 10) + "-" * (10 - int(score / 10))
                print(f"  {component:15s}: {bar} {score:6.1f}/100")
        
        if 'issues' in report and report['issues']:
            print(f"\n[발견된 문제: {len(report['issues'])}개]")
            for issue in report['issues']:
                print(f"  [경고] {issue}")
        
        if 'recommendations' in report:
            print("\n[개선 권고사항]")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
    
    print(f"\n{'='*80}")
    print(" [완료] 로그 분석 완료")
    print(f"{'='*80}")


def show_pipeline_flow():
    """파이프라인 플로우 다이어그램"""
    
    flow = """
    ================================================================
                   RAG 파이프라인 데이터 플로우                    
    ================================================================
    
    [PDF 파일]
        |
        v
    +-------------------------+
    | 1. PDF -> 텍스트 변환   | <- PyMuPDF4LLM
    |   - 레이아웃 보존       |
    |   - 테이블 추출         |
    +-------------------------+
        | 31,674 문자
        v
    +-------------------------+
    | 2. 텍스트 정제          | <- TextCleaner
    |   - 메타데이터 제거     |
    |   - 반복 패턴 제거      |
    +-------------------------+
        |
        v
    +-------------------------+
    | 3. 텍스트 청킹          | <- DocumentChunker
    |   - 500자 단위          |
    |   - 50자 오버랩         |
    +-------------------------+
        | 78개 청크
        v
    +-------------------------+
    | 4. 임베딩 생성          | <- E5 Embedding (시뮬레이션)
    |   - 384차원 벡터        |
    |   - FAISS 인덱싱        |
    +-------------------------+
        |
        v
    +-------------------------+
    | 5. 검색 (RAG)           | <- 하이브리드 검색
    |   - 질문 -> 컨텍스트   |
    |   - Top-5 선택          |
    +-------------------------+
        | 5개 컨텍스트
        v
    +-------------------------+
    | 6. 프롬프트 생성        | <- FinancialPromptManager
    |   - 금융 도메인 특화    |
    |   - Few-shot 예시       |
    +-------------------------+
        | 1,976 문자
        v
    +-------------------------+
    | 7. 답변 생성            | <- LLM (시뮬레이션)
    |   - 질문 유형별 처리    |
    |   - 도메인 지식 활용    |
    +-------------------------+
        |
        v
    +-------------------------+
    | 8. 품질 평가            | <- RAGQualityChecker
    |   - 15가지 품질 지표    |
    |   - 자동 권고사항       |
    +-------------------------+
        |
        v
    [최종 답변 + 품질 점수]
    """
    
    print(flow)


def main():
    """메인 함수"""
    
    print("\n" + "="*80)
    print(" RAG 파이프라인 로그 뷰어")
    print("="*80)
    
    print("\n[1] 최신 로그 분석")
    print("[2] 파이프라인 플로우 보기")
    print("[3] 모두 보기")
    
    choice = input("\n선택 (기본값: 3): ").strip() or "3"
    
    if choice == "1":
        view_latest_logs()
    elif choice == "2":
        show_pipeline_flow()
    else:
        show_pipeline_flow()
        view_latest_logs()


if __name__ == "__main__":
    main()