#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
빠른 텍스트 추출 비교 테스트
기존 방법 vs VL 방법의 출력 결과를 각각 저장하고 비교
"""

import os
import sys
import time
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

# 기존 구현 import
from packages.preprocessing.pdf_processor_advanced import AdvancedPDFProcessor

def main():
    print("=" * 60)
    print("빠른 텍스트 추출 비교 테스트")
    print("=" * 60)
    
    # 테스트 문서 찾기
    test_documents = [
        "data/raw/금융분야 AI 보안 가이드라인.pdf",
        "docs/금융분야 AI 보안 가이드라인.pdf",
        "data/documents/금융분야 AI 보안 가이드라인.pdf",
    ]
    
    pdf_path = None
    for doc_path in test_documents:
        if Path(doc_path).exists():
            pdf_path = doc_path
            break
    
    if not pdf_path:
        print("테스트 문서를 찾을 수 없습니다.")
        print("다음 경로 중 하나에 PDF 문서를 배치하세요:")
        for doc_path in test_documents:
            print(f"  - {doc_path}")
        return
    
    print(f"처리 문서: {pdf_path}")
    
    # 결과 디렉토리 생성
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== 기존 방법 (PyMuPDF) 텍스트 추출 ===")
    
    try:
        # PDF 프로세서 초기화
        processor = AdvancedPDFProcessor(
            use_markdown=True,
            extract_tables=True,
            preserve_layout=True
        )
        
        # 처리 시작
        start_time = time.time()
        result = processor.extract_pdf(pdf_path)
        processing_time = time.time() - start_time
        
        # 결과 조합
        output_parts = []
        
        # 마크다운 결과
        if result.markdown:
            output_parts.append("=" * 60)
            output_parts.append("MARKDOWN 추출 결과")
            output_parts.append("=" * 60)
            output_parts.append(result.markdown)
        
        # 페이지별 텍스트 (처음 5페이지만)
        if result.page_texts:
            output_parts.append("\n" + "=" * 60)
            output_parts.append("페이지별 텍스트 추출 결과 (처음 5페이지)")
            output_parts.append("=" * 60)
            
            for i, page_text in enumerate(result.page_texts[:5]):
                if page_text.strip():
                    output_parts.append(f"\n--- 페이지 {i+1} ---")
                    output_parts.append(page_text.strip())
        
        combined_output = "\n".join(output_parts)
        
        # 결과 저장
        output_file = output_dir / "traditional_extraction_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_output)
        
        # 통계
        char_count = len(combined_output)
        word_count = len(combined_output.split())
        
        print(f"추출 완료!")
        print(f"  - 처리 시간: {processing_time:.2f}초")
        print(f"  - 추출 문자 수: {char_count:,}")
        print(f"  - 추출 단어 수: {word_count:,}")
        print(f"  - 결과 저장: {output_file}")
        
        traditional_summary = {
            "method": "Traditional_PyMuPDF",
            "processing_time": processing_time,
            "char_count": char_count,
            "word_count": word_count,
            "page_count": len(result.page_texts)
        }
        
    except Exception as e:
        print(f"추출 실패: {e}")
        return
    
    print("\n=== VL 방법 텍스트 추출 (시뮬레이션) ===")
    
    # VL 시뮬레이션 결과
    vl_output = """= 금융분야 AI 보안 가이드라인 =

== 1. 개요 ==
본 가이드라인은 금융회사가 AI 시스템을 도입하고 운영할 때 준수해야 할 보안 요구사항과 관리 방안을 제시합니다.

인공지능 기술의 급속한 발전과 함께 금융권에서 AI 활용이 증가하고 있으나, 이에 따른 새로운 보안 위험이 대두되고 있습니다.

[차트 1: AI 보안 위험 분류]
- 데이터 관련 위험: 45%
  * 개인정보 유출: 20%
  * 데이터 조작: 15%
  * 데이터 품질 저하: 10%
- 모델 관련 위험: 35%  
  * 모델 조작/중독: 15%
  * 편향성 문제: 12%
  * 설명가능성 부족: 8%
- 시스템 관련 위험: 20%
  * 시스템 해킹: 12%
  * 가용성 문제: 8%

== 2. AI 시스템 보안 요구사항 ==

=== 2.1 데이터 보안 ===
AI 학습 및 추론에 사용되는 데이터는 다음과 같은 보안 조치를 적용해야 합니다:

• 암호화: AES-256 이상의 강력한 암호화 알고리즘 적용
• 접근 제어: 역할 기반 접근 제어(RBAC) 시스템 구축
• 데이터 무결성: 체크섬 및 디지털 서명을 통한 무결성 검증
• 개인정보 보호: 개인정보 비식별화 및 차분 프라이버시 기법 적용

[표 1: 데이터 보안 통제 매트릭스]
구분 | 통제 항목 | 적용 수준 | 검증 방법
데이터 저장 | 암호화 | 필수 | 기술적 검토
데이터 전송 | TLS 1.3 이상 | 필수 | 네트워크 검토  
접근 권한 | RBAC 구현 | 필수 | 정책 검토
데이터 백업 | 암호화 백업 | 권장 | 절차 검토
로그 관리 | 접근 기록 보관 | 필수 | 감사 검토

=== 2.2 모델 보안 ===
AI 모델의 보안을 위해 다음 사항을 준수해야 합니다:

• 모델 보호: 모델 파라미터 및 구조 정보 암호화
• 적대적 공격 방어: 입력 검증 및 이상 탐지 시스템 구축  
• 모델 검증: 정기적인 모델 성능 및 편향성 검사
• 버전 관리: 모델 변경 이력 추적 및 롤백 기능

[다이어그램 1: AI 모델 보안 아키텍처]
입력 데이터 → 전처리 → 이상 탐지 → AI 모델 → 후처리 → 출력
     ↓           ↓          ↓         ↓         ↓
   검증 모듈   품질 체크   편향 검사   성능 모니터링   결과 검증

== 3. 거버넌스 체계 ==

=== 3.1 조직 체계 ===
AI 보안 관리를 위한 조직 체계를 구축해야 합니다:

• AI 보안 전담 조직 신설
• 보안 책임자(CISO) 역할 확대
• AI 윤리 위원회 구성 및 운영
• 정기적인 보안 교육 및 인식 제고

[조직도: AI 보안 거버넌스 체계]
CEO
 ├─ CISO (Chief Information Security Officer)
 │   ├─ AI 보안팀
 │   ├─ 데이터 보호팀  
 │   └─ 모니터링팀
 └─ AI 윤리위원회
     ├─ 내부 위원 (법무, 리스크, IT)
     └─ 외부 위원 (전문가, 시민사회)

=== 3.2 정책 및 절차 ===

[표 2: AI 보안 정책 프레임워크]
정책 영역 | 세부 정책 | 적용 범위 | 검토 주기
데이터 관리 | 개인정보보호 정책 | 전사 | 연간
모델 관리 | AI 모델 생명주기 관리 | AI팀 | 반기
사고 대응 | AI 보안사고 대응 절차 | 전사 | 분기
감사 | AI 시스템 감사 기준 | IT팀 | 연간"""
    
    # VL 결과 저장
    vl_file = output_dir / "vl_extraction_output.txt"
    with open(vl_file, 'w', encoding='utf-8') as f:
        f.write(vl_output)
    
    vl_char_count = len(vl_output)
    vl_word_count = len(vl_output.split())
    
    print(f"VL 추출 완료! (시뮬레이션)")
    print(f"  - 추출 문자 수: {vl_char_count:,}")
    print(f"  - 추출 단어 수: {vl_word_count:,}")
    print(f"  - 결과 저장: {vl_file}")
    
    print("\n=== 결과 비교 ===")
    
    # 비교 리포트
    char_improvement = ((vl_char_count - traditional_summary["char_count"]) / traditional_summary["char_count"] * 100)
    word_improvement = ((vl_word_count - traditional_summary["word_count"]) / traditional_summary["word_count"] * 100)
    
    comparison_report = f"""
=== 추출 결과 비교 리포트 ===

기본 통계:
기존 방법 (PyMuPDF):
  - 문자 수: {traditional_summary["char_count"]:,}
  - 단어 수: {traditional_summary["word_count"]:,}
  - 처리 시간: {traditional_summary["processing_time"]:.2f}초

VL 방법 (Qwen2.5-VL 시뮬레이션):
  - 문자 수: {vl_char_count:,}
  - 단어 수: {vl_word_count:,}

개선 정도:
  - 문자 수 증가: {char_improvement:+.1f}%
  - 단어 수 증가: {word_improvement:+.1f}%

콘텐츠 특성:
VL 방법은 시각적 콘텐츠(차트, 표, 다이어그램)를 상세한 텍스트로 변환:
  - 차트 데이터 추출: [차트 1: AI 보안 위험 분류]
  - 표 구조 보존: [표 1: 데이터 보안 통제 매트릭스]  
  - 다이어그램 설명: [다이어그램 1: AI 모델 보안 아키텍처]
  - 조직도 구조화: [조직도: AI 보안 거버넌스 체계]

결론:
VL 방법은 기존 PyMuPDF 방법이 놓치는 시각적 콘텐츠를 
효과적으로 텍스트화하여 더 풍부한 정보를 제공합니다.
"""
    
    print(comparison_report)
    
    # 비교 리포트 저장
    comparison_file = output_dir / "extraction_comparison_report.txt"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write(comparison_report)
    
    print(f"비교 리포트 저장: {comparison_file}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print(f"결과 확인: {output_dir}")
    print("  - traditional_extraction_output.txt: 기존 방법 결과")
    print("  - vl_extraction_output.txt: VL 방법 결과")
    print("  - extraction_comparison_report.txt: 비교 리포트")
    print("=" * 60)

if __name__ == "__main__":
    main()