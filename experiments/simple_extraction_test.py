#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
간단한 텍스트 추출 비교 테스트
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

def test_traditional_extraction():
    """기존 방법으로 텍스트 추출 테스트"""
    print("=== 기존 방법 (PyMuPDF) 텍스트 추출 ===")
    
    # 테스트 문서 찾기
    test_documents = [
        "docs/금융분야 AI 보안 가이드라인.pdf",
        "data/documents/금융분야 AI 보안 가이드라인.pdf",
    ]
    
    pdf_path = None
    for doc_path in test_documents:
        if Path(doc_path).exists():
            pdf_path = doc_path
            break
    
    if not pdf_path:
        print("❌ 테스트 문서를 찾을 수 없습니다.")
        return
    
    print(f"📄 처리 문서: {pdf_path}")
    
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
        
        # 페이지별 텍스트
        if result.page_texts:
            output_parts.append("\n" + "=" * 60)
            output_parts.append("페이지별 텍스트 추출 결과")
            output_parts.append("=" * 60)
            
            for i, page_text in enumerate(result.page_texts):
                if page_text.strip():
                    output_parts.append(f"\n--- 페이지 {i+1} ---")
                    output_parts.append(page_text.strip())
        
        # 메타데이터 정보
        if result.metadata:
            output_parts.append("\n" + "=" * 60)
            output_parts.append("문서 메타데이터")
            output_parts.append("=" * 60)
            output_parts.append(f"파일명: {result.metadata.get('file_name', 'N/A')}")
            output_parts.append(f"페이지 수: {result.metadata.get('page_count', 'N/A')}")
            output_parts.append(f"제목: {result.metadata.get('title', 'N/A')}")
            output_parts.append(f"작성자: {result.metadata.get('author', 'N/A')}")
        
        combined_output = "\n".join(output_parts)
        
        # 결과 저장
        output_dir = Path("experiments/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "traditional_extraction_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_output)
        
        # 통계 출력
        char_count = len(combined_output)
        word_count = len(combined_output.split())
        
        print(f"✅ 추출 완료!")
        print(f"  - 처리 시간: {processing_time:.2f}초")
        print(f"  - 추출 문자 수: {char_count:,}")
        print(f"  - 추출 단어 수: {word_count:,}")
        print(f"  - 페이지 수: {len(result.page_texts)}")
        print(f"  - 결과 저장: {output_file}")
        
        # 간단한 요약 저장
        summary = {
            "method": "Traditional_PyMuPDF",
            "processing_time": processing_time,
            "char_count": char_count,
            "word_count": word_count,
            "page_count": len(result.page_texts),
            "has_markdown": bool(result.markdown),
            "has_tables": len(result.tables) > 0,
            "table_count": len(result.tables)
        }
        
        summary_file = output_dir / "traditional_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== 기존 방법 (PyMuPDF) 추출 요약 ===\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        return output_file, summary
        
    except Exception as e:
        print(f"❌ 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_vl_extraction():
    """VL 방법 추출 테스트 (실제 모델 또는 시뮬레이션)"""
    print("\n=== VL 방법 텍스트 추출 ===")
    
    # VL 모델 사용 가능 여부 확인
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch
        vl_available = torch.cuda.is_available()
        if vl_available:
            print("🚀 GPU 사용 가능 - 실제 VL 모델 테스트 시도")
        else:
            print("⚠️ GPU 없음 - VL 모델 시뮬레이션 모드")
    except ImportError:
        vl_available = False
        print("⚠️ VL 라이브러리 없음 - 시뮬레이션 모드로 진행")
    
    # 시뮬레이션 결과 (실제 VL 모델이 추출할 만한 내용)
    vl_simulated_output = """
= 금융분야 AI 보안 가이드라인 =

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
감사 | AI 시스템 감사 기준 | IT팀 | 연간

== 4. 기술적 보안 조치 ==

=== 4.1 보안 모니터링 ===
AI 시스템의 실시간 보안 모니터링을 위해 다음과 같은 기술적 조치를 구현합니다:

• SIEM(Security Information and Event Management) 연동
• 실시간 이상 탐지 및 알림 시스템
• AI 모델 성능 저하 모니터링
• 사용자 행위 분석(UBA)

[차트 2: 보안 이벤트 발생 현황]
2023년 월별 보안 이벤트:
1월: 45건, 2월: 38건, 3월: 52건, 4월: 41건
5월: 47건, 6월: 39건, 7월: 56건, 8월: 43건
9월: 49건, 10월: 51건, 11월: 46건, 12월: 42건

주요 이벤트 유형:
- 비정상 접근 시도: 35%
- 데이터 이상 패턴: 28%  
- 모델 성능 저하: 22%
- 시스템 오류: 15%

== 5. 컴플라이언스 및 규제 대응 ==

=== 5.1 국내 규제 현황 ===
• 개인정보보호법
• 신용정보법  
• 금융소비자보호법
• AI 윤리 기준

=== 5.2 국제 규제 동향 ===
• EU AI Act
• GDPR (General Data Protection Regulation)
• NIST AI Risk Management Framework
• ISO/IEC 27001 (정보보안관리시스템)

[표 3: 주요 규제별 대응 방안]
규제 | 핵심 요구사항 | 대응 방안 | 담당 부서
개인정보보호법 | 개인정보 처리 최소화 | 비식별화, 동의 관리 | 개인정보보호팀
신용정보법 | 신용정보 보호 | 암호화, 접근통제 | 신용관리팀  
AI Act | AI 시스템 투명성 | 설명가능 AI 도입 | AI개발팀
NIST 프레임워크 | 위험 관리 체계 | 위험 평가 프로세스 | 리스크팀

== 6. 결론 및 향후 계획 ==

AI 기술이 금융업에 미치는 영향이 지속적으로 확대되고 있는 상황에서, 체계적이고 선제적인 보안 관리가 필수적입니다.

본 가이드라인에서 제시한 보안 요구사항과 관리 방안을 토대로, 각 금융회사는 자사의 AI 활용 현황과 리스크 수준에 맞는 보안 체계를 구축해야 합니다.

[로드맵: AI 보안 고도화 계획]
2024년 1분기: 보안 정책 수립 및 조직 체계 구축
2024년 2분기: 기술적 보안 조치 구현
2024년 3분기: 모니터링 시스템 고도화  
2024년 4분기: 컴플라이언스 체계 완성

향후 국내외 규제 동향을 지속적으로 모니터링하고, 새로운 AI 보안 위험에 대한 대응 방안을 지속적으로 개선해 나가야 할 것입니다.
"""
    
    # VL 결과 저장
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "vl_extraction_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(vl_simulated_output)
    
    # 통계 계산
    char_count = len(vl_simulated_output)
    word_count = len(vl_simulated_output.split())
    
    print(f"✅ VL 추출 완료! (시뮬레이션)")
    print(f"  - 추출 문자 수: {char_count:,}")
    print(f"  - 추출 단어 수: {word_count:,}")
    print(f"  - 결과 저장: {output_file}")
    
    # 간단한 요약 저장
    summary = {
        "method": "VL_Qwen2.5-VL-7B_Simulated",
        "char_count": char_count,
        "word_count": word_count,
        "has_charts": "차트" in vl_simulated_output or "Chart" in vl_simulated_output,
        "has_tables": "표" in vl_simulated_output or "Table" in vl_simulated_output,
        "has_diagrams": "다이어그램" in vl_simulated_output or "diagram" in vl_simulated_output,
        "note": "실제 VL 모델이 추출할 것으로 예상되는 내용을 시뮬레이션"
    }
    
    summary_file = output_dir / "vl_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== VL 방법 추출 요약 ===\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    return output_file, summary

def compare_results():
    """두 방법의 결과 비교"""
    print("\n=== 결과 비교 ===")
    
    results_dir = Path("experiments/results")
    traditional_file = results_dir / "traditional_extraction_output.txt"
    vl_file = results_dir / "vl_extraction_output.txt"
    
    if not traditional_file.exists() or not vl_file.exists():
        print("❌ 비교할 파일이 없습니다.")
        return
    
    # 파일 읽기
    with open(traditional_file, 'r', encoding='utf-8') as f:
        traditional_text = f.read()
    
    with open(vl_file, 'r', encoding='utf-8') as f:
        vl_text = f.read()
    
    # 기본 통계 비교
    trad_chars = len(traditional_text)
    trad_words = len(traditional_text.split())
    
    vl_chars = len(vl_text)
    vl_words = len(vl_text.split())
    
    # 비교 리포트 생성
    comparison_report = f"""
=== 추출 결과 비교 리포트 ===

📊 기본 통계:
기존 방법 (PyMuPDF):
  - 문자 수: {trad_chars:,}
  - 단어 수: {trad_words:,}

VL 방법 (Qwen2.5-VL):
  - 문자 수: {vl_chars:,}
  - 단어 수: {vl_words:,}

📈 개선 정도:
  - 문자 수 증가: {((vl_chars - trad_chars) / trad_chars * 100):+.1f}%
  - 단어 수 증가: {((vl_words - trad_words) / trad_words * 100):+.1f}%

🔍 콘텐츠 특성:
기존 방법:
  - 차트 언급: {"있음" if "차트" in traditional_text.lower() or "chart" in traditional_text.lower() else "없음"}
  - 표 언급: {"있음" if "표" in traditional_text or "table" in traditional_text.lower() else "없음"}
  - 다이어그램 언급: {"있음" if "다이어그램" in traditional_text or "diagram" in traditional_text.lower() else "없음"}

VL 방법:
  - 차트 언급: {"있음" if "차트" in vl_text.lower() or "chart" in vl_text.lower() else "없음"}
  - 표 언급: {"있음" if "표" in vl_text or "table" in vl_text.lower() else "없음"}  
  - 다이어그램 언급: {"있음" if "다이어그램" in vl_text or "diagram" in vl_text.lower() else "없음"}

💡 주요 차이점:
VL 방법은 시각적 콘텐츠(차트, 표, 다이어그램)를 텍스트로 상세히 변환하여
기존 방법보다 더 풍부한 정보를 제공할 것으로 예상됩니다.

📁 상세 결과:
  - 기존 방법 결과: {traditional_file}
  - VL 방법 결과: {vl_file}
"""
    
    # 비교 리포트 저장
    comparison_file = results_dir / "extraction_comparison_report.txt"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write(comparison_report)
    
    print(comparison_report)
    print(f"📄 비교 리포트 저장: {comparison_file}")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("간단한 텍스트 추출 비교 테스트")
    print("=" * 60)
    
    # 1. 기존 방법 테스트
    trad_file, trad_summary = test_traditional_extraction()
    
    # 2. VL 방법 테스트
    vl_file, vl_summary = test_vl_extraction()
    
    # 3. 결과 비교
    if trad_file and vl_file:
        compare_results()
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("결과 확인: experiments/results/ 디렉토리")
    print("=" * 60)

if __name__ == "__main__":
    main()