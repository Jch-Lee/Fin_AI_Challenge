# 데이터 수집 및 전처리

- 1.2.1. Hugging Face Datasets API를 사용해 후보 데이터셋 다운로드 스크립트 작성
- 1.2.2. 데이터 라이선스 검증 및 출처 기록 자동화 스크립트 작성
- 1.2.3. **Vision V2 기반 고품질 PDF 텍스트 추출 파서 구현** (41.2% 개선)
  - 주요 방법: Qwen2.5-VL-7B-Instruct 모델 사용
  - Fallback: PyMuPDF를 이용한 PDF 텍스트 및 표 추출
  - 3-Tier 안전망: Vision V2 → Traditional PyMuPDF → Basic PyMuPDF
- 1.2.4. BeautifulSoup4를 이용한 HTML 파서 구현
- 1.2.5. 국영문 혼합 텍스트 정제 클래스 (KoreanEnglishTextProcessor) 구현