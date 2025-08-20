# RAG System Diagnostic Test

## 📋 개요

RAG 파이프라인의 품질을 진단하고 검색 방식의 중요도를 분석하는 실험 도구입니다.

### 실험 목적
1. **벡터DB 자료의 질 점검**: 현재 지식 베이스가 금융보안 문제를 풀기에 충분한지 검증
2. **검색 방식 중요도 판단**: 키워드 기반(BM25) vs 의미 기반(Vector) 검색의 상대적 중요도 분석

## 🗂️ 파일 구조

```
tests/experiments/rag_diagnostic/
├── README.md                    # 이 파일
├── extract_questions.py         # 테스트 문제 추출 스크립트
├── test_questions_20.csv        # 추출된 20개 테스트 문제
├── diagnostic_prompts.py        # 진단용 프롬프트 템플릿
├── run_diagnostic_test.py       # 메인 진단 실행 스크립트
├── analyze_results.py           # 결과 분석 스크립트
├── diagnostic_results_*.json    # 진단 결과 (JSON)
├── diagnostic_summary_*.md      # 진단 요약 (Markdown)
└── analysis_report_*.md         # 상세 분석 보고서
```

## 🚀 실행 방법

### 1. 로컬 환경에서 테스트

```bash
# 1단계: 테스트 문제 추출 (이미 완료됨)
python tests/experiments/rag_diagnostic/extract_questions.py

# 2단계: 진단 테스트 실행
python tests/experiments/rag_diagnostic/run_diagnostic_test.py

# 3단계: 결과 분석
python tests/experiments/rag_diagnostic/analyze_results.py
```

### 2. 원격 서버에서 실행

```bash
# 파일 동기화 (로컬 → 서버)
scp -r tests/experiments/rag_diagnostic/ user@server:/path/to/project/tests/experiments/

# 서버에서 실행
ssh user@server
cd /path/to/project
python tests/experiments/rag_diagnostic/run_diagnostic_test.py

# 결과 가져오기 (서버 → 로컬)
scp user@server:/path/to/project/tests/experiments/rag_diagnostic/diagnostic_*.json ./
```

## 📊 출력 형식

### 진단 결과 (JSON)
```json
{
  "question_id": "TEST_XXX",
  "question_type": "multiple_choice|descriptive",
  "retrieval_results": {
    "top_5_documents": [...],
    "score_analysis": {
      "bm25_dominant": true/false,
      "vector_dominant": true/false,
      "avg_bm25_score": 0.XX,
      "avg_vector_score": 0.XX
    }
  },
  "generation_process": {
    "thought_process": "...",
    "evidence": "...",
    "score_preference": "BM25|Vector",
    "diagnostic_answer": "...",
    "simple_answer": "..."
  }
}
```

### 분석 보고서 주요 지표

1. **검색 품질**
   - 평균 Top-1 점수
   - 평균 Top-5 점수
   - 문서 활용률

2. **점수 패턴**
   - BM25 dominant 비율
   - Vector dominant 비율
   - 균형 잡힌 케이스 비율

3. **답변 품질**
   - 사고 과정 제공 비율
   - 근거 인용 비율
   - 진단/단순 답변 일치율

## 🔍 주요 분석 내용

### 현재 설정
- **BM25 가중치**: 70%
- **Vector 가중치**: 30%
- **청킹 크기**: 2300자
- **검색 문서 수**: 5개

### 평가 기준

#### 자료 충분성
- 검색된 문서의 관련성 점수
- 답변 생성에 실제 사용된 문서 비율
- 답변 품질과 검색 품질의 상관관계

#### 검색 방식 중요도
- BM25 점수가 높은 경우의 비율
- Vector 점수가 높은 경우의 비율
- 문제 유형별 선호 검색 방식

## 💡 예상 인사이트

1. **BM25 Dominant (>60%)**
   - 현재 70% 가중치가 적절함
   - 키워드 매칭이 중요한 도메인

2. **Vector Dominant (>40%)**
   - Vector 가중치 증가 고려 (30% → 40-50%)
   - 의미적 유사성이 중요

3. **문서 활용률 낮음 (<50%)**
   - 지식 베이스 확장 필요
   - 청킹 전략 재검토

## 🛠️ 커스터마이징

### 테스트 문제 수 변경
`extract_questions.py`에서 선택 개수 조정:
```python
selected_mc = mc_questions[:10]  # 객관식 10개
selected_desc = desc_questions[:10]  # 주관식 10개
```

### 검색 문서 수 변경
`run_diagnostic_test.py`에서 k 값 조정:
```python
contexts, scores = self.retrieve_with_scores(question, k=5)  # 기본 5개
```

### 프롬프트 수정
`diagnostic_prompts.py`에서 프롬프트 템플릿 수정

## 📝 참고사항

- GPU 메모리: 최소 16GB 권장 (Qwen2.5-7B 모델 사용)
- 실행 시간: 20문제 기준 약 10-15분
- 디스크 공간: 결과 파일 약 100MB

## 🔄 개선 계획

1. **실시간 모니터링**: 진행 상황 시각화
2. **배치 처리**: 대량 문제 처리 최적화
3. **비교 분석**: 다른 가중치 설정과 비교
4. **자동 튜닝**: 최적 가중치 자동 탐색