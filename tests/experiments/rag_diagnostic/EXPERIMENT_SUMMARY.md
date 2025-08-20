# RAG 진단 실험 요약

## 🎯 실험 목표

### 1. 벡터DB 자료의 질 점검
- 현재 지식 베이스(2300자 청킹, 50개 PDF)가 금융보안 문제를 해결하기에 충분한가?
- 검색된 문서가 실제 답변 생성에 얼마나 활용되는가?

### 2. 검색 방식 중요도 판단  
- BM25 (키워드 매칭) vs Vector (의미 유사도) 중 어느 것이 더 중요한가?
- 현재 설정 (BM25=70%, Vector=30%)이 적절한가?

## 📊 현재 RAG 시스템 구성

```yaml
version: "2300"
chunking:
  method: "hybrid"
  chunk_size: 2300
  chunk_overlap: 200
  
search:
  bm25_weight: 0.7  # 70% 키워드 기반
  vector_weight: 0.3  # 30% 의미 기반
  normalization_method: "min_max"
  
retrieval:
  initial_k: 30  
  final_k: 5     # 최종 5개 문서 사용
```

## 🔬 실험 방법

### 테스트 데이터
- **객관식**: 10문제 (TEST_000 ~ TEST_009)
- **주관식**: 10문제 (TEST_004, TEST_007, TEST_082 등)
- **총 20문제**로 진단 수행

### 분석 관점

#### 1. 점수 분포 분석
```python
# 각 문서별로 수집되는 정보
{
  "bm25_score": 0.XX,     # 키워드 매칭 점수
  "vector_score": 0.XX,    # 의미 유사도 점수  
  "hybrid_score": 0.XX,    # 가중 평균 점수
  "retrieval_methods": ["bm25", "vector"]
}
```

#### 2. Dominance 판단 기준
- **BM25 Dominant**: BM25/Vector 비율 > 1.5
- **Vector Dominant**: BM25/Vector 비율 < 0.67
- **Balanced**: 0.67 ≤ 비율 ≤ 1.5

#### 3. 문서 활용도 측정
- 생성된 답변에서 검색 문서 인용 여부
- 사고 과정(Chain of Thought)에서 문서 참조 패턴

## 💡 예상 시나리오 및 대응

### 시나리오 1: BM25 Dominant (>60%)
- **의미**: 키워드 매칭이 더 효과적
- **현재 설정**: ✅ 적절함 (BM25=70%)
- **권장사항**: 현재 가중치 유지

### 시나리오 2: Vector Dominant (>40%)
- **의미**: 의미적 유사성이 더 중요
- **현재 설정**: ⚠️ 조정 필요
- **권장사항**: Vector 가중치 증가 (30% → 40-50%)

### 시나리오 3: 문서 활용률 낮음 (<50%)
- **의미**: 지식 베이스 부족
- **권장사항**:
  1. 더 많은 PDF 문서 추가
  2. 청킹 크기 조정 (2300 → 1500-2000)
  3. 청킹 overlap 증가 (200 → 300-400)

## 🚀 실행 명령

### 로컬 테스트
```bash
# Quick test (2문제)
python tests/experiments/rag_diagnostic/quick_test.py

# Full test (20문제)  
python tests/experiments/rag_diagnostic/run_diagnostic_test.py

# 결과 분석
python tests/experiments/rag_diagnostic/analyze_results.py
```

### 서버 실행
```bash
# 서버에 파일 전송
scp -r tests/experiments/rag_diagnostic/ user@server:/path/

# 서버에서 실행
ssh user@server
cd /path/to/project
bash tests/experiments/rag_diagnostic/run_on_server.sh
```

## 📈 성능 지표

### 핵심 메트릭
1. **Top-1 평균 점수**: 가장 관련성 높은 문서의 평균 점수
2. **Top-5 평균 점수**: 상위 5개 문서의 평균 점수
3. **문서 활용률**: 답변 생성시 실제 사용된 문서 비율
4. **BM25/Vector 비율**: 점수 우세 패턴

### 목표 수치
- Top-1 점수: > 0.6
- 문서 활용률: > 70%
- 답변 정확도: > 80%

## 🔄 개선 방향

### 단기 개선 (1주)
1. 가중치 조정 실험 (BM25:Vector 비율 변경)
2. 청킹 크기 최적화
3. 리랭킹 모델 도입 검토

### 중기 개선 (2-3주)
1. 지식 베이스 확장 (100+ PDF)
2. 도메인 특화 임베딩 모델 파인튜닝
3. 하이브리드 청킹 전략 고도화

### 장기 개선 (1개월+)
1. RAG 파이프라인 전면 개편
2. 멀티스테이지 검색 시스템
3. 질문 유형별 맞춤 전략

## 📝 참고사항

- **GPU 메모리**: 16GB+ 필요 (Qwen2.5-7B)
- **실행 시간**: 20문제 기준 10-15분
- **디스크 공간**: 결과 파일 ~100MB

## ✅ 체크리스트

- [x] 테스트 문제 20개 준비
- [x] 진단 프롬프트 템플릿 작성
- [x] 메인 진단 스크립트 구현
- [x] 결과 분석 도구 개발
- [x] 서버 실행 스크립트 작성
- [x] 문서화 완료
- [ ] 서버 테스트 대기

---

**작성일**: 2025-08-20
**버전**: 1.0.0