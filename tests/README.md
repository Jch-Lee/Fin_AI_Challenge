# 테스트 구조 및 실행 가이드

*Last Updated: 2025-08-23*

## 📌 현재 테스트 현황

### 구현된 메인 파이프라인 테스트
- **8-bit 추론 파이프라인**: `generate_submission_standalone.py`에서 자체 테스트 모드 제공
  ```bash
  # 10개 샘플 테스트
  python scripts/generate_submission_standalone.py --test_mode --num_samples 10
  ```
- **테스트 커버리지**: 주요 추론 경로 커버, 단위 테스트는 부분적 구현
- **통합 테스트**: RAG 시스템 통합 테스트 부분 구현

### 테스트 전략
- 메인 추론 파이프라인은 자체 테스트 모드로 검증
- 개별 컴포넌트는 단위 테스트로 검증 예정
- Vision 모듈은 현재 메인 파이프라인에서 미사용

## 📂 디렉토리 구조

```
tests/
├── unit/                      # 단위 테스트
│   ├── preprocessing/        # 전처리 관련 테스트
│   │   ├── test_kiwi_tokenizer.py      # Kiwi 토크나이저
│   │   ├── test_hierarchical_chunker.py # 계층적 청킹
│   │   └── test_semantic_enhancer.py    # 시맨틱 강화
│   │
│   ├── embeddings/           # 임베딩 관련 테스트
│   │   └── test_kure_v1.py              # KURE-v1 임베딩
│   │
│   ├── retrieval/            # 검색 관련 테스트
│   │   └── test_reranker.py             # Reranker
│   │
│   └── vision/               # Vision 관련 테스트
│       └── test_vision_processor.py     # Vision 프로세서
│
├── integration/              # 통합 테스트
│   ├── test_rag_complete_system.py     # RAG 전체 시스템
│   ├── test_rag_full_pipeline.py       # RAG 파이프라인
│   ├── test_rag_simple_pipeline.py     # 간단한 RAG 파이프라인
│   ├── test_rag_full_system.py         # 전체 RAG 시스템
│   └── test_rag_with_pdf.py            # PDF 처리 포함 RAG
│
├── experiments/              # 실험 코드
│   ├── rag_pipeline_experiment.py      # RAG 파이프라인 실험
│   ├── tokenizer_comparison.py         # 토크나이저 성능 비교
│   └── konlpy_comparison.py            # KoNLPy 비교 실험
│
├── benchmarks/               # 성능 벤치마크
│   ├── kiwi_performance.py             # Kiwi 성능 측정
│   ├── embedding_benchmark.py          # 임베딩 벤치마크
│   └── vision_benchmark.py             # Vision 벤치마크
│
└── fixtures/                # 테스트 데이터
    ├── sample_pdfs/         # 샘플 PDF 파일
    ├── sample_texts/        # 샘플 텍스트 파일
    └── expected_outputs/    # 예상 출력 결과
```

## 🚀 테스트 실행 방법

### 단위 테스트 실행
```bash
# 전체 단위 테스트
python -m pytest tests/unit/

# 특정 모듈 테스트
python -m pytest tests/unit/preprocessing/
python -m pytest tests/unit/embeddings/
python -m pytest tests/unit/retrieval/
python -m pytest tests/unit/vision/

# 개별 테스트 파일
python tests/unit/preprocessing/test_kiwi_tokenizer.py
```

### 통합 테스트 실행
```bash
# 전체 통합 테스트
python -m pytest tests/integration/

# 주요 통합 테스트
python tests/integration/test_rag_complete_system.py
python tests/integration/test_rag_with_pdf.py
```

### 실험 코드 실행
```bash
# 토크나이저 비교 실험
python tests/experiments/tokenizer_comparison.py

# RAG 파이프라인 실험
python tests/experiments/rag_pipeline_experiment.py
```

### 벤치마크 실행
```bash
# Kiwi 성능 벤치마크
python tests/benchmarks/kiwi_performance.py

# 임베딩 벤치마크
python tests/benchmarks/embedding_benchmark.py

# Vision 벤치마크
python tests/benchmarks/vision_benchmark.py
```

## 📝 테스트 작성 가이드

### 단위 테스트
- 단일 기능/메서드에 집중
- 외부 의존성 최소화 (Mock 사용)
- 빠른 실행 시간 유지 (<1초)

### 통합 테스트
- 여러 컴포넌트 간 상호작용 검증
- 실제 데이터와 유사한 테스트 데이터 사용
- End-to-end 시나리오 테스트

### 실험 코드
- 새로운 아이디어나 접근법 테스트
- 성능/품질 비교 분석
- 결과 시각화 및 리포트 생성

### 벤치마크
- 성능 측정 (속도, 메모리 사용량)
- 품질 평가 (정확도, F1-score 등)
- 회귀 테스트 (성능 저하 감지)

## 🔧 CI/CD 통합

GitHub Actions에서 자동 실행:
```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run unit tests
        run: pytest tests/unit/ --cov
      - name: Run integration tests
        run: pytest tests/integration/
```

## 📊 커버리지 목표

- 단위 테스트: 80% 이상
- 통합 테스트: 주요 워크플로우 100% 커버
- 전체 커버리지: 70% 이상

## 🐛 디버깅 팁

1. **개별 테스트 실행**
   ```bash
   python -m pytest tests/unit/preprocessing/test_kiwi_tokenizer.py::TestKiwiTokenizer::test_basic_tokenization -v
   ```

2. **로그 레벨 조정**
   ```bash
   python tests/integration/test_rag_complete_system.py --log-level=DEBUG
   ```

3. **실패 시 중단**
   ```bash
   python -m pytest tests/ -x
   ```

4. **마지막 실패만 재실행**
   ```bash
   python -m pytest tests/ --lf
   ```

## 📚 관련 문서

- [프로젝트 구조](../PROJECT_STRUCTURE.md)
- [개발 가이드](../docs/README.md)
- [RAG 시스템 아키텍처](../docs/architecture/1-high-level-architecture.md)