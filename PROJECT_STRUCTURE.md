# 프로젝트 구조

## 📁 디렉토리 구조

```
Fin_AI_Challenge/
│
├── packages/               # 핵심 구현 모듈
│   ├── preprocessing/     # 전처리 (PDF, 청킹, 임베딩)
│   ├── retrieval/        # 검색 (BM25, 하이브리드)
│   ├── llm/              # LLM 관련 (프롬프트)
│   ├── rag/              # RAG 시스템
│   └── vision/           # Vision 처리 (Qwen2.5-VL)
│
├── scripts/               # 실행 스크립트
│   ├── utils/            # 유틸리티 도구
│   │   ├── check_encoding.py         # 인코딩 체크
│   │   ├── cleanup_files.py          # 파일 정리
│   │   ├── set_encoding.py           # 인코딩 설정
│   │   ├── view_pipeline_logs.py     # 로그 뷰어
│   │   ├── save_pipeline_results.py  # 파이프라인 결과 저장
│   │   ├── view_intermediate_results.py # 중간 결과 조회
│   │   └── process_vision_texts.py   # Vision 텍스트 처리
│   │
│   ├── build_rag_system.py           # RAG 시스템 빌드
│   ├── integrate_qwen_llm.py         # Qwen LLM 통합
│   └── check_qwen_requirements.py    # Qwen 요구사항 체크
│
├── tests/                 # 테스트 코드 (체계적으로 재구성됨)
│   ├── unit/             # 단위 테스트
│   │   ├── preprocessing/
│   │   │   ├── test_kiwi_tokenizer.py
│   │   │   ├── test_hierarchical_chunker.py
│   │   │   └── test_semantic_enhancer.py
│   │   ├── embeddings/
│   │   │   └── test_kure_v1.py
│   │   ├── retrieval/
│   │   │   └── test_reranker.py
│   │   └── vision/
│   │       └── test_vision_processor.py
│   │
│   ├── integration/      # 통합 테스트
│   │   ├── test_rag_complete_system.py
│   │   ├── test_rag_full_pipeline.py
│   │   ├── test_rag_simple_pipeline.py
│   │   ├── test_rag_full_system.py
│   │   └── test_rag_with_pdf.py
│   │
│   ├── experiments/      # 실험 코드
│   │   ├── rag_pipeline_experiment.py
│   │   ├── tokenizer_comparison.py
│   │   └── konlpy_comparison.py
│   │
│   ├── benchmarks/       # 성능 벤치마크
│   │   ├── kiwi_performance.py
│   │   ├── embedding_benchmark.py
│   │   └── vision_benchmark.py
│   │
│   ├── fixtures/         # 테스트 데이터
│   │   ├── sample_pdfs/
│   │   ├── sample_texts/
│   │   └── expected_outputs/
│   │
│   └── README.md         # 테스트 실행 가이드
│
├── data/                  # 데이터 및 인덱스
│   ├── competition/      # 경진대회 데이터
│   │   ├── test.csv      # 평가 질문 (515개)
│   │   └── sample_submission.csv  # 제출 형식
│   └── e5_embeddings/    # E5 임베딩 인덱스
│
├── docs/                  # 프로젝트 문서
│   ├── Architecture.md   # 시스템 아키텍처
│   ├── Pipeline.md       # 개발 파이프라인
│   └── PROJECT_PLAN.md   # 프로젝트 계획
│
├── baseline_code/         # 참조 구현
├── rag_results/          # 실행 결과
│
├── setup.py              # 패키지 설정
├── requirements.txt      # 의존성
├── pyproject.toml        # 프로젝트 설정
├── CLAUDE.md             # Claude Code 가이드
└── README.md             # 프로젝트 소개
```

## 🚀 실행 방법

### 테스트 실행
```bash
# 전체 테스트 실행
python -m pytest tests/

# 단위 테스트만 실행
python -m pytest tests/unit/

# 통합 테스트만 실행
python -m pytest tests/integration/

# 특정 테스트 파일 실행
python tests/integration/test_rag_complete_system.py

# 벤치마크 실행
python tests/benchmarks/embedding_benchmark.py
```

### 유틸리티 사용
```bash
# 인코딩 체크
python scripts/utils/check_encoding.py

# 파일 정리
python scripts/utils/cleanup_files.py

# 파이프라인 결과 저장
python scripts/utils/save_pipeline_results.py

# Vision 텍스트 처리
python scripts/utils/process_vision_texts.py
```

## 📌 중요 사항

- 모든 테스트 파일은 **상대경로**를 사용하여 프로젝트 루트를 참조합니다
- `Path(__file__).parent.parent.parent` 패턴으로 루트 경로 설정
- 실행은 어느 위치에서든 가능하도록 설계됨
- 테스트는 목적별로 체계적으로 분류되어 있습니다:
  - **unit/**: 개별 컴포넌트 테스트
  - **integration/**: 시스템 통합 테스트
  - **experiments/**: 실험적 코드 및 비교 분석
  - **benchmarks/**: 성능 측정 및 평가