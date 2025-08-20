# 프로젝트 구조

## 📁 디렉토리 구조

```
Fin_AI_Challenge/
│
├── packages/               # 핵심 구현 모듈
│   ├── preprocessing/     # 전처리 (PDF, 청킹, 임베딩)
│   ├── rag/              # RAG 시스템
│   │   ├── embeddings/   # 임베딩 (KURE)
│   │   ├── retrieval/    # 검색 (BM25, 하이브리드)
│   │   └── reranking/    # 리랭킹 (Qwen3)
│   ├── llm/              # LLM 관련 (프롬프트, Qwen2.5)
│   ├── vision/           # Vision 처리 (Qwen2.5-VL)
│   ├── training/         # 학습 관련
│   └── evaluation/       # 평가 도구
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
│   ├── build_hybrid_rag_2300.py      # RAG 시스템 빌드 (2300자)
│   ├── generate_final_submission_bm25_070.py  # BM25 0.7 최종 제출
│   ├── load_rag_v2.py                # RAG v2.0 로더
│   ├── process_all_pdfs.py           # PDF 일괄 처리
│   ├── add_new_pdfs.py               # PDF 추가
│   ├── build_vision_knowledge_base.py # Vision KB 구축
│   └── check_kb.py                   # 지식베이스 확인
│
├── tests/                 # 테스트 코드
│   ├── unit/             # 단위 테스트
│   │   ├── preprocessing/
│   │   ├── embeddings/
│   │   ├── retrieval/
│   │   └── vision/
│   │
│   ├── integration/      # 통합 테스트
│   ├── experiments/      # 실험 코드
│   │   ├── chunking/     # 청킹 실험 (2300자 개발)
│   │   └── (기타 실험)
│   ├── benchmarks/       # 성능 벤치마크
│   ├── fixtures/         # 테스트 데이터
│   │   └── sample_data/  # 샘플 데이터
│   ├── results/          # 테스트 결과
│   │   └── chunking/     # 청킹 결과
│   └── README.md
│
├── configs/               # 설정 파일
│   ├── rag_config.yaml  # RAG 설정 (BM25 0.7)
│   ├── inference_config.yaml
│   ├── model_config.py
│   └── vision/           # Vision 설정
│
├── data/                  # 데이터 및 인덱스
│   ├── competition/      # 경진대회 데이터
│   │   ├── test.csv      # 평가 질문 (515개)
│   │   └── sample_submission.csv
│   ├── raw/              # 원본 PDF (60개)
│   ├── processed/        # 처리된 텍스트
│   ├── rag/              # RAG 인덱스
│   │   ├── chunks_2300.json
│   │   ├── embeddings_2300.npy
│   │   ├── faiss_index_2300.index
│   │   └── bm25_index_2300.pkl
│   └── knowledge_base/   # 지식베이스
│
├── docs/                  # 프로젝트 문서
│   ├── architecture/     # 시스템 아키텍처
│   ├── pipeline/         # 개발 파이프라인
│   ├── project-plan/     # 프로젝트 계획
│   ├── git-workflow/     # Git 워크플로우
│   ├── reports/          # 분석 보고서
│   │   ├── kiwi_analysis_final_report.md
│   │   ├── model_comparison_report.md
│   │   └── vision/       # Vision 보고서
│   └── 요구사항정의서/
│
├── models/                # 모델 파일
│   └── models--Qwen--Qwen2.5-7B-Instruct/
│
├── evaluation_results/    # 모델 평가 결과
├── remote_results/        # 원격 제출 결과
├── test_results/          # 테스트 실행 결과
│   ├── 2025-08-17/      # 날짜별 결과
│   ├── pipeline/         # 파이프라인 결과
│   │   └── 2025-08-12/
│   └── rag_validation/   # RAG 검증
│
├── logs/                  # 로그 파일
├── venv/                  # 가상환경
│
├── requirements.txt       # 의존성
├── pyproject.toml        # 프로젝트 설정
├── CLAUDE.md             # Claude Code 가이드
├── README.md             # 프로젝트 소개
├── PROJECT_STRUCTURE.md  # 이 문서
├── activate_env.bat      # 환경 활성화
└── setup_utf8.bat        # UTF-8 설정
```

## 🚀 실행 방법

### 환경 설정
```bash
# 가상환경 활성화
activate_env.bat

# UTF-8 인코딩 설정
setup_utf8.bat
```

### RAG 시스템 구축 및 실행
```bash
# RAG 시스템 빌드 (2300자 청킹)
python scripts/build_hybrid_rag_2300.py

# 최종 제출 생성 (BM25 0.7 가중치)
python scripts/generate_final_submission_bm25_070.py

# RAG 시스템 로드 테스트
python scripts/load_rag_v2.py
```

### 테스트 실행
```bash
# 전체 테스트 실행
python -m pytest tests/

# 단위 테스트만 실행
python -m pytest tests/unit/

# 통합 테스트만 실행
python -m pytest tests/integration/

# 청킹 실험 실행
python tests/experiments/chunking/test_chunking_realistic.py

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

### 현재 시스템 구성
- **RAG 버전**: v2.0 (2300자 청킹)
- **검색 가중치**: BM25 0.7, Vector 0.3
- **임베딩 모델**: KURE-v1 (1024차원)
- **LLM**: Qwen2.5-7B-Instruct (16-bit)
- **리랭킹**: 비활성화 (성능 최적화)

### 개발 가이드라인
- 모든 테스트 파일은 **상대경로**를 사용하여 프로젝트 루트를 참조합니다
- `Path(__file__).parent.parent` 패턴으로 루트 경로 설정
- 실행은 어느 위치에서든 가능하도록 설계됨
- Git 워크플로우는 `docs/git-workflow/` 참조

### 디렉토리 분류
- **packages/**: 핵심 비즈니스 로직
- **scripts/**: 실행 가능한 스크립트
- **tests/**: 테스트 및 실험 코드
  - **unit/**: 개별 컴포넌트 테스트
  - **integration/**: 시스템 통합 테스트
  - **experiments/**: 실험적 코드 및 비교 분석
  - **benchmarks/**: 성능 측정 및 평가
- **configs/**: 설정 파일
- **data/**: 데이터 및 인덱스
- **docs/**: 프로젝트 문서
- **results 디렉토리**:
  - **evaluation_results/**: 모델 평가 메트릭
  - **remote_results/**: 실제 제출 파일
  - **test_results/**: 개발 중 테스트 결과