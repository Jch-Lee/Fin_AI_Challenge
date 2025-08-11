# 프로젝트 구조

## 📁 디렉토리 구조

```
Fin_AI_Challenge/
│
├── packages/               # 핵심 구현 모듈
│   ├── preprocessing/     # 전처리 (PDF, 청킹, 임베딩)
│   ├── retrieval/        # 검색 (BM25, 하이브리드)
│   ├── llm/              # LLM 관련 (프롬프트)
│   └── rag/              # RAG 시스템
│
├── scripts/               # 실행 스크립트
│   ├── utils/            # 유틸리티 도구
│   │   ├── check_encoding.py      # 인코딩 체크
│   │   ├── cleanup_files.py       # 파일 정리
│   │   ├── set_encoding.py        # 인코딩 설정
│   │   └── view_pipeline_logs.py  # 로그 뷰어
│   │
│   ├── integrate_qwen_llm.py      # Qwen LLM 통합
│   └── qwen_quantized.py          # Qwen 양자화
│
├── tests/                 # 테스트 코드
│   ├── integration/      # 통합 테스트
│   │   ├── test_rag_complete_system.py  # 전체 시스템 검증
│   │   └── test_rag_full_pipeline.py    # 파이프라인 테스트
│   │
│   └── experiments/      # 실험 코드
│       └── rag_pipeline_experiment.py   # RAG 실험
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
# 통합 테스트 실행 (프로젝트 루트에서)
python tests/integration/test_rag_complete_system.py

# 또는 테스트 디렉토리에서 실행
cd tests/integration
python test_rag_complete_system.py
```

### 유틸리티 사용
```bash
# 인코딩 체크
python scripts/utils/check_encoding.py

# 파일 정리
python scripts/utils/cleanup_files.py
```

## 📌 중요 사항

- 모든 테스트 파일은 **상대경로**를 사용하여 프로젝트 루트를 참조합니다
- `Path(__file__).parent.parent.parent` 패턴으로 루트 경로 설정
- 실행은 어느 위치에서든 가능하도록 설계됨