# 4. Components - Current Implementation Status

## 현재 구현된 컴포넌트 구조

### 실제 구현 아키텍처
현재 프로젝트는 문서에 기술된 10개 독립 컴포넌트 대신, `packages/rag/` 중심의 통합 구조로 구현되어 있습니다.

## 구현된 핵심 컴포넌트

### 1. RAG Pipeline (`packages/rag/rag_pipeline.py`)
**역할**: 전체 RAG 시스템 오케스트레이션
- Document embedding 관리
- Knowledge base 연동
- Query processing
- Document retrieval (리랭킹 옵션 제거)
- Context preparation for generation

**주요 메서드**:
```python
- add_documents(): 문서 추가 및 임베딩
- retrieve(): 관련 문서 검색 (직접 Top-5 반환)
- generate_context(): LLM용 컨텍스트 생성
- save/load_knowledge_base(): 지식 베이스 저장/로드
- load_from_v2_format(): RAGSystemV2 데이터 호환
- create_simple_hybrid_retriever(): 간소화된 하이브리드 검색기
```

**최신 설정** (2025-08-20 업데이트 - 🏆 0.55 점수):
- **검색 방식**: `retriever_type="combined_top"` (기본값)
- **BM25+Vector Top3**: 각 방법에서 독립적으로 상위 3개씩 선택
- **총 6개 문서**: 중복 허용으로 풍부한 컨텍스트
- **리랭킹 비활성화**: 독립 선택 방식이 더 효과적
- **설정 자동 로드**: `configs/rag_config.yaml v3.0`

### 2. Knowledge Base (`packages/rag/knowledge_base.py`)
**역할**: FAISS 기반 벡터 저장 및 검색
- FAISS 인덱스 관리 (Flat, IVF, HNSW)
- 문서 및 메타데이터 저장
- 벡터 검색 및 배치 검색
- Legacy 데이터 호환성 지원

**인덱스 타입**:
- Flat: 소규모 데이터셋 (기본값)
- IVF: 중규모 데이터셋
- HNSW: 대규모 데이터셋

### 3. Embedders (`packages/rag/embeddings/`)
**구현된 임베딩 모델**:
- **KUREEmbedder**: nlpai-lab/KURE-v1 (한국어 특화 임베딩, SentenceTransformer 기반)
- **E5Embedder**: intfloat/multilingual-e5-large (768차원)
- **BaseEmbedder**: 추상 기본 클래스

**토크나이저 전략**:
- **벡터 임베딩**: KURE 내장 토크나이저 (SentencePiece 기반, 476배 빠른 속도)
- **BM25 검색**: Kiwipiepy 형태소 분석기 (의미 단위 정확도 우선)

### 4. Retrievers (`packages/rag/retrieval/`)
**검색 전략**:
- **VectorRetriever**: FAISS 기반 밀집 벡터 검색 (KURE 임베딩 사용)
- **BM25Retriever**: BM25S 라이브러리 기반 희소 검색
  - Kiwipiepy 형태소 분석기 통합
  - 품사 필터링: 명사(N), 동사(V), 형용사(VA), 외국어(SL)
  - 10-30배 빠른 토크나이징 속도
- **HybridRetriever**: Vector(KURE) + BM25 앙상블 (Legacy)
  - 가중치 통합: BM25 70%, Vector 30%
  - Min-Max 정규화 적용
- **🏆 CombinedTopRetriever**: BM25+Vector 독립 선택 (현재 기본값)
  - **리더보드 0.55 달성**: 기존 0.46 대비 +19.6% 향상
  - BM25에서 상위 3개, Vector에서 상위 3개 독립 선택
  - 중복 허용으로 정보 다양성 극대화
  - 각 검색 방법의 강점을 희석시키지 않음
- **RerankingRetriever**: ~~Qwen3 기반 재순위화~~ (비활성화됨)

### 5. Model Loaders (`models/model_loader.py`)
**모델 관리 시스템**:
- **TeacherModelLoader**: Qwen2.5-7B-Instruct
- **StudentModelLoader**: Qwen2.5-1.5B-Instruct  
- **SyntheticDataModelLoader**: Qwen2.5-14B-Instruct (Fallback)

**주요 기능**:
- 양자화 지원 (4-bit, 8-bit)
- 메모리 사용량 추정
- 대회 규정 준수 검증
- 배치 생성 지원

### 6. Vision Processor (`packages/vision/` & `packages/preprocessing/`)
**역할**: Vision-Language 모델 기반 고품질 PDF 텍스트 추출
- **VisionTextExtractor**: Qwen2.5-VL-7B-Instruct 기반 텍스트 추출
- **VisionPDFProcessor**: PDF 페이지를 이미지로 변환 후 VL 모델 처리
- **41.2% 텍스트 추출 품질 향상** (PyMuPDF 대비, 56페이지 검증 완료)

**주요 기능**:
- 표/차트/그래프 의미론적 해석
- Version 2 최적화 프롬프트 적용
- GPU 환경 자동 감지 및 최적화
- 실시간 메모리 관리

### 7. PDF Processing Pipeline (`packages/preprocessing/data_preprocessor.py`)
**3-Tier Fallback 구조**: 안정성과 품질을 동시 확보
1. **Vision V2** (Primary): GPU 환경, 최고 품질 (41.2% 개선)
2. **Traditional PyMuPDF** (Fallback): PyMuPDF4LLM 향상된 추출
3. **Basic PyMuPDF** (Final): 원시 텍스트 추출, 최종 안전망

**처리 흐름**:
```python
GPU 가용성 확인 → Vision V2 시도 → 실패 시 Traditional → 최종 Basic
```

### 8. Question Classifier (`packages/preprocessing/question_classifier.py`)
**질문 분류 기능**:
- 객관식/주관식 구분
- 질문과 선택지 분리
- 프롬프트 템플릿 생성

## 계획된 컴포넌트 (미구현)

### 아직 구현되지 않은 컴포넌트들:
1. **ModelTrainer**: Distill-M 2 학습 컴포넌트
2. **InferenceOrchestrator**: 통합 추론 오케스트레이터
3. **ModelOptimizer**: 양자화 및 최적화
4. **EvaluationMonitor**: 평가 및 모니터링
5. **CacheLayer**: 추론 캐싱 시스템
6. **MultiStageRetriever**: 다단계 검색 (현재는 단일 단계)

## 디렉토리 구조

```
packages/
├── rag/                     # RAG 시스템 (핵심)
│   ├── __init__.py
│   ├── rag_pipeline.py      # 메인 파이프라인 (v2 호환성 추가)
│   ├── knowledge_base.py    # FAISS 지식 베이스
│   ├── embeddings/          # 임베딩 모델들
│   │   ├── base_embedder.py
│   │   ├── kure_embedder.py
│   │   └── e5_embedder.py (deprecated)
│   ├── retrieval/           # 검색 전략들
│   │   ├── base_retriever.py
│   │   ├── vector_retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── combined_top_retriever.py # 🏆 0.55 점수 달성
│   │   └── reranking_retriever.py (비활성화)
│   └── reranking/           # 리랭킹 시스템 (비활성화)
│       ├── base_reranker.py
│       ├── qwen3_reranker.py
│       └── reranker_config.py
├── vision/                  # Vision-Language 모듈 (신규)
│   ├── __init__.py
│   └── vision_extraction.py # VL 모델 기반 텍스트 추출
├── preprocessing/           # 전처리 시스템
│   ├── data_preprocessor.py # 통합 전처리 (Vision V2 메인)
│   ├── pdf_processor_vision.py      # Vision 기반 PDF 프로세서
│   ├── pdf_processor_traditional.py # PyMuPDF 기반 (Fallback)
│   ├── embedder.py         # 하위 호환성 래퍼
│   └── question_classifier.py
└── data_processing/        # 데이터 처리
    └── korean_english_processor.py

models/
├── model_loader.py         # 모델 로딩 시스템
└── (모델 가중치 파일들)

configs/
└── model_config.py         # 모델 설정
```

## 통합 진행 상황

### 완료된 작업 ✅
- [x] RAG 시스템 구조 재편성
- [x] 임베딩 모델 통합 (KURE-v1)
- [x] 지식 베이스 구축 (FAISS)
- [x] 검색 시스템 구현
- [x] 모델 설정 시스템
- [x] 하위 호환성 유지
- [x] **Vision V2 통합 완료** (2025-08-14)
- [x] **PDF 처리 파이프라인 3-Tier 구조 구축**
- [x] **텍스트 추출 품질 41.2% 개선** (56페이지 벤치마크 검증)
- [x] **BM25 0.7 파이프라인 마이그레이션** (2025-08-20)
- [x] **리랭킹 제거로 40-50% 속도 향상**
- [x] **RAGSystemV2 호환성 메서드 추가**
- [x] **🏆 BM25+Vector Top3 방식 구현** (2025-08-20)
- [x] **리더보드 점수 0.55 달성** (기존 0.46 대비 +19.6%)
- [x] **CombinedTopRetriever 구현 및 통합**

### 진행 중 🔄
- [ ] Teacher-Student 응답 생성
- [ ] Distill-M 2 학습 파이프라인

### 계획됨 📋
- [ ] 추론 오케스트레이터
- [ ] 캐싱 시스템
- [ ] 평가 및 모니터링
- [ ] 최종 최적화

## 아키텍처 차이점 분석

### 원래 설계 vs 현재 구현
| 측면 | 원래 설계 (문서) | 현재 구현 |
|------|-----------------|-----------|
| 구조 | 10개 독립 컴포넌트 | RAG 중심 통합 구조 |
| 인터페이스 | ABC 추상 클래스 | 일부만 ABC 사용 |
| 모듈화 | 완전 분리 | 부분적 통합 |
| 확장성 | 높음 (설계상) | 중간 (실용적) |

### 현재 접근 방식의 장점
1. **빠른 개발**: 통합 구조로 신속한 프로토타이핑
2. **실용성**: 실제 필요한 기능 중심 구현
3. **유지보수**: 단순한 구조로 관리 용이
4. **호환성**: 레거시 코드와의 원활한 통합

### 향후 리팩토링 제안
1. 필요 시 컴포넌트 분리 강화
2. 인터페이스 정의 추가
3. 의존성 주입 패턴 도입
4. 테스트 커버리지 확대

---
*Last Updated: 2025-08-20 - BM25 0.7 파이프라인 마이그레이션 완료*