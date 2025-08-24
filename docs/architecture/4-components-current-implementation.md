# 4. Components - Current Implementation Status

## 현재 구현된 컴포넌트 구조 (2025-08-23 최종 업데이트)

### 실제 구현 아키텍처
현재 프로젝트는 문서에 기술된 10개 독립 컴포넌트 대신, 통합 추론 파이프라인으로 구현되어 있습니다.

## 구현된 핵심 컴포넌트

### 1. 통합 추론 파이프라인 (`scripts/generate_submission_*.py`)
**메인 스크립트**:
- `generate_submission_remote_8bit_fixed.py`: 원격 서버용 (8-bit 양자화)
- `generate_submission_standalone.py`: 독립 실행 버전 (문서 전체 사용)

**핵심 기능**:
- Question Classification (is_multiple_choice)
- Hybrid Retrieval (BM25 + FAISS)
- Prompt Engineering (Chat Template)
- 8-bit Quantized Inference
- Answer Post-processing

**주요 메서드**:
```python
- add_documents(): 문서 추가 및 임베딩
- retrieve(): 관련 문서 검색 (직접 Top-5 반환)
- generate_context(): LLM용 컨텍스트 생성
- save/load_knowledge_base(): 지식 베이스 저장/로드
- load_from_v2_format(): RAGSystemV2 데이터 호환
- create_simple_hybrid_retriever(): 간소화된 하이브리드 검색기
```

**현재 설정** (2025-08-23 검증 완료):
- **검색 방식**: Combined Top-3 (BM25 + FAISS 독립 선택)
- **BM25**: get_top_n() 메서드로 상위 3개
- **FAISS**: 코사인 유사도 기반 상위 3개
- **총 6개 컨텍스트**: 중복 허용으로 정보 다양성
- **청크 크기**: 2,300자 단위 (8,756개 청크)

### 2. 지식 베이스 (`data/rag/`)
**저장된 인덱스 파일**:
- `chunks_2300.json`: 8,756개 청크 (평균 764.6자)
- `bm25_index_2300.pkl`: BM25 인덱스 (Kiwi 형태소 기반)
- `faiss_index_2300.index`: FAISS 벡터 인덱스 (KURE-v1 1024차원)
- `metadata_2300.json`: 문서 메타데이터

**인덱스 타입**:
- Flat: 소규모 데이터셋 (기본값)
- IVF: 중규모 데이터셋
- HNSW: 대규모 데이터셋

### 3. 임베딩 및 토크나이저
**임베딩 모델**:
- **KURE-v1**: nlpai-lab/KURE-v1 (1024차원 한국어 특화)
- SentenceTransformer 기반
- CUDA GPU 사용

**토크나이저**:
- **BM25**: Kiwipiepy 형태소 분석기
- **Vector**: KURE-v1 내장 토크나이저

### 4. 검색 시스템 (하이브리드 RAG)
**현재 구현** (`retrieve_combined_contexts()`):
- **BM25 검색** (`search_bm25()`):
  - rank-bm25 라이브러리의 get_top_n() 사용
  - Kiwi 형태소 분석기로 질문 토크화
  - 상위 3개 문서 선택
  
- **Vector 검색** (`search_vector()`):
  - FAISS 코사인 유사도 검색
  - KURE-v1 임베딩 (1024차원)
  - 상위 3개 문서 선택
  
- **결합 전략**: 
  - 각 방법에서 독립적으로 3개씩 선택
  - 총 6개 컨텍스트 (중복 허용)

### 5. 추론 모델 (8-bit 양자화)
**현재 모델**: Qwen2.5-7B-Instruct
- **양자화**: BitsAndBytesConfig (8-bit)
- **메모리 사용**: 10.7GB VRAM (24GB 제한 내)
- **추론 파라미터**:
  - temperature: 0.05 (매우 보수적)
  - top_p: 0.9, top_k: 5
  - do_sample: False (결정론적)
  - max_new_tokens: 32(객관식) / 256(주관식)

### 6. 프롬프트 엔지니어링 (`create_prompt()`)
**Chat Template 형식**:
- System Role + User Role 구조
- tokenizer.apply_chat_template() 사용

**객관식 프롬프트**:
- System: "객관식 문제의 정답 번호만 답하세요"
- User: 참고자료(전체) + 질문 + 선택지
- 지침: "1부터 N까지 중 하나의 숫자만"

**주관식 프롬프트 (5가지 지침)**:
1. 참고 문서 기반 정확한 한국어 답변
2. 순수 텍스트만 사용
3. 이미지/URL/링크/마크다운 금지
4. 도표는 텍스트 설명으로 대체
5. 핵심 2-3문장 요약


### 7. 후처리 시스템 (`extract_answer()`)
**주요 기능**:
- 정규식으로 이미지/URL 패턴 제거
- 중국어/일본어 문자 제거
- 객관식: 첫 번째 숫자만 추출
- 주관식: 최대 500자 제한
### 8. 성능 메트릭 (실측값)
**처리 성능**:
- 속도: 5.75초/질문
- 메모리: 10.7GB VRAM
- 전체 시간: 515개 질문 약 49분 예상
- 대회 제한: 4.5시간 내 충족 (5.5배 여유)

## 미구현 컴포넌트

### Teacher-Student Distillation (계획됨)
1. **Teacher 모델**: Qwen2.5-14B-Instruct
2. **Student 모델**: Qwen2.5-1.5B-Instruct
3. **학습 방법**: Distill-M 2 (Contrastive Distillation)
4. **학습 데이터**: 3,000개 합성 질문-답변 쌍

### 캐싱 레이어 (계획됨)
- 자주 나오는 질문 캐싱
- 임베딩 결과 재사용
- 검색 결과 캐싱

## 프로젝트 구조 (현재)

```
Fin_AI_Challenge/
├── scripts/
│   ├── generate_submission_remote_8bit_fixed.py  # 메인 추론
│   ├── generate_submission_standalone.py         # 독립 버전
│   ├── generate_bulk_3000.py                     # 합성 데이터 생성
│   └── build_hybrid_rag_2300.py                  # RAG 구축
├── data/
│   ├── rag/
│   │   ├── chunks_2300.json      # 8,756개 청크
│   │   ├── bm25_index_2300.pkl   # BM25 인덱스
│   │   ├── faiss_index_2300.index # FAISS 인덱스  
│   │   └── metadata_2300.json    # 메타데이터
│   └── synthetic_questions/
│       ├── combined_3000_questions.csv  # 3,000개 합성 Q&A
│       └── generation_report.json       # 생성 보고서
├── test.csv                      # 515개 질문
└── sample_submission.csv         # 제출 형식
```

## 구현 완료 항목 ✅

### Epic 1: 데이터 파이프라인
- [x] 73개 PDF 문서 수집 및 전처리
- [x] 8,756개 청크 (2,300자 단위) 생성
- [x] BM25 인덱스 구축 (Kiwi 형태소 분석)
- [x] FAISS 인덱스 구축 (KURE-v1 1024차원)
- [x] 3,000개 합성 질문-답변 쌍 생성

### Epic 3: 추론 파이프라인
- [x] Question Classifier 구현 (객관식/주관식)
- [x] Multi-Stage Retriever (BM25 + FAISS)
- [x] Qwen2.5-7B-Instruct 8-bit 양자화
- [x] Chat Template 프롬프트 엔지니어링
- [x] 이미지/URL 차단 및 후처리
- [x] 5.75초/질문 성능 달성
- [x] 10.7GB VRAM 사용 (24GB 제한 충족)
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

## 8-bit 양자화 추론 파이프라인 (2025-08-23 신규)

### QwenUpdatedDBPredictor 클래스
**위치**: `scripts/generate_submission_remote_8bit_fixed.py`

**핵심 특징**:
- **8-bit 양자화**: BitsAndBytesConfig로 메모리 75% 절감
- **이미지 생성 차단**: bad_words_ids로 이미지 토큰 필터링
- **한국어 최적화**: Kiwi 형태소 분석 + KURE-v1 임베딩

**주요 메서드**:
```python
class QwenUpdatedDBPredictor:
    def setup(self):
        # 시스템 초기화
        - 8,756개 청크 로드
        - BM25 인덱스 (pickle)
        - FAISS 인덱스 로드
        - KURE-v1 임베더 초기화
        - Qwen2.5-7B-Instruct 8-bit 로드
    
    def retrieve_combined_contexts(self, question):
        # BM25 Top-3 + Vector Top-3 독립 검색
        - BM25: Kiwi 토크나이저 사용
        - Vector: KURE-v1 임베딩 → FAISS 검색
        - 총 6개 컨텍스트 반환 (중복 허용)
    
    def generate_answer(self, prompt, question):
        # 8-bit 모델 추론
        - temperature: 0.05 (결정론적)
        - top_p: 0.9, top_k: 5
        - bad_words_ids: 이미지 토큰 차단
        - max_new_tokens: 32(객관식) / 256(주관식)
```

**메모리 최적화**:
- 8-bit 양자화: ~8GB VRAM 사용
- 100개마다 가비지 컬렉션
- RTX 4090 24GB 제약 충족

### 향후 리팩토링 제안
1. Teacher-Student Distillation 구현
2. 4-bit 양자화 추가 최적화
3. 캐싱 시스템 도입
4. 앙상블 전략 테스트

---
*Last Updated: 2025-08-23 - 8-bit 양자화 추론 파이프라인 완성 및 3,000개 합성 데이터 생성*