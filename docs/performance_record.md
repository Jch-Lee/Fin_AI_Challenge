# 성과 기록 (Performance Record)

*Last Updated: 2025-08-23*

## 현재 구현 성능 (8-bit 양자화)

### 기본 정보
- **최종 검증 일자**: 2025년 8월 23일
- **테스트 환경**: RTX 4090 24GB (대회 사양)
- **메인 스크립트**: `generate_submission_standalone.py`

### 핵심 설정

#### 모델 구성
- **LLM**: Qwen2.5-7B-Instruct
- **양자화**: 8-bit (BitsAndBytesConfig)
- **임베딩**: nlpai-lab/KURE-v1 (1024차원)

#### RAG 시스템
- **문서**: 73개 PDF
- **청크**: 8,756개 (2,300자 단위)
- **검색 전략**: BM25+Vector Combined Top-3
  - BM25 상위 3개 (get_top_n 메서드)
  - Vector 상위 3개 (FAISS 코사인 유사도)
  - 중복 허용, 총 6개 컨텍스트

#### 생성 파라미터 (현재 설정)
```python
temperature=0.05     # 매우 보수적 (기존 0.1에서 하향)
top_p=0.9           # 제한적 샘플링
top_k=5             # 상위 5개만
do_sample=False     # 결정론적 생성
max_new_tokens=32   # 객관식
max_new_tokens=256  # 주관식
repetition_penalty=1.1
```

### 성능 지표 (실측값)
- **평균 처리 속도**: 5.75초/질문
- **전체 처리 시간**: 약 49분 (515문제)
- **GPU 메모리 사용**: 10.7GB (8-bit 양자화)
- **대회 제한 충족**: 4.5시간 내 (5.5배 여유)

### 양자화 비교

| 설정 | 양자화 | 메모리 사용 | 처리 속도 | RTX 4090 호환 |
|------|--------|------------|-----------|---------------|
| **8-bit (현재)** | BitsAndBytes 8-bit | 10.7GB | 5.75초/질문 | ✅ 충족 |
| 4-bit | BitsAndBytes 4-bit | ~6GB | 약 4초/질문 | ✅ 충족 |
| 16-bit | float16 | ~14GB | 약 3초/질문 | ✅ 충족 |
| 32-bit | float32 | ~28GB | 약 7초/질문 | ❌ 초과 |

### 핵심 구현 특징

#### 1. 하이브리드 검색
- **BM25**: Kiwi 형태소 분석 기반 키워드 매칭
- **FAISS**: KURE-v1 임베딩 기반 의미 검색
- **독립 선택**: 각 방법에서 3개씩 독립적으로 선택

#### 2. 프롬프트 엔지니어링
- **Chat Template**: System + User Role 구조
- **객관식**: 숫자만 출력 유도
- **주관식**: 5가지 상세 지침
- **문서 전체 사용**: 길이 제한 제거

#### 3. 후처리 최적화
- **이미지/URL 제거**: 정규식 필터링
- **bad_words_ids**: 이미지 토큰 차단
- **길이 제한**: 주관식 최대 500자

### 파일 구조
```
Fin_AI_Challenge/
├── scripts/
│   ├── generate_submission_standalone.py     # 독립 실행 버전
│   └── generate_submission_remote_8bit_fixed.py # 원격 서버용
├── data/
│   ├── rag/
│   │   ├── chunks_2300.json         # 8,756 청크
│   │   ├── bm25_index_2300.pkl     # BM25 인덱스  
│   │   ├── faiss_index_2300.index  # FAISS 인덱스
│   │   └── metadata_2300.json      # 메타데이터
│   └── synthetic_questions/
│       └── combined_3000_questions.csv # 3,000개 합성 Q&A
└── test.csv                          # 515개 테스트 질문
```

### 실행 방법
```bash
# 테스트 모드 (10개 샘플)
python scripts/generate_submission_standalone.py \
    --test_mode \
    --num_samples 10 \
    --output_file test_result.csv

# 전체 추론 (515개)
python scripts/generate_submission_standalone.py \
    --input_file test.csv \
    --output_file submission.csv
```

## 이전 성과 기록

### 최고 점수: 0.579 (2025-08-21)
- **환경**: RTX 5090 32GB
- **양자화**: 16-bit
- **검색**: Combined Top-3
- **Temperature**: 0.1

### 다음 단계 계획
- [ ] Teacher-Student Distillation 구현
- [ ] Qwen2.5-14B Teacher 모델 학습
- [ ] 3,000개 합성 데이터 활용
- [ ] 4-bit 양자화 최적화