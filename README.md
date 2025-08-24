# Financial Security Knowledge Understanding (FSKU) AI Challenge

*Last Updated: 2025-08-23*

## 🎯 프로젝트 개요
금융 보안 관련 질문에 대한 한국어 답변을 생성하는 AI 시스템 개발

### 핵심 목표
- **객관식 문제**: 정확한 선택지 번호 예측
- **주관식 문제**: 설명적이고 정확한 한국어 답변 생성
- **성능 요구사항**: RTX 4090 24GB에서 4.5시간 내 전체 데이터셋 추론
- **메모리 제약**: 24GB VRAM 이내 동작

## 🏗️ 현재 구현 시스템

### 8-bit 양자화 추론 파이프라인
```
질문 입력 (515개)
    ↓
질문 분류 (객관식/주관식)
    ↓
하이브리드 RAG 검색
├── BM25: Kiwi 토크나이저 → Top-3
└── FAISS: KURE-v1 임베딩 → Top-3
    ↓
6개 컨텍스트 (중복 허용)
    ↓
Qwen2.5-7B-Instruct (8-bit)
    ↓
답변 생성 및 후처리
```

### 주요 구성요소
1. **추론 모델**
   - Qwen2.5-7B-Instruct (8-bit 양자화)
   - BitsAndBytesConfig 최적화
   - 10.7GB VRAM 사용

2. **RAG System**
   - **문서**: 73개 PDF → 8,756개 청크 (2,300자 단위)
     - Vision 모델(Qwen2.5-VL)로 이미지/차트 텍스트화 처리
   - **BM25 검색**: Kiwipiepy 형태소 분석 기반
   - **Vector 검색**: FAISS + KURE-v1 (1024차원)
   - **하이브리드**: 각 방법에서 독립적으로 Top-3 선택

3. **합성 데이터** (Teacher-Student 학습용)
   - 3,000개 고품질 Q&A 쌍 생성 완료
   - 6가지 질문 유형 균등 분배
   - Qwen2.5-14B Teacher 모델 사용

## 📊 성능 지표

### 현재 달성 성능 (8-bit 양자화)
| 지표 | 측정값 | 대회 제한 | 상태 |
|------|--------|----------|------|
| **처리 속도** | 5.75초/질문 | <31.5초 | ✅ 충족 |
| **전체 시간** | ~49분 (515문제) | 4.5시간 | ✅ 충족 (5.5배 여유) |
| **메모리 사용** | 10.7GB | 24GB | ✅ 충족 |
| **추론 안정성** | 100% | - | ✅ 안정 |

### 이전 최고 성과
- **점수**: 0.579 (2025.08.21)
- **환경**: RTX 5090 32GB, 16-bit 양자화
- **속도**: 0.2초/문제

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# Python 3.10 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 추론 실행
```bash
# 테스트 모드 (10개 샘플)
python scripts/generate_submission_standalone.py \
    --test_mode \
    --num_samples 10 \
    --output_file test_result.csv

# 전체 추론 (515문제)
python scripts/generate_submission_standalone.py \
    --input_file test.csv \
    --output_file submission.csv \
    --data_dir ./data/rag
```

### 3. 주요 파라미터
```python
# 생성 설정
temperature = 0.05      # 매우 보수적
top_p = 0.9            # 제한적 샘플링
top_k = 5              # 상위 5개만
do_sample = False      # 결정론적 생성

# 토큰 제한
max_new_tokens = 32    # 객관식
max_new_tokens = 256   # 주관식
```

## 📁 프로젝트 구조
```
Fin_AI_Challenge/
├── scripts/                          # 실행 스크립트
│   ├── generate_submission_standalone.py     # 메인 추론 (독립 실행)
│   ├── generate_submission_remote_8bit_fixed.py  # 원격 서버용
│   ├── generate_bulk_3000.py               # 합성 데이터 생성
│   └── build_hybrid_rag_2300.py           # RAG 구축
├── data/
│   ├── rag/                         # RAG 인덱스
│   │   ├── chunks_2300.json        # 8,756개 청크
│   │   ├── bm25_index_2300.pkl     # BM25 인덱스
│   │   ├── faiss_index_2300.index  # FAISS 벡터 인덱스
│   │   └── metadata_2300.json      # 메타데이터
│   └── synthetic_questions/         # 합성 데이터
│       └── combined_3000_questions.csv
├── docs/                            # 프로젝트 문서
│   ├── CURRENT_STATUS_2025_08_23.md
│   ├── architecture/
│   ├── pipeline-auto/
│   └── competition_environment_setup.md
├── test.csv                         # 515개 테스트 질문
└── requirements.txt                 # 의존성 목록
```

## 🔧 핵심 기술 스택

### 필수 라이브러리
```python
torch==2.1.0                    # PyTorch (대회 요구사항)
transformers==4.41.2            # Hugging Face Transformers
bitsandbytes==0.43.1           # 8-bit 양자화
accelerate==0.30.1             # GPU 가속
faiss-cpu==1.8.0               # 벡터 검색
rank-bm25==0.2.2               # BM25 검색
sentence-transformers==2.7.0   # KURE-v1 임베더
kiwipiepy==0.18.0              # 한국어 형태소 분석
```

## 🎯 향후 개발 계획

### Teacher-Student Distillation (계획 중)
```
Qwen2.5-14B-Instruct Teacher
    ↓ 3,000개 합성 데이터
Distill-M 2 학습
    ↓
Qwen2.5-7B-Instruct Student (최적화)
```

### 예상 개선사항
- Student 모델 (Qwen2.5-7B-Instruct) 최적화로 추론 속도 향상
- 답변 품질 개선 (Teacher 모델 지식 전수)
- 더 안정적인 답변 생성

## 🏆 대회 제약사항 충족
- ✅ 단일 모델만 사용 (앙상블 불가)
- ✅ 오픈소스 모델 (Apache 2.0 라이선스)
- ✅ 오프라인 환경 추론 가능
- ✅ RTX 4090 24GB VRAM 제한 충족
- ✅ 4.5시간 추론 시간 제한 충족

## 📝 관련 문서
- [프로젝트 현황](./docs/CURRENT_STATUS_2025_08_23.md)
- [추론 파이프라인 상세](./docs/pipeline-auto/32-추론-파이프라인-구축.md)
- [아키텍처 구현 현황](./docs/architecture/4-components-current-implementation.md)
- [성능 기록](./docs/performance_record.md)
- [대회 환경 재현 가이드](./docs/competition_environment_setup.md)

## 🤝 기여
이슈와 Pull Request를 통해 참여해주세요.

## 📄 라이센스
Apache 2.0 License (대회 요구사항 준수)