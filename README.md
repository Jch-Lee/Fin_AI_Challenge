# Financial Security Knowledge Understanding (FSKU) AI Challenge

*Last Updated: 2025-09-05*

## 🎯 프로젝트 개요
금융 보안 관련 질문에 대한 한국어 답변을 생성하는 AI 시스템 개발 - **프로젝트 완료**

### 핵심 목표 ✅
- **객관식 문제**: 정확한 선택지 번호 예측
- **주관식 문제**: 설명적이고 정확한 한국어 답변 생성
- **성능 요구사항**: RTX 4090 24GB에서 4.5시간 내 전체 데이터셋 추론 (달성: ~49분)
- **메모리 제약**: 24GB VRAM 이내 동작 (달성: 10.7GB)

## 🏆 최종 구현 시스템

### 최종 성과
- **최종 점수**: 0.59 (대회 평가 기준)
- **추론 시간**: 49분 (515문제, 5.75초/문제)
- **메모리 사용**: 10.7GB VRAM (8-bit 양자화)
- **모델**: Qwen2.5-7B-Instruct
- **검색 시스템**: 하이브리드 RAG (BM25 + FAISS)
- **데이터**: 73개 PDF → 8,756개 청크
- **합성 데이터**: 3,000개 Q&A 쌍 생성 완료

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
   - 3,000개 고품질 Q&A 쌍 생성 완료 ✅
   - 6가지 질문 유형 균등 분배 (각 500개)
   - Qwen2.5-14B-Instruct Teacher 모델 사용
   - 100% 생성 성공률 달성

4. **DistiLLM-2 프레임워크 통합**
   - ICML 2025 Oral 논문 기반 구현
   - Contrastive Distillation 방법론 적용 준비
   - Teacher-Student 학습 파이프라인 구축

## 📊 성능 지표

### 최종 달성 성능 (8-bit 양자화)
| 지표 | 측정값 | 대회 제한 | 상태 |
|------|--------|----------|------|
| **처리 속도** | 5.75초/질문 | <31.5초 | ✅ 충족 |
| **전체 시간** | ~49분 (515문제) | 4.5시간 | ✅ 충족 (5.5배 여유) |
| **메모리 사용** | 10.7GB | 24GB | ✅ 충족 |
| **추론 안정성** | 100% | - | ✅ 안정 |
| **제출 파일 생성** | 6개 버전 | - | ✅ 완료 |

### 주요 마일스톤
- **2025.08.21**: 초기 모델 점수 0.579 달성
- **2025.08.23**: 8-bit 양자화 파이프라인 완성
- **2025.08.24**: 3,000개 합성 데이터 생성 완료
- **2025.09.05**: DistiLLM-2 통합 및 최종 제출 (최종 점수: 0.59)

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
├── distillm-2/                      # DistiLLM-2 프레임워크
│   ├── generate/
│   │   ├── generate_teacher_with_rag.py    # Teacher 모델 추론
│   │   └── generate_student_baseline.py    # Student 모델 추론
│   ├── src/
│   │   └── run_distillm.py                # Distillation 학습
│   └── README.md                           # DistiLLM-2 문서
├── data/
│   ├── rag/                         # RAG 인덱스
│   │   ├── chunks_2300.json        # 8,756개 청크
│   │   ├── bm25_index_2300.pkl     # BM25 인덱스
│   │   ├── faiss_index_2300.index  # FAISS 벡터 인덱스
│   │   └── metadata_2300.json      # 메타데이터
│   ├── synthetic_questions/         # 합성 데이터
│   │   └── combined_3000_questions.csv     # 3,000개 Q&A
│   └── teacher_responses_3000/      # Teacher 응답 데이터
├── docs/                            # 프로젝트 문서
│   ├── CURRENT_STATUS_2025_08_23.md
│   ├── architecture/
│   ├── pipeline-auto/
│   └── competition_environment_setup.md
├── test.csv                         # 515개 테스트 질문
├── submission_*.csv                 # 제출 파일들
├── requirements.txt                 # 의존성 목록
└── CLAUDE.md                        # Claude Code 가이드라인
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

## ✅ 완료된 개발 내용

### 1. 추론 파이프라인 완성
- 8-bit 양자화 Qwen2.5-7B-Instruct 모델
- 하이브리드 RAG (BM25 + FAISS)
- 515개 문제 49분 내 처리
- 10.7GB VRAM 사용

### 2. 데이터 생성 완료
- 3,000개 고품질 Q&A 쌍
- 6가지 질문 유형 균등 분배
- 100% 생성 성공률

### 3. DistiLLM-2 통합
```
Qwen2.5-14B-Instruct (Teacher 모델)
    ↓ 3,000개 합성 데이터 생성 (완료)
DistiLLM-2 Contrastive Learning
    ↓
Qwen2.5-7B-Instruct (Student 모델)
```

### 4. 제출 파일 생성
- `submission_8bit_full_context_20250823_233908.csv`
- `submission_distilled_final.csv`
- `submission_distilled_v2_final.csv`
- 기타 실험 버전 다수

## 🏆 대회 제약사항 충족 현황
| 제약사항 | 요구사항 | 달성 현황 | 상태 |
|---------|---------|----------|------|
| **모델 제한** | 단일 모델, 앙상블 불가 | Qwen2.5-7B 단일 모델 | ✅ |
| **라이선스** | 오픈소스 (Apache 2.0) | Apache 2.0 준수 | ✅ |
| **환경** | 오프라인 추론 | 독립 실행 스크립트 | ✅ |
| **메모리** | RTX 4090 24GB | 10.7GB 사용 | ✅ |
| **시간** | 4.5시간 이내 | ~49분 소요 | ✅ |
| **PyTorch** | v2.1.0 | v2.1.0 사용 | ✅ |

## 📝 프로젝트 문서
- [프로젝트 최종 현황](./docs/CURRENT_STATUS_2025_08_23.md)
- [DistiLLM-2 프레임워크](./distillm-2/README.md)
- [Claude Code 가이드라인](./CLAUDE.md)
- [추론 파이프라인 상세](./docs/pipeline-auto/32-추론-파이프라인-구축.md)
- [아키텍처 구현 현황](./docs/architecture/4-components-current-implementation.md)
- [대회 환경 재현 가이드](./docs/competition_environment_setup.md)

## 💡 핵심 기술 특징

### 1. 하이브리드 검색 시스템
- **BM25**: Kiwi 형태소 분석기 기반 정확한 키워드 매칭
- **FAISS**: KURE-v1 (1024차원) 한국어 임베딩으로 의미적 유사성 포착
- **독립 선택**: 각 방법에서 Top-3씩 선택하여 정보 다양성 극대화

### 2. 8-bit 양자화 최적화
- BitsAndBytesConfig를 통한 효율적 메모리 사용
- 16-bit 대비 75% 메모리 절감
- 추론 성능 저하 최소화

### 3. 한국어 특화 처리
- Kiwi 형태소 분석기로 정확한 한국어 토큰화
- KURE-v1 한국어 전용 임베더 사용
- 한국어 금융보안 도메인 특화 프롬프트

### 4. DistiLLM-2 Knowledge Distillation
- ICML 2025 Oral 선정 최신 기법
- Contrastive Learning 기반 효율적 지식 전달
- Teacher (Qwen2.5-14B-Instruct) → Student (Qwen2.5-7B-Instruct) 모델 압축

## 🎓 학습된 교훈

1. **청킹 크기의 중요성**: 2,300자가 최적 (500자보다 우수)
2. **검색 방법론**: 하이브리드가 단일 방법보다 효과적
3. **양자화 효과**: 8-bit로 충분한 성능 + 대폭적 메모리 절감
4. **프롬프트 엔지니어링**: 명확한 지침이 환각 현상 감소
5. **합성 데이터 품질**: Temperature 로테이션으로 다양성 확보

## 🤝 기여
이 프로젝트는 금융보안 AI 경진대회를 위해 개발되었습니다.

## 📄 라이센스
Apache 2.0 License (대회 요구사항 준수)

## 👥 팀
개인 참가 프로젝트

---

**프로젝트 상태**: ✅ 완료 (2025.09.05)
**최종 성과**: 대회 모든 제약사항 충족 및 6개 제출 파일 생성 완료