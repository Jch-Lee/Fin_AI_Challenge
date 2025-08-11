# Financial Security Knowledge Understanding (FSKU) AI Challenge

## 🎯 프로젝트 개요
금융 보안 관련 질문에 대한 한국어 답변을 생성하는 AI 시스템 개발

### 핵심 목표
- **객관식 문제**: 정확한 선택지 번호 예측
- **주관식 문제**: 설명적이고 정확한 한국어 답변 생성
- **성능 요구사항**: RTX 4090에서 4.5시간 내 전체 데이터셋 추론

## 🏗️ 시스템 아키텍처

### Distill-M 2 (Contrastive Distillation) 접근법
```
Teacher Models (70B/7B) → Knowledge Distillation → Student Models (1.5B-10.7B)
                            ↓
                    RAG System Integration
                            ↓
                    Optimized Inference
```

### 주요 구성요소
1. **Teacher Models**
   - Llama-3.1-70B-Instruct
   - Qwen2.5-7B-Instruct

2. **Student Models**
   - Mistral-7B-Instruct
   - Solar-10.7B-Instruct  
   - Qwen2.5-1.5B-Instruct

3. **RAG System**
   - FAISS Vector Database
   - BM25 Hybrid Search
   - Korean Sentence Embeddings

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 활성화 (Windows)
activate_env.bat

# 의존성 설치
pip install -r requirements.txt

# 개발 모드 설치
pip install -e .
```

### 2. 베이스라인 실행
```bash
# 빠른 테스트 (10개 샘플)
python baseline_code/run_baseline_quick.py

# 전체 베이스라인
python baseline_code/run_baseline.py
```

### 3. 학습 실행
```bash
# Teacher 모델로 데이터 생성
python src/data/generate_teacher_data.py

# Student 모델 학습
python src/train.py --config configs/train_config.yaml

# 평가
python src/evaluate.py --model_path ./checkpoints/best_model
```

## 📁 프로젝트 구조
```
Fin_AI_Challenge/
├── baseline_code/       # 대회 제공 베이스라인
├── packages/            # 핵심 파이썬 소스 코드 패키지
│   ├── preprocessing/  # 데이터 전처리 및 지식 베이스 구축
│   ├── training/       # Distill-M 2 모델 훈련
│   ├── inference/      # 최종 추론 파이프라인
│   └── rag/           # RAG 시스템 컴포넌트
├── scripts/            # 유틸리티 및 실행 스크립트
├── docs/              # 핵심 참고 문서
│   ├── Architecture.md # 시스템 아키텍처 정의
│   ├── Pipeline.md     # 상세 개발 파이프라인
│   └── PROJECT_PLAN.md # 개발 계획 및 우선순위
├── data/              # 데이터 자산 폴더
│   ├── raw/           # 원본 외부 데이터
│   ├── processed/     # 정제 및 청킹된 데이터
│   ├── finetune/      # 합성된 Q&A 학습 데이터셋
│   └── knowledge_base/ # 생성된 FAISS 인덱스 파일
├── models/            # 모델 가중치 폴더
│   ├── student/       # 최종 파인튜닝된 학생 모델
│   └── teacher/       # 교사 모델
├── tests/             # 테스트 코드
│   ├── unit/          # 단위 테스트
│   └── integration/   # 통합 테스트
└── outputs/           # 출력 결과
```

## 📊 평가 지표
- **객관식**: Exact Match Accuracy
- **주관식**: ROUGE-L, BERTScore
- **종합**: FSKU Evaluation Metric

## 🔧 개발 가이드

### 코드 스타일
```bash
# 포맷팅
black src/

# 린팅
ruff check src/

# 테스트
pytest tests/
```

### Git Workflow
```bash
# 기능 브랜치 생성
git checkout -b feature/rag-implementation

# 커밋
git add .
git commit -m "feat: RAG 시스템 구현"

# PR 생성
git push origin feature/rag-implementation
```

## 📈 성능 최적화
- **4-bit Quantization**: 메모리 사용량 75% 감소
- **Flash Attention**: 추론 속도 2x 향상
- **Batch Processing**: 처리량 3x 증가
- **KV Cache**: 토큰 생성 속도 향상

## 🏆 대회 제약사항
- ✅ 단일 모델만 사용 (앙상블 불가)
- ✅ 오픈소스 모델 (2025.08.01 이전 공개)
- ✅ 오프라인 환경 추론
- ✅ 24GB VRAM 제한
- ✅ 4.5시간 추론 시간 제한

## 📝 라이센스
MIT License

## 🤝 기여
기여를 환영합니다! Issue와 Pull Request를 통해 참여해주세요.