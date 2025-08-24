# 대회 환경 재현 가이드 (Competition Environment Setup Guide)

## 개요
이 문서는 금융보안 AI 경진대회 환경을 원격 서버에서 정확하게 재현하기 위한 상세 가이드입니다.
RTX 4090 24GB GPU 환경에서 8-bit 양자화를 사용한 Qwen2.5-7B 모델 추론 시스템을 구축합니다.

## 필수 사양

### 하드웨어 요구사항
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) 또는 동등 사양
- **RAM**: 32GB 이상 권장
- **Storage**: 40GB 이상 여유 공간
- **OS**: Ubuntu 22.04 LTS 또는 Ubuntu 24.04 LTS

### 소프트웨어 버전 (대회 공식 환경)
- **Python**: 3.10.x (3.10.13 권장)
- **PyTorch**: 2.1.0+cu118
- **CUDA**: 11.8 (PyTorch 호환)
- **NumPy**: 1.26.4 (중요: 2.x 버전 사용 금지)

## Step 1: Python 3.10 환경 설치

### 1.1 시스템 패키지 업데이트
```bash
apt update && apt upgrade -y
apt install -y build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libffi-dev liblzma-dev python3-openssl git
```

### 1.2 pyenv를 사용한 Python 3.10 설치
```bash
# pyenv 설치
curl https://pyenv.run | bash

# 환경 변수 설정
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

# Python 3.10.13 설치
pyenv install 3.10.13
pyenv global 3.10.13
```

### 1.3 가상환경 생성
```bash
# 프로젝트 디렉토리 생성
mkdir -p /workspace
cd /workspace

# 가상환경 생성
python -m venv competition_env
source competition_env/bin/activate
```

## Step 2: PyTorch 및 핵심 라이브러리 설치

### 2.1 PyTorch 2.1.0+cu118 설치
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### 2.2 NumPy 호환성 해결
```bash
# NumPy 다운그레이드 (중요!)
pip install numpy==1.26.4
```

### 2.3 핵심 의존성 설치
```bash
# Transformers 및 관련 라이브러리
pip install transformers==4.41.2
pip install accelerate==0.30.1
pip install bitsandbytes==0.43.1
pip install scipy

# 한국어 처리
pip install kiwipiepy
pip install konlpy

# RAG 구성요소
pip install sentence-transformers==2.7.0
pip install faiss-cpu
pip install rank-bm25==0.2.2

# 유틸리티
pip install pandas
pip install tqdm
pip install datasets
```

## Step 3: CUDA 11.8 라이브러리 설정 (8-bit 양자화용)

### 3.1 CUDA 11.8 런타임 라이브러리 설치
```bash
# bitsandbytes가 필요로 하는 CUDA 11.8 라이브러리 설치
pip install nvidia-cusparse-cu11==11.7.4.91

# 환경 변수 설정 (가상환경 활성화 시 자동 적용)
echo 'export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH' >> $VIRTUAL_ENV/bin/activate
```

### 3.2 설치 확인
```bash
# bitsandbytes 로드 테스트
python -c "import bitsandbytes as bnb; print(f'bitsandbytes version: {bnb.__version__}')"
```

## Step 4: 프로젝트 파일 구조 설정

### 4.1 디렉토리 구조 생성
```bash
mkdir -p /workspace/Fin_AI_Challenge/{data,scripts,docs}
mkdir -p /workspace/Fin_AI_Challenge/data/{rag,competition}
```

### 4.2 필수 파일 전송
```
/workspace/Fin_AI_Challenge/
├── data/
│   ├── competition/
│   │   ├── test.csv                 # 테스트 질문 (515개)
│   │   └── sample_submission.csv    # 제출 형식
│   └── rag/
│       ├── chunks_2300.json         # RAG 청크 (8,756개)
│       ├── faiss_index_2300.index   # FAISS 인덱스
│       ├── bm25_index_2300.pkl      # BM25 인덱스
│       └── metadata_2300.json       # 메타데이터
└── scripts/
    └── generate_submission_standalone.py  # 추론 스크립트
```

## Step 5: 모델 및 임베더 초기화

### 5.1 KURE-v1 임베더 다운로드
```bash
python -c "
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('nlpai-lab/KURE-v1')
print('KURE-v1 loaded successfully')
"
```

### 5.2 Qwen2.5-7B-Instruct 모델 다운로드
```bash
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print('Tokenizer loaded successfully')
"
```

## Step 6: 추론 스크립트 실행

### 6.1 빠른 테스트 (10개 샘플)
```bash
cd /workspace
source competition_env/bin/activate
python generate_submission_standalone.py \
    --test_mode \
    --num_samples 10 \
    --data_dir /workspace/Fin_AI_Challenge/data/rag \
    --input_file /workspace/Fin_AI_Challenge/test.csv \
    --output_file /workspace/test_10_samples.csv
```

### 6.2 전체 추론 실행
```bash
python generate_submission_standalone.py \
    --data_dir /workspace/Fin_AI_Challenge/data/rag \
    --input_file /workspace/Fin_AI_Challenge/test.csv \
    --output_file /workspace/submission.csv
```

## 검증 단계

### GPU 메모리 사용량 확인
```bash
nvidia-smi
# 예상: ~10.7GB / 24GB 사용
```

### 성능 지표 확인
- **메모리 사용**: 10-11GB (8-bit 양자화)
- **처리 속도**: 5-10초/문제
- **전체 시간**: 515문제 × 6초 = 약 51분

### 결과 형식 검증
```bash
head -5 submission.csv
# 형식: ID,Answer
# 객관식: 숫자 (1-5)
# 주관식: 한국어 텍스트
```

## 트러블슈팅

### 1. CUDA 라이브러리 오류
**문제**: `libcusparse.so.11: version 'libcusparse.so.11' not found`

**해결**:
```bash
pip install nvidia-cusparse-cu11==11.7.4.91
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
```

### 2. NumPy 호환성 오류
**문제**: `module 'numpy' has no attribute 'bool'`

**해결**:
```bash
pip uninstall numpy -y
pip install numpy==1.26.4
```

### 3. 메모리 부족
**문제**: CUDA out of memory

**해결**:
- 8-bit 양자화 설정 확인
- batch_size 감소
- GPU 캐시 정리: `torch.cuda.empty_cache()`

### 4. 디스크 공간 부족
**문제**: No space left on device

**해결**:
```bash
# 불필요한 파일 정리
rm -rf ~/.cache/huggingface/hub/*
rm -rf /tmp/*
```

## 환경 변수 요약

```bash
# .bashrc 또는 .profile에 추가
export PYTHONPATH=/workspace/Fin_AI_Challenge:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 가상환경 activate 스크립트에 추가
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
```

## 주요 파일 체크섬 (검증용)

```bash
# RAG 데이터베이스 검증
wc -l /workspace/Fin_AI_Challenge/data/rag/chunks_2300.json
# 예상: 8756 lines

# FAISS 인덱스 크기
ls -lh /workspace/Fin_AI_Challenge/data/rag/faiss_index_2300.index
# 예상: 약 34MB

# BM25 인덱스 크기
ls -lh /workspace/Fin_AI_Challenge/data/rag/bm25_index_2300.pkl
# 예상: 약 76MB
```

## 성공 기준

1. ✅ GPU 메모리 사용량: 10-11GB (24GB 중)
2. ✅ 처리 속도: 5-10초/문제
3. ✅ 전체 추론 시간: 4.5시간 이내
4. ✅ 출력 형식: CSV (ID, Answer)
5. ✅ 객관식 정확도: 숫자만 출력
6. ✅ 주관식 품질: 한국어 텍스트, 이미지 링크 없음

## 참고 자료

- [PyTorch 2.1.0 Documentation](https://pytorch.org/docs/2.1/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

**마지막 업데이트**: 2025-08-23
**작성자**: Claude Code Assistant
**버전**: 1.0