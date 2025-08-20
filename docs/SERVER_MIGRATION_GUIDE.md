# 서버 마이그레이션 가이드

## 개요
이 문서는 Fin_AI_Challenge 프로젝트를 새로운 서버로 마이그레이션하는 상세한 과정을 설명합니다.
현재 검증된 파이프라인: **BM25+Vector Top3 방식 (리더보드 점수 0.55)**

## 목차
1. [사전 준비 사항](#사전-준비-사항)
2. [필수 파일 목록](#필수-파일-목록)
3. [단계별 마이그레이션 절차](#단계별-마이그레이션-절차)
4. [문제 해결 가이드](#문제-해결-가이드)
5. [검증 절차](#검증-절차)

---

## 사전 준비 사항

### 로컬 환경 확인
```bash
# 프로젝트 루트 확인
cd C:\Fin_AI_Challenge

# 필수 파일 존재 확인
ls data/rag/chunks_2300.json
ls data/rag/faiss_index_2300.index
ls data/rag/bm25_index_2300.pkl
ls data/competition/test.csv
ls scripts/generate_submission_*.py
```

### SSH 연결 정보
- 기본 연결: `ssh -p [PORT] root@[IP_ADDRESS]`
- 포트 포워딩 포함: `ssh -p [PORT] root@[IP_ADDRESS] -L 8080:localhost:8080`

---

## 필수 파일 목록

### 1. 핵심 데이터 파일 (data/rag/)
```
chunks_2300.json       # 19.6MB - RAG 문서 청크 (8,270개)
faiss_index_2300.index # 33.9MB - FAISS 벡터 인덱스
bm25_index_2300.pkl    # 14.6MB - BM25 검색 인덱스
```

### 2. 실험 스크립트 (scripts/)
```
generate_submission_bm25_vector_top3.py    # Qwen2.5-7B + BM25+Vector Top3 (0.55 점수)
generate_submission_gemma_combined_top3.py # Gemma-Ko-7B + BM25+Vector Top3 (실험용)
```

### 3. 테스트 데이터
```
data/competition/test.csv  # 515개 문제 (객관식 500개, 주관식 15개)
```

### 4. 의존성 파일
```
requirements.txt  # Python 패키지 목록
```

---

## 단계별 마이그레이션 절차

### Step 1: SSH 연결 테스트
```bash
# 연결 테스트
ssh -p [PORT] root@[IP_ADDRESS] "echo 'Connection successful' && pwd"
```

### Step 2: 디렉토리 구조 생성
```bash
ssh -p [PORT] root@[IP_ADDRESS] "mkdir -p /root/Fin_AI_Challenge/{scripts,data/rag,logs,submissions}"
```

### Step 3: RAG 데이터 압축 및 전송

#### 옵션 A: tar 압축 방식 (권장)
```bash
# 로컬에서 압축
cd data/rag
tar -czf rag_data.tar.gz chunks_2300.json faiss_index_2300.index bm25_index_2300.pkl
cd ../..

# 서버로 전송
scp -P [PORT] data/rag/rag_data.tar.gz root@[IP_ADDRESS]:/root/Fin_AI_Challenge/data/rag/

# 서버에서 압축 해제
ssh -p [PORT] root@[IP_ADDRESS] "cd /root/Fin_AI_Challenge/data/rag && tar -xzf rag_data.tar.gz && ls -la"
```

#### 옵션 B: 개별 파일 전송 (네트워크 불안정 시)
```bash
# 각 파일 개별 전송
scp -P [PORT] data/rag/chunks_2300.json root@[IP_ADDRESS]:/root/Fin_AI_Challenge/data/rag/
scp -P [PORT] data/rag/faiss_index_2300.index root@[IP_ADDRESS]:/root/Fin_AI_Challenge/data/rag/
scp -P [PORT] data/rag/bm25_index_2300.pkl root@[IP_ADDRESS]:/root/Fin_AI_Challenge/data/rag/
```

#### 옵션 C: cat을 통한 전송 (scp 실패 시)
```bash
# chunks 파일 (큰 파일은 split 필요할 수 있음)
cat data/rag/chunks_2300.json | ssh -p [PORT] root@[IP_ADDRESS] "cat > /root/Fin_AI_Challenge/data/rag/chunks_2300.json"

# FAISS 인덱스
cat data/rag/faiss_index_2300.index | ssh -p [PORT] root@[IP_ADDRESS] "cat > /root/Fin_AI_Challenge/data/rag/faiss_index_2300.index"

# BM25 인덱스
cat data/rag/bm25_index_2300.pkl | ssh -p [PORT] root@[IP_ADDRESS] "cat > /root/Fin_AI_Challenge/data/rag/bm25_index_2300.pkl"
```

### Step 4: 스크립트 전송
```bash
# Qwen 스크립트
cat scripts/generate_submission_bm25_vector_top3.py | ssh -p [PORT] root@[IP_ADDRESS] "cat > /root/Fin_AI_Challenge/scripts/generate_submission_bm25_vector_top3.py"

# Gemma 스크립트
cat scripts/generate_submission_gemma_combined_top3.py | ssh -p [PORT] root@[IP_ADDRESS] "cat > /root/Fin_AI_Challenge/scripts/generate_submission_gemma_combined_top3.py"
```

### Step 5: 테스트 데이터 전송
```bash
cat data/competition/test.csv | ssh -p [PORT] root@[IP_ADDRESS] "cat > /root/Fin_AI_Challenge/test.csv"
```

### Step 6: Python 환경 설정

#### 기본 패키지 설치
```bash
ssh -p [PORT] root@[IP_ADDRESS] "pip install --upgrade pip"

# 필수 패키지 (단계별 설치 권장)
ssh -p [PORT] root@[IP_ADDRESS] "pip install pandas numpy tqdm"
ssh -p [PORT] root@[IP_ADDRESS] "pip install torch --index-url https://download.pytorch.org/whl/cu121"
ssh -p [PORT] root@[IP_ADDRESS] "pip install transformers accelerate"
ssh -p [PORT] root@[IP_ADDRESS] "pip install sentence-transformers"
ssh -p [PORT] root@[IP_ADDRESS] "pip install kiwipiepy kiwipiepy-model"
ssh -p [PORT] root@[IP_ADDRESS] "pip install faiss-cpu"  # GPU 서버면 faiss-gpu
ssh -p [PORT] root@[IP_ADDRESS] "pip install bitsandbytes"
ssh -p [PORT] root@[IP_ADDRESS] "pip install rank-bm25"
```

#### requirements.txt 사용 (선택사항)
```bash
# requirements.txt 전송
cat requirements.txt | ssh -p [PORT] root@[IP_ADDRESS] "cat > /root/Fin_AI_Challenge/requirements.txt"

# 설치
ssh -p [PORT] root@[IP_ADDRESS] "cd /root/Fin_AI_Challenge && pip install -r requirements.txt"
```

### Step 7: GPU 확인 (중요)
```bash
ssh -p [PORT] root@[IP_ADDRESS] "nvidia-smi"
ssh -p [PORT] root@[IP_ADDRESS] "python3 -c 'import torch; print(torch.cuda.is_available())'"
```

---

## 실험 실행

### Qwen2.5-7B 모델 (검증된 0.55 점수)
```bash
# 백그라운드 실행
ssh -p [PORT] root@[IP_ADDRESS] "cd /root/Fin_AI_Challenge && nohup python3 scripts/generate_submission_bm25_vector_top3.py > qwen_log.txt 2>&1 &"

# 진행 상황 모니터링
ssh -p [PORT] root@[IP_ADDRESS] "tail -f /root/Fin_AI_Challenge/qwen_log.txt"
```

### Gemma-Ko-7B 모델 (실험)
```bash
# 백그라운드 실행
ssh -p [PORT] root@[IP_ADDRESS] "cd /root/Fin_AI_Challenge && nohup python3 scripts/generate_submission_gemma_combined_top3.py > gemma_log.txt 2>&1 &"

# 진행 상황 모니터링
ssh -p [PORT] root@[IP_ADDRESS] "tail -f /root/Fin_AI_Challenge/gemma_log.txt"
```

### 프로세스 확인
```bash
ssh -p [PORT] root@[IP_ADDRESS] "ps aux | grep python"
```

---

## 결과 파일 다운로드

### 생성된 파일 확인
```bash
ssh -p [PORT] root@[IP_ADDRESS] "ls -la /root/Fin_AI_Challenge/submission*.csv"
```

### 파일 다운로드
```bash
# Qwen 결과
scp -P [PORT] root@[IP_ADDRESS]:/root/Fin_AI_Challenge/submission_bm25_vector_top3_*.csv ./

# Gemma 결과
scp -P [PORT] root@[IP_ADDRESS]:/root/Fin_AI_Challenge/submission_gemma_combined_top3_*.csv ./
```

---

## 문제 해결 가이드

### 1. CUDA/GPU 문제
```bash
# CUDA 버전 확인
ssh -p [PORT] root@[IP_ADDRESS] "nvcc --version"

# PyTorch CUDA 호환성 확인
ssh -p [PORT] root@[IP_ADDRESS] "python3 -c 'import torch; print(torch.version.cuda)'"

# GPU 메모리 부족 시
# - 4-bit 양자화 사용 (Gemma 스크립트에 이미 적용됨)
# - batch_size 줄이기
# - 모델을 CPU로 변경 (매우 느림)
```

### 2. 패키지 설치 실패
```bash
# pip 업그레이드
ssh -p [PORT] root@[IP_ADDRESS] "python3 -m pip install --upgrade pip"

# 캐시 클리어
ssh -p [PORT] root@[IP_ADDRESS] "pip cache purge"

# 특정 버전 설치
ssh -p [PORT] root@[IP_ADDRESS] "pip install torch==2.1.0"
```

### 3. 메모리 부족
```bash
# 시스템 메모리 확인
ssh -p [PORT] root@[IP_ADDRESS] "free -h"

# 스왑 추가 (임시)
ssh -p [PORT] root@[IP_ADDRESS] "fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile"
```

### 4. 네트워크 타임아웃
```bash
# pip 타임아웃 증가
ssh -p [PORT] root@[IP_ADDRESS] "pip install --default-timeout=100 [package_name]"

# 미러 사이트 사용
ssh -p [PORT] root@[IP_ADDRESS] "pip install -i https://pypi.douban.com/simple [package_name]"
```

---

## 검증 절차

### 1. 파일 무결성 확인
```bash
# 파일 크기 비교
ssh -p [PORT] root@[IP_ADDRESS] "ls -lh /root/Fin_AI_Challenge/data/rag/"

# 예상 크기:
# chunks_2300.json: ~19.6MB
# faiss_index_2300.index: ~33.9MB
# bm25_index_2300.pkl: ~14.6MB
```

### 2. 간단한 테스트 실행
```bash
# Python 임포트 테스트
ssh -p [PORT] root@[IP_ADDRESS] "python3 -c 'import torch, transformers, sentence_transformers, kiwipiepy, faiss'"
```

### 3. 부분 실행 테스트
첫 10개 문제만 테스트하는 스크립트를 만들어 빠르게 검증할 수 있습니다.

---

## 성능 메트릭 (참고)

### 예상 실행 시간
- **Qwen2.5-7B (16-bit)**: 약 4-5분 (515문제)
- **Gemma-Ko-7B (4-bit)**: 약 4분 (515문제)

### GPU 메모리 사용량
- **Qwen2.5-7B**: 약 15-18GB VRAM
- **Gemma-Ko-7B (4-bit)**: 약 8-10GB VRAM

### 처리 속도
- 평균 2-3 문제/초
- 100개마다 가비지 컬렉션

---

## 자동화 스크립트

### 전체 마이그레이션 스크립트 (migrate.sh)
```bash
#!/bin/bash
# Usage: ./migrate.sh [PORT] [IP_ADDRESS]

PORT=$1
IP=$2

echo "Starting migration to $IP:$PORT..."

# 1. Create directories
ssh -p $PORT root@$IP "mkdir -p /root/Fin_AI_Challenge/{scripts,data/rag,logs,submissions}"

# 2. Compress and transfer RAG data
cd data/rag
tar -czf rag_data.tar.gz chunks_2300.json faiss_index_2300.index bm25_index_2300.pkl
scp -P $PORT rag_data.tar.gz root@$IP:/root/Fin_AI_Challenge/data/rag/
ssh -p $PORT root@$IP "cd /root/Fin_AI_Challenge/data/rag && tar -xzf rag_data.tar.gz"
rm rag_data.tar.gz
cd ../..

# 3. Transfer scripts
scp -P $PORT scripts/generate_submission_*.py root@$IP:/root/Fin_AI_Challenge/scripts/

# 4. Transfer test data
scp -P $PORT data/competition/test.csv root@$IP:/root/Fin_AI_Challenge/

echo "Migration complete!"
```

---

## 체크리스트

### 마이그레이션 전
- [ ] SSH 키 설정 완료
- [ ] 로컬 파일 무결성 확인
- [ ] 대상 서버 GPU 확인
- [ ] 네트워크 안정성 확인

### 마이그레이션 중
- [ ] 디렉토리 구조 생성
- [ ] RAG 데이터 전송 (3개 파일)
- [ ] 스크립트 전송
- [ ] test.csv 전송
- [ ] Python 패키지 설치
- [ ] GPU 작동 확인

### 마이그레이션 후
- [ ] 파일 크기 검증
- [ ] Import 테스트
- [ ] 짧은 테스트 실행
- [ ] 전체 실행
- [ ] 결과 파일 다운로드

---

## 주의사항

1. **데이터 백업**: 로컬에 모든 파일의 백업본을 유지하세요.
2. **서버 시간**: 장시간 실행 시 서버 연결이 끊어질 수 있으므로 nohup이나 tmux 사용을 권장합니다.
3. **GPU 호환성**: 서버의 CUDA 버전과 PyTorch 버전 호환성을 확인하세요.
4. **네트워크**: 큰 파일 전송 시 네트워크 안정성이 중요합니다.

---

**작성일**: 2025-08-20
**최종 검증 점수**: Qwen2.5-7B + BM25+Vector Top3 = 0.55
**문서 버전**: 1.0