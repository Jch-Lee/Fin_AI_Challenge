# RAG 파이프라인 사용 가이드

## 📋 개요
이 문서는 금융 보안 문서 RAG 시스템의 PDF 처리 파이프라인 사용법을 설명합니다.

## 🚀 빠른 시작

### 1. 새로운 PDF 추가 처리 (증분 처리)
```bash
# 새 PDF 확인만
python scripts/add_new_pdfs.py --check-only

# 로컬에서 새 PDF 처리 (Vision V2 사용)
python scripts/add_new_pdfs.py --use-vision

# 원격 서버에서 처리
python scripts/add_new_pdfs.py --use-vision --remote --remote-host root@86.127.233.28:34270
```

### 2. 전체 PDF 재처리
```bash
# 로컬 처리
python scripts/process_all_pdfs.py --use-vision --batch-size 2

# 테스트 (3개 파일만)
python scripts/process_all_pdfs.py --use-vision --test
```

## 📂 디렉토리 구조
```
Fin_AI_Challenge/
├── data/
│   ├── raw/                  # 원본 PDF 파일 위치 (새 PDF 추가)
│   ├── processed/             # 추출된 텍스트 파일
│   ├── knowledge_base/        # FAISS 인덱스 및 벡터 DB
│   └── processed_files.json   # 처리 이력 추적
├── packages/
│   ├── vision/               # Vision V2 텍스트 추출
│   ├── preprocessing/        # 청킹 및 전처리
│   └── rag/                 # RAG 컴포넌트
└── scripts/
    ├── process_all_pdfs.py   # 전체 처리 스크립트
    └── add_new_pdfs.py       # 증분 처리 스크립트
```

## 🔧 주요 컴포넌트

### Vision V2 텍스트 추출
- **모델**: Qwen2.5-VL-7B-Instruct
- **품질**: PyMuPDF 대비 41.2% 향상
- **속도**: 페이지당 3-6초 (최적화 후)
- **프롬프트**: 70줄 최적화 버전

### 처리 파이프라인
1. **PDF → 이미지 변환**: PyMuPDF 사용
2. **텍스트 추출**: Vision V2 모델
3. **청킹**: HierarchicalMarkdownChunker
4. **임베딩**: KURE-v1 (1024차원)
5. **인덱싱**: FAISS + BM25

## 📊 성능 최적화

### 메모리 관리
```bash
# GPU 메모리 최적화 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 배치 크기 조정
- **메모리 충분**: `--batch-size 3`
- **일반적**: `--batch-size 2`
- **메모리 부족**: `--batch-size 1`

## 🔄 증분 처리 워크플로우

1. **새 PDF 추가**: `data/raw/` 디렉토리에 PDF 복사
2. **확인**: `python scripts/add_new_pdfs.py --check-only`
3. **처리**: `python scripts/add_new_pdfs.py --use-vision`
4. **결과**: 
   - 텍스트: `data/processed/`
   - 벡터 DB: `data/knowledge_base/`
   - 이력: `data/processed_files.json`

## 🛠️ 문제 해결

### GPU 메모리 부족
```bash
# 프로세스 종료
pkill -f process_all_pdfs

# 메모리 정리
python -c "import torch; torch.cuda.empty_cache()"

# 배치 크기 줄여서 재시작
python scripts/process_all_pdfs.py --use-vision --batch-size 1
```

### 원격 서버 연결 문제
```bash
# SSH 키 확인
ssh-keygen -t rsa -b 4096

# 연결 테스트
ssh -p 34270 root@86.127.233.28 "echo 'Connected'"
```

## 📈 모니터링

### 처리 진행 상황
```bash
# 로그 확인
tail -f processing.log

# 처리된 파일 수
ls data/processed/*.txt | wc -l

# GPU 사용률
nvidia-smi -l 1
```

### 품질 검증
```python
# 추출된 텍스트 확인
from pathlib import Path

processed_dir = Path("data/processed")
for txt_file in processed_dir.glob("*.txt"):
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"{txt_file.name}: {len(content):,} chars")
```

## 📝 환경 설정

### 필수 패키지
```bash
pip install -r requirements.txt
```

### 주요 의존성
- PyTorch 2.1.0
- Transformers 4.41.2
- FAISS-CPU
- sentence-transformers
- PyMuPDF
- Qwen-VL-Utils

## 🔐 보안 고려사항

- PDF 파일명에 특수문자 사용 주의
- 민감 정보 포함 PDF는 별도 관리
- 처리 후 원본 PDF 백업 권장

## 📞 지원

문제 발생 시:
1. `processing.log` 확인
2. GPU 메모리 상태 확인
3. 프롬프트 길이 확인 (70줄 유지)