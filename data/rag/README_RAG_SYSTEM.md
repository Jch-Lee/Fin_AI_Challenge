# RAG System Documentation

## 현재 버전: 2300자 하이브리드 청킹 (v2.0)
**생성일**: 2025-08-17 23:40

## 시스템 개요

### 주요 특징
- **청킹 방식**: 하이브리드 청킹 (섹션 인식 + 경계 오버랩)
- **청크 크기**: 2,300자 (이전 512자 대비 4.5배 증가)
- **오버랩**: 200자 (일반), 300자 (섹션 경계)
- **임베딩 모델**: KURE-v1 (1024차원, 한국어 특화)
- **검색 방식**: 하이브리드 (BM25 50% + Vector 50%)

### 통계
- **총 문서**: 59개
- **총 청크**: 8,270개
  - 일반 청크: 4,679개
  - 경계 청크: 3,591개
- **평균 청크 크기**: 765자
- **총 문자 수**: 6,323,638자

## 파일 구조

### 현재 사용 파일 (2300 버전)
```
data/rag/
├── chunks_2300.json         # 청크 데이터 (19.7MB)
├── embeddings_2300.npy      # KURE 임베딩 (33.9MB)
├── faiss_index_2300.index   # 벡터 인덱스 (33.9MB)
├── bm25_index_2300.pkl      # 키워드 인덱스 (14.6MB)
└── metadata_2300.json       # 메타데이터
```

### 이전 버전 (백업)
```
data/rag/old_versions_backup/
├── chunks.json              # 구 버전 (512자 청킹)
├── embeddings.npy           # 구 버전 임베딩
├── faiss.index              # 구 버전 인덱스
└── bm25_index.pkl           # 구 버전 BM25
```

## 사용 방법

### 1. RAG 시스템 로드
```python
from pathlib import Path
import json
import pickle
import numpy as np
import faiss

# 파일 경로
rag_dir = Path("data/rag")
chunks_file = rag_dir / "chunks_2300.json"
embeddings_file = rag_dir / "embeddings_2300.npy"
faiss_file = rag_dir / "faiss_index_2300.index"
bm25_file = rag_dir / "bm25_index_2300.pkl"

# 데이터 로드
with open(chunks_file, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

embeddings = np.load(embeddings_file)
faiss_index = faiss.read_index(str(faiss_file))

with open(bm25_file, 'rb') as f:
    bm25_index = pickle.load(f)
```

### 2. 하이브리드 검색 설정
```python
from packages.rag.retrieval.hybrid_retriever import HybridRetriever

# 하이브리드 검색기 초기화 (5:5 가중치)
hybrid_retriever = HybridRetriever(
    bm25_retriever=bm25_retriever,
    vector_retriever=vector_retriever,
    embedder=embedder,
    bm25_weight=0.5,  # 50% 키워드 기반
    vector_weight=0.5, # 50% 벡터 기반
    normalization_method="min_max"
)
```

### 3. 검색 실행
```python
# 검색 실행
query = "개인정보 보호법의 주요 내용은?"
results = hybrid_retriever.search(query, k=5)

for result in results:
    print(f"Score: {result.hybrid_score:.4f}")
    print(f"Content: {result.content[:200]}...")
    print(f"Source: {result.metadata.get('source_file')}")
```

## 주요 개선사항 (v2.0)

### 1. 청킹 개선
- **이전**: 512자 단순 청킹, 섹션 경계 손실
- **현재**: 2300자 하이브리드 청킹, 섹션 경계 보존
- **효과**: 4.5배 더 많은 문맥 정보 보존

### 2. 검색 균형 조정
- **이전**: BM25 30% + Vector 70%
- **현재**: BM25 50% + Vector 50%
- **효과**: 키워드 검색과 의미 검색의 균형

### 3. 경계 처리
- **이전**: 섹션 간 0자 오버랩
- **현재**: 섹션 경계 300자 특별 오버랩
- **효과**: 섹션 전환 시 문맥 연속성 유지

## 테스트 스크립트

### 검색 테스트
```bash
python scripts/test_hybrid_search_2300.py
```

### RAG 시스템 재구축
```bash
python scripts/build_hybrid_rag_2300.py --batch-size 16
```

## 성능 메트릭

### 처리 시간
- 문서 로드: 0.05초
- 청킹: 2.21초
- 임베딩 생성: 387.99초 (6.5분)
- 인덱스 구축: 2초
- **총 시간**: 397.35초 (6.62분)

### 메모리 사용량
- 총 디스크 사용: 약 102MB
- GPU 메모리: 임베딩 생성 시 약 4GB

## 주의사항

1. **파일 버전 확인**: 항상 `_2300` 접미사가 붙은 파일 사용
2. **가중치 설정**: RAG 파이프라인에서 5:5 가중치 유지
3. **청킹 일관성**: 새 문서 추가 시 동일한 2300자 설정 사용
4. **백업**: 이전 버전은 `old_versions_backup/` 폴더에 보관

## 문제 해결

### Q: 검색 결과가 이상한 경우
- 파일 버전 확인 (2300 버전 사용 중인지)
- 가중치 설정 확인 (5:5로 설정되어 있는지)
- 인덱스 재구축 필요 시 `build_hybrid_rag_2300.py` 실행

### Q: 메모리 부족 오류
- 배치 크기 줄이기: `--batch-size 8`
- GPU 대신 CPU 사용: 코드에서 device='cpu' 설정

### Q: 이전 버전으로 롤백
- `old_versions_backup/` 폴더의 파일들을 복원
- 파일명에서 _2300 제거