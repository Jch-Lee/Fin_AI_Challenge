# 1.3. RAG - 청킹 및 임베딩

## 청킹 전략 (최신화: 2025-08-15)

### 1.3.1. 계층적 마크다운 청킹 (기본)
- **HierarchicalMarkdownChunker** 사용 (Vision V2 출력에 최적화)
  - 마크다운 헤더 구조 인식 (# ~ ####)
  - 계층별 다른 청크 크기:
    - Level 1: 1024자, 오버랩 100자
    - Level 2: 512자, 오버랩 50자
    - Level 3-4: 256자, 오버랩 20-30자
  - LangChain 기반 구현

### 1.3.1-fallback. 기본 청킹 (폴백)
- **DocumentChunker** 사용 (LangChain 미설치 시)
  - RecursiveCharacterTextSplitter 방식
  - 청크 크기: 512자, 오버랩: 50자

### 1.3.2. 한국어 형태소 분석
- **Kiwipiepy** 사용 (BM25 검색용)
  - 품사 필터링: 명사(N), 동사(V), 형용사(VA), 외국어(SL)
  - 띄어쓰기 교정 기능
  - 10-30배 빠른 토크나이징 속도

### 1.3.3. 임베딩 모델
- **KURE-v1** (nlpai-lab/KURE-v1) - 기본
  - 768차원 벡터
  - SentenceTransformer 기반
  - 한국어 특화 임베딩
- **E5** (intfloat/multilingual-e5-large) - 대체 옵션
  - 768차원 벡터
  - 다국어 지원
