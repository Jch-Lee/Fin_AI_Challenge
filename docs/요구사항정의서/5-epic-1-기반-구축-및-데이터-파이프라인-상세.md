# 5. Epic 1: 기반 구축 및 데이터 파이프라인 (상세)

## **Story 1.1: `프로젝트 초기화 및 환경 설정`**

- **As a** developer, **I want** to '모노레포' 구조를 설정하고 핵심 라이브러리 의존성을 정의하여, **so that** 모든 팀원이 일관되고 재현 가능한 개발 환경에서 작업할 수 있다.
- **Acceptance Criteria:**
    1. 모노레포 폴더 구조가 생성된다.
    2. `requirements.txt` 파일에 초기 라이브러리 목록과 버전이 명시된다.
    3. 기본 린터(linter) 및 포매터(formatter) 설정이 완료된다.

## **Story 1.2: `PDF 데이터 전처리 파이프라인 구축`**

- **As a** data engineer, **I want** to `/data/raw`의 PDF 파일들을 고품질로 파싱하고 정제하여, **so that** 깨끗하고 신뢰성 있는 텍스트 데이터를 확보할 수 있다.
- **Acceptance Criteria:**
    1. **Vision V2 PDF 파서**가 41.2% 개선된 텍스트 추출을 수행한다.
    2. **3-Tier Fallback** 시스템이 구현된다 (Vision V2 → Traditional → Basic PyMuPDF).
    3. **배치 처리 파이프라인**이 `/data/raw`의 모든 PDF를 순차 처리한다.
    4. `KoreanEnglishTextProcessor` 로직에 기반한 국영문 혼합 텍스트 정제 기능이 구현된다.
    5. 메모리 효율적인 대용량 파일 처리가 가능하다.
    6. 데이터 처리 과정을 추적할 수 있는 로그 또는 메타데이터가 생성된다.

## **Story 1.3: `RAG 지식 베이스(FAISS) 구축`**

- **As a** data scientist, **I want** to 정제된 텍스트를 청크로 분할하고 벡터로 변환하여 FAISS 인덱스를 구축하여, **so that** RAG 시스템이 관련 정보를 빠르고 정확하게 검색할 수 있다.
- **Acceptance Criteria:**
    1. 스크립트는 텍스트를 적절한 크기의 청크로 분할한다.
    2. 임베딩 모델을 사용하여 각 청크를 벡터로 변환한다.
    3. FAISS 인덱스 파일이 디스크에 저장된다.
    4. 샘플 질문으로 인덱스가 정상 작동하는지 확인하는 테스트 함수가 포함된다.

## **Story 1.4: `합성 Q&A 학습 데이터셋 생성`**

- **As a** machine learning engineer, **I want** to 고성능 '교사 모델'을 사용하여 고품질 Q&A 쌍을 대량으로 생성하여, **so that** '학생 모델'을 파인튜닝하기 위한 맞춤형 학습 데이터셋을 확보할 수 있다.
- **Acceptance Criteria:**
    1. 스크립트는 (제약 없는 환경에서) '교사 모델'을 로드한다.
    2. 스크립트는 텍스트 청크를 입력하여 관련된 질문과 답변 쌍을 생성한다.
    3. 생성된 Q&A 쌍은 `/data/finetune` 폴더에 구조화된 형식으로 저장된다.
    4. 부적절한 데이터를 필터링하는 품질 검증 로직이 포함된다.

---
