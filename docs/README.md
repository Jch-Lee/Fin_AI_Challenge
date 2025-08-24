# 프로젝트 문서 모음

*Last Updated: 2025-08-23*

이 디렉토리는 프로젝트 진행에 필요한 핵심 참고 문서들을 보관합니다.

## 📌 최신 상태
- **[CURRENT_STATUS_2025_08_23.md](./CURRENT_STATUS_2025_08_23.md)** - 🎯 **8-bit 양자화 추론 파이프라인 완성**
  - ✅ Qwen2.5-7B-Instruct 8-bit 양자화로 10.7GB VRAM 사용
  - ✅ 5.75초/질문 처리 속도 (4.5시간 제한 대비 5.5배 여유)
  - ✅ BM25(Kiwi) + FAISS(KURE-v1) Combined Top-3 하이브리드 검색
  - ✅ 3,000개 합성 Q&A 데이터 생성 완료
- **[성능 기록: performance_record.md](./performance_record.md)** - 📈 **실측 성능 메트릭**
- **[이전 상태: CURRENT_STATUS_2025_08_20.md](./CURRENT_STATUS_2025_08_20.md)**

## 📋 완전 분할된 문서들 (Auto-Generated)

### 🏗️ Architecture (docs/architecture/)
- **완전 자동 분할됨**: 11개 파일
- [📑 index.md](./architecture/index.md) - 상세 목차
- [1. High Level Architecture](./architecture/1-high-level-architecture.md)
- [2. Tech Stack](./architecture/2-tech-stack.md)  
- [3. Data Models](./architecture/3-data-models.md)
- [4. Components - Current Implementation](./architecture/4-components-current-implementation.md) ⭐ **Updated 08-23**
- [5. External APIs](./architecture/5-external-apis.md)
- [6. Core Workflows](./architecture/6-core-workflows.md)
- [7. Source Tree](./architecture/7-source-tree.md)
- [8. Infrastructure and Deployment](./architecture/8-infrastructure-and-deployment.md)
- [Appendix A: 데이터 파이프라인 기술 가이드](./architecture/appendix-a-데이터-파이프라인-기술-가이드.md)
- [Appendix B: Distillm-2의 핵심 컴포넌트 원본코드](./architecture/appendix-b-distillm-2의-핵심-컴포넌트-원본코드.md)

### 📋 Pipeline (docs/pipeline-auto/)
- **완전 자동 분할됨**: 14개 파일
- [📑 index.md](./pipeline-auto/index.md) - 상세 목차
- Epic 1: [1.1](./pipeline-auto/11-프로젝트-초기화.md) | [1.2](./pipeline-auto/12-데이터-수집-및-전처리.md) | [1.3](./pipeline-auto/13-rag-청킹-및-임베딩.md) | [1.4](./pipeline-auto/14-rag-지식-베이스-구축.md) | [1.5](./pipeline-auto/15-학습-데이터-준비.md) ✅
- Epic 2: [2.1](./pipeline-auto/21-응답-생성-logits-generation.md)
- Epic 3: [3.1](./pipeline-auto/31-최종-훈련-distill-m-2.md) | [3.2](./pipeline-auto/32-추론-파이프라인-구축.md) ⭐ **Updated 08-23** | [3.3](./pipeline-auto/33-예측-및-제출.md) | [3.4](./pipeline-auto/34-최종화-finalization.md)
- [📋 체크리스트](./pipeline-auto/단계별-완료-기준-체크리스트.md)

### 🏆 Competition Info (docs/competition-info-auto/)
- **완전 자동 분할됨**: 5개 파일
- [📑 index.md](./competition-info-auto/index.md) - 상세 목차
- [1. 프로젝트 목표](./competition-info-auto/1-프로젝트-목표.md)
- [2. 절대 규칙](./competition-info-auto/2-절대-규칙-critical-rules.md) ⭐
- [3. 과제 명세](./competition-info-auto/3-과제-명세-task-specification-상세-버전.md)
- [4. 제출 규칙](./competition-info-auto/4-제출-규칙-상세-가이드라인.md)

### 📄 Requirements Definition (docs/requirements-definition/)
- [📑 index.md](./requirements-definition/index.md) - PDF 문서 참조 가이드

## 📦 구현 진행 상황

### ✅ 완료된 작업
- **Epic 1 (데이터 파이프라인)**: 100% 완료
  - 73개 PDF → 8,756개 청크 (2,300자 단위)
  - BM25 + FAISS 인덱스 구축
  - 3,000개 합성 Q&A 생성

- **Epic 3 (추론 파이프라인)**: 100% 완료
  - Question Classifier (객관식/주관식)
  - Multi-Stage Retriever (Combined Top-3)
  - 8-bit 양자화 추론
  - 성능 검증 완료

### ⏳ 진행 예정
- **Epic 2 (Teacher-Student Distillation)**: 계획 단계
  - Qwen2.5-14B Teacher 모델
  - Qwen2.5-1.5B Student 모델
  - Distill-M 2 학습

## 🔄 문서 업데이트 규칙

1. **Architecture**: 컴포넌트 인터페이스 변경 시 업데이트
2. **Pipeline**: 작업 완료 시 체크리스트 업데이트  
3. **Competition**: 대회 규칙 변경 시 업데이트

## ⚙️ 자동화 설정

### 🛠️ Markdown Tree Parser 설치됨
- 전역 설치: `@kayvan/markdown-tree-parser`
- 자동 분할: `md-tree explode <source> <target>`
- BMad 통합: `.bmad-core/core-config.yaml`에서 `markdownExploder: true`

### 📖 참조 방법

**즉시 참조 파일** (`.bmad-core/core-config.yaml`에 설정됨):
- 🔧 [기술 스택](./architecture/2-tech-stack.md)
- 🧩 [컴포넌트 인터페이스](./architecture/4-components-interface-definitions.md) 
- 📁 [소스 트리](./architecture/7-source-tree.md)
- ⚖️ [대회 규칙](./competition-info-auto/2-절대-규칙-critical-rules.md)
- 📊 [데이터 증강 규칙](./pipeline-auto/15-학습-데이터-준비.md)

## 🎯 핵심 원칙

- **Architecture 우선**: 모든 구현은 Architecture.md 기준
- **Pipeline 준수**: 완료 기준 100% 충족
- **Competition 준수**: 대회 규칙 완전 준수
- **자동화 활용**: 문서 변경 시 자동 분할 도구 사용

## 🚀 프로젝트 진행 현황

### ✅ 완료된 작업 (2025-08-23 기준)
**Epic 1: 데이터 파이프라인**
- ✅ 73개 문서 → 8,756개 청크 변환
- ✅ BM25 + FAISS 하이브리드 검색 시스템 구축
- ✅ 3,000개 합성 질문-답변 쌍 생성

**Epic 3: 추론 파이프라인**
- ✅ 8-bit 양자화 Qwen2.5-7B-Instruct 추론 시스템
- ✅ Question Classifier 구현
- ✅ Multi-Stage Retriever 구현
- ✅ 이미지 생성 방지 및 한국어 최적화

### ⏳ 진행 예정 작업
**Epic 2: Teacher-Student Distillation**
1. Teacher 모델 답변 생성 (3,000개 질문)
2. Distill-M 2 학습 파이프라인 구축
3. Student 모델 (Qwen2.5-1.5B) 학습

### 📊 주요 성과
- 메모리 사용량 75% 감소 (8-bit 양자화)
- RTX 4090 24GB 제약 충족
- 처리 속도: 2-3 문제/초
- 3,000개 고품질 학습 데이터 확보