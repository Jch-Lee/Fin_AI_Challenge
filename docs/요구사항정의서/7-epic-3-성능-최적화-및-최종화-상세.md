# 7. Epic 3: 성능 최적화 및 최종화 (상세)

## **Story 3.1: `Distill-M 2 최종 훈련 구현`**

- **As a** machine learning engineer, **I want** to 생성된 교사-학생 응답 쌍을 입력받아 DistiLLM 훈련 형식으로 재포맷하고, `DistiLLMTrainer`를 사용하여 최종 모델을 훈련하는 스크립트를 구현하여, **so that** 대조적 증류 방식을 통해 학생 모델의 성능을 교사 모델 수준으로 끌어올릴 수 있다.
- **Acceptance Criteria:**
    1. 스크립트는 Story 2.1에서 생성된 교사-학생 응답 쌍 데이터를 로드한다.
    2. 데이터를 DistiLLM 훈련에 맞는 형식으로 재포맷한다.
    3. `DistiLLMTrainer`와 `distillm_v2` 손실 함수를 사용하여 모델 훈련을 실행한다.
    4. 최종 훈련된 모델의 LoRA 가중치가 `/models` 디렉토리에 저장된다.
    5. Distill-M 2의 주요 하이퍼파라미터(alpha, beta, temperature) 튜닝을 위한 실험 로직이 포함된다.

## **Story 3.2: `LLM-as-Reranker 구현 및 통합`**

- **As a** machine learning engineer, **I want** to 1차 검색 결과의 순위를 **'단일 학생 LLM'** 을 활용하여 재평가하고 순위를 재조정하는 `LLM-as-Reranker` 컴포넌트를 구현하고 통합하여, **so that** 최종 LLM에 전달되는 컨텍스트의 품질을 극대화하고 '단일 LLM' 규칙을 완벽하게 준수할 수 있다.
- **Acceptance Criteria:**
    1. `Multi-Stage Retriever`가 1차 검색 결과를 학생 LLM에 전달하도록 수정된다.
    2. 학생 LLM이 후보 문서들의 순위를 재조정하는 프롬프트 기반 로직이 구현된다.
    3. 재조정된 최종 컨텍스트가 `Inference Orchestrator`로 전달된다.

## **Story 3.3: `End-to-End 추론 및 제출 파일 생성`**

- **As a** developer, **I want** to 최종 훈련 및 Reranker가 통합된 모델과 RAG 시스템을 통해 `test.csv` 전체를 처리하고 `submission.csv` 파일을 생성하여, **so that** 대회에 제출할 수 있는 완전한 결과물을 만들어낼 수 있다.
- **Acceptance Criteria:**
    1. 스크립트는 `test.csv` 파일을 입력으로 읽고 모든 행을 순회한다.
    2. 각 질문에 대해 Reranker가 포함된 RAG 검색, 프롬프트 구성, 모델 추론을 수행한다.
    3. 결과를 취합하여 규칙에 맞는 `submission.csv` 파일을 생성한다.
    4. 총 실행 시간을 측정하고 기록한다.

## **Story 3.4: `최종 양자화 및 성능 튜닝`**

- **As a** machine learning engineer, **I want** to 최종 훈련된 모델에 양자화 전략을 적용하고 미세 조정하여, **so that** 모델의 점수 하락을 최소화하면서 추론 속도와 리소스 사용량을 최적화할 수 있다.
- **Acceptance Criteria:**
    1. 다양한 양자화 수준을 모델에 적용하고 테스트한다.
    2. 각 수준별 최종 점수와 추론 시간을 측정하고 기록한다.
    3. 4.5시간 제한을 안정적으로 충족하면서 가장 높은 점수를 내는 양자화 전략을 최종 선택한다.
    4. 최종 양자화된 모델 가중치가 제출을 위해 저장된다.

## **Story 3.5: `최종 제출 패키지 구성 및 검증`**

- **As a** developer, **I want** to 대회의 모든 규칙에 따라 최종 제출 패키지를 구성하고, 재현성 검증을 완료하여, **so that** 우리의 제출물이 유효하며 심사위원들이 성공적으로 우리의 점수를 복원할 수 있도록 보장할 수 있다.
- **Acceptance Criteria:**
    1. 제출에 필요한 모든 파일(코드, 모델, `requirements.txt`, 증빙 자료, 보고서)을 포함한 최종 디렉토리가 구성된다.
    2. 모든 코드 내 파일 경로가 상대 경로인지 최종 확인된다.
    3. 깨끗한 환경에서 재현성을 검증하는 스크립트가 포함된다.
    4. 최종 결과 보고서가 작성되고 패키지에 포함된다.