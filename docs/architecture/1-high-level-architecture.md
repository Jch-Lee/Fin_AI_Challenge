# 1. High Level Architecture

## High Level Project Diagram (v3)

'Distill-M 2'의 대조적 증류 파이프라인을 반영하여, 학습 단계를 **'응답 생성 → 데이터 재포맷 → 최종 훈련'** 의 명확한 단일 흐름으로 구성합니다.

```python
graph TD
    subgraph "Phase 1: 학습 과정 (Unconstrained Environment)"
        A[External Data] --> B(Data Processing);
        B --> C[FAISS Index Builder];
        B --> D[Fine-tuning Dataset];
        
        %% Distill-M 2 Workflow
        D --> E{Response Generation};
        E -- Teacher Model --> E;
        E -- Student Model --> E;
        E --> F[Reformat Data];
        F --> G(DistiLLM-2 Training);
        
        %% Validation
        G --> V[Validation];
        V -->|Score Check| G;
    end

    subgraph "Phase 2: 추론 과정 (Competition Environment)"
        H(Quantized Student Model) --> J[Inference Engine];
        I[FAISS Index] --> J;
        K(test.csv) --> J;
        J --> L(submission.csv);
    end

    G --> Q[Quantization];
    Q --> H;
    C --> I;
```

## Architectural and Design Patterns

- **`Contrastive Distillation (Distill-M 2)`**: 교사-학생 모델의 응답을 모두 생성하고, 그 차이를 학습하여 학생 모델의 성능을 극대화하는 핵심 전략입니다.
- **`Retrieval-Augmented Generation (RAG)`**: 외부 지식 베이스를 활용하여 답변의 사실 근거를 마련하는 핵심 패턴입니다.
- **`Modular Pipeline`**: 개발 및 테스트 용이성을 위한 모듈형 스크립트 구조.
- **`Monorepo`**: 재현성과 관리 용이성을 위한 단일 저장소 구조.
- **`Adapter Pattern (LoRA)`**: PEFT 라이브러리를 활용한 효율적인 파인튜닝.
- **`Circuit Breaker`**: 개별 질문 처리 시간 초과 방지를 통해 전체 추론 시간 준수.
- **`Cache-Aside`**: 임베딩 및 검색 결과 캐싱으로 반복 연산 감소 및 속도 향상.

---
