# 6. Core Workflows

## Workflow 1: 모델 개발

```python
sequenceDiagram
    participant User
    participant DataPrep as Data Preprocessing
    participant SynGen as Synthetic Data Gen
    participant Eval as Evaluation
    participant FineTune as Model Fine-tuning
    participant Optim as Optimization

    User->>DataPrep: 원본 문서 제공
    DataPrep->>SynGen: DocumentChunk 전달
    
    SynGen->>SynGen: Teacher Model로 생성
    SynGen->>Eval: 품질 평가 요청
    
    alt 품질 < 7.0
        Eval-->>SynGen: 재생성 요청
    else 품질 >= 7.0
        Eval-->>FineTune: 승인된 데이터셋
    end
    
    loop Distill-M 2 Iterations
        FineTune->>FineTune: 재학습
    end
    
    FineTune->>Optim: 최종 모델 전달
    Optim->>Optim: 4-bit Quantization
    Optim-->>User: 경량화된 최종 모델
```

## Workflow 2: RAG 추론

```python
sequenceDiagram
    participant Runner as Competition Runner
    participant Orchestrator as Inference Orchestrator
    participant Cache as Cache Layer
    participant Retriever as Multi-Stage Retriever
    participant Model as Fine-tuned Model
    participant Fallback as Fallback Handler

    Runner->>Orchestrator: Batch[CompetitionQuestion]
    
    loop For each question
        Orchestrator->>Cache: 쿼리
        alt Cache Miss
            Cache->>Retriever: 검색 요청
            Retriever-->>Cache: 컨텍스트
            Cache->>Cache: 저장
        end
        
        Orchestrator->>Model: Generate with timeout
        
        alt 정상 처리
            Model-->>Orchestrator: 답변
        else Timeout or Low Confidence
            Orchestrator->>Fallback: 긴급 처리
            Fallback-->>Orchestrator: 대체 답변
        end
    end
    
    Orchestrator-->>Runner: List[SubmissionRow]
```

---
