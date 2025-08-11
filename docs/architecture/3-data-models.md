# 3. Data Models

시스템의 각 컴포넌트가 주고받는 데이터의 형식을 명확히 정의합니다.

```python
from typing import TypedDict, List, Optional, Dict

class DocumentChunk_v2(TypedDict):
    chunk_id: str
    content: str
    source: str
    domain: str
    embedding: Optional[List[float]]
    quality_score: float

class SyntheticQAPair_v2(TypedDict):
    qa_id: str
    question: str
    answer: str
    context_chunk_ids: List[str]
    teacher_model: str
    overall_quality_score: float
    difficulty_level: str
    student_loss: Optional[float]

class CompetitionQuestion_v2(TypedDict):
    ID: str
    Question: str
    question_type: str
    parsed_question: str
    choices: Optional[Dict[str, str]]

class SubmissionRow_v2(TypedDict):
    ID: str
    answer: str
    confidence: float
    inference_time: float
    used_fallback: bool
```

---
