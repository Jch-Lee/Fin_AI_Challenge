# 4. Components & Interface Definitions

시스템은 단일 책임 원칙(SRP)에 따라 10개의 독립적인 컴포넌트로 구성됩니다.

## 4.1. 컴포넌트 목록
1. **데이터 전처리 컴포넌트** (`DataPreprocessor`)
2. **지식 베이스 컴포넌트** (`KnowledgeBase`)
3. **합성 데이터 생성 컴포넌트** (`SyntheticDataGenerator`)
4. **모델 파인튜닝 컴포넌트** (`ModelTrainer`)
5. **추론 오케스트레이터** (`InferenceOrchestrator`)
6. **최적화 컴포넌트** (`ModelOptimizer`)
7. **평가 및 모니터링 컴포넌트** (`EvaluationMonitor`)
8. **캐싱 컴포넌트** (`CacheLayer`)
9. **질문 분류 컴포넌트** (`QuestionClassifier`)
10. **다단계 검색 컴포넌트** (`MultiStageRetriever`)

## 4.2. 컴포넌트 간 인터페이스 정의

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
