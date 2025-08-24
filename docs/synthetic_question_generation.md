# Synthetic Question Generation Documentation

## 최종 선택 방법론

### 모델 및 설정
- **모델**: Qwen2.5-14B-Instruct (16-bit quantization)
- **방법**: 단일 질문 생성 방식 (한 번에 하나씩)
- **GPU**: RTX 5090 (33.7GB VRAM)

### 최적화된 파라미터
```python
temperature = 0.3      # 낮은 온도로 일관성 향상
max_new_tokens = 256   # 충분한 길이 확보
top_p = 0.9           # 적절한 다양성 유지
repetition_penalty = 1.2
```

### 불용어 처리 정책
- **완화된 필터링**: 특수문자와 명백한 메타텍스트만 제거
- **유지 항목**: 
  - 물음표 (?)
  - 쉼표 (,)
  - 괄호 ()
- **제거 항목**:
  - 메타 참조: "자료 1", "위 내용", "이 문서" 등
  - 레이블: "질문:", "답변:", "Q:", "A:" 등
  - 특수문자: ###, ---, ***, 등

## 성능 지표

### 최종 실험 결과 (2025-08-21)
- **생성 개수**: 30개 질문
- **성공률**: 85.7% (35번 시도 중 30개 성공)
- **평균 생성 시간**: 10.6초/질문
- **총 소요 시간**: 5분 18초

### 질문 유형 분포
| 유형 | 개수 | 설명 |
|------|------|------|
| definition | 6 | 핵심 개념이나 용어의 정의 |
| process | 6 | 절차, 프로세스, 방법론 |
| regulation | 6 | 법률, 규정, 기준 |
| example | 4 | 구체적 사례나 실제 적용 |
| comparison | 4 | 차이점이나 비교 |
| application | 4 | 활용 방법이나 대응 방안 |

## 파일 구조

### 최종 버전
- **스크립트**: `scripts/generate_14b_relaxed.py`
- **결과**: `data/synthetic_questions/relaxed_14b_questions_20250821_133849.csv`

### 아카이브 구조
```
archive/experiments_20250821/
├── scripts/       # 실험용 스크립트들
├── results/       # 실험 결과 CSV 파일들
└── logs/         # 실행 로그들
```

## 품질 평가

### 우수한 점
- ✅ 100% 한국어로 작성 (외국어 혼재 없음)
- ✅ 메타텍스트 참조 완전 제거
- ✅ 구체적이고 명확한 질문
- ✅ 금융보안 전문 용어 적절히 사용
- ✅ 다양한 문서 소스 활용 (71개 문서)

### 개선 사항
- 일부 질문의 길이 최적화
- 복잡한 문장 구조 단순화

## 사용 방법

### 실행 명령
```bash
python scripts/generate_14b_relaxed.py
```

### 필요 패키지
- torch
- transformers
- accelerate
- pandas
- tqdm

### 입력 데이터
- `data/rag/chunks_2300.json`: 8,756개 청크 (71개 문서)

### 출력 형식
CSV 파일 (UTF-8 with BOM)
- id: 질문 번호 (1-30)
- question: 생성된 질문 텍스트