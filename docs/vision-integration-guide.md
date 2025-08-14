# Vision-Language 모델 통합 가이드

## 📚 개요

본 가이드는 Qwen2.5-VL-7B Vision-Language 모델을 금융 보안 AI 시스템에 통합하는 방법을 설명합니다. 
VL 모델을 활용하여 PDF 문서의 이미지, 차트, 테이블 등 시각적 콘텐츠를 텍스트로 변환하여 RAG 시스템의 검색 품질을 향상시킵니다.

## 🎯 목표

- **정보 추출률 향상**: PyMuPDF 대비 49.3% 더 많은 정보 추출
- **RAG 검색 품질 개선**: 시각적 콘텐츠 포함 쿼리에서 최대 76.9% 성능 향상
- **오프라인 추론 지원**: 지식베이스 사전 구축으로 온라인 의존성 제거

## 🏗️ 아키텍처

```
PDF 문서
    ↓
[PDF 프로세서]
    ├── PyMuPDF: 텍스트 추출
    └── Qwen2.5-VL: 이미지/차트/테이블 분석
         ↓
[텍스트 통합]
    ↓
[임베딩 생성]
    ↓
[FAISS 인덱스]
    ↓
[RAG 검색]
```

## 📦 설치

### 필수 패키지

```bash
pip install transformers==4.41.2
pip install torch==2.1.0
pip install accelerate==0.30.1
pip install bitsandbytes==0.42.0
pip install pymupdf==1.24.0
pip install Pillow==10.0.0
pip install numpy==1.24.3
```

### Qwen2.5-VL 모델 설치

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 모델 다운로드 (최초 1회)
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)
```

## 💻 사용법

### 1. 기본 사용

```python
from packages.vision import QwenVLProcessor

# VL 프로세서 초기화
vl_processor = QwenVLProcessor(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    quantization="4-bit",  # 메모리 최적화
    batch_size=4,
    device="cuda"
)

# 이미지 처리
image_path = "path/to/image.png"
extracted_text = vl_processor.process_image(
    image_path,
    prompt_type="chart"  # general, chart, table, diagram 중 선택
)

print(extracted_text)
```

### 2. PDF 문서 처리

```python
from packages.vision import VisionKnowledgeBuilder

# 지식베이스 빌더 초기화
kb_builder = VisionKnowledgeBuilder(
    vl_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    output_dir="data/knowledge_base"
)

# PDF 처리 및 지식베이스 구축
pdf_path = "docs/financial_report.pdf"
knowledge_base = kb_builder.build_from_pdf(
    pdf_path,
    selective_processing=True,  # 시각적 콘텐츠가 있는 페이지만 VL 처리
    cache_enabled=True
)

# 결과 저장
kb_builder.save_knowledge_base(knowledge_base, "financial_kb.json")
```

### 3. 프롬프트 템플릿

다양한 시각적 콘텐츠 유형에 최적화된 프롬프트:

```python
# 차트/그래프 전용
CHART_PROMPT = """이 차트/그래프의 모든 데이터를 추출하세요.
- X축과 Y축의 모든 레이블과 값
- 모든 데이터 포인트의 수치
- 범례에 있는 모든 텍스트
- 제목과 부제목
형태 설명 없이 데이터만 나열하세요."""

# 테이블 전용
TABLE_PROMPT = """이 테이블의 모든 셀 내용을 순서대로 읽어주세요.
- 첫 행(헤더)부터 시작
- 각 행의 모든 셀 내용을 순서대로
- 빈 셀은 '빈칸'으로 표시
- 병합된 셀은 한 번만 읽기
구조 설명 없이 내용만 추출하세요."""

# 금융 문서 특화
FINANCIAL_PROMPT = """이 금융 문서 이미지의 모든 정보를 추출하세요.
특히 다음 항목에 주의하세요:
- 모든 수치 데이터 (금액, 비율, 날짜)
- 계정 번호나 코드
- 거래 내역
- 규제 관련 텍스트
정확성이 중요하므로 모든 숫자를 정확히 읽어주세요."""
```

## ⚙️ 최적화 설정

### 메모리 제약 환경 (24GB VRAM)

```python
config = {
    "quantization": "4-bit",
    "batch_size": 4,
    "selective_processing": True,
    "caching": True,
    "parallel_workers": 4
}
```

### 성능별 권장 설정

| 환경 | 양자화 | 배치크기 | GPU메모리 | 처리속도 | 정확도 |
|------|--------|---------|-----------|----------|--------|
| 메모리 제약 | 4-bit | 4 | 3.5GB | 2-3 img/s | 93% |
| 품질 우선 | 8-bit | 2 | 7GB | 1-2 img/s | 94% |
| 속도 우선 | 4-bit | 8 | 7GB | 4-5 img/s | 93% |

## 🚀 고급 기능

### 1. 선택적 처리

시각적 콘텐츠가 있는 페이지만 VL 모델로 처리:

```python
def should_use_vl(page_text: str, page_num: int) -> bool:
    """VL 처리 필요 여부 판단"""
    visual_hints = ["그림", "표", "차트", "그래프", "Figure", "Table"]
    has_visual = any(hint in page_text for hint in visual_hints)
    is_important = page_num < 5  # 첫 몇 페이지는 항상 처리
    return has_visual or is_important
```

### 2. 병렬 처리

대량 문서 처리를 위한 병렬화:

```python
from concurrent.futures import ThreadPoolExecutor

def process_pages_parallel(pages, num_workers=4):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(vl_processor.process_image, pages)
    return list(results)
```

### 3. 캐싱 전략

중복 처리 방지를 위한 캐싱:

```python
import hashlib
import json
from pathlib import Path

def get_cached_result(image_hash: str, cache_dir: Path):
    cache_file = cache_dir / f"{image_hash}.json"
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_to_cache(image_hash: str, result: str, cache_dir: Path):
    cache_file = cache_dir / f"{image_hash}.json"
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({"content": result}, f, ensure_ascii=False)
```

## 📊 성능 벤치마크

### 실험 환경
- **문서**: 금융분야 AI 보안 가이드라인.pdf (56페이지)
- **GPU**: RTX 4090 (24GB VRAM)
- **모델**: Qwen2.5-VL-7B-Instruct (4-bit 양자화)

### 결과

| 메트릭 | PyMuPDF | VL 모델 | 개선율 |
|--------|---------|---------|--------|
| 추출 문자 수 | 32,140 | 47,999 | +49.3% |
| 토큰 수 | 15,647 | 22,456 | +43.5% |
| 청크 수 | 79 | 140 | +77.2% |
| RAG 검색 품질 | 0.79 | 0.86 | +15.4% |

### 시각적 콘텐츠별 성능

| 콘텐츠 유형 | 검색 개선율 | 주요 이점 |
|------------|------------|-----------|
| 차트/그래프 | +76.9% | 데이터 포인트 추출 |
| 테이블 | +45.2% | 구조적 정보 보존 |
| 다이어그램 | +38.5% | 관계 정보 추출 |
| 수식 | +62.3% | 변수/상수 인식 |

## 🔧 트러블슈팅

### 1. CUDA Out of Memory

```python
# 배치 크기 감소
config["batch_size"] = 2

# 더 공격적인 양자화
config["quantization"] = "3-bit"

# 캐시 정리
torch.cuda.empty_cache()
```

### 2. 느린 처리 속도

```python
# 선택적 처리 활성화
config["selective_processing"] = True

# 병렬 처리 증가
config["parallel_workers"] = 8

# 캐싱 활성화
config["caching"] = True
```

### 3. 낮은 정확도

```python
# 양자화 수준 완화
config["quantization"] = "8-bit"

# Temperature 조정
generation_config["temperature"] = 0.1

# 프롬프트 개선
prompt = FINANCIAL_PROMPT  # 도메인 특화 프롬프트 사용
```

## 📝 베스트 프랙티스

1. **사전 처리**: 오프라인에서 모든 VL 처리를 완료하고 인덱스만 사용
2. **선택적 처리**: 시각적 콘텐츠가 없는 페이지는 PyMuPDF만 사용
3. **캐싱 활용**: 동일한 이미지 반복 처리 방지
4. **배치 처리**: 여러 이미지를 한 번에 처리하여 효율성 향상
5. **프롬프트 최적화**: 콘텐츠 유형별 전용 프롬프트 사용

## 🤝 기여하기

Vision 통합 개선을 위한 제안이나 버그 리포트는 GitHub Issues에 등록해주세요.

## 📄 라이센스

본 프로젝트는 금융 보안 AI 대회용으로 개발되었으며, 비상업적 목적으로만 사용 가능합니다.