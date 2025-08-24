# Vision Processing Module

*Last Updated: 2025-08-23*

## 📌 현재 상태

**이 모듈은 벡터DB 구축 과정에서 활용되었습니다.**

- **사용 시점**: RAG 시스템의 벡터DB 구축 단계
- **역할**: PDF 문서 내 이미지, 차트, 테이블을 텍스트로 변환
- **기여**: 8,756개 청크 생성 시 이미지 컨텐츠를 텍스트화하여 검색 가능하게 변환
- **현재**: 추론 파이프라인에서는 직접 사용되지 않음 (이미 변환된 텍스트 청크 사용)

---

Qwen2.5-VL을 활용한 이미지-텍스트 변환 모듈로, 금융 문서의 이미지, 차트, 테이블을 텍스트로 변환하여 RAG 시스템의 검색 성능을 향상시킵니다.

## 구성 요소

### 1. BaseVisionProcessor
- Vision 프로세서의 추상 베이스 클래스
- 모든 Vision 프로세서가 구현해야 하는 인터페이스 정의

### 2. QwenVisionProcessor
- Qwen/Qwen2.5-VL-7B-Instruct 모델 기반 Vision 프로세서
- 4-bit 양자화로 메모리 최적화
- 배치 처리 지원
- 금융 문서 특화 프롬프트 활용

### 3. PDFImageExtractor
- PDF 문서에서 이미지를 추출하는 클래스
- 임베디드 이미지와 렌더링된 이미지 모두 지원
- 이미지 품질 향상 기능 내장
- 주변 텍스트 컨텍스트 추출

### 4. ImagePreprocessor
- 이미지 전처리 클래스
- 크기 조정, 품질 개선, 포맷 변환
- 종횡비 유지 옵션

### 5. VisionProcessingPipeline
- 통합 처리 파이프라인
- PDF → 이미지 추출 → 전처리 → Vision 모델 → 텍스트 설명
- 배치 처리 및 병렬 처리 지원
- 결과 캐싱 기능

### 6. AsyncVisionPipeline
- 비동기 처리 파이프라인
- 다중 PDF 동시 처리
- 높은 처리량 달성

## 사용법

### 기본 사용법

```python
from packages.vision import create_vision_processor, create_pipeline

# Vision 프로세서 생성
processor = create_vision_processor({
    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
    "quantization": True,
    "max_tokens": 512
})

# 이미지 처리
from PIL import Image
image = Image.open("financial_chart.png")
description = processor.process_image(image, prompt_type="chart")
print(description)
```

### 파이프라인 사용법

```python
from packages.vision import create_pipeline

# 파이프라인 생성
pipeline = create_pipeline({
    "vision_model": {
        "quantization": True,
        "max_tokens": 512
    },
    "batch_size": 4
})

# PDF 문서 처리
result = pipeline.process_pdf_document(
    "financial_report.pdf",
    output_dir="output/"
)

print(f"추출된 이미지: {result['total_images']}개")
for item in result['results']:
    print(f"페이지 {item['page']}: {item['description']}")
```

### 지식베이스 구축

```python
# 스크립트 실행
python scripts/build_vision_knowledge_base.py \
    --input-dir ./data/financial_docs \
    --output-dir ./knowledge_base/vision \
    --batch-size 4 \
    --test
```

## 프롬프트 템플릿

### 금융 문서 특화 프롬프트
- `general`: 일반적인 이미지 설명
- `chart`: 차트 및 그래프 분석
- `table`: 테이블 구조 및 데이터 추출
- `diagram`: 다이어그램 및 플로차트 설명
- `text`: OCR 텍스트 추출
- `security`: 보안 가이드라인 분석
- `formula`: 수식 및 계산 설명

### 한국어 최적화
- 한국어 응답 품질 향상
- 금융 용어 및 표현 최적화
- 문화적 맥락 고려

## 성능 최적화

### 메모리 최적화
- 4-bit 양자화 (BitsAndBytesConfig)
- 배치 처리로 메모리 효율성 향상
- 자동 GPU 메모리 정리

### 처리 속도 최적화
- 배치 처리 지원
- 병렬 워커 활용
- 결과 캐싱

### 품질 최적화
- 이미지 전처리 및 향상
- 컨텍스트 기반 프롬프트
- 후처리를 통한 결과 정제

## 설정 옵션

### Vision 모델 설정
```python
{
    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
    "quantization": True,  # 4-bit 양자화 사용
    "max_tokens": 512,     # 최대 생성 토큰 수
    "temperature": 0.3,    # 생성 온도
    "cache_dir": "./models/vision_cache"
}
```

### 이미지 추출 설정
```python
{
    "min_width": 100,      # 최소 이미지 너비
    "min_height": 100,     # 최소 이미지 높이
    "enhance_quality": True, # 이미지 품질 향상
    "extract_embedded": True,  # 임베디드 이미지 추출
    "extract_rendered": True   # 렌더링된 이미지 추출
}
```

### 파이프라인 설정
```python
{
    "batch_size": 4,       # 배치 크기
    "max_workers": 2,      # 워커 수
    "cache_enabled": True, # 캐싱 활성화
    "output_format": "json" # 출력 형식
}
```

## 테스트

### 단위 테스트
```bash
python tests/test_vision_integration.py TestQwenVisionProcessor
python tests/test_vision_integration.py TestImageExtractor
python tests/test_vision_integration.py TestVisionPipeline
```

### 통합 테스트
```bash
python tests/test_vision_integration.py --system
```

### 설정 테스트
```bash
python scripts/build_vision_knowledge_base.py --test-setup
```

## 요구사항

### 필수 패키지
- torch>=2.1.0
- transformers>=4.51.0
- Pillow>=10.0.0
- PyMuPDF>=1.24.1
- numpy
- bitsandbytes>=0.43.1

### 선택적 패키지
- qwen-vl-utils (Qwen Vision 최적화)
- accelerate (모델 로딩 가속화)

### 하드웨어 요구사항
- GPU: 8GB+ VRAM 권장 (4-bit 양자화 사용 시)
- RAM: 16GB+ 권장
- 디스크: 모델 저장용 20GB+ 여유 공간

## 라이선스

이 프로젝트는 Qwen 모델의 라이선스를 따릅니다.

## 기여

버그 리포트와 기능 제안은 이슈를 통해 해주세요.