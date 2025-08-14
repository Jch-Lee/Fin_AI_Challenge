#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vision-Language 모델에 전달할 프롬프트 예제
실제 Qwen2.5-VL 모델 사용 시 적용할 프롬프트 템플릿
"""

# ============================================================================
# VL 모델 프롬프트 템플릿
# ============================================================================

# 1. 범용 콘텐츠 추출 프롬프트
GENERAL_EXTRACTION_PROMPT = """이 이미지의 모든 텍스트와 데이터를 추출하세요.
형태나 구조 설명은 하지 말고, 오직 내용만 추출하세요.
한국어와 영어가 섞여있다면 원문 그대로 추출하세요.
숫자는 정확하게 읽어주세요."""

# 2. 차트/그래프 전용 프롬프트
CHART_EXTRACTION_PROMPT = """이 차트/그래프의 모든 데이터를 추출하세요.
- X축과 Y축의 모든 레이블과 값
- 모든 데이터 포인트의 수치
- 범례에 있는 모든 텍스트
- 제목과 부제목
형태 설명 없이 데이터만 나열하세요."""

# 3. 테이블 전용 프롬프트
TABLE_EXTRACTION_PROMPT = """이 테이블의 모든 셀 내용을 순서대로 읽어주세요.
- 첫 행(헤더)부터 시작
- 각 행의 모든 셀 내용을 순서대로
- 빈 셀은 '빈칸'으로 표시
- 병합된 셀은 한 번만 읽기
구조 설명 없이 내용만 추출하세요."""

# 4. 다이어그램/플로차트 전용 프롬프트
DIAGRAM_EXTRACTION_PROMPT = """이 다이어그램의 모든 텍스트를 추출하세요.
- 모든 박스/노드 안의 텍스트
- 화살표나 연결선 위의 레이블
- 범례나 설명 텍스트
- 제목과 캡션
연결 관계는 '→' 기호로 표현하세요."""

# 5. 수식/공식 전용 프롬프트
FORMULA_EXTRACTION_PROMPT = """이 수식/공식을 텍스트로 변환하세요.
- 모든 변수와 상수
- 연산자와 함수
- 위첨자와 아래첨자
- 특수 기호
LaTeX 형식이 아닌 일반 텍스트로 표현하세요."""

# 6. 금융 문서 특화 프롬프트
FINANCIAL_DOC_PROMPT = """이 금융 문서 이미지의 모든 정보를 추출하세요.
특히 다음 항목에 주의하세요:
- 모든 수치 데이터 (금액, 비율, 날짜)
- 계정 번호나 코드
- 거래 내역
- 규제 관련 텍스트
정확성이 중요하므로 모든 숫자를 정확히 읽어주세요."""

# ============================================================================
# 실제 사용 예제 (Qwen2.5-VL 모델)
# ============================================================================

def extract_with_vl_model(image_path: str, prompt_type: str = "general"):
    """
    실제 VL 모델을 사용한 콘텐츠 추출 예제
    
    Args:
        image_path: 이미지 파일 경로
        prompt_type: 프롬프트 유형 (general, chart, table, diagram, formula, financial)
    """
    
    # 프롬프트 선택
    prompts = {
        "general": GENERAL_EXTRACTION_PROMPT,
        "chart": CHART_EXTRACTION_PROMPT,
        "table": TABLE_EXTRACTION_PROMPT,
        "diagram": DIAGRAM_EXTRACTION_PROMPT,
        "formula": FORMULA_EXTRACTION_PROMPT,
        "financial": FINANCIAL_DOC_PROMPT
    }
    
    selected_prompt = prompts.get(prompt_type, GENERAL_EXTRACTION_PROMPT)
    
    # 실제 모델 사용 코드 (의사코드)
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    
    # 모델 로드
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    # 이미지 로드
    image = Image.open(image_path)
    
    # 메시지 구성
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": selected_prompt}
            ]
        }
    ]
    
    # 프로세싱
    text = processor.apply_chat_template(messages, tokenize=False)
    inputs = processor(text, images=image, return_tensors="pt").to(model.device)
    
    # 생성
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.1,  # 낮은 temperature로 정확성 향상
        do_sample=True
    )
    
    # 디코딩
    result = processor.decode(outputs[0], skip_special_tokens=True)
    
    return result
    """
    
    # 시뮬레이션 결과
    return f"[VL 모델이 추출한 내용]\n프롬프트: {selected_prompt[:50]}...\n추출된 텍스트: ..."

# ============================================================================
# 배치 처리 예제
# ============================================================================

def batch_extract_from_pdf(pdf_path: str):
    """
    PDF의 모든 이미지에서 콘텐츠 추출
    """
    import pymupdf
    
    results = []
    doc = pymupdf.open(pdf_path)
    
    for page_num, page in enumerate(doc):
        # 페이지를 이미지로 변환
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        
        # 페이지 텍스트로 콘텐츠 유형 추론
        page_text = page.get_text().lower()
        
        # 적절한 프롬프트 선택
        if "table" in page_text or "표" in page_text:
            prompt_type = "table"
        elif "chart" in page_text or "그래프" in page_text:
            prompt_type = "chart"
        elif "diagram" in page_text or "다이어그램" in page_text:
            prompt_type = "diagram"
        else:
            prompt_type = "general"
        
        # VL 모델로 추출 (실제로는 extract_with_vl_model 사용)
        extracted_content = f"페이지 {page_num + 1} - {prompt_type} 콘텐츠 추출됨"
        
        results.append({
            "page": page_num + 1,
            "prompt_type": prompt_type,
            "content": extracted_content
        })
    
    doc.close()
    return results

# ============================================================================
# 프롬프트 최적화 팁
# ============================================================================

"""
VL 모델 프롬프트 최적화 가이드:

1. 명확하고 구체적인 지시
   - "텍스트를 읽어주세요" (X)
   - "모든 텍스트를 순서대로 정확히 읽어주세요" (O)

2. 출력 형식 지정
   - "데이터를 추출하세요" (X)
   - "각 데이터를 '레이블: 값' 형식으로 추출하세요" (O)

3. 중요 정보 강조
   - 금융 문서: "특히 숫자와 날짜는 정확히"
   - 테이블: "헤더와 데이터 구분하여"
   - 차트: "축 레이블과 데이터 포인트 모두"

4. 언어 처리
   - "한국어와 영어 원문 그대로"
   - "번역하지 말고 있는 그대로"

5. 에러 처리
   - "읽을 수 없는 부분은 [불명확]로 표시"
   - "빈 칸은 '빈칸'으로 표시"

6. Temperature 설정
   - 정확성 중요: 0.1~0.3
   - 창의성 필요: 0.7~0.9
   - 금융 문서는 항상 낮은 temperature 사용
"""

if __name__ == "__main__":
    # 테스트
    print("=== VL 모델 프롬프트 템플릿 ===\n")
    
    print("1. 범용 추출:")
    print(GENERAL_EXTRACTION_PROMPT)
    print("\n2. 차트 추출:")
    print(CHART_EXTRACTION_PROMPT)
    print("\n3. 테이블 추출:")
    print(TABLE_EXTRACTION_PROMPT)
    
    # 시뮬레이션 실행
    print("\n=== 추출 시뮬레이션 ===")
    result = extract_with_vl_model("sample.png", "table")
    print(result)