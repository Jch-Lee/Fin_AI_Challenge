"""
Korean Language Specific Prompts

한국어 문서 처리를 위한 특화 프롬프트
"""

KOREAN_PROMPTS = {
    "ocr_korean": """
이미지의 한국어 텍스트를 정확히 읽어주세요:
1. 한글 텍스트를 정확히 OCR
2. 한자가 있다면 한글 독음과 함께 표기
3. 영문 약어는 원문 그대로 유지
4. 띄어쓰기와 줄바꿈 유지
5. 특수 기호나 문장 부호 포함

주의사항:
- 오타나 철자 오류 수정하지 말 것
- 원문 그대로 추출할 것
""",
    
    "mixed_language": """
이 이미지에는 한국어와 영어가 혼재되어 있습니다.
다음과 같이 처리해주세요:
1. 한국어는 정확히 읽기
2. 영어 용어는 원문 유지
3. 약어는 풀어서 설명 (예: AI → Artificial Intelligence, 인공지능)
4. 기술 용어는 한/영 병기
""",
    
    "korean_table": """
한국어 테이블을 분석해주세요:
1. 열 제목을 한국어로 정확히 읽기
2. 숫자는 천 단위 구분 (예: 1,000,000원)
3. 단위 명시 (원, 개, %, 건 등)
4. 빈 셀은 '-'로 표시
5. 병합된 셀 구조 설명

마크다운 테이블로 표현:
| 항목 | 값 | 단위 |
|------|-----|------|
""",
    
    "korean_financial_terms": """
이 이미지의 한국 금융 용어를 중점적으로 추출해주세요:
1. 금융 전문 용어 식별
2. 약어는 풀어서 설명
3. 영문 병기된 경우 함께 표시
4. 법규나 규정 명칭 정확히 표기
5. 기관명은 전체 명칭으로

예시:
- 금감원 → 금융감독원
- KYC → Know Your Customer (고객확인제도)
""",
    
    "korean_compliance": """
한국 금융 규제 관련 내용을 분석해주세요:
1. 규제 기관명 (금융위원회, 금융감독원 등)
2. 법령이나 규정 명칭
3. 조항 번호와 내용
4. 벌칙이나 제재 사항
5. 시행일이나 유예 기간

규제 준수 관점에서 중요한 사항을 강조해주세요.
""",
    
    "korean_form": """
한국어 양식/서식을 분석해주세요:
1. 양식 제목과 문서 번호
2. 필수 입력 항목 (★, * 표시)
3. 각 필드의 레이블과 입력 형식
4. 체크박스나 선택 항목
5. 서명란이나 날인 위치
6. 유의사항이나 안내 문구

양식의 구조를 체계적으로 설명해주세요.
"""
}

def get_korean_prompt(prompt_type: str, additional_info: str = "") -> str:
    """
    한국어 처리에 특화된 프롬프트 생성
    
    Args:
        prompt_type: 프롬프트 유형
        additional_info: 추가 정보
        
    Returns:
        완성된 한국어 프롬프트
    """
    base_prompt = KOREAN_PROMPTS.get(prompt_type, KOREAN_PROMPTS["ocr_korean"])
    
    if additional_info:
        return f"{base_prompt}\n\n추가 정보: {additional_info}"
    
    return base_prompt

def create_bilingual_prompt(korean_prompt: str, english_context: str = "") -> str:
    """
    한영 혼용 프롬프트 생성
    
    Args:
        korean_prompt: 한국어 프롬프트
        english_context: 영어 컨텍스트
        
    Returns:
        한영 혼용 프롬프트
    """
    if english_context:
        return f"""
{korean_prompt}

Additional Context (English):
{english_context}

응답은 한국어로 작성하되, 영어 용어는 원문 그대로 사용해주세요.
"""
    return korean_prompt