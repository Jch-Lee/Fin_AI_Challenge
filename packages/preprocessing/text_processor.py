"""
텍스트 전처리 모듈
한국어/영어 혼합 텍스트 정제 및 정규화
"""
import re
from typing import List, Optional
import unicodedata


class KoreanEnglishTextProcessor:
    """한국어/영어 혼합 텍스트 처리 클래스"""
    
    def __init__(self):
        self.korean_pattern = re.compile('[가-힣]+')
        self.english_pattern = re.compile('[a-zA-Z]+')
        self.number_pattern = re.compile(r'\d+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
    def clean_text(self, text: str) -> str:
        """텍스트 정제 메인 함수"""
        if not text:
            return ""
            
        # Unicode 정규화
        text = unicodedata.normalize('NFKC', text)
        
        # URL 제거 (필요시 [URL]로 대체)
        text = self.url_pattern.sub('[URL]', text)
        
        # 이메일 제거 (필요시 [EMAIL]로 대체)
        text = self.email_pattern.sub('[EMAIL]', text)
        
        # 특수문자 정리 (한글, 영어, 숫자, 기본 문장부호만 유지)
        text = re.sub(r'[^\w\s가-힣a-zA-Z0-9.,!?;:\-\(\)\[\]\'\"％＆／]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """공백 정규화"""
        # 탭, 줄바꿈 등을 공백으로 변환
        text = re.sub(r'[\t\n\r\f\v]', ' ', text)
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_html_tags(self, text: str) -> str:
        """HTML 태그 제거"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def extract_korean(self, text: str) -> List[str]:
        """한국어만 추출"""
        return self.korean_pattern.findall(text)
    
    def extract_english(self, text: str) -> List[str]:
        """영어만 추출"""
        return self.english_pattern.findall(text)
    
    def extract_numbers(self, text: str) -> List[str]:
        """숫자만 추출"""
        return self.number_pattern.findall(text)
    
    def split_sentences(self, text: str) -> List[str]:
        """문장 분리"""
        # 한국어와 영어 문장 종결 부호 기준으로 분리
        sentences = re.split(r'[.!?。！？]\s+', text)
        # 빈 문장 제거
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def remove_duplicate_chars(self, text: str, max_repeat: int = 3) -> str:
        """연속된 중복 문자 제거"""
        # ㅋㅋㅋㅋ, ㅎㅎㅎㅎ 같은 반복 제거
        pattern = re.compile(r'(.)\1{' + str(max_repeat) + ',}')
        return pattern.sub(r'\1' * max_repeat, text)
    
    def process_financial_text(self, text: str) -> str:
        """금융 도메인 특화 전처리"""
        # 금융 용어 정규화
        text = re.sub(r'％', '%', text)
        text = re.sub(r'＄', '$', text)
        text = re.sub(r'￦', '원', text)
        
        # 숫자 포맷 정규화 (1,000 -> 1000)
        text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
        
        return text
    
    def process(self, text: str, 
                remove_html: bool = True,
                normalize_ws: bool = True,
                remove_duplicates: bool = True,
                financial_domain: bool = True) -> str:
        """통합 전처리 파이프라인"""
        
        if remove_html:
            text = self.remove_html_tags(text)
        
        text = self.clean_text(text)
        
        if normalize_ws:
            text = self.normalize_whitespace(text)
        
        if remove_duplicates:
            text = self.remove_duplicate_chars(text)
        
        if financial_domain:
            text = self.process_financial_text(text)
        
        return text


if __name__ == "__main__":
    # 테스트
    processor = KoreanEnglishTextProcessor()
    
    test_texts = [
        "안녕하세요!!! 금융보안 AI 챌린지입니다~~~",
        "<p>HTML 태그가 포함된 텍스트</p>",
        "이메일: test@example.com, URL: https://example.com",
        "연속된    공백과\n\n줄바꿈이 있는 텍스트",
        "금액은 ￦1,000,000원 (약 $1,000)입니다.",
    ]
    
    for text in test_texts:
        processed = processor.process(text)
        print(f"원본: {text}")
        print(f"처리: {processed}")
        print("-" * 50)