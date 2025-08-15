"""
Kiwi 기반 한국어 텍스트 전처리 모듈
- 사용자 사전 없이 순수 Kiwi 기능만 사용
- 대회 규정 100% 준수 (외부 데이터 없음)
"""

from typing import List, Dict, Optional, Tuple
import logging
from .text_processor import KoreanEnglishTextProcessor

logger = logging.getLogger(__name__)

# Kiwi 사용 가능 여부 확인
KIWI_AVAILABLE = False
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
    logger.info("Kiwi is available for Korean text processing")
except ImportError:
    logger.warning("Kiwi not available. Install with: pip install kiwipiepy")


class KiwiTextProcessor(KoreanEnglishTextProcessor):
    """
    Kiwi 기본 기능만 사용하는 텍스트 전처리기
    - 사용자 사전 없음 (대회 규정 준수)
    - 외부 데이터 없음
    - 순수 Kiwi 성능만 활용
    """
    
    def __init__(self, 
                 num_workers: int = 1,
                 apply_spacing: bool = True,
                 apply_normalization: bool = True,
                 verbose: bool = False):
        """
        Args:
            num_workers: 병렬 처리 워커 수
            apply_spacing: 띄어쓰기 교정 활성화
            apply_normalization: 텍스트 정규화 활성화
            verbose: 상세 로그 출력
        """
        super().__init__()
        
        if not KIWI_AVAILABLE:
            raise ImportError("Kiwi is not installed. Please install with: pip install kiwipiepy")
        
        # Kiwi 초기화 - 기본 설정만 사용 (사용자 사전 없음)
        self.kiwi = Kiwi(num_workers=num_workers)
        self.apply_spacing = apply_spacing
        self.apply_normalization = apply_normalization
        self.verbose = verbose
        
        # 처리 통계
        self.stats = {
            'processed_docs': 0,
            'total_chars': 0,
            'spacing_corrections': 0,
            'processing_errors': 0
        }
        
        if self.verbose:
            logger.info(f"KiwiTextProcessor initialized (workers={num_workers}, spacing={apply_spacing})")
    
    def correct_spacing(self, text: str) -> str:
        """
        Kiwi 내장 띄어쓰기 교정
        - 외부 데이터 없이 모델 자체 지식만 사용
        - 한국어 띄어쓰기 규칙 자동 적용
        """
        if not text or not self.apply_spacing:
            return text
            
        try:
            # reset_whitespace=True: 기존 띄어쓰기 무시하고 완전히 새로 교정
            # 이렇게 하면 붙어있는 텍스트도 제대로 분리
            corrected = self.kiwi.space(text, reset_whitespace=True)
            
            # 통계 업데이트
            if corrected != text:
                self.stats['spacing_corrections'] += 1
                if self.verbose:
                    logger.debug(f"Spacing corrected: '{text[:50]}...' -> '{corrected[:50]}...'")
            
            return corrected
            
        except Exception as e:
            logger.warning(f"Spacing correction failed: {e}")
            self.stats['processing_errors'] += 1
            return text
    
    def normalize_text(self, text: str) -> str:
        """
        Kiwi 내장 정규화 기능
        - 한글 자모 정규화
        - 중복 공백 제거
        - 기본적인 정규화만 수행
        """
        if not text or not self.apply_normalization:
            return text
            
        try:
            # reset_whitespace=False: 기존 띄어쓰기 유지하면서 정규화
            normalized = self.kiwi.space(text, reset_whitespace=False)
            return normalized
            
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            self.stats['processing_errors'] += 1
            return text
    
    def extract_morphemes(self, text: str) -> List[Tuple[str, str]]:
        """
        형태소 분석 (정보 추출용)
        
        Returns:
            List of (형태소, 품사) tuples
        """
        if not text:
            return []
            
        try:
            tokens = self.kiwi.tokenize(text)
            return [(token.form, token.tag) for token in tokens]
            
        except Exception as e:
            logger.error(f"Morpheme extraction failed: {e}")
            self.stats['processing_errors'] += 1
            return []
    
    def extract_nouns(self, text: str) -> List[str]:
        """
        명사만 추출 (키워드 추출용)
        """
        try:
            tokens = self.kiwi.tokenize(text)
            nouns = [
                token.form for token in tokens 
                if token.tag.startswith('N')  # 모든 명사 태그 (NNG, NNP, NNB 등)
            ]
            return nouns
            
        except Exception as e:
            logger.error(f"Noun extraction failed: {e}")
            return []
    
    def extract_keywords(self, text: str, min_length: int = 2) -> List[str]:
        """
        키워드 추출 (명사, 동사 어간, 형용사)
        
        Args:
            text: 입력 텍스트
            min_length: 최소 키워드 길이
        """
        try:
            tokens = self.kiwi.tokenize(text)
            keywords = []
            
            for token in tokens:
                # 의미 있는 품사만 선택
                if (token.tag.startswith('N') or      # 명사
                    token.tag.startswith('V') or      # 동사
                    token.tag == 'VA' or              # 형용사
                    token.tag.startswith('SL')):      # 외국어
                    
                    # 길이 필터링 (외국어는 예외)
                    if len(token.form) >= min_length or token.tag.startswith('SL'):
                        keywords.append(token.form.lower())
            
            return keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def process_paragraph(self, paragraph: str) -> str:
        """
        문단 단위 처리 (메모리 효율적)
        """
        if not paragraph or not paragraph.strip():
            return ""
        
        # 띄어쓰기 교정
        if self.apply_spacing:
            paragraph = self.correct_spacing(paragraph)
        
        # 정규화
        if self.apply_normalization:
            paragraph = self.normalize_text(paragraph)
        
        return paragraph
    
    def process(self, 
                text: str,
                preserve_paragraphs: bool = True,
                remove_html: bool = True,
                financial_processing: bool = True) -> str:
        """
        통합 전처리 파이프라인
        
        Args:
            text: 입력 텍스트
            preserve_paragraphs: 문단 구조 보존
            remove_html: HTML 태그 제거
            financial_processing: 금융 도메인 특화 처리
        """
        if not text or not text.strip():
            return ""
        
        # Step 1: 부모 클래스의 기본 정제
        if remove_html:
            text = self.remove_html_tags(text)
        
        text = super().clean_text(text)
        
        # Step 2: Kiwi 띄어쓰기 교정 및 정규화
        if preserve_paragraphs:
            # 문단 단위로 처리 (메모리 효율적)
            paragraphs = text.split('\n\n')
            processed_paragraphs = []
            
            for para in paragraphs:
                if para.strip():
                    processed_para = self.process_paragraph(para)
                    if processed_para:
                        processed_paragraphs.append(processed_para)
            
            text = '\n\n'.join(processed_paragraphs)
        else:
            # 전체 텍스트 한번에 처리
            text = self.process_paragraph(text)
        
        # Step 3: 부모 클래스의 금융 텍스트 처리
        # (숫자 포맷, 통화 기호 등 정규표현식 기반)
        if financial_processing:
            text = super().process_financial_text(text)
        
        # 통계 업데이트
        self.stats['processed_docs'] += 1
        self.stats['total_chars'] += len(text)
        
        return text
    
    def batch_process(self, texts: List[str]) -> List[str]:
        """
        배치 처리 (효율성 향상)
        """
        processed = []
        for text in texts:
            try:
                result = self.process(text)
                processed.append(result)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                processed.append(text)  # 실패 시 원본 반환
        
        return processed
    
    def get_stats(self) -> Dict:
        """처리 통계 반환"""
        return self.stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            'processed_docs': 0,
            'total_chars': 0,
            'spacing_corrections': 0,
            'processing_errors': 0
        }


# 테스트 코드
if __name__ == "__main__":
    # 기본 동작 테스트
    processor = KiwiTextProcessor(verbose=True)
    
    test_cases = [
        "금융보안은매우중요합니다",
        "한국은행이기준금리를인상했습니다",
        "디지털자산거래소가새로운규제를발표했다",
        "인터넷뱅킹을사용할때는반드시보안에주의해야합니다"
    ]
    
    print("\n=== Kiwi 텍스트 처리 테스트 ===\n")
    
    for text in test_cases:
        processed = processor.process(text)
        print(f"원본: {text}")
        print(f"처리: {processed}")
        
        # 키워드 추출 테스트
        keywords = processor.extract_keywords(processed)
        print(f"키워드: {keywords}")
        print("-" * 50)
    
    # 통계 출력
    stats = processor.get_stats()
    print(f"\n처리 통계: {stats}")