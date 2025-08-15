"""
데이터 전처리 및 지식 베이스 구축 관련 모듈
"""
import logging

logger = logging.getLogger(__name__)

# 기본 imports
from .chunker import DocumentChunker
from .embedder import EmbeddingGenerator

# Kiwi 사용 가능 여부 확인 및 조건부 import
KIWI_AVAILABLE = False
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
    logger.info("Kiwi is available - using KiwiTextProcessor for enhanced Korean processing")
except ImportError:
    logger.info("Kiwi not available - using basic KoreanEnglishTextProcessor")

# 조건부 TextProcessor 선택
if KIWI_AVAILABLE:
    from .kiwi_text_processor import KiwiTextProcessor as TextProcessor
    from .kiwi_text_processor import KiwiTextProcessor
else:
    from .text_processor import KoreanEnglishTextProcessor as TextProcessor

# 항상 기본 프로세서도 export (호환성)
from .text_processor import KoreanEnglishTextProcessor

__all__ = [
    'TextProcessor',  # 자동 선택된 최적 프로세서
    'KoreanEnglishTextProcessor',  # 기본 프로세서
    'DocumentChunker',
    'EmbeddingGenerator',
    'KIWI_AVAILABLE'  # Kiwi 사용 가능 여부
]

# Kiwi 사용 가능한 경우 추가 export
if KIWI_AVAILABLE:
    __all__.append('KiwiTextProcessor')