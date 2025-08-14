"""
Vision Processing Module for Financial Documents

이 모듈은 PDF 문서의 이미지, 차트, 테이블 등을 텍스트로 변환하여
RAG 시스템의 검색 성능을 향상시킵니다.

Components:
- BaseVisionProcessor: 기본 Vision 프로세서 인터페이스
- QwenVisionProcessor: Qwen2.5-VL 모델 기반 Vision 프로세서
- PDFImageExtractor: PDF에서 이미지 추출
- ImagePreprocessor: 이미지 전처리
- VisionProcessingPipeline: 통합 처리 파이프라인
- AsyncVisionPipeline: 비동기 처리 파이프라인
"""

from .base_processor import BaseVisionProcessor
from .qwen_vision import QwenVisionProcessor, create_vision_processor, QwenVisionCache
from .image_extractor import PDFImageExtractor, ImagePreprocessor, extract_images_from_directory
from .vision_pipeline import VisionProcessingPipeline, AsyncVisionPipeline, create_pipeline

__all__ = [
    # Base classes
    'BaseVisionProcessor',
    
    # Qwen Vision
    'QwenVisionProcessor', 
    'create_vision_processor',
    'QwenVisionCache',
    
    # Image extraction
    'PDFImageExtractor',
    'ImagePreprocessor',
    'extract_images_from_directory',
    
    # Pipeline
    'VisionProcessingPipeline',
    'AsyncVisionPipeline',
    'create_pipeline'
]

__version__ = '0.1.0'