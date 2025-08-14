"""
Base Vision Processor Abstract Class

Vision 프로세서의 공통 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np


class BaseVisionProcessor(ABC):
    """Vision 프로세서 추상 베이스 클래스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 프로세서 설정
        """
        self.config = config or {}
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """모델 및 프로세서 초기화"""
        pass
    
    @abstractmethod
    def process_image(
        self,
        image: Union[bytes, Image.Image, np.ndarray],
        context: Optional[str] = None,
        prompt_type: str = "general"
    ) -> str:
        """
        이미지를 텍스트 설명으로 변환
        
        Args:
            image: 처리할 이미지 (bytes, PIL Image, numpy array)
            context: 이미지 주변 텍스트 컨텍스트
            prompt_type: 프롬프트 유형 (general, chart, table, diagram)
            
        Returns:
            이미지에 대한 텍스트 설명
        """
        pass
    
    @abstractmethod
    def batch_process(
        self,
        images: List[Union[bytes, Image.Image, np.ndarray]],
        contexts: Optional[List[str]] = None,
        batch_size: int = 4
    ) -> List[str]:
        """
        여러 이미지를 배치로 처리
        
        Args:
            images: 처리할 이미지 리스트
            contexts: 각 이미지의 컨텍스트
            batch_size: 배치 크기
            
        Returns:
            각 이미지에 대한 텍스트 설명 리스트
        """
        pass
    
    def preprocess_image(
        self,
        image: Union[bytes, Image.Image, np.ndarray]
    ) -> Image.Image:
        """
        이미지 전처리
        
        Args:
            image: 원본 이미지
            
        Returns:
            전처리된 PIL Image
        """
        if isinstance(image, bytes):
            from io import BytesIO
            return Image.open(BytesIO(image))
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def detect_image_type(self, image: Image.Image) -> str:
        """
        이미지 유형 감지 (차트, 테이블, 다이어그램 등)
        
        Args:
            image: PIL Image
            
        Returns:
            이미지 유형 (chart, table, diagram, text, general)
        """
        # 간단한 휴리스틱 기반 감지
        # 실제로는 더 정교한 방법 사용 가능
        
        # 이미지 크기와 비율 확인
        width, height = image.size
        aspect_ratio = width / height if height > 0 else 1
        
        # 색상 분포 확인
        img_array = np.array(image)
        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
        
        # 간단한 규칙 기반 분류
        if 0.8 < aspect_ratio < 1.2 and unique_colors < 100:
            return "chart"  # 정사각형에 가깝고 색상이 적음
        elif aspect_ratio > 1.5 and unique_colors < 50:
            return "table"  # 가로가 길고 색상이 매우 적음
        elif unique_colors < 20:
            return "diagram"  # 매우 적은 색상
        elif unique_colors < 200:
            return "text"  # 텍스트 이미지
        else:
            return "general"  # 일반 이미지
    
    def postprocess_description(self, description: str) -> str:
        """
        생성된 설명 후처리
        
        Args:
            description: 원본 설명
            
        Returns:
            후처리된 설명
        """
        # 불필요한 공백 제거
        description = " ".join(description.split())
        
        # 특수 문자 정리
        description = description.strip()
        
        return description
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            모델 이름, 크기, 설정 등의 정보
        """
        pass
    
    def validate_input(self, image: Any) -> bool:
        """
        입력 이미지 유효성 검사
        
        Args:
            image: 검사할 이미지
            
        Returns:
            유효한 경우 True
        """
        try:
            processed_image = self.preprocess_image(image)
            return processed_image is not None
        except:
            return False
    
    def __repr__(self) -> str:
        model_info = self.get_model_info()
        return f"{self.__class__.__name__}(model={model_info.get('model_name', 'unknown')})"