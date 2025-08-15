"""
Qwen2.5-VL Vision Model Wrapper

Qwen/Qwen2.5-VL-7B-Instruct 모델을 사용한 Vision 프로세서 구현
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image
import numpy as np
import torch
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoTokenizer, 
    AutoProcessor,
    BitsAndBytesConfig
)
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    # qwen_vl_utils가 없는 경우 더미 함수 사용
    def process_vision_info(messages):
        """더미 process_vision_info 함수"""
        image_inputs = []
        video_inputs = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for content in msg["content"]:
                    if content.get("type") == "image":
                        image_inputs.append(content["image"])
        return image_inputs, video_inputs

from .base_processor import BaseVisionProcessor
from .prompts.financial import get_financial_prompt
from .prompts.korean import get_korean_prompt

logger = logging.getLogger(__name__)


class QwenVisionProcessor(BaseVisionProcessor):
    """Qwen2.5-VL 모델 기반 Vision 프로세서"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 프로세서 설정
                - model_name: 모델 이름 (default: "Qwen/Qwen2.5-VL-7B-Instruct")
                - quantization: 양자화 사용 여부 (default: True)
                - cache_dir: 모델 캐시 디렉토리
                - max_tokens: 최대 생성 토큰 수 (default: 512)
                - temperature: 생성 온도 (default: 0.3)
                - device_map: 디바이스 매핑 (default: "auto")
        """
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = None
        super().__init__(config)
    
    def _setup(self):
        """모델 및 프로세서 초기화"""
        logger.info("Qwen2.5-VL 모델 설정 중...")
        
        # 설정 값 가져오기
        model_name = self.config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        use_quantization = self.config.get("quantization", True)
        cache_dir = self.config.get("cache_dir", None)
        device_map = self.config.get("device_map", "auto")
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            # 양자화 설정
            quantization_config = None
            if use_quantization and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("4-bit 양자화 설정 완료")
            
            # 모델 로드
            logger.info(f"모델 로딩: {model_name}")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto" if not use_quantization else torch.float16,
                device_map=device_map,
                quantization_config=quantization_config,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # 프로세서 및 토크나이저 로드
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # 패드 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Qwen2.5-VL 모델 로딩 완료")
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {str(e)}")
            raise
    
    def process_image(
        self,
        image: Union[bytes, Image.Image, np.ndarray],
        context: Optional[str] = None,
        prompt_type: str = "general"
    ) -> str:
        """
        이미지를 텍스트 설명으로 변환
        
        Args:
            image: 처리할 이미지
            context: 이미지 주변 텍스트 컨텍스트
            prompt_type: 프롬프트 유형
            
        Returns:
            이미지에 대한 텍스트 설명
        """
        if not self.validate_input(image):
            raise ValueError("Invalid image input")
        
        try:
            # 이미지 전처리
            pil_image = self.preprocess_image(image)
            
            # 이미지 유형 감지 (필요시)
            if prompt_type == "auto":
                prompt_type = self.detect_image_type(pil_image)
            
            # 프롬프트 생성
            prompt = self._create_prompt(prompt_type, context)
            
            # 모델 추론
            description = self._inference_single(pil_image, prompt)
            
            # 후처리
            return self.postprocess_description(description)
            
        except Exception as e:
            logger.error(f"이미지 처리 실패: {str(e)}")
            return f"이미지 처리 중 오류 발생: {str(e)}"
    
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
        results = []
        contexts = contexts or [None] * len(images)
        
        # 배치 단위로 처리
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            batch_results = []
            for image, context in zip(batch_images, batch_contexts):
                try:
                    result = self.process_image(image, context)
                    batch_results.append(result)
                except Exception as e:
                    logger.warning(f"배치 처리 중 오류 (인덱스 {i}): {str(e)}")
                    batch_results.append(f"처리 실패: {str(e)}")
            
            results.extend(batch_results)
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def _create_prompt(self, prompt_type: str, context: Optional[str] = None) -> str:
        """프롬프트 생성"""
        # 금융 특화 프롬프트 사용
        prompt = get_financial_prompt(prompt_type, context or "")
        
        # 한국어 특화 설정 추가
        korean_enhancement = get_korean_prompt("enhancement")
        if korean_enhancement:
            prompt += "\n\n" + korean_enhancement
        
        return prompt
    
    def _inference_single(self, image: Image.Image, prompt: str) -> str:
        """단일 이미지 추론"""
        try:
            # 메시지 형식 구성
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ]
            
            # 프로세서를 통한 입력 준비
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 이미지와 텍스트 정보 처리
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # 생성 설정
            generation_config = {
                "max_new_tokens": self.config.get("max_tokens", 512),
                "temperature": self.config.get("temperature", 0.3),
                "top_p": 0.8,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # 추론 실행
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 입력 토큰 제거하고 디코딩
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return response
            
        except Exception as e:
            logger.error(f"추론 실패: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
            "quantization": self.config.get("quantization", True),
            "device": str(self.device) if self.device else "unknown",
            "max_tokens": self.config.get("max_tokens", 512),
            "temperature": self.config.get("temperature", 0.3)
        }
    
    def _cleanup_memory(self):
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """소멸자 - 메모리 정리"""
        try:
            self._cleanup_memory()
        except:
            pass


class QwenVisionCache:
    """Qwen Vision 결과 캐싱 클래스"""
    
    def __init__(self, max_size: int = 100):
        """
        Args:
            max_size: 캐시 최대 크기
        """
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def _get_image_hash(self, image: Image.Image) -> str:
        """이미지 해시 생성"""
        import hashlib
        img_bytes = image.tobytes()
        return hashlib.md5(img_bytes).hexdigest()
    
    def get(self, image: Image.Image, prompt: str) -> Optional[str]:
        """캐시에서 결과 조회"""
        key = self._get_image_hash(image) + "_" + hashlib.md5(prompt.encode()).hexdigest()
        
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        
        return None
    
    def put(self, image: Image.Image, prompt: str, result: str):
        """결과를 캐시에 저장"""
        import hashlib
        
        key = self._get_image_hash(image) + "_" + hashlib.md5(prompt.encode()).hexdigest()
        
        # 캐시 크기 제한
        if len(self.cache) >= self.max_size:
            # LRU 방식으로 오래된 항목 제거
            oldest_key = min(self.access_times.keys(), key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = result
        self.access_times[key] = time.time()
    
    def clear(self):
        """캐시 초기화"""
        self.cache.clear()
        self.access_times.clear()


# 전역 캐시 인스턴스
_vision_cache = QwenVisionCache()


def create_vision_processor(config: Optional[Dict[str, Any]] = None) -> QwenVisionProcessor:
    """
    Qwen Vision 프로세서 생성 함수
    
    Args:
        config: 프로세서 설정
        
    Returns:
        설정된 QwenVisionProcessor 인스턴스
    """
    default_config = {
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "quantization": True,
        "max_tokens": 512,
        "temperature": 0.3,
        "device_map": "auto"
    }
    
    if config:
        default_config.update(config)
    
    return QwenVisionProcessor(default_config)


def test_vision_processor():
    """간단한 테스트 함수"""
    try:
        # 테스트용 설정
        test_config = {
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "quantization": True,
            "max_tokens": 128,
            "temperature": 0.3
        }
        
        processor = create_vision_processor(test_config)
        
        # 간단한 테스트 이미지 생성 (100x100 빨간 사각형)
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # 이미지 처리 테스트
        result = processor.process_image(test_image, prompt_type="general")
        
        print(f"테스트 결과: {result}")
        print(f"모델 정보: {processor.get_model_info()}")
        
        return True
        
    except Exception as e:
        print(f"테스트 실패: {str(e)}")
        return False


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 실행
    print("Qwen2.5-VL Vision Processor 테스트 시작...")
    success = test_vision_processor()
    print(f"테스트 {'성공' if success else '실패'}")