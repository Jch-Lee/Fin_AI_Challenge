"""
Model Loader Component with Quantization Support
Architecture.md의 IModelLoader 인터페이스 구현
4-bit/8-bit 양자화 및 오프라인 모드 지원
"""

import os
import torch
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """모델 설정 데이터 클래스"""
    model_name: str
    model_path: Optional[str] = None
    quantization: Optional[str] = None  # "4bit", "8bit", None
    device_map: str = "auto"
    max_memory: Optional[Dict[str, str]] = None
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    use_cache: bool = True
    offline_mode: bool = True  # 대회 규칙: 오프라인 환경


class IModelLoader(ABC):
    """모델 로더 인터페이스"""
    
    @abstractmethod
    def load_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """모델과 토크나이저 로드"""
        pass
    
    @abstractmethod
    def prepare_for_inference(self, model: Any) -> Any:
        """추론을 위한 모델 준비"""
        pass
    
    @abstractmethod
    def validate_model(self, model: Any) -> bool:
        """모델 검증"""
        pass


class ModelLoader(IModelLoader):
    """
    양자화 지원 모델 로더
    Architecture.md Tech Stack: auto-gptq, bitsandbytes
    Pipeline.md 3.4.1: 모델 크기 < 15GB + 성능 유지
    """
    
    def __init__(self):
        self.loaded_models = {}
        self.supported_quantization = ["4bit", "8bit", "gptq", None]
        
        # 오프라인 모드 설정 (대회 환경)
        self._setup_offline_mode()
        
        logger.info("ModelLoader initialized with quantization support")
    
    def _setup_offline_mode(self):
        """오프라인 환경 설정 (Pipeline.md 3.2.5)"""
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        logger.info("Offline mode enabled for competition environment")
    
    def load_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """
        모델과 토크나이저 로드
        
        Args:
            config: 모델 설정
            
        Returns:
            Tuple[model, tokenizer]: 로드된 모델과 토크나이저
        """
        logger.info(f"Loading model: {config.model_name}")
        
        # 캐시 확인
        cache_key = f"{config.model_name}_{config.quantization}"
        if cache_key in self.loaded_models:
            logger.info(f"Using cached model: {cache_key}")
            return self.loaded_models[cache_key]
        
        # 모델 경로 결정
        model_path = config.model_path or f"./models/{config.model_name}"
        if not Path(model_path).exists() and not config.offline_mode:
            model_path = config.model_name  # HuggingFace Hub에서 로드
        
        try:
            # 양자화 설정
            quantization_config = self._get_quantization_config(config.quantization)
            
            # 토크나이저 로드
            tokenizer = self._load_tokenizer(model_path, config)
            
            # 모델 로드
            model = self._load_model_with_quantization(
                model_path, 
                config, 
                quantization_config
            )
            
            # 추론 준비
            model = self.prepare_for_inference(model)
            
            # 검증
            if not self.validate_model(model):
                raise ValueError(f"Model validation failed: {config.model_name}")
            
            # 캐시 저장
            self.loaded_models[cache_key] = (model, tokenizer)
            
            # 메모리 사용량 로깅
            self._log_memory_usage(model)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {config.model_name}: {e}")
            raise
    
    def _get_quantization_config(self, quantization: Optional[str]) -> Optional[BitsAndBytesConfig]:
        """
        양자화 설정 생성
        
        Args:
            quantization: 양자화 유형 ("4bit", "8bit", "gptq", None)
            
        Returns:
            BitsAndBytesConfig or None
        """
        if quantization is None:
            return None
        
        if quantization == "4bit":
            # 4-bit 양자화 (NF4)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True  # Double quantization for more compression
            )
        
        elif quantization == "8bit":
            # 8-bit 양자화
            return BitsAndBytesConfig(
                load_in_8bit=True,
                int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                llm_int8_enable_fp32_cpu_offload=True
            )
        
        elif quantization == "gptq":
            # GPTQ 양자화는 별도 처리 필요
            logger.info("GPTQ quantization will be handled separately")
            return None
        
        else:
            raise ValueError(f"Unsupported quantization type: {quantization}")
    
    def _load_tokenizer(self, model_path: str, config: ModelConfig) -> AutoTokenizer:
        """토크나이저 로드"""
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.offline_mode
        )
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        return tokenizer
    
    def _load_model_with_quantization(self, 
                                     model_path: str, 
                                     config: ModelConfig,
                                     quantization_config: Optional[BitsAndBytesConfig]) -> Any:
        """양자화 적용하여 모델 로드"""
        
        # 모델 설정 로드
        model_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.offline_mode
        )
        
        # torch dtype 결정
        if config.torch_dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype = getattr(torch, config.torch_dtype)
        
        # 모델 로드 인자
        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "config": model_config,
            "device_map": config.device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": config.trust_remote_code,
            "local_files_only": config.offline_mode,
            "use_cache": config.use_cache
        }
        
        # 양자화 설정 추가
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        
        # 메모리 제한 설정
        if config.max_memory:
            load_kwargs["max_memory"] = config.max_memory
        
        # 모델 로드
        logger.info(f"Loading model with quantization: {config.quantization}")
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        
        return model
    
    def prepare_for_inference(self, model: Any) -> Any:
        """
        추론을 위한 모델 준비
        
        Args:
            model: 로드된 모델
            
        Returns:
            준비된 모델
        """
        # Evaluation 모드 설정
        model.eval()
        
        # Gradient 비활성화
        for param in model.parameters():
            param.requires_grad = False
        
        # CUDA 사용 가능 시 최적화
        if torch.cuda.is_available():
            # Memory efficient attention
            if hasattr(model.config, "use_flash_attention_2"):
                model.config.use_flash_attention_2 = True
                logger.info("Flash Attention 2 enabled")
            
            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, "compile") and torch.cuda.get_device_capability()[0] >= 7:
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
        
        return model
    
    def validate_model(self, model: Any) -> bool:
        """
        모델 검증
        
        Args:
            model: 검증할 모델
            
        Returns:
            bool: 검증 통과 여부
        """
        try:
            # 기본 속성 확인
            if not hasattr(model, 'forward'):
                logger.error("Model missing forward method")
                return False
            
            if not hasattr(model, 'config'):
                logger.error("Model missing config")
                return False
            
            # 간단한 추론 테스트
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            
            # 더미 입력으로 테스트
            dummy_input = torch.tensor([[1, 2, 3]], device=device)
            
            with torch.no_grad():
                try:
                    output = model(dummy_input)
                    if output is None:
                        logger.error("Model returned None")
                        return False
                except Exception as e:
                    logger.error(f"Model inference test failed: {e}")
                    return False
            
            logger.info("Model validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return False
    
    def _log_memory_usage(self, model: Any):
        """모델 메모리 사용량 로깅"""
        if torch.cuda.is_available():
            # GPU 메모리 사용량
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
            
            # 모델 파라미터 수
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
            
            # 모델 크기 추정 (바이트)
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            model_size_gb = model_size / 1024**3
            
            logger.info(f"Estimated model size: {model_size_gb:.2f} GB")
            
            # Pipeline.md 3.4.1 요구사항 확인: 모델 크기 < 15GB
            if model_size_gb > 15:
                logger.warning(f"Model size {model_size_gb:.2f} GB exceeds 15GB limit!")
    
    def load_student_model(self, quantization: str = "4bit") -> Tuple[Any, Any]:
        """학생 모델 로드 (경쟁용)"""
        config = ModelConfig(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            model_path="./models/mistral-7b-instruct",
            quantization=quantization,
            offline_mode=True
        )
        return self.load_model(config)
    
    def load_teacher_model(self) -> Tuple[Any, Any]:
        """교사 모델 로드 (학습용)"""
        config = ModelConfig(
            model_name="Meta-Llama-3.1-70B-Instruct",
            model_path="./models/llama-70b-instruct",
            quantization="8bit",  # 교사 모델은 8-bit로 메모리 절약
            offline_mode=False  # 학습 시에는 온라인 가능
        )
        return self.load_model(config)
    
    def cleanup(self):
        """메모리 정리"""
        self.loaded_models.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Model cache cleared and GPU memory freed")


def main():
    """테스트 및 데모"""
    loader = ModelLoader()
    
    # 테스트 설정
    test_config = ModelConfig(
        model_name="beomi/gemma-ko-2b",  # 작은 모델로 테스트
        quantization="4bit",
        offline_mode=False  # 테스트 시에는 온라인 허용
    )
    
    try:
        # 모델 로드
        model, tokenizer = loader.load_model(test_config)
        
        # 간단한 추론 테스트
        text = "금융 AI 시스템의 보안은"
        inputs = tokenizer(text, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        
        # 메모리 정리
        loader.cleanup()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    main()