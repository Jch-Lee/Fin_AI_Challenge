"""
Qwen2.5-7B 4-bit 양자화 모듈
메모리 효율적인 추론을 위한 최적화
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class QuantizedQwenLLM:
    """
    4-bit 양자화된 Qwen2.5-7B 모델
    RTX 4090 24GB에 최적화
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        quantization_type: str = "4bit",  # 4bit, 8bit, none
        cache_dir: str = "./models"
    ):
        """
        Args:
            model_id: HuggingFace 모델 ID
            quantization_type: 양자화 타입
            cache_dir: 모델 캐시 디렉토리
        """
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.quantization_type = quantization_type
        
        # 양자화 설정
        self.quant_config = self._get_quantization_config()
        
        # 모델 로드
        self._load_model()
        
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """양자화 설정 반환"""
        
        if self.quantization_type == "4bit":
            logger.info("Setting up 4-bit quantization (NF4)")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # 추가 압축
            )
        elif self.quantization_type == "8bit":
            logger.info("Setting up 8-bit quantization")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=False
            )
        else:
            logger.info("No quantization (full precision)")
            return None
    
    def _load_model(self):
        """모델 로드 및 최적화"""
        
        logger.info(f"Loading {self.model_id} with {self.quantization_type}")
        
        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 모델 로드 옵션
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "max_memory": {0: "22GB"},  # RTX 4090 24GB 중 22GB 사용
        }
        
        # 양자화 설정 추가
        if self.quant_config:
            model_kwargs["quantization_config"] = self.quant_config
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        
        # 메모리 사용량 로깅
        self._log_memory_usage()
        
    def _log_memory_usage(self):
        """메모리 사용량 로깅"""
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            # 모델 크기 추정
            param_count = sum(p.numel() for p in self.model.parameters())
            
            if self.quantization_type == "4bit":
                model_size = param_count * 0.5 / 1024**3  # 4-bit = 0.5 bytes per param
            elif self.quantization_type == "8bit":
                model_size = param_count * 1 / 1024**3  # 8-bit = 1 byte per param
            else:
                model_size = param_count * 2 / 1024**3  # fp16 = 2 bytes per param
            
            logger.info(f"Estimated model size: {model_size:.2f}GB")
            logger.info(f"Total parameters: {param_count/1e9:.2f}B")
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """메모리 사용량 반환"""
        
        if not torch.cuda.is_available():
            return {"status": "CPU mode"}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "free_gb": (torch.cuda.get_device_properties(0).total_memory - 
                       torch.cuda.memory_allocated()) / 1024**3
        }
    
    @torch.no_grad()
    def generate_optimized(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        use_cache: bool = True
    ) -> str:
        """
        최적화된 생성
        
        Args:
            prompt: 입력 프롬프트
            max_new_tokens: 최대 생성 토큰
            temperature: 생성 온도
            use_cache: KV 캐시 사용 여부
        
        Returns:
            생성된 텍스트
        """
        # 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # 생성 설정
        gen_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": use_cache  # KV 캐시로 속도 향상
        }
        
        # 생성
        outputs = self.model.generate(**inputs, **gen_config)
        
        # 디코딩
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


def benchmark_quantization():
    """양자화 성능 벤치마크"""
    
    import time
    
    print("="*60)
    print(" Qwen2.5-7B 양자화 벤치마크")
    print("="*60)
    
    test_prompt = """당신은 금융 전문가입니다.
    
질문: AI 시스템의 보안을 강화하는 방법은?

답변:"""
    
    results = {}
    
    # 4-bit 테스트
    if torch.cuda.is_available():
        print("\n[1] 4-bit 양자화 테스트...")
        
        model_4bit = QuantizedQwenLLM(quantization_type="4bit")
        memory_4bit = model_4bit.get_memory_footprint()
        
        start = time.time()
        response_4bit = model_4bit.generate_optimized(test_prompt)
        time_4bit = time.time() - start
        
        results["4bit"] = {
            "memory": memory_4bit,
            "time": time_4bit,
            "response_preview": response_4bit[:100]
        }
        
        print(f"  - 메모리: {memory_4bit['allocated_gb']:.2f}GB")
        print(f"  - 생성 시간: {time_4bit:.2f}초")
        print(f"  - 응답: {response_4bit[:50]}...")
        
        del model_4bit  # 메모리 해제
        torch.cuda.empty_cache()
    
    # 결과 요약
    print("\n" + "="*60)
    print(" 벤치마크 결과 요약")
    print("="*60)
    
    for quant_type, result in results.items():
        print(f"\n{quant_type}:")
        print(f"  - 메모리: {result['memory']['allocated_gb']:.2f}GB")
        print(f"  - 속도: {result['time']:.2f}초")
    
    return results


if __name__ == "__main__":
    # 환경 체크
    if not torch.cuda.is_available():
        print("⚠️ GPU를 사용할 수 없습니다. CPU 모드는 매우 느립니다.")
        
    # 벤치마크 실행
    benchmark_quantization()