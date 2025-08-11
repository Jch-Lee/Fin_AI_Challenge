#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5-7B-Instruct LLM 통합 스크립트
RAG 시스템을 위한 답변 생성 모듈
"""

import torch
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer
)
import sys

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenLLM:
    """
    Qwen2.5-7B-Instruct 기반 답변 생성 모듈
    4-bit 양자화 및 금융 도메인 최적화
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        use_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Args:
            model_id: HuggingFace 모델 ID
            use_4bit: 4-bit 양자화 사용 여부
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 온도
            device: 연산 디바이스
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # 디바이스 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing Qwen2.5-7B-Instruct on {self.device}")
        
        # 4-bit 양자화 설정
        if use_4bit and self.device == "cuda":
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            logger.info("Using 4-bit quantization (NF4)")
        else:
            self.quantization_config = None
            
        # 모델 로드
        self._load_model()
        
    def _load_model(self):
        """모델과 토크나이저 로드"""
        try:
            logger.info(f"Loading model: {self.model_id}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로드 인자
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None
            }
            
            # 양자화 설정 추가
            if self.quantization_config:
                model_kwargs["quantization_config"] = self.quantization_config
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            # CPU 사용 시 명시적 이동
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # 모델 정보 출력
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded: {total_params/1e9:.2f}B parameters")
            
            # 메모리 사용량 체크
            if self.device == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU memory used: {memory_used:.2f} GB")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def create_prompt(
        self,
        query: str,
        contexts: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        RAG용 프롬프트 생성
        
        Args:
            query: 사용자 질문
            contexts: 검색된 컨텍스트 리스트
            system_prompt: 시스템 프롬프트
        
        Returns:
            완성된 프롬프트
        """
        if system_prompt is None:
            system_prompt = """당신은 금융 보안 전문가입니다. 
제공된 문서를 바탕으로 정확하고 신뢰할 수 있는 답변을 제공합니다.
답변은 한국어로 작성하며, 전문적이고 명확해야 합니다."""
        
        # 컨텍스트 결합
        context_text = "\n\n".join([
            f"[문서 {i+1}]\n{ctx}" 
            for i, ctx in enumerate(contexts)
        ])
        
        # Qwen 포맷 프롬프트
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""다음 문서들을 참고하여 질문에 답변해주세요.

참고 문서:
{context_text}

질문: {query}

답변:"""}
        ]
        
        # 토크나이저 채팅 템플릿 적용
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    @torch.no_grad()
    def generate(
        self,
        query: str,
        contexts: List[str],
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        답변 생성
        
        Args:
            query: 사용자 질문
            contexts: 검색된 컨텍스트
            system_prompt: 시스템 프롬프트
            stream: 스트리밍 출력 여부
        
        Returns:
            생성된 답변
        """
        # 프롬프트 생성
        prompt = self.create_prompt(query, contexts, system_prompt)
        
        # 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        # 디바이스로 이동
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 생성 설정
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        # 스트리밍 설정
        if stream:
            streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            gen_kwargs["streamer"] = streamer
        
        # 생성 시작
        start_time = time.time()
        
        outputs = self.model.generate(
            **inputs,
            **gen_kwargs
        )
        
        # 디코딩
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # 생성 시간 로깅
        gen_time = time.time() - start_time
        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        logger.info(f"Generated {tokens_generated} tokens in {gen_time:.2f}s")
        logger.info(f"Speed: {tokens_generated/gen_time:.2f} tokens/s")
        
        return response.strip()
    
    def generate_batch(
        self,
        queries: List[str],
        contexts_list: List[List[str]],
        batch_size: int = 4
    ) -> List[str]:
        """
        배치 생성 (성능 최적화)
        
        Args:
            queries: 질문 리스트
            contexts_list: 각 질문에 대한 컨텍스트 리스트
            batch_size: 배치 크기
        
        Returns:
            답변 리스트
        """
        responses = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_contexts = contexts_list[i:i + batch_size]
            
            for query, contexts in zip(batch_queries, batch_contexts):
                response = self.generate(query, contexts)
                responses.append(response)
                
        return responses


def test_qwen_integration():
    """Qwen 통합 테스트"""
    
    print("="*70)
    print(" Qwen2.5-7B-Instruct 통합 테스트")
    print("="*70)
    
    # 1. 모델 초기화
    print("\n[1] 모델 초기화...")
    
    llm = QwenLLM(
        use_4bit=torch.cuda.is_available(),  # GPU 있으면 4-bit 사용
        max_new_tokens=256,
        temperature=0.3
    )
    
    # 2. 테스트 데이터
    test_query = "금융 AI 시스템의 보안을 위해 어떤 조치가 필요한가?"
    
    test_contexts = [
        "금융 AI 시스템의 보안을 위해서는 데이터 암호화, 접근 통제, 정기적인 보안 감사가 필수적입니다. 특히 개인정보를 포함한 학습 데이터의 안전한 관리가 중요합니다.",
        "AI 모델의 적대적 공격에 대한 방어 메커니즘을 구축해야 합니다. 입력 검증, 모델 강건성 테스트, 이상 탐지 시스템 등을 구현해야 합니다.",
        "금융 AI 서비스는 규제 준수가 중요합니다. GDPR, 개인정보보호법 등 관련 법규를 준수하고, 설명 가능한 AI 구현이 필요합니다."
    ]
    
    # 3. 답변 생성
    print("\n[2] 답변 생성...")
    print(f"질문: {test_query}")
    print("\n참고 문서:")
    for i, ctx in enumerate(test_contexts, 1):
        print(f"  [{i}] {ctx[:60]}...")
    
    print("\n생성 중...")
    response = llm.generate(
        query=test_query,
        contexts=test_contexts,
        stream=False  # 스트리밍 비활성화
    )
    
    print("\n[생성된 답변]")
    print("-"*60)
    print(response)
    print("-"*60)
    
    # 4. 성능 측정
    print("\n[3] 성능 벤치마크...")
    
    # 단일 추론 시간
    start = time.time()
    _ = llm.generate(test_query, test_contexts)
    single_time = time.time() - start
    print(f"  - 단일 추론 시간: {single_time:.2f}초")
    
    # 메모리 사용량
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  - GPU 메모리 사용: {memory_used:.2f}GB / {memory_reserved:.2f}GB")
    
    print("\n" + "="*70)
    print(" [완료] Qwen2.5-7B 통합 성공!")
    print("="*70)
    
    return llm


if __name__ == "__main__":
    try:
        # 통합 테스트 실행
        llm = test_qwen_integration()
        
        print("\n[SUCCESS] LLM 모듈이 RAG 시스템에 통합 준비 완료!")
        print("다음 단계: python test_complete_rag.py 실행")
        
    except Exception as e:
        logger.error(f"통합 실패: {e}")
        import traceback
        traceback.print_exc()