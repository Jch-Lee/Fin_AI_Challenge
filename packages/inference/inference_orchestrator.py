#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference Orchestrator with Enhanced Prompt Templates
프롬프트 템플릿이 통합된 추론 오케스트레이터
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
import time
import re
from .prompt_templates import FinancialPromptManager, QuestionType, PromptOptimizer

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """추론 결과 데이터 클래스"""
    question_id: str
    question: str
    answer: str
    confidence: float
    question_type: str
    processing_time: float
    context_used: Optional[List[str]] = None
    model_name: Optional[str] = None


class InferenceOrchestrator:
    """
    향상된 프롬프트 템플릿을 사용하는 추론 오케스트레이터
    """
    
    def __init__(self, 
                 model_name: str = "beomi/gemma-ko-7b",
                 use_quantization: bool = True,
                 device: str = "cuda",
                 use_enhanced_prompts: bool = True):
        """
        Args:
            model_name: 사용할 모델명
            use_quantization: 양자화 사용 여부
            device: 실행 디바이스
            use_enhanced_prompts: 향상된 프롬프트 템플릿 사용 여부
        """
        self.model_name = model_name
        self.device = device
        self.use_enhanced_prompts = use_enhanced_prompts
        
        # 프롬프트 매니저 초기화
        if use_enhanced_prompts:
            self.prompt_manager = FinancialPromptManager()
            self.prompt_optimizer = PromptOptimizer(self.prompt_manager)
        
        # 모델 로드 (실제 환경에서)
        self.model = None
        self.tokenizer = None
        
        # 시뮬레이션 모드 플래그
        self.simulation_mode = True
        
        logger.info(f"InferenceOrchestrator initialized with {model_name}")
        if use_enhanced_prompts:
            logger.info("Enhanced prompt templates enabled")
    
    def load_model(self):
        """모델과 토크나이저 로드"""
        try:
            # 양자화 설정
            if self.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = None
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.simulation_mode = False
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Running in simulation mode.")
            self.simulation_mode = True
    
    def generate_answer(self,
                       question: str,
                       context: Optional[List[str]] = None,
                       max_new_tokens: int = 256,
                       temperature: float = 0.3,
                       top_p: float = 0.8) -> str:
        """
        질문에 대한 답변 생성
        
        Args:
            question: 입력 질문
            context: RAG 검색 컨텍스트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 온도
            top_p: Top-p 샘플링 값
            
        Returns:
            생성된 답변
        """
        
        # 프롬프트 생성
        if self.use_enhanced_prompts:
            # 질문 유형 분류
            question_type = self.prompt_manager.classify_question(question)
            
            # 컨텍스트 준비
            context_str = "\n".join(context) if context else ""
            
            # 프롬프트 생성
            system_prompt, user_prompt = self.prompt_manager.create_prompt(
                question=question,
                context=context_str,
                question_type=question_type
            )
            
            # 모델별 최적화
            if not self.simulation_mode:
                system_prompt = self.prompt_optimizer.optimize_for_model(
                    system_prompt, 
                    self.model_name
                )
            
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
        else:
            # 기본 프롬프트 (이전 방식)
            full_prompt = self._create_basic_prompt(question, context)
            question_type = self._detect_basic_question_type(question)
        
        # 시뮬레이션 모드
        if self.simulation_mode:
            return self._simulate_generation(question, question_type)
        
        # 실제 모델 생성
        try:
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 답변 추출
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            return self._post_process_answer(generated_text, question_type)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._get_fallback_answer(question_type)
    
    def process_question(self,
                        question_id: str,
                        question: str,
                        context: Optional[List[str]] = None) -> InferenceResult:
        """
        단일 질문 처리
        
        Args:
            question_id: 질문 ID
            question: 질문 텍스트
            context: RAG 검색 결과
            
        Returns:
            InferenceResult 객체
        """
        
        start_time = time.time()
        
        # 질문 유형 분류
        if self.use_enhanced_prompts:
            question_type = self.prompt_manager.classify_question(question)
        else:
            question_type = self._detect_basic_question_type(question)
        
        # 답변 생성
        answer = self.generate_answer(question, context)
        
        # 신뢰도 계산 (시뮬레이션)
        confidence = self._calculate_confidence(answer, question_type)
        
        # 처리 시간
        processing_time = time.time() - start_time
        
        return InferenceResult(
            question_id=question_id,
            question=question,
            answer=answer,
            confidence=confidence,
            question_type=str(question_type.value) if hasattr(question_type, 'value') else str(question_type),
            processing_time=processing_time,
            context_used=context,
            model_name=self.model_name
        )
    
    def _create_basic_prompt(self, question: str, context: Optional[List[str]]) -> str:
        """기본 프롬프트 생성 (이전 방식)"""
        
        base_prompt = "당신은 금융 보안 전문가입니다.\n\n"
        
        if context:
            base_prompt += "참고 자료:\n"
            for i, ctx in enumerate(context[:3]):
                base_prompt += f"{i+1}. {ctx[:200]}...\n"
            base_prompt += "\n"
        
        base_prompt += f"질문: {question}\n답변:"
        
        return base_prompt
    
    def _detect_basic_question_type(self, question: str) -> str:
        """기본 질문 유형 감지"""
        lines = question.strip().split('\n')
        choice_pattern = re.compile(r'^\s*[1-9]\d*[\s\)\.]\s*.+')
        
        choices = [line for line in lines if choice_pattern.match(line)]
        
        if len(choices) >= 2:
            return "multiple_choice"
        else:
            return "descriptive"
    
    def _simulate_generation(self, question: str, question_type) -> str:
        """시뮬레이션 모드 답변 생성"""
        
        if question_type == QuestionType.MULTIPLE_CHOICE or question_type == "multiple_choice":
            # 객관식 시뮬레이션
            import random
            choices = re.findall(r'^\s*([1-9]\d*)[\s\)\.]\s*.+', question, re.MULTILINE)
            if choices:
                return random.choice(choices)
            return "1"
        
        else:
            # 서술형 시뮬레이션
            templates = [
                "금융 AI 시스템의 보안은 데이터 보호, 모델 무결성, 접근 통제를 포함한 다층적 접근이 필요합니다.",
                "해당 사항은 금융위원회 가이드라인에 따라 정기적인 모니터링과 감사가 요구됩니다.",
                "AI 모델의 안전성 확보를 위해 지속적인 검증과 업데이트가 필수적입니다.",
                "금융 데이터의 특성상 암호화와 익명화 처리가 기본적으로 적용되어야 합니다."
            ]
            
            import random
            base_answer = random.choice(templates)
            
            # 질문 키워드 기반 커스터마이징
            if "편향" in question:
                return f"{base_answer} 특히 편향성 문제는 공정성 지표를 통해 지속적으로 모니터링해야 합니다."
            elif "공격" in question:
                return f"{base_answer} 적대적 공격에 대한 방어 메커니즘을 구축하는 것이 중요합니다."
            elif "규제" in question:
                return f"{base_answer} 관련 규제 준수를 위한 체크리스트를 유지해야 합니다."
            
            return base_answer
    
    def _post_process_answer(self, answer: str, question_type) -> str:
        """답변 후처리"""
        
        # 공백 정리
        answer = answer.strip()
        
        # 객관식 답변 추출
        if question_type == QuestionType.MULTIPLE_CHOICE or question_type == "multiple_choice":
            # 숫자만 추출
            numbers = re.findall(r'\d+', answer)
            if numbers:
                return numbers[0]
            return "1"
        
        # 서술형 답변 정리
        # 불필요한 반복 제거
        lines = answer.split('\n')
        unique_lines = []
        for line in lines:
            if line and line not in unique_lines:
                unique_lines.append(line)
        
        return '\n'.join(unique_lines)
    
    def _calculate_confidence(self, answer: str, question_type) -> float:
        """답변 신뢰도 계산"""
        
        # 시뮬레이션 신뢰도
        if self.simulation_mode:
            return 0.7
        
        # 실제 모델 기반 신뢰도 (추후 구현)
        # - 생성 확률 기반
        # - 답변 길이와 일관성
        # - 키워드 매칭
        
        base_confidence = 0.8
        
        # 답변 길이 기반 조정
        if question_type == QuestionType.DESCRIPTIVE:
            if len(answer) < 50:
                base_confidence -= 0.2
            elif len(answer) > 500:
                base_confidence += 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _get_fallback_answer(self, question_type) -> str:
        """실패 시 기본 답변"""
        
        if question_type == QuestionType.MULTIPLE_CHOICE or question_type == "multiple_choice":
            return "1"
        else:
            return "답변을 생성할 수 없습니다. 질문을 다시 확인해 주세요."
    
    def batch_process(self,
                     questions: List[Dict[str, str]],
                     contexts: Optional[Dict[str, List[str]]] = None,
                     batch_size: int = 8) -> List[InferenceResult]:
        """
        배치 처리
        
        Args:
            questions: [{"id": "Q001", "question": "..."}, ...]
            contexts: {"Q001": ["context1", ...], ...}
            batch_size: 배치 크기
            
        Returns:
            InferenceResult 리스트
        """
        
        results = []
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            
            for item in batch:
                question_id = item['id']
                question = item['question']
                context = contexts.get(question_id) if contexts else None
                
                result = self.process_question(question_id, question, context)
                results.append(result)
                
                logger.info(f"Processed {question_id}: {result.question_type} - "
                          f"Confidence: {result.confidence:.2f}")
        
        return results


# 사용 예시
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 오케스트레이터 초기화
    orchestrator = InferenceOrchestrator(
        model_name="beomi/gemma-ko-7b",
        use_enhanced_prompts=True
    )
    
    # 테스트 질문들
    test_questions = [
        {
            "id": "Q001",
            "question": """AI 모델의 보안 위협 중 가장 심각한 것은?
            1. 데이터 유출
            2. 모델 역공학
            3. 적대적 공격
            4. 모든 것이 동일하게 심각"""
        },
        {
            "id": "Q002", 
            "question": "금융 AI 시스템에서 편향성을 줄이는 방법을 설명하시오."
        }
    ]
    
    # 시뮬레이션 컨텍스트
    test_contexts = {
        "Q001": ["AI 보안 위협은 다양한 형태로 나타납니다..."],
        "Q002": ["편향성은 데이터와 알고리즘 모두에서 발생할 수 있습니다..."]
    }
    
    # 배치 처리
    results = orchestrator.batch_process(test_questions, test_contexts)
    
    # 결과 출력
    for result in results:
        print(f"\n{'='*60}")
        print(f"ID: {result.question_id}")
        print(f"질문 유형: {result.question_type}")
        print(f"답변: {result.answer}")
        print(f"신뢰도: {result.confidence:.2f}")
        print(f"처리 시간: {result.processing_time:.3f}초")