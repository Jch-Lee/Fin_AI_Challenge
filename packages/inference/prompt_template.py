"""
Prompt Template Component
Architecture.md의 IPromptTemplate 인터페이스 구현
금융 보안 도메인 특화 프롬프트 생성
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """프롬프트 유형"""
    MULTIPLE_CHOICE = "multiple_choice"
    OPEN_ENDED = "open_ended"
    RERANKING = "reranking"
    SYNTHETIC_QA = "synthetic_qa"
    VALIDATION = "validation"


@dataclass
class PromptConfig:
    """프롬프트 설정"""
    max_context_length: int = 2048
    max_answer_length: int = 512
    include_examples: bool = True
    language: str = "ko"  # ko, en, mixed
    domain: str = "financial_security"
    temperature_hint: float = 0.3


class IPromptTemplate(ABC):
    """프롬프트 템플릿 인터페이스"""
    
    @abstractmethod
    def create_prompt(self, 
                     question: str, 
                     context: Optional[str] = None,
                     prompt_type: PromptType = PromptType.OPEN_ENDED) -> str:
        """프롬프트 생성"""
        pass
    
    @abstractmethod
    def create_system_prompt(self, role: str = "assistant") -> str:
        """시스템 프롬프트 생성"""
        pass


class PromptTemplate(IPromptTemplate):
    """
    금융 보안 도메인 특화 프롬프트 템플릿
    baseline_code의 make_prompt_auto() 개선 버전
    """
    
    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig()
        
        # 도메인별 시스템 프롬프트
        self.system_prompts = {
            "financial_security": """당신은 금융 보안 분야의 전문가입니다. 
금융 규제, AI 시스템 보안, 개인정보 보호, 사이버 보안 등에 대한 깊은 지식을 가지고 있습니다.
답변은 정확하고 전문적이며, 한국 금융 규제 환경을 고려하여 작성해주세요.""",
            
            "financial_ai": """당신은 금융 AI 시스템 전문가입니다.
머신러닝 모델의 금융 적용, 리스크 관리, 알고리즘 트레이딩, 신용 평가 AI 등에 정통합니다.
기술적 정확성과 실무 적용 가능성을 모두 고려하여 답변해주세요.""",
            
            "compliance": """당신은 금융 규제 준수 전문가입니다.
금융위원회 규정, 개인정보보호법, 전자금융거래법 등 한국 금융 규제를 깊이 이해하고 있습니다.
규제 요구사항과 실무 적용 방안을 명확하게 설명해주세요."""
        }
        
        # Few-shot 예제
        self.few_shot_examples = {
            PromptType.MULTIPLE_CHOICE: [
                {
                    "question": "다음 중 금융 AI 시스템의 보안 위협이 아닌 것은?\n1) 데이터 중독\n2) 모델 추출\n3) 데이터 증강\n4) 적대적 공격",
                    "answer": "3"
                },
                {
                    "question": "개인정보보호법상 민감정보에 해당하지 않는 것은?\n1) 건강정보\n2) 유전정보\n3) 연락처\n4) 생체정보",
                    "answer": "3"
                }
            ],
            PromptType.OPEN_ENDED: [
                {
                    "question": "금융 AI 시스템에서 설명가능성(Explainability)이 중요한 이유를 설명하시오.",
                    "answer": "금융 AI 시스템의 설명가능성은 1) 규제 준수: 금융당국의 AI 의사결정 설명 요구, 2) 고객 신뢰: 대출 거절 등 중요 결정에 대한 투명성 제공, 3) 리스크 관리: 모델의 편향성과 오류 발견 및 수정, 4) 법적 책임: 분쟁 시 의사결정 근거 제시 필요 등의 이유로 필수적입니다."
                }
            ]
        }
        
        logger.info(f"PromptTemplate initialized for domain: {self.config.domain}")
    
    def create_prompt(self, 
                     question: str, 
                     context: Optional[str] = None,
                     prompt_type: PromptType = PromptType.OPEN_ENDED,
                     **kwargs) -> str:
        """
        프롬프트 생성
        
        Args:
            question: 질문 텍스트
            context: RAG 검색 컨텍스트
            prompt_type: 프롬프트 유형
            **kwargs: 추가 파라미터
            
        Returns:
            생성된 프롬프트
        """
        if prompt_type == PromptType.MULTIPLE_CHOICE:
            return self._create_multiple_choice_prompt(question, context)
        elif prompt_type == PromptType.OPEN_ENDED:
            return self._create_open_ended_prompt(question, context)
        elif prompt_type == PromptType.RERANKING:
            return self._create_reranking_prompt(question, context, **kwargs)
        elif prompt_type == PromptType.SYNTHETIC_QA:
            return self._create_synthetic_qa_prompt(context)
        elif prompt_type == PromptType.VALIDATION:
            return self._create_validation_prompt(question, kwargs.get('answer', ''))
        else:
            return self._create_open_ended_prompt(question, context)
    
    def create_system_prompt(self, role: str = "financial_security") -> str:
        """
        시스템 프롬프트 생성
        
        Args:
            role: 역할 (financial_security, financial_ai, compliance)
            
        Returns:
            시스템 프롬프트
        """
        return self.system_prompts.get(role, self.system_prompts["financial_security"])
    
    def _create_multiple_choice_prompt(self, question: str, context: Optional[str]) -> str:
        """객관식 질문 프롬프트"""
        prompt_parts = []
        
        # 시스템 지시
        prompt_parts.append("다음은 객관식 문제입니다. 정답 번호만 출력하세요.")
        
        # Few-shot 예제
        if self.config.include_examples:
            prompt_parts.append("\n예제:")
            for example in self.few_shot_examples[PromptType.MULTIPLE_CHOICE][:2]:
                prompt_parts.append(f"질문: {example['question']}")
                prompt_parts.append(f"정답: {example['answer']}\n")
        
        # 컨텍스트
        if context:
            context_truncated = self._truncate_context(context)
            prompt_parts.append(f"참고 정보:\n{context_truncated}\n")
        
        # 실제 질문
        prompt_parts.append(f"질문: {question}")
        prompt_parts.append("정답:")
        
        return "\n".join(prompt_parts)
    
    def _create_open_ended_prompt(self, question: str, context: Optional[str]) -> str:
        """주관식 질문 프롬프트"""
        prompt_parts = []
        
        # 시스템 지시
        prompt_parts.append(
            "다음 질문에 대해 정확하고 전문적인 답변을 작성하세요. "
            f"답변은 {self.config.max_answer_length}자 이내로 작성하세요."
        )
        
        # Few-shot 예제
        if self.config.include_examples and PromptType.OPEN_ENDED in self.few_shot_examples:
            prompt_parts.append("\n예제:")
            example = self.few_shot_examples[PromptType.OPEN_ENDED][0]
            prompt_parts.append(f"질문: {example['question']}")
            prompt_parts.append(f"답변: {example['answer']}\n")
        
        # 컨텍스트
        if context:
            context_truncated = self._truncate_context(context)
            prompt_parts.append("다음 정보를 참고하여 답변하세요:")
            prompt_parts.append(f"{context_truncated}\n")
        
        # 실제 질문
        prompt_parts.append(f"질문: {question}")
        prompt_parts.append("답변:")
        
        return "\n".join(prompt_parts)
    
    def _create_reranking_prompt(self, query: str, documents: str, **kwargs) -> str:
        """재순위화 프롬프트 (LLM-as-Reranker)"""
        prompt = f"""다음 질문과 가장 관련성이 높은 문서를 순위대로 정렬하세요.

질문: {query}

문서 목록:
{documents}

각 문서의 관련성을 평가하고, 가장 관련성이 높은 순서대로 문서 번호를 나열하세요.
출력 형식: [1, 3, 2, 4, 5] (예시)

순위:"""
        return prompt
    
    def _create_synthetic_qa_prompt(self, context: str) -> str:
        """합성 Q&A 생성 프롬프트"""
        prompt = f"""다음 텍스트를 읽고 금융 보안 관련 질문과 답변을 3개 생성하세요.

텍스트:
{self._truncate_context(context, max_length=1000)}

각 Q&A는 다음 형식으로 작성하세요:
Q1: [질문]
A1: [답변]

Q2: [질문]
A2: [답변]

Q3: [질문]
A3: [답변]

금융 규제, AI 보안, 개인정보 보호 등의 주제를 포함하세요.

생성된 Q&A:"""
        return prompt
    
    def _create_validation_prompt(self, question: str, answer: str) -> str:
        """답변 검증 프롬프트"""
        prompt = f"""다음 질문과 답변의 정확성과 적절성을 평가하세요.

질문: {question}
답변: {answer}

평가 기준:
1. 사실적 정확성 (0-10)
2. 완전성 (0-10)
3. 명확성 (0-10)
4. 관련성 (0-10)

각 항목을 평가하고 총점을 제시하세요.

평가:"""
        return prompt
    
    def _truncate_context(self, context: str, max_length: Optional[int] = None) -> str:
        """컨텍스트 길이 제한"""
        max_len = max_length or self.config.max_context_length
        
        if len(context) <= max_len:
            return context
        
        # 중요한 부분을 보존하면서 자르기
        # 앞부분과 뒷부분을 보존
        half_len = max_len // 2 - 20
        truncated = context[:half_len] + "\n...[중략]...\n" + context[-half_len:]
        
        return truncated
    
    def create_chain_of_thought_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Chain-of-Thought 프롬프트 생성"""
        prompt_parts = []
        
        prompt_parts.append("다음 질문에 대해 단계별로 생각하며 답변하세요.")
        prompt_parts.append("먼저 문제를 분석하고, 관련 개념을 정리한 후, 논리적으로 답변을 도출하세요.\n")
        
        if context:
            prompt_parts.append(f"참고 정보:\n{self._truncate_context(context)}\n")
        
        prompt_parts.append(f"질문: {question}\n")
        prompt_parts.append("단계별 분석:")
        prompt_parts.append("1. 문제 이해:")
        prompt_parts.append("2. 핵심 개념:")
        prompt_parts.append("3. 분석:")
        prompt_parts.append("4. 결론:")
        prompt_parts.append("\n최종 답변:")
        
        return "\n".join(prompt_parts)
    
    def create_zero_shot_cot_prompt(self, question: str) -> str:
        """Zero-shot Chain-of-Thought 프롬프트"""
        return f"""{question}

Let's think step by step. 차근차근 생각해봅시다.

답변:"""
    
    def format_for_model(self, 
                        prompt: str, 
                        model_type: str = "mistral") -> Dict[str, Any]:
        """
        모델별 포맷팅
        
        Args:
            prompt: 원본 프롬프트
            model_type: 모델 유형 (mistral, llama, solar)
            
        Returns:
            포맷된 프롬프트 딕셔너리
        """
        if model_type == "mistral":
            return {
                "prompt": f"<s>[INST] {prompt} [/INST]",
                "temperature": self.config.temperature_hint,
                "max_tokens": self.config.max_answer_length
            }
        elif model_type == "llama":
            system_prompt = self.create_system_prompt()
            return {
                "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
                "temperature": self.config.temperature_hint,
                "max_tokens": self.config.max_answer_length
            }
        else:
            # 기본 포맷
            return {
                "prompt": prompt,
                "temperature": self.config.temperature_hint,
                "max_tokens": self.config.max_answer_length
            }


def main():
    """테스트 및 데모"""
    # 프롬프트 템플릿 생성
    template = PromptTemplate(
        config=PromptConfig(
            max_context_length=1024,
            max_answer_length=256,
            include_examples=True
        )
    )
    
    # 테스트 케이스
    test_cases = [
        {
            "question": "금융 AI 시스템의 편향성을 줄이는 방법은?\n1) 더 많은 데이터 수집\n2) 공정성 메트릭 적용\n3) 모델 크기 증가\n4) 학습률 조정",
            "type": PromptType.MULTIPLE_CHOICE,
            "context": "AI 시스템의 편향성은 데이터와 알고리즘 모두에서 발생할 수 있습니다."
        },
        {
            "question": "개인정보 가명처리와 익명처리의 차이점을 설명하시오.",
            "type": PromptType.OPEN_ENDED,
            "context": "가명처리는 추가 정보 사용 시 개인 식별이 가능하지만, 익명처리는 불가능합니다."
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"Test Case {i+1}: {test['type'].value}")
        print(f"{'='*50}")
        
        prompt = template.create_prompt(
            question=test["question"],
            context=test.get("context"),
            prompt_type=test["type"]
        )
        
        print(prompt)
        
        # 모델별 포맷팅 테스트
        formatted = template.format_for_model(prompt, "mistral")
        print(f"\nFormatted for Mistral:")
        print(f"Temperature: {formatted['temperature']}")
        print(f"Max tokens: {formatted['max_tokens']}")
    
    # Chain-of-Thought 테스트
    print(f"\n{'='*50}")
    print("Chain-of-Thought Example")
    print(f"{'='*50}")
    
    cot_prompt = template.create_chain_of_thought_prompt(
        "블록체인 기술이 금융 보안에 미치는 영향은?"
    )
    print(cot_prompt)


if __name__ == "__main__":
    main()