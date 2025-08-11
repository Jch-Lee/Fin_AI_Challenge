#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Financial Domain-Specific Prompt Templates
금융 보안 도메인 특화 프롬프트 템플릿
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class QuestionType(Enum):
    """질문 유형 분류"""
    MULTIPLE_CHOICE = "multiple_choice"
    DESCRIPTIVE = "descriptive"
    DEFINITION = "definition"
    COMPARISON = "comparison"
    PROCESS = "process"
    REGULATION = "regulation"
    TECHNICAL = "technical"


@dataclass
class PromptTemplate:
    """프롬프트 템플릿 데이터 클래스"""
    system_prompt: str
    user_prompt_template: str
    context_prompt_template: Optional[str] = None
    examples: Optional[List[Dict[str, str]]] = None
    
    def format(self, **kwargs) -> str:
        """템플릿에 변수 삽입"""
        return self.user_prompt_template.format(**kwargs)


class FinancialPromptManager:
    """금융 도메인 특화 프롬프트 관리자"""
    
    def __init__(self):
        """금융 보안 전문가 프롬프트 초기화"""
        
        # 기본 시스템 프롬프트 (금융 보안 전문가)
        self.base_system_prompt = """당신은 한국 금융위원회와 금융보안원에서 20년 이상 근무한 금융 보안 및 AI 규제 전문가입니다.

전문 분야:
1. 금융 AI 시스템의 보안 요구사항과 규제
2. 금융위원회, 금융보안원, 한국은행의 가이드라인 및 규정
3. AI 모델의 취약점 분석 및 보안 대책
4. 금융 데이터 보호 및 프라이버시
5. 금융 AI 시스템의 리스크 관리 및 거버넌스

답변 원칙:
- 정확성: 금융 규제와 보안 가이드라인에 기반한 정확한 정보 제공
- 실용성: 실무에 적용 가능한 구체적인 방안 제시
- 최신성: 최신 금융 AI 보안 트렌드와 규제 반영
- 명확성: 전문 용어는 필요시 쉽게 설명"""

        # 질문 유형별 템플릿 초기화
        self._init_templates()
        
        # Few-shot 예시 초기화
        self._init_examples()
        
    def _init_templates(self):
        """질문 유형별 템플릿 초기화"""
        
        self.templates = {
            # 객관식 질문용 템플릿
            QuestionType.MULTIPLE_CHOICE: PromptTemplate(
                system_prompt=self.base_system_prompt + """

객관식 답변 시:
- 반드시 정답 번호만 출력하세요 (예: 3)
- 추가 설명은 하지 마세요
- 확실하지 않으면 가장 적절한 답을 선택하세요""",
                
                user_prompt_template="""아래 금융 보안 관련 질문을 읽고 가장 적절한 답을 선택하세요.

질문: {question}

{choices}

참고 자료:
{context}

정답 번호:""",
                
                context_prompt_template="""관련 금융 보안 가이드라인:
{retrieved_context}

핵심 규정:
{key_regulations}"""
            ),
            
            # 서술형 질문용 템플릿
            QuestionType.DESCRIPTIVE: PromptTemplate(
                system_prompt=self.base_system_prompt + """

서술형 답변 시:
- 핵심 내용을 명확하고 구조적으로 설명하세요
- 관련 규제나 가이드라인을 인용하세요
- 실무 적용 방안을 포함하세요
- 필요시 예시를 들어 설명하세요""",
                
                user_prompt_template="""아래 금융 보안 관련 질문에 전문가 관점에서 답변하세요.

질문: {question}

참고 자료:
{context}

답변:""",
                
                context_prompt_template="""관련 문서:
{retrieved_context}

주요 키워드: {keywords}
규제 기관: {regulatory_body}"""
            ),
            
            # 정의 설명 질문
            QuestionType.DEFINITION: PromptTemplate(
                system_prompt=self.base_system_prompt,
                user_prompt_template="""다음 금융 보안 용어나 개념을 정의하고 설명하세요.

용어/개념: {question}

참고 자료:
{context}

정의 및 설명:
1. 정의:
2. 주요 특징:
3. 금융 분야 적용:
4. 관련 규제:""",
            ),
            
            # 비교 분석 질문
            QuestionType.COMPARISON: PromptTemplate(
                system_prompt=self.base_system_prompt,
                user_prompt_template="""다음 항목들을 금융 보안 관점에서 비교 분석하세요.

질문: {question}

참고 자료:
{context}

비교 분석:
1. 공통점:
2. 차이점:
3. 각각의 장단점:
4. 금융 분야 적용 시 고려사항:""",
            ),
            
            # 프로세스/절차 질문
            QuestionType.PROCESS: PromptTemplate(
                system_prompt=self.base_system_prompt,
                user_prompt_template="""다음 금융 보안 프로세스나 절차를 단계별로 설명하세요.

질문: {question}

참고 자료:
{context}

프로세스 설명:
[단계 1]
[단계 2]
[단계 3]
...

주의사항:""",
            ),
            
            # 규제/컴플라이언스 질문
            QuestionType.REGULATION: PromptTemplate(
                system_prompt=self.base_system_prompt + """

규제 관련 답변 시:
- 관련 법령과 가이드라인을 명시하세요
- 규제 기관과 발행일을 포함하세요
- 실무 준수 방안을 제시하세요""",
                
                user_prompt_template="""다음 금융 규제나 컴플라이언스 관련 질문에 답하세요.

질문: {question}

참고 자료:
{context}

답변:
관련 규제:
준수 요구사항:
실무 적용 방안:""",
            ),
            
            # 기술적 질문
            QuestionType.TECHNICAL: PromptTemplate(
                system_prompt=self.base_system_prompt,
                user_prompt_template="""다음 금융 AI 시스템의 기술적 질문에 답하세요.

질문: {question}

참고 자료:
{context}

기술적 답변:
1. 기술 개요:
2. 구현 방법:
3. 보안 고려사항:
4. 모니터링 방안:""",
            ),
        }
    
    def _init_examples(self):
        """Few-shot 학습용 예시 초기화"""
        
        self.examples = {
            QuestionType.MULTIPLE_CHOICE: [
                {
                    "question": "금융 AI 시스템에서 가장 중요한 보안 요소는?",
                    "choices": "1. 성능\n2. 데이터 보호\n3. 사용자 경험\n4. 비용",
                    "answer": "2"
                },
                {
                    "question": "AI 모델의 적대적 공격 대응 방법으로 적절하지 않은 것은?",
                    "choices": "1. 입력 검증\n2. 모델 앙상블\n3. 데이터 공개\n4. 이상 탐지",
                    "answer": "3"
                }
            ],
            
            QuestionType.DESCRIPTIVE: [
                {
                    "question": "금융 AI 시스템의 편향성 문제와 해결 방안을 설명하시오.",
                    "answer": """금융 AI 시스템의 편향성은 학습 데이터의 불균형이나 알고리즘 설계상의 문제로 발생하며, 
대출 심사, 신용 평가 등에서 특정 집단에 불리한 결정을 내릴 수 있습니다.

주요 해결 방안:
1. 데이터 다양성 확보: 다양한 인구 집단을 대표하는 균형잡힌 학습 데이터 구축
2. 공정성 지표 모니터링: Demographic Parity, Equal Opportunity 등 공정성 메트릭 적용
3. 정기적 감사: AI 모델의 의사결정 패턴을 주기적으로 검토하고 편향 탐지
4. 설명가능한 AI: 모델의 의사결정 과정을 투명하게 공개하여 편향 식별"""
                }
            ]
        }
    
    def classify_question(self, question: str) -> QuestionType:
        """질문 유형 자동 분류"""
        
        # 객관식 패턴 감지
        if self._is_multiple_choice(question):
            return QuestionType.MULTIPLE_CHOICE
        
        # 키워드 기반 분류
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ["정의", "무엇", "개념"]):
            return QuestionType.DEFINITION
        elif any(keyword in question_lower for keyword in ["비교", "차이", "vs", "대비"]):
            return QuestionType.COMPARISON
        elif any(keyword in question_lower for keyword in ["절차", "프로세스", "단계", "방법"]):
            return QuestionType.PROCESS
        elif any(keyword in question_lower for keyword in ["규제", "규정", "법령", "가이드라인", "준수"]):
            return QuestionType.REGULATION
        elif any(keyword in question_lower for keyword in ["구현", "기술", "알고리즘", "시스템", "아키텍처"]):
            return QuestionType.TECHNICAL
        else:
            return QuestionType.DESCRIPTIVE
    
    def _is_multiple_choice(self, question: str) -> bool:
        """객관식 질문 여부 판단"""
        lines = question.strip().split('\n')
        choice_pattern = re.compile(r'^\s*[1-9]\d*[\s\)\.]\s*.+')
        
        choices = []
        for line in lines:
            if choice_pattern.match(line):
                choices.append(line)
        
        return len(choices) >= 2
    
    def create_prompt(self, 
                     question: str,
                     context: str = "",
                     question_type: Optional[QuestionType] = None,
                     use_few_shot: bool = True) -> Tuple[str, str]:
        """
        질문에 맞는 프롬프트 생성
        
        Args:
            question: 입력 질문
            context: RAG에서 검색된 컨텍스트
            question_type: 질문 유형 (None이면 자동 분류)
            use_few_shot: Few-shot 예시 사용 여부
            
        Returns:
            (system_prompt, user_prompt) 튜플
        """
        
        # 질문 유형 결정
        if question_type is None:
            question_type = self.classify_question(question)
        
        # 템플릿 선택
        template = self.templates[question_type]
        
        # 객관식인 경우 선택지 분리
        if question_type == QuestionType.MULTIPLE_CHOICE:
            question_text, choices = self._extract_choices(question)
            user_prompt = template.format(
                question=question_text,
                choices=choices,
                context=context
            )
        else:
            user_prompt = template.format(
                question=question,
                context=context
            )
        
        # Few-shot 예시 추가
        system_prompt = template.system_prompt
        if use_few_shot and question_type in self.examples:
            examples_text = "\n\n예시:\n"
            for example in self.examples[question_type][:2]:  # 최대 2개 예시
                examples_text += f"Q: {example['question']}\n"
                if 'choices' in example:
                    examples_text += f"{example['choices']}\n"
                examples_text += f"A: {example['answer']}\n\n"
            system_prompt += examples_text
        
        return system_prompt, user_prompt
    
    def _extract_choices(self, question: str) -> Tuple[str, str]:
        """객관식 질문에서 질문과 선택지 분리"""
        lines = question.strip().split('\n')
        choice_pattern = re.compile(r'^\s*[1-9]\d*[\s\)\.]\s*.+')
        
        question_lines = []
        choice_lines = []
        
        for line in lines:
            if choice_pattern.match(line):
                choice_lines.append(line)
            elif choice_lines:  # 선택지가 시작된 후의 비선택지 라인
                break
            else:
                question_lines.append(line)
        
        question_text = '\n'.join(question_lines).strip()
        choices_text = '\n'.join(choice_lines).strip()
        
        return question_text, choices_text
    
    def enhance_with_context(self, 
                            prompt: str,
                            retrieved_docs: List[str],
                            keywords: List[str] = None) -> str:
        """
        RAG 검색 결과로 프롬프트 강화
        
        Args:
            prompt: 기본 프롬프트
            retrieved_docs: 검색된 문서들
            keywords: 주요 키워드
            
        Returns:
            강화된 프롬프트
        """
        
        # 검색 문서 요약
        context_summary = "\n\n".join([
            f"[참고 {i+1}] {doc[:500]}..." if len(doc) > 500 else f"[참고 {i+1}] {doc}"
            for i, doc in enumerate(retrieved_docs[:3])  # 상위 3개만 사용
        ])
        
        # 키워드 강조
        if keywords:
            keyword_text = f"\n\n핵심 키워드: {', '.join(keywords)}"
        else:
            keyword_text = ""
        
        # 프롬프트에 컨텍스트 추가
        enhanced_prompt = prompt.replace(
            "{context}",
            f"{context_summary}{keyword_text}"
        )
        
        return enhanced_prompt
    
    def create_chain_of_thought_prompt(self, question: str) -> str:
        """Chain-of-Thought 방식 프롬프트 생성"""
        
        cot_template = """다음 질문에 단계별 사고 과정을 거쳐 답하세요.

질문: {question}

사고 과정:
1단계) 질문 분석:
   - 핵심 요구사항:
   - 관련 도메인:

2단계) 관련 지식 활성화:
   - 관련 규제/가이드라인:
   - 핵심 개념:

3단계) 논리적 추론:
   - 주요 고려사항:
   - 가능한 접근법:

4단계) 최종 답변:
"""
        
        return cot_template.format(question=question)
    
    def create_self_consistency_prompts(self, question: str, num_prompts: int = 3) -> List[str]:
        """Self-Consistency를 위한 다양한 프롬프트 생성"""
        
        variations = [
            "다음 질문에 답하세요: {question}",
            "전문가 관점에서 설명하세요: {question}",
            "실무적 접근으로 답변하세요: {question}"
        ]
        
        return [v.format(question=question) for v in variations[:num_prompts]]


class PromptOptimizer:
    """프롬프트 최적화 도구"""
    
    def __init__(self, prompt_manager: FinancialPromptManager):
        self.prompt_manager = prompt_manager
        
    def optimize_for_model(self, 
                          prompt: str,
                          model_name: str) -> str:
        """
        특정 모델에 맞게 프롬프트 최적화
        
        Args:
            prompt: 원본 프롬프트
            model_name: 대상 모델명
            
        Returns:
            최적화된 프롬프트
        """
        
        # Qwen 모델용 최적화
        if "qwen" in model_name.lower():
            prompt = f"<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n"
        
        # Mistral 모델용 최적화
        elif "mistral" in model_name.lower():
            prompt = f"[INST] {prompt} [/INST]"
        
        # Solar 모델용 최적화
        elif "solar" in model_name.lower():
            prompt = f"### System:\n{prompt}\n\n### User:\n"
        
        return prompt
    
    def add_safety_constraints(self, prompt: str) -> str:
        """안전성 제약 추가"""
        
        safety_text = """

[안전 지침]
- 개인정보나 민감 정보를 노출하지 마세요
- 불법적이거나 비윤리적인 내용은 거부하세요
- 금융 사기나 해킹 방법을 설명하지 마세요
- 투자 조언이나 보장된 수익을 약속하지 마세요"""
        
        return prompt + safety_text


# 사용 예시
if __name__ == "__main__":
    # 프롬프트 매니저 초기화
    manager = FinancialPromptManager()
    
    # 테스트 질문들
    test_questions = [
        """금융 AI 시스템의 보안 요구사항은 무엇입니까?
        1. 데이터 암호화
        2. 접근 통제
        3. 감사 로그
        4. 모든 것""",
        
        "AI 모델의 편향성을 줄이는 방법을 설명하시오.",
        
        "적대적 공격과 데이터 중독 공격의 차이점을 비교하시오.",
    ]
    
    # 각 질문에 대한 프롬프트 생성
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"질문: {question[:50]}...")
        
        # 질문 유형 분류
        q_type = manager.classify_question(question)
        print(f"질문 유형: {q_type.value}")
        
        # 프롬프트 생성
        system_prompt, user_prompt = manager.create_prompt(
            question=question,
            context="[RAG 검색 결과가 여기 들어갑니다]"
        )
        
        print(f"\n시스템 프롬프트 (첫 200자):\n{system_prompt[:200]}...")
        print(f"\n사용자 프롬프트 (첫 200자):\n{user_prompt[:200]}...")