"""
금융 도메인 특화 프롬프트 템플릿
Qwen2.5-7B에 최적화된 프롬프트 엔지니어링
"""

from typing import List, Dict, Optional
from enum import Enum


class QuestionType(Enum):
    """질문 유형 분류"""
    DEFINITION = "definition"  # 정의/개념
    PROCEDURE = "procedure"    # 절차/방법
    COMPARISON = "comparison"  # 비교/대조
    ANALYSIS = "analysis"      # 분석/평가
    COMPLIANCE = "compliance"  # 규제/준수
    SECURITY = "security"      # 보안/위험


class FinancePromptTemplate:
    """금융 도메인 프롬프트 템플릿"""
    
    # 기본 시스템 프롬프트
    BASE_SYSTEM = """당신은 한국 금융 보안 분야의 전문가입니다.
다음 원칙을 따라 답변하세요:
1. 정확성: 제공된 문서의 내용을 정확히 인용
2. 전문성: 금융 용어와 개념을 정확히 사용
3. 명확성: 구조화된 답변으로 이해하기 쉽게 설명
4. 규제 준수: 한국 금융 규제 및 가이드라인 준수"""
    
    # 질문 유형별 프롬프트
    TYPE_PROMPTS = {
        QuestionType.DEFINITION: """
제공된 문서를 바탕으로 다음 개념을 명확히 정의하고 설명하세요:
- 정의: 핵심 개념을 간단명료하게
- 특징: 주요 특징과 구성 요소
- 예시: 실제 적용 사례 (있다면)
""",
        QuestionType.PROCEDURE: """
제공된 문서를 바탕으로 다음 절차를 단계별로 설명하세요:
1. 각 단계를 번호로 구분
2. 구체적인 실행 방법 포함
3. 주의사항이나 팁 추가
""",
        QuestionType.COMPARISON: """
제공된 문서를 바탕으로 비교 분석하세요:
- 공통점: 유사한 특징들
- 차이점: 구별되는 특징들
- 장단점: 각각의 강점과 약점
- 권장사항: 상황별 적합한 선택
""",
        QuestionType.ANALYSIS: """
제공된 문서를 바탕으로 심층 분석하세요:
- 현황: 현재 상태와 문제점
- 원인: 근본 원인 분석
- 영향: 예상되는 영향과 리스크
- 해결책: 구체적인 개선 방안
""",
        QuestionType.COMPLIANCE: """
제공된 문서와 관련 규제를 바탕으로 답변하세요:
- 관련 규제: 적용되는 법규와 가이드라인
- 요구사항: 충족해야 할 구체적 요건
- 준수 방법: 실제 구현 방안
- 벌칙/리스크: 미준수 시 결과
""",
        QuestionType.SECURITY: """
제공된 문서를 바탕으로 보안 관점에서 답변하세요:
- 위협 요소: 잠재적 보안 위험
- 취약점: 시스템의 약점
- 대응 방안: 구체적인 보안 조치
- 모범 사례: 업계 베스트 프랙티스
"""
    }
    
    @classmethod
    def detect_question_type(cls, question: str) -> QuestionType:
        """질문 유형 자동 감지"""
        
        # 키워드 기반 분류
        type_keywords = {
            QuestionType.DEFINITION: ["무엇", "정의", "개념", "의미", "란"],
            QuestionType.PROCEDURE: ["어떻게", "방법", "절차", "단계", "과정"],
            QuestionType.COMPARISON: ["차이", "비교", "대조", "vs", "장단점"],
            QuestionType.ANALYSIS: ["분석", "평가", "검토", "영향", "효과"],
            QuestionType.COMPLIANCE: ["규제", "법규", "준수", "요구사항", "가이드라인"],
            QuestionType.SECURITY: ["보안", "위험", "취약", "공격", "방어", "보호"]
        }
        
        question_lower = question.lower()
        
        for q_type, keywords in type_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return q_type
        
        # 기본값
        return QuestionType.ANALYSIS
    
    @classmethod
    def create_prompt(
        cls,
        question: str,
        contexts: List[str],
        question_type: Optional[QuestionType] = None,
        include_citations: bool = True
    ) -> Dict[str, str]:
        """
        프롬프트 생성
        
        Args:
            question: 사용자 질문
            contexts: 검색된 컨텍스트
            question_type: 질문 유형 (자동 감지 가능)
            include_citations: 인용 포함 여부
        
        Returns:
            시스템 프롬프트와 사용자 프롬프트
        """
        # 질문 유형 감지
        if question_type is None:
            question_type = cls.detect_question_type(question)
        
        # 컨텍스트 포맷팅
        formatted_contexts = "\n\n".join([
            f"[문서 {i+1}]\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])
        
        # 시스템 프롬프트
        system_prompt = cls.BASE_SYSTEM
        
        # 유형별 추가 지시
        type_instruction = cls.TYPE_PROMPTS.get(
            question_type,
            "제공된 문서를 바탕으로 정확하고 상세하게 답변하세요."
        )
        
        # 인용 지시
        citation_instruction = ""
        if include_citations:
            citation_instruction = "\n\n답변 시 관련 문서 번호를 [문서 N] 형식으로 인용하세요."
        
        # 사용자 프롬프트
        user_prompt = f"""참고 문서:
{formatted_contexts}

{type_instruction}

질문: {question}
{citation_instruction}

답변:"""
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    @classmethod
    def create_multi_turn_prompt(
        cls,
        conversation_history: List[Dict[str, str]],
        new_question: str,
        contexts: List[str]
    ) -> Dict[str, str]:
        """
        다중 턴 대화 프롬프트 생성
        
        Args:
            conversation_history: 이전 대화 기록
            new_question: 새 질문
            contexts: 새로운 컨텍스트
        
        Returns:
            프롬프트
        """
        # 대화 기록 포맷팅
        history_text = ""
        for turn in conversation_history[-3:]:  # 최근 3턴만
            history_text += f"사용자: {turn['question']}\n"
            history_text += f"어시스턴트: {turn['answer']}\n\n"
        
        # 컨텍스트 포맷팅
        formatted_contexts = "\n\n".join([
            f"[문서 {i+1}]\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])
        
        user_prompt = f"""이전 대화:
{history_text}

새로운 참고 문서:
{formatted_contexts}

이전 대화의 맥락을 고려하여 다음 질문에 답변하세요.

질문: {new_question}

답변:"""
        
        return {
            "system": cls.BASE_SYSTEM,
            "user": user_prompt
        }
    
    @classmethod
    def create_cot_prompt(
        cls,
        question: str,
        contexts: List[str]
    ) -> Dict[str, str]:
        """
        Chain-of-Thought 프롬프트 생성
        복잡한 추론이 필요한 경우
        
        Args:
            question: 질문
            contexts: 컨텍스트
        
        Returns:
            CoT 프롬프트
        """
        formatted_contexts = "\n\n".join([
            f"[문서 {i+1}]\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])
        
        user_prompt = f"""참고 문서:
{formatted_contexts}

다음 질문에 대해 단계별로 추론하여 답변하세요:

질문: {question}

답변 과정:
1. 질문 분석: 질문의 핵심 요구사항 파악
2. 정보 추출: 문서에서 관련 정보 식별
3. 논리 전개: 추출된 정보를 바탕으로 논리적 연결
4. 결론 도출: 최종 답변 정리

단계별 추론:
"""
        
        return {
            "system": cls.BASE_SYSTEM + "\n\n단계별로 명확하게 추론 과정을 보여주세요.",
            "user": user_prompt
        }


class PromptOptimizer:
    """프롬프트 최적화 도구"""
    
    @staticmethod
    def truncate_context(
        contexts: List[str],
        max_tokens: int = 1500,
        tokenizer = None
    ) -> List[str]:
        """
        컨텍스트 길이 최적화
        
        Args:
            contexts: 원본 컨텍스트
            max_tokens: 최대 토큰 수
            tokenizer: 토크나이저
        
        Returns:
            잘린 컨텍스트
        """
        if tokenizer is None:
            # 간단한 추정: 한글 3자 = 1토큰
            max_chars = max_tokens * 3
            
            truncated = []
            total_chars = 0
            
            for ctx in contexts:
                if total_chars + len(ctx) <= max_chars:
                    truncated.append(ctx)
                    total_chars += len(ctx)
                else:
                    remaining = max_chars - total_chars
                    if remaining > 100:  # 최소 100자는 포함
                        truncated.append(ctx[:remaining] + "...")
                    break
            
            return truncated
        else:
            # 실제 토크나이저 사용
            truncated = []
            total_tokens = 0
            
            for ctx in contexts:
                tokens = tokenizer.encode(ctx)
                if total_tokens + len(tokens) <= max_tokens:
                    truncated.append(ctx)
                    total_tokens += len(tokens)
                else:
                    remaining_tokens = max_tokens - total_tokens
                    if remaining_tokens > 50:
                        truncated_tokens = tokens[:remaining_tokens]
                        truncated_text = tokenizer.decode(truncated_tokens)
                        truncated.append(truncated_text + "...")
                    break
            
            return truncated
    
    @staticmethod
    def add_few_shot_examples(
        prompt: str,
        examples: List[Dict[str, str]]
    ) -> str:
        """
        Few-shot 예제 추가
        
        Args:
            prompt: 원본 프롬프트
            examples: 예제 리스트
        
        Returns:
            예제가 추가된 프롬프트
        """
        example_text = "다음은 좋은 답변의 예시입니다:\n\n"
        
        for i, example in enumerate(examples, 1):
            example_text += f"예시 {i}:\n"
            example_text += f"질문: {example['question']}\n"
            example_text += f"답변: {example['answer']}\n\n"
        
        return example_text + "\n" + prompt


if __name__ == "__main__":
    # 템플릿 테스트
    print("="*60)
    print(" 프롬프트 템플릿 테스트")
    print("="*60)
    
    # 테스트 데이터
    test_question = "AI 모델의 적대적 공격을 방어하는 방법은?"
    test_contexts = [
        "적대적 공격은 AI 모델을 속이기 위해 의도적으로 조작된 입력을 사용하는 공격입니다.",
        "방어 방법으로는 입력 검증, 적대적 훈련, 모델 앙상블 등이 있습니다."
    ]
    
    # 질문 유형 감지
    q_type = FinancePromptTemplate.detect_question_type(test_question)
    print(f"\n감지된 질문 유형: {q_type.value}")
    
    # 프롬프트 생성
    prompts = FinancePromptTemplate.create_prompt(
        test_question,
        test_contexts
    )
    
    print("\n생성된 프롬프트:")
    print("-"*60)
    print("[System]")
    print(prompts["system"])
    print("\n[User]")
    print(prompts["user"][:500] + "...")
    
    print("\n✅ 프롬프트 템플릿 준비 완료!")