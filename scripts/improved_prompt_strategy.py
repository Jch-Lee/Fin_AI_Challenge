#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
개선된 프롬프트 전략 - 참고자료와 모델 지식의 균형
"""

from typing import List, Optional

class ImprovedPromptStrategy:
    """참고자료 의존도를 균형있게 조절하는 프롬프트 전략"""
    
    @staticmethod
    def create_balanced_prompt(
        question: str,
        contexts: List[str],
        is_mc: bool = False,
        confidence_threshold: float = 0.7
    ) -> str:
        """
        균형잡힌 프롬프트 생성
        - 참고자료가 있을 때: 참고하되 비판적 검토
        - 참고자료가 없을 때: 모델의 지식 활용
        """
        
        if is_mc:
            # 객관식 문제
            if contexts and len(contexts) > 0:
                # 참고자료가 있는 경우
                context_text = "\n\n".join(contexts[:3])
                prompt = f"""당신은 한국 금융보안 분야의 전문가입니다.

[참고 문서]
{context_text}

[중요 지침]
1. 위 참고 문서를 우선 검토하되, 금융보안 전문지식도 함께 활용하세요.
2. 참고 문서가 불완전하거나 관련성이 낮다면, 당신의 전문지식을 바탕으로 답하세요.
3. 한국 금융 규제와 최신 보안 트렌드를 고려하세요.

[질문]
{question}

위 객관식 문제에 대해:
1. 먼저 각 선택지를 신중히 검토하세요
2. 참고 문서와 전문지식을 종합하여 판단하세요
3. 최종 답변은 숫자만 출력하세요

답변:"""
            else:
                # 참고자료가 없는 경우
                prompt = f"""당신은 한국 금융보안 분야의 전문가입니다.

[중요 지침]
다음 원칙에 따라 답변하세요:
1. 한국 금융 규제 및 보안 표준 (전자금융거래법, 정보보호법 등)
2. 금융보안원, 금융위원회의 최신 가이드라인
3. ISMS-P, ISO 27001 등 보안 인증 기준
4. 업계 베스트 프랙티스와 실무 경험

[질문]
{question}

금융보안 전문지식을 바탕으로 가장 적절한 답을 선택하세요.
정답 번호만 출력하세요.

답변:"""
            
        else:
            # 주관식 문제
            if contexts and len(contexts) > 0:
                context_text = "\n\n".join(contexts[:3])
                prompt = f"""당신은 한국 금융보안 분야의 전문가입니다.

[참고 문서]
{context_text}

[답변 지침]
1. 참고 문서의 정보를 기반으로 하되, 부족한 부분은 전문지식으로 보완하세요.
2. 한국 금융 환경과 규제 특성을 반영하세요.
3. 실무적이고 구체적인 답변을 제공하세요.

[질문]
{question}

구조화된 답변:
• 핵심 답변: (간단명료하게)
• 근거: (참고 문서 또는 전문지식 기반)
• 보충 설명: (필요시 추가 정보)

답변:"""
            else:
                # 참고자료가 없는 경우
                prompt = f"""당신은 한국 금융보안 분야의 전문가입니다.

[전문지식 활용 지침]
다음 분야의 전문지식을 활용하여 답변하세요:
1. 정보보호 관리체계 (ISMS-P)
2. 개인정보보호법 및 금융 규제
3. 사이버 보안 위협 및 대응
4. 금융 IT 시스템 보안
5. 재해복구 및 업무연속성

[질문]
{question}

금융보안 전문가로서 정확하고 실용적인 답변을 제공하세요.

답변:"""
        
        return prompt
    
    @staticmethod
    def create_cot_balanced_prompt(
        question: str,
        contexts: List[str],
        is_mc: bool = False
    ) -> str:
        """Chain-of-Thought 추론을 활용한 균형잡힌 프롬프트"""
        
        if contexts and len(contexts) > 0:
            context_text = "\n\n".join(contexts[:3])
            base_context = f"""[참고 문서]
{context_text}

[분석 과정]
Step 1. 참고 문서 검토: 제공된 문서에서 관련 정보 추출
Step 2. 전문지식 활용: 문서에 없는 정보는 금융보안 지식으로 보완
Step 3. 종합 판단: 두 정보원을 결합하여 최적 답안 도출"""
        else:
            base_context = """[분석 과정]
Step 1. 질문 분석: 핵심 요구사항 파악
Step 2. 전문지식 활용: 금융보안 관련 지식 적용
Step 3. 답안 구성: 체계적이고 정확한 답변 작성"""
        
        if is_mc:
            prompt = f"""당신은 한국 금융보안 분야의 전문가입니다.

{base_context}

[질문]
{question}

각 선택지를 단계별로 분석한 후, 최종 답만 숫자로 출력하세요.

분석:
"""
        else:
            prompt = f"""당신은 한국 금융보안 분야의 전문가입니다.

{base_context}

[질문]
{question}

단계별 분석을 통해 답변을 도출하세요.

분석 및 답변:
"""
        
        return prompt
    
    @staticmethod
    def create_confidence_aware_prompt(
        question: str,
        contexts: List[str],
        retrieval_scores: List[float],
        is_mc: bool = False
    ) -> str:
        """검색 신뢰도를 고려한 프롬프트"""
        
        # 평균 검색 점수 계산
        avg_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0
        
        if avg_score > 0.8:
            confidence_instruction = "참고 문서의 신뢰도가 높습니다. 문서 내용을 중심으로 답변하세요."
        elif avg_score > 0.5:
            confidence_instruction = "참고 문서를 활용하되, 전문지식으로 보완하여 답변하세요."
        else:
            confidence_instruction = "참고 문서의 관련성이 낮습니다. 주로 전문지식을 활용하세요."
        
        context_text = "\n\n".join(contexts[:3]) if contexts else ""
        
        if is_mc:
            prompt = f"""당신은 한국 금융보안 분야의 전문가입니다.

[참고 문서] (평균 관련성: {avg_score:.2f})
{context_text}

[지침]
{confidence_instruction}

[질문]
{question}

정답 번호만 출력하세요.

답변:"""
        else:
            prompt = f"""당신은 한국 금융보안 분야의 전문가입니다.

[참고 문서] (평균 관련성: {avg_score:.2f})
{context_text}

[지침]
{confidence_instruction}

[질문]
{question}

답변:"""
        
        return prompt


class FallbackStrategies:
    """참고자료가 부족할 때의 대체 전략"""
    
    @staticmethod
    def domain_knowledge_prompt(question: str, is_mc: bool = False) -> str:
        """도메인 지식 중심 프롬프트"""
        
        domain_areas = """
        1. 정보보호 관리체계 (ISMS-P): 인증 기준, 통제 항목
        2. 개인정보보호: 개인정보보호법, GDPR, 수집/이용/제공 원칙
        3. 금융 보안: 전자금융거래법, 금융보안원 가이드라인
        4. 사이버 보안: 랜섬웨어, APT, 제로데이 공격 대응
        5. 암호화: 대칭/비대칭 암호, 해시 함수, 전자서명
        6. 네트워크 보안: 방화벽, IDS/IPS, VPN, 제로트러스트
        7. 재해복구: BCP, DRP, RTO/RPO
        """
        
        if is_mc:
            return f"""당신은 한국 금융보안 전문가입니다.

[전문 분야]
{domain_areas}

[질문]
{question}

위 전문 분야의 지식을 활용하여 정답을 선택하세요.
답변은 숫자만 출력하세요.

답변:"""
        else:
            return f"""당신은 한국 금융보안 전문가입니다.

[전문 분야]
{domain_areas}

[질문]
{question}

위 전문 분야의 지식을 활용하여 상세히 답변하세요.

답변:"""
    
    @staticmethod
    def reasoning_chain_prompt(question: str, is_mc: bool = False) -> str:
        """추론 체인을 통한 답변 생성"""
        
        if is_mc:
            return f"""당신은 금융보안 전문가입니다.

[질문]
{question}

다음 추론 과정을 따르세요:
1. 질문의 핵심 개념 파악
2. 각 선택지의 의미 분석
3. 금융보안 원칙과 규제 적용
4. 논리적 배제를 통한 답 도출

추론 과정: (간단히)
최종 답: (숫자만)

답변:"""
        else:
            return f"""당신은 금융보안 전문가입니다.

[질문]
{question}

다음 구조로 답변하세요:
1. 개념 정의
2. 한국 금융 환경에서의 적용
3. 실무적 고려사항
4. 결론

답변:"""


if __name__ == "__main__":
    # 테스트
    print("="*60)
    print("개선된 프롬프트 전략 테스트")
    print("="*60)
    
    # 테스트 질문
    test_question = "개인정보 영향평가는 언제 수행해야 하는가?"
    test_contexts = [
        "개인정보 영향평가는 대규모 개인정보 처리 시 필요합니다.",
        "신규 시스템 도입 전에 평가를 실시해야 합니다."
    ]
    test_scores = [0.75, 0.65]
    
    # 균형잡힌 프롬프트
    balanced = ImprovedPromptStrategy.create_balanced_prompt(
        test_question, test_contexts, is_mc=False
    )
    print("\n[균형잡힌 프롬프트]")
    print(balanced[:500] + "...")
    
    # 신뢰도 기반 프롬프트
    confidence = ImprovedPromptStrategy.create_confidence_aware_prompt(
        test_question, test_contexts, test_scores, is_mc=False
    )
    print("\n[신뢰도 기반 프롬프트]")
    print(confidence[:500] + "...")
    
    # 도메인 지식 프롬프트
    domain = FallbackStrategies.domain_knowledge_prompt(
        test_question, is_mc=False
    )
    print("\n[도메인 지식 프롬프트]")
    print(domain[:500] + "...")
    
    print("\n✅ 프롬프트 전략 테스트 완료!")