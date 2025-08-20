#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic prompts for RAG system analysis
These prompts are designed to reveal the reasoning process and document usage
"""

def create_diagnostic_prompt_mc(question: str, contexts: list, scores: list) -> str:
    """
    Create diagnostic prompt for multiple choice questions
    
    Args:
        question: The question text
        contexts: List of retrieved documents
        scores: List of score information for each document
            Each item should have: {'bm25_score', 'vector_score', 'hybrid_score'}
    """
    
    # Format retrieved documents with scores
    context_section = ""
    if contexts and scores:
        doc_parts = []
        for i, (ctx, score) in enumerate(zip(contexts[:5], scores[:5])):
            doc_parts.append(f"""[문서 {i+1}]
- BM25 점수: {score.get('bm25_score', 0):.4f}
- Vector 점수: {score.get('vector_score', 0):.4f}  
- 종합 점수: {score.get('hybrid_score', 0):.4f}
- 내용: {ctx[:500]}...""")
        
        context_section = "\n\n".join(doc_parts)
    
    system_prompt = """당신은 한국 금융보안 분야의 전문가입니다. 
답변 과정을 단계별로 설명하면서 최종 답을 제시해야 합니다."""
    
    user_prompt = f"""[검색된 참고 문서]
{context_section}

[분석 지침]
1. 각 문서의 관련성을 평가하세요
2. BM25 점수(키워드 매칭)와 Vector 점수(의미 유사도) 중 어느 것이 더 유용한지 판단하세요
3. 문서에서 답변에 필요한 핵심 정보를 추출하세요
4. 사고 과정을 명확히 보여주세요

[질문]
{question}

[답변 형식]
### 문서 관련성 분석
- 가장 유용한 문서: [문서 번호와 이유]
- BM25 vs Vector: [어느 점수가 더 신뢰할 만한지]

### 사고 과정
[단계별 추론 과정을 설명]

### 근거
[어떤 문서의 어떤 부분을 참고했는지]

### 최종 답
[숫자만 출력 (1-5 중 하나)]"""
    
    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant"""


def create_diagnostic_prompt_desc(question: str, contexts: list, scores: list) -> str:
    """
    Create diagnostic prompt for descriptive questions
    
    Args:
        question: The question text
        contexts: List of retrieved documents
        scores: List of score information for each document
    """
    
    # Format retrieved documents with scores
    context_section = ""
    if contexts and scores:
        doc_parts = []
        for i, (ctx, score) in enumerate(zip(contexts[:5], scores[:5])):
            doc_parts.append(f"""[문서 {i+1}]
- BM25 점수: {score.get('bm25_score', 0):.4f}
- Vector 점수: {score.get('vector_score', 0):.4f}
- 종합 점수: {score.get('hybrid_score', 0):.4f}
- 내용: {ctx[:500]}...""")
        
        context_section = "\n\n".join(doc_parts)
    
    system_prompt = """당신은 한국 금융보안 분야의 전문가입니다.
답변 과정을 단계별로 설명하면서 최종 답을 제시해야 합니다."""
    
    user_prompt = f"""[검색된 참고 문서]
{context_section}

[분석 지침]
1. 각 문서의 관련성을 평가하세요
2. BM25 점수(키워드 매칭)와 Vector 점수(의미 유사도) 중 어느 것이 더 유용한지 판단하세요
3. 문서에서 답변에 필요한 핵심 정보를 추출하세요
4. 문서 정보가 부족한 경우, 일반적인 금융보안 지식을 활용하세요

[질문]
{question}

[답변 형식]
### 문서 관련성 분석
- 가장 유용한 문서: [문서 번호와 이유]
- BM25 vs Vector: [어느 점수가 더 신뢰할 만한지]
- 문서 정보 충분성: [충분/부족 및 이유]

### 사고 과정
[단계별 추론 과정을 설명]

### 근거
[어떤 문서의 어떤 부분을 참고했는지, 또는 일반 지식 활용 여부]

### 최종 답변
[간결하고 명확한 답변 (100-300자)]"""
    
    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant"""


def extract_diagnostic_answer(response: str, question_type: str) -> dict:
    """
    Extract structured information from diagnostic response
    
    Returns:
        dict with keys:
        - final_answer: The actual answer
        - document_analysis: Analysis of document relevance
        - thought_process: Step-by-step reasoning
        - evidence: Which documents were used
        - score_preference: BM25 vs Vector preference
    """
    
    result = {
        'final_answer': '',
        'document_analysis': '',
        'thought_process': '',
        'evidence': '',
        'score_preference': '',
        'raw_response': response
    }
    
    # Extract sections from response
    sections = response.split('###')
    
    for section in sections:
        section = section.strip()
        if section.startswith('문서 관련성 분석'):
            result['document_analysis'] = section.replace('문서 관련성 분석', '').strip()
            # Extract score preference
            if 'BM25 vs Vector:' in section:
                score_part = section.split('BM25 vs Vector:')[1].split('\n')[0].strip()
                result['score_preference'] = score_part
        elif section.startswith('사고 과정'):
            result['thought_process'] = section.replace('사고 과정', '').strip()
        elif section.startswith('근거'):
            result['evidence'] = section.replace('근거', '').strip()
        elif section.startswith('최종 답'):
            answer_text = section.replace('최종 답', '').strip()
            if question_type == 'multiple_choice':
                # Extract number
                import re
                numbers = re.findall(r'\d+', answer_text)
                if numbers:
                    result['final_answer'] = numbers[0]
                else:
                    result['final_answer'] = '1'  # Default
            else:
                result['final_answer'] = answer_text
        elif section.startswith('최종 답변'):
            result['final_answer'] = section.replace('최종 답변', '').strip()
    
    return result


def create_simple_prompt(question: str, contexts: list) -> str:
    """
    Create simple prompt for comparison (without diagnostic features)
    This is the standard prompt used in production
    """
    
    context_section = ""
    if contexts:
        context_text = "\n\n".join(contexts[:3])
        context_section = f"""[참고 문서]
{context_text}

"""
    
    # Check if multiple choice
    import re
    lines = question.strip().split('\n')
    is_mc = False
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and len(line) > 2:
            if line[1] in ['.', ')', ' ', ':']:
                is_mc = True
                break
    
    if is_mc:
        prompt = f"""{context_section}[질문]
{question}

정답 번호만 출력하세요."""
    else:
        prompt = f"""{context_section}[질문]
{question}

간결하고 명확한 답변을 작성하세요."""
    
    return prompt