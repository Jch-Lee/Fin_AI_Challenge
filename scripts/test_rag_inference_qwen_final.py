#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG v2.0을 활용한 10개 샘플 추론 테스트 - Qwen2.5-7B-Instruct 최종 개선 버전
- 단순화된 답변 추출 로직
- simple_improved_prompt.py 기반 프롬프트
- 중국어 출력 방지
"""

import os
import sys
import json
import time
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
import re
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_inference_qwen_final.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_environment():
    """환경 설정"""
    import torch
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        logger.info(f"GPU Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("GPU not available, using CPU")
    
    return torch.cuda.is_available()


def load_test_data(test_file: str = "test.csv", num_samples: int = 10) -> pd.DataFrame:
    """테스트 데이터 로드"""
    logger.info(f"Loading test data from {test_file}")
    
    # CSV 파일 로드
    df = pd.read_csv(test_file)
    
    # 처음 N개 샘플만 선택
    df_sample = df.head(num_samples)
    logger.info(f"Selected {len(df_sample)} samples for testing")
    
    return df_sample


def initialize_rag_system():
    """RAG 시스템 초기화"""
    logger.info("Initializing RAG system v2.0...")
    
    # RAG v2 로더 임포트
    from scripts.load_rag_v2 import RAGSystemV2
    
    # RAG 시스템 로드
    rag = RAGSystemV2()
    rag.load_all()
    
    # 하이브리드 검색기 생성
    retriever = rag.create_hybrid_retriever()
    
    logger.info("RAG system initialized successfully")
    return rag, retriever


def initialize_llm():
    """LLM 초기화 - Qwen2.5-7B-Instruct 사용"""
    logger.info("Initializing LLM...")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    # Qwen2.5-7B-Instruct 모델 사용
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 모델 로드
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Qwen 모델은 특별한 패딩 설정이 필요할 수 있음
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("LLM initialized successfully")
    return model, tokenizer


def retrieve_context(retriever, query: str, k: int = 5) -> List[str]:
    """RAG 검색으로 컨텍스트 추출 - 리스트 형태로 반환"""
    try:
        # 하이브리드 검색 실행
        results = retriever.search(query, k=k)
        
        # 컨텍스트를 리스트로 구성
        contexts = []
        for result in results:
            contexts.append(result.content)
        
        return contexts
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []


def is_multiple_choice(text: str) -> bool:
    """객관식 문제 판별"""
    lines = text.strip().split('\n')
    options = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and len(line) > 2:
            if line[1] in ['.', ')', ' ', ':']:
                options.append(line)
    return len(options) >= 2


def get_max_choice_number(question: str) -> int:
    """객관식 문제의 최대 보기 번호 추출"""
    lines = question.strip().split('\n')
    max_num = 4  # 기본값
    
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            # 첫 번째 숫자 추출
            match = re.match(r'^(\d+)', line)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
    
    return max_num


def create_improved_prompt(question: str, contexts: List[str]) -> str:
    """simple_improved_prompt.py 기반 개선된 프롬프트 생성"""
    
    is_mc = is_multiple_choice(question)
    
    if is_mc:
        # 객관식 - 보기 범위 동적 결정
        max_choice = get_max_choice_number(question)
        
        # 참고 문서 섹션 - 상위 3개 사용 (최적 밸런스)
        context_section = ""
        if contexts and len(contexts) > 0:
            # 상위 3개 사용 (너무 많은 컨텍스트는 혼란 유발)
            context_text = "\n\n".join(contexts[:3])
            context_section = f"""[참고 문서]
{context_text}

"""
        
        # Qwen 모델용 시스템 프롬프트
        system_prompt = "당신은 한국 금융보안 분야의 전문가입니다. 한국어로만 답변하세요."
        
        # 개선된 프롬프트 - 이전 버전 기반
        user_prompt = f"""{context_section}[중요 지침]
• 참고 문서와 전문지식을 종합하여 가장 설득력 있는 답을 도출하세요
• 한국 금융 규제와 보안 표준을 고려하세요
• 확신을 가지고 판단하며, "모른다"는 표현을 사용하지 마세요
• 답변은 반드시 1~{max_choice} 범위의 단일 숫자만 출력하세요
• 절대 여러 개의 숫자나 쉼표를 사용하지 마세요
• 한국어로만 답변하고 중국어나 다른 언어를 사용하지 마세요
• 완전한 한글 문장으로 답변하되, 숫자만 출력하세요

[질문]
{question}

정답 번호(1~{max_choice} 중 하나)만 출력하세요.

답변:"""
        
    else:
        # 주관식
        context_section = ""
        if contexts and len(contexts) > 0:
            context_text = "\n\n".join(contexts[:3])
            context_section = f"""[참고 문서]
{context_text}

"""
        
        system_prompt = "당신은 한국 금융보안 분야의 전문가입니다. 한국어로만 답변하세요."
        
        user_prompt = f"""{context_section}[중요 지침]
• 참고 문서와 전문지식을 종합하여 명확한 답변을 제시하세요
• 한국 금융 환경의 특성을 반영하세요
• 질문의 핵심 요구사항을 모두 답하되, 간결한 답변 생성
• 10자 이상 500자 이하로 작성하세요
• "모른다"는 표현을 피하고 확신 있는 답변을 제공하세요
• 반드시 한국어로만 답변하고 중국어나 다른 언어를 사용하지 마세요
• 완전한 한글 문장으로 답변하세요. 중국어 문자(，、。：)는 절대 사용하지 마세요

[질문]
{question}

완전한 한글 문장으로 간결하고 정확하게 답변하세요.

답변:"""
    
    # Qwen 형식 프롬프트 - assistant 태그 제거하여 프롬프트 유출 방지
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant"""
    
    return prompt


def extract_answer_simple(response: str, question: str) -> str:
    """단순화된 답변 추출 로직"""
    
    # 빈 응답 처리
    if not response or response.strip() == "":
        if is_multiple_choice(question):
            return "1"
        else:
            return "답변을 생성할 수 없습니다."
    
    # 중국어 및 중국어 문장부호 제거
    # 중국어 문자 범위: \u4e00-\u9fff
    # 중국어 문장부호: ，、。：
    response = re.sub(r'[\u4e00-\u9fff]+', '', response)
    response = response.replace('，', ', ').replace('、', ', ').replace('。', '. ').replace('：', ': ')
    
    answer = response.strip()
    
    if is_multiple_choice(question):
        # 객관식: 첫 번째 단일 숫자만 추출
        max_choice = get_max_choice_number(question)
        
        # 모든 숫자 찾기
        numbers = re.findall(r'\d+', answer)
        
        if numbers:
            # 첫 번째 숫자를 int로 변환
            first_num = int(numbers[0])
            
            # 보기 범위 내의 숫자인지 확인
            if 1 <= first_num <= max_choice:
                return str(first_num)
            
            # 범위 밖이면 다른 숫자 찾기
            for num_str in numbers[1:]:
                num = int(num_str)
                if 1 <= num <= max_choice:
                    return str(num)
        
        # 숫자를 찾지 못하면 기본값
        return "1"
    
    else:
        # 주관식
        # "답변:" 이후의 내용 추출
        if "답변:" in answer:
            answer = answer.split("답변:")[-1].strip()
        
        # 프롬프트 유출 제거 - "assistant", "user" 같은 프리픽스 제거
        if answer.startswith("assistant"):
            answer = answer[9:].strip()
        elif answer.startswith("user"):
            answer = answer[4:].strip()
        
        # 너무 짧은 답변 체크
        if len(answer) < 10:
            return "답변을 생성할 수 없습니다."
        
        # 최대 길이 제한 - 500자
        if len(answer) > 500:
            answer = answer[:500]
        
        return answer


def generate_answer(model, tokenizer, prompt: str, question: str, max_length: int = 512) -> str:
    """LLM으로 답변 생성 - 최종 개선 버전"""
    try:
        import torch
        
        # 토큰화
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 생성 파라미터 조정
        if is_multiple_choice(question):
            # 객관식은 매우 짧게, deterministic하게
            generation_config = {
                "max_new_tokens": 10,  # 매우 짧게
                "temperature": 0.01,   # 거의 deterministic
                "top_p": 0.95,
                "do_sample": False,    # Greedy decoding
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.2  # 반복 방지
            }
        else:
            # 주관식
            generation_config = {
                "max_new_tokens": 512,  # 충분한 토큰 확보
                "temperature": 0.3,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.1
            }
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
        
        # 디코딩
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 제거하고 답변만 추출 - 더 정확한 분리
        if "<|im_start|>assistant" in full_response:
            # assistant 태그 이후 내용만 추출
            parts = full_response.split("<|im_start|>assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
            else:
                response = full_response.strip()
        elif "답변:" in full_response:
            # "답변:" 이후 부분만 추출
            parts = full_response.split("답변:")
            if len(parts) > 1:
                response = parts[-1].strip()
            else:
                response = full_response.strip()
        else:
            # 입력 프롬프트 길이만큼 제거
            input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            if full_response.startswith(input_text):
                response = full_response[len(input_text):].strip()
            else:
                response = full_response.strip()
        
        # 답변 추출
        answer = extract_answer_simple(response, question)
        
        logger.debug(f"Raw response: {response[:100]}...")
        logger.debug(f"Extracted answer: {answer}")
        
        return answer
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        if is_multiple_choice(question):
            return "1"
        else:
            return "답변 생성 중 오류가 발생했습니다."


def process_samples(df: pd.DataFrame, rag, retriever, model, tokenizer) -> pd.DataFrame:
    """샘플 처리"""
    results = []
    
    for idx, row in df.iterrows():
        question_id = row['ID']
        question = row['Question']
        
        logger.info(f"\nProcessing {idx+1}/{len(df)}: ID={question_id}")
        logger.info(f"Question type: {'Multiple Choice' if is_multiple_choice(question) else 'Open-ended'}")
        if is_multiple_choice(question):
            max_choice = get_max_choice_number(question)
            logger.info(f"Max choice number: {max_choice}")
        logger.info(f"Question: {question[:100]}...")
        
        start_time = time.time()
        
        # 1. RAG 검색 (리스트 형태로) - 상위 5개
        contexts = retrieve_context(retriever, question, k=5)
        logger.info(f"Retrieved {len(contexts)} contexts")
        
        # 검색 결과 점수 출력
        search_results = retriever.search(question, k=5)
        for i, result in enumerate(search_results):
            # HybridResult 객체에서 올바른 속성 참조
            hybrid_score = getattr(result, 'hybrid_score', 0.0)
            bm25_score = getattr(result, 'bm25_score', 0.0)
            vector_score = getattr(result, 'vector_score', 0.0)
            
            # 메타데이터에서 소스 정보 추출
            metadata = getattr(result, 'metadata', {})
            if metadata:
                # source, source_file, chunk_id 순서로 확인
                source = metadata.get('source', 
                        metadata.get('source_file', 
                        metadata.get('chunk_id', 'unknown')))
                # 파일명에서 경로 제거 (파일명만 표시)
                if source != 'unknown' and '\\' in source:
                    source = source.split('\\')[-1]
                if source != 'unknown' and '/' in source:
                    source = source.split('/')[-1]
                # .txt 확장자 제거
                if source.endswith('.txt'):
                    source = source[:-4]
            else:
                source = 'unknown'
            
            # 점수 정보 로깅
            logger.info(f"Context {i+1}: Hybrid={hybrid_score:.4f} (BM25={bm25_score:.4f}, Vector={vector_score:.4f}), Source={str(source)[:50]}...")
        
        # 2. 프롬프트 생성 (개선된 버전)
        prompt = create_improved_prompt(question, contexts)
        
        # 3. 답변 생성
        answer = generate_answer(model, tokenizer, prompt, question)
        logger.info(f"Generated answer: {answer}")
        
        elapsed = time.time() - start_time
        logger.info(f"Processing time: {elapsed:.2f}s")
        
        results.append({
            'ID': question_id,
            'Answer': answer
        })
    
    return pd.DataFrame(results)


def main():
    """메인 함수"""
    logger.info("="*60)
    logger.info("RAG v2.0 Inference Test - 20 Samples (Qwen2.5-7B-Instruct Final)")
    logger.info("="*60)
    
    # 환경 설정
    has_gpu = setup_environment()
    if not has_gpu:
        logger.warning("No GPU available, this may be slow")
    
    # 테스트 데이터 로드
    test_df = load_test_data("test.csv", num_samples=20)
    
    # RAG 시스템 초기화
    rag, retriever = initialize_rag_system()
    
    # LLM 초기화 (Qwen2.5-7B-Instruct)
    model, tokenizer = initialize_llm()
    
    # 추론 실행
    logger.info("\n" + "="*60)
    logger.info("Starting inference...")
    logger.info("="*60)
    
    start_time = time.time()
    results_df = process_samples(test_df, rag, retriever, model, tokenizer)
    total_time = time.time() - start_time
    
    # 결과 저장
    output_file = "rag_v2_qwen_final_with_scores_20samples.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"\nResults saved to {output_file}")
    
    # 통계 출력
    logger.info("\n" + "="*60)
    logger.info("Test Summary")
    logger.info("="*60)
    logger.info(f"Total samples: {len(results_df)}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average time per sample: {total_time/len(results_df):.2f}s")
    logger.info(f"Model used: Qwen2.5-7B-Instruct")
    logger.info("="*60)
    
    # 결과 미리보기
    logger.info("\nResults preview:")
    for idx, row in results_df.head(10).iterrows():
        logger.info(f"ID: {row['ID']}, Answer: {row['Answer']}")
    
    # 객관식 답변 검증
    logger.info("\n" + "="*60)
    logger.info("Answer Validation")
    logger.info("="*60)
    
    for idx, row in results_df.iterrows():
        question = test_df.iloc[idx]['Question']
        if is_multiple_choice(question):
            answer = row['Answer']
            max_choice = get_max_choice_number(question)
            
            # 답변이 숫자인지 확인
            if answer.isdigit():
                num = int(answer)
                if 1 <= num <= max_choice:
                    logger.info(f"✓ {row['ID']}: Valid answer {answer} (range: 1-{max_choice})")
                else:
                    logger.warning(f"✗ {row['ID']}: Out of range {answer} (range: 1-{max_choice})")
            else:
                logger.warning(f"✗ {row['ID']}: Non-numeric answer '{answer}'")


if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()
    main()