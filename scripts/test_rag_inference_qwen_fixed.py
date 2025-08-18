#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG v2.0을 활용한 10개 샘플 추론 테스트 - Qwen2.5-7B-Instruct 개선 버전
답변 추출 로직 및 프롬프트 개선
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
        logging.FileHandler('rag_inference_qwen_fixed.log'),
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


def retrieve_context(retriever, query: str, k: int = 5) -> str:
    """RAG 검색으로 컨텍스트 추출"""
    try:
        # 하이브리드 검색 실행
        results = retriever.search(query, k=k)
        
        # 컨텍스트 구성
        contexts = []
        for i, result in enumerate(results, 1):
            contexts.append(f"[참고 {i}]\n{result.content}")
        
        context = "\n\n".join(contexts)
        return context[:4000]  # 최대 4000자로 제한
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return ""


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


def create_prompt_with_context(question: str, context: str) -> str:
    """RAG 컨텍스트를 포함한 프롬프트 생성 - 개선된 Qwen 형식"""
    
    # Qwen2.5 모델을 위한 시스템 프롬프트
    system_prompt = "당신은 한국의 금융 보안 전문가입니다. 제공된 참고 자료를 정확히 분석하여 답변하세요."
    
    if is_multiple_choice(question):
        # 객관식 프롬프트 - 더 명확한 지시
        user_prompt = f"""다음 참고 자료를 읽고 질문에 답하세요.

참고 자료:
{context}

질문:
{question}

지시사항: 위 객관식 문제의 정답을 하나만 선택하세요. 반드시 단일 숫자로만 답하세요.
예시: 정답이 3번이면 "3"이라고만 답하세요.

정답 번호:"""
        
    else:
        # 주관식 프롬프트
        user_prompt = f"""다음 참고 자료를 읽고 질문에 답하세요.

참고 자료:
{context}

질문:
{question}

답변:"""
    
    # Qwen 모델 형식에 맞는 프롬프트 구성
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
    
    return prompt


def extract_answer_from_response(response: str, question: str) -> str:
    """개선된 답변 추출 로직"""
    
    # 빈 응답 처리
    if not response or response.strip() == "":
        if is_multiple_choice(question):
            return "1"
        else:
            return "답변을 생성할 수 없습니다."
    
    # 답변 정제
    answer = response.strip()
    
    if is_multiple_choice(question):
        # 객관식 답변 추출 - 첫 번째 숫자만 추출
        
        # 패턴 1: "정답 번호: X" 형식
        match = re.search(r'정답\s*번호\s*[:：]?\s*(\d+)', answer)
        if match:
            return match.group(1)
        
        # 패턴 2: 단독 숫자 (줄의 시작 부분)
        match = re.match(r'^(\d+)', answer)
        if match:
            return match.group(1)
        
        # 패턴 3: "X번" 형식
        match = re.search(r'(\d+)\s*번', answer)
        if match:
            return match.group(1)
        
        # 패턴 4: 쉼표로 구분된 경우 첫 번째 숫자만
        if ',' in answer:
            parts = answer.split(',')
            first_part = parts[0].strip()
            match = re.search(r'(\d+)', first_part)
            if match:
                return match.group(1)
        
        # 패턴 5: 전체 텍스트에서 첫 번째 숫자
        all_numbers = re.findall(r'\d+', answer)
        if all_numbers:
            # 1-5 범위의 숫자 우선
            for num in all_numbers:
                if 1 <= int(num) <= 5:
                    return num
            # 없으면 첫 번째 숫자
            return all_numbers[0]
        
        # 기본값
        return "1"
    
    else:
        # 주관식 답변
        # "답변:" 이후의 내용 추출
        if "답변:" in answer:
            answer = answer.split("답변:")[-1].strip()
        
        # 너무 짧은 답변 체크
        if len(answer) < 10:
            return "답변을 생성할 수 없습니다."
        
        # 최대 길이 제한
        if len(answer) > 1000:
            answer = answer[:1000] + "..."
        
        return answer


def generate_answer(model, tokenizer, prompt: str, question: str, max_length: int = 512) -> str:
    """LLM으로 답변 생성 - 개선된 버전"""
    try:
        import torch
        
        # 토큰화
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 생성 파라미터 조정
        if is_multiple_choice(question):
            # 객관식은 짧게
            generation_config = {
                "max_new_tokens": 50,
                "temperature": 0.1,  # 더 deterministic하게
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
        else:
            # 주관식은 길게
            generation_config = {
                "max_new_tokens": max_length,
                "temperature": 0.3,
                "top_p": 0.8,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
        
        # 디코딩
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 제거하고 답변만 추출
        if "<|im_start|>assistant" in full_response:
            response = full_response.split("<|im_start|>assistant")[-1].strip()
        elif "assistant" in full_response:
            response = full_response.split("assistant")[-1].strip()
        else:
            # 입력 프롬프트 길이만큼 제거
            input_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
            response = full_response[input_length:].strip()
        
        # 답변 추출
        answer = extract_answer_from_response(response, question)
        
        logger.debug(f"Raw response: {response[:200]}...")
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
        logger.info(f"Question: {question[:100]}...")
        
        start_time = time.time()
        
        # 1. RAG 검색
        context = retrieve_context(retriever, question, k=5)
        logger.info(f"Retrieved context: {len(context)} chars")
        
        # 2. 프롬프트 생성
        prompt = create_prompt_with_context(question, context)
        
        # 3. 답변 생성
        answer = generate_answer(model, tokenizer, prompt, question)
        logger.info(f"Generated answer: {answer[:100]}...")
        
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
    logger.info("RAG v2.0 Inference Test - 10 Samples (Qwen2.5-7B-Instruct Fixed)")
    logger.info("="*60)
    
    # 환경 설정
    has_gpu = setup_environment()
    if not has_gpu:
        logger.warning("No GPU available, this may be slow")
    
    # 테스트 데이터 로드
    test_df = load_test_data("test.csv", num_samples=10)
    
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
    output_file = "rag_v2_qwen_fixed_10samples.csv"
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
        answer_preview = row['Answer'][:50] + "..." if len(row['Answer']) > 50 else row['Answer']
        logger.info(f"ID: {row['ID']}, Answer: {answer_preview}")


if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()
    main()