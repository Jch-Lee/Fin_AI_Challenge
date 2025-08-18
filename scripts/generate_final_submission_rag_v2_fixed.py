#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG v2.0 Final Submission Generator - Fixed Version
test_rag_inference_qwen_final.py와 동일한 파이프라인 사용
"""

import os
import sys
import time
import logging
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import re

warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('final_submission_rag_v2.log', encoding='utf-8'),
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


def load_test_data(test_file: str = "test.csv") -> pd.DataFrame:
    """전체 테스트 데이터 로드"""
    logger.info(f"Loading test data from {test_file}")
    
    # CSV 파일 로드
    df = pd.read_csv(test_file)
    logger.info(f"Loaded {len(df)} samples")
    
    return df


def initialize_rag_system():
    """RAG 시스템 초기화 - v2.0 (2300자 청킹)"""
    logger.info("Initializing RAG system v2.0...")
    
    # RAG 시스템 로드
    from scripts.load_rag_v2 import RAGSystemV2
    rag = RAGSystemV2()
    rag.load_all()
    
    # 하이브리드 검색기 생성
    retriever = rag.create_hybrid_retriever()
    
    logger.info("RAG system initialized successfully")
    return rag, retriever


def initialize_llm():
    """Qwen2.5-7B-Instruct 모델 초기화 (4-bit 양자화)"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    logger.info("Initializing Qwen2.5-7B-Instruct model...")
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # 4-bit 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Model loaded successfully")
    return model, tokenizer


def is_multiple_choice(text: str) -> bool:
    """객관식 문제인지 판별 - 테스트 스크립트와 동일"""
    lines = text.strip().split('\n')
    options_count = 0
    
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and len(line) > 1:
            # "1." 또는 "1)" 형식 확인
            if line[1] in '.':
                options_count += 1
            elif len(line) > 2 and line[1:3] in ['. ', ') ']:
                options_count += 1
    
    return options_count >= 2


def retrieve_context(retriever, query: str, k: int = 5) -> List[str]:
    """RAG 검색 수행"""
    try:
        results = retriever.search(query, k=k)
        contexts = []
        for result in results:
            content = getattr(result, 'content', '')
            if content:
                contexts.append(content)
        return contexts
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []


def create_improved_prompt(question: str, contexts: List[str]) -> str:
    """개선된 프롬프트 생성 - 테스트 스크립트와 동일"""
    
    # 컨텍스트 통합
    context_text = "\n\n".join([f"[참고 {i+1}]\n{ctx[:800]}" for i, ctx in enumerate(contexts)])
    
    is_mc = is_multiple_choice(question)
    
    if is_mc:
        # 객관식 프롬프트 (테스트와 동일)
        prompt = f"""당신은 금융보안 전문가입니다. 다음 참고자료를 바탕으로 질문에 대한 정답을 선택하세요.

참고자료:
{context_text}

질문: {question}

지시사항:
1. 제공된 선택지 중 가장 적절한 답을 선택하세요
2. 반드시 숫자만 출력하세요 (예: 1, 2, 3, 4, 5)
3. 설명이나 추가 텍스트 없이 숫자만 출력하세요

정답:"""
    else:
        # 주관식 프롬프트 (테스트와 동일)
        prompt = f"""당신은 금융보안 전문가입니다. 다음 참고자료를 바탕으로 질문에 답하세요.

참고자료:
{context_text}

질문: {question}

지시사항:
1. 전문적이고 정확한 답변을 제공하세요
2. 10자 이상 500자 이하로 작성하세요
3. 한국어로 명확하게 답변하세요

답변:"""
    
    return prompt


def extract_answer_simple(text: str, question: str) -> str:
    """답변 추출 - 테스트 스크립트와 동일한 로직"""
    if is_multiple_choice(question):
        # 객관식: 숫자만 추출
        numbers = re.findall(r'\b([1-9]|[1-9][0-9])\b', text)
        if numbers:
            return numbers[0]
        return "1"
    else:
        # 주관식: 텍스트 정리
        text = text.strip()
        # 따옴표 제거
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        # 콜론으로 시작하는 경우 제거
        if text.startswith(':'):
            text = text[1:].strip()
        
        return text if text else "답변을 생성할 수 없습니다."


def generate_answer_rag(model, tokenizer, question: str, retriever, max_new_tokens: int = 128) -> str:
    """RAG를 사용한 답변 생성 - 테스트 스크립트와 동일"""
    import torch
    
    try:
        # 1. RAG 검색 (리스트 형태로) - 상위 5개
        contexts = retrieve_context(retriever, question, k=5)
        logger.debug(f"Retrieved {len(contexts)} contexts")
        
        # 2. 프롬프트 생성 (개선된 버전)
        prompt = create_improved_prompt(question, contexts)
        
        # 3. 토큰화
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 4. 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 5. 디코딩 - 전체 응답
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


def process_batch(batch_df: pd.DataFrame, retriever, model, tokenizer) -> pd.DataFrame:
    """배치 처리"""
    results = []
    
    for idx, row in batch_df.iterrows():
        question_id = row['ID']
        question = row['Question']
        
        start_time = time.time()
        
        logger.info(f"Processing {idx + 1}/{len(batch_df)}: ID={question_id}")
        
        # 답변 생성
        answer = generate_answer_rag(model, tokenizer, question, retriever)
        
        elapsed = time.time() - start_time
        logger.info(f"Generated answer: {answer[:50]}...")
        logger.info(f"Processing time: {elapsed:.2f}s")
        
        results.append({
            'ID': question_id,
            'Answer': answer
        })
    
    return pd.DataFrame(results)


def main():
    """메인 함수"""
    logger.info("="*60)
    logger.info("RAG v2.0 Final Submission Generation")
    logger.info("="*60)
    
    # 환경 설정
    has_gpu = setup_environment()
    if not has_gpu:
        logger.error("No GPU available! This will be very slow.")
        return
    
    # 테스트 데이터 로드
    test_df = load_test_data("test.csv")
    total_samples = len(test_df)
    
    # RAG 시스템 초기화
    rag, retriever = initialize_rag_system()
    
    # LLM 초기화 (Qwen2.5-7B-Instruct)
    model, tokenizer = initialize_llm()
    
    # 추론 실행
    logger.info("\n" + "="*60)
    logger.info(f"Starting inference for {total_samples} samples...")
    logger.info("="*60)
    
    # 배치 크기 설정
    batch_size = 50  # 더 큰 배치로 처리
    all_results = []
    
    start_time = time.time()
    
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        batch_df = test_df.iloc[i:batch_end]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
        logger.info(f"Samples {i+1} to {batch_end}")
        logger.info('='*60)
        
        # 배치 처리
        batch_results = process_batch(batch_df, retriever, model, tokenizer)
        all_results.append(batch_results)
        
        # 진행 상황 출력
        elapsed = time.time() - start_time
        progress = batch_end / total_samples
        eta = elapsed / progress - elapsed if progress > 0 else 0
        
        logger.info(f"\nProgress: {batch_end}/{total_samples} ({progress:.1%})")
        logger.info(f"Elapsed time: {elapsed:.1f}s")
        logger.info(f"ETA: {eta:.1f}s")
        
        # 메모리 정리
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 결과 병합
    results_df = pd.concat(all_results, ignore_index=True)
    
    total_time = time.time() - start_time
    
    # 결과 저장
    output_file = f"submission_rag_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    logger.info("\n" + "="*60)
    logger.info("Generation Complete!")
    logger.info("="*60)
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average time per sample: {total_time/total_samples:.2f}s")
    logger.info(f"Results saved to: {output_file}")
    
    # 검증
    logger.info("\n" + "="*60)
    logger.info("Answer Validation")
    logger.info("="*60)
    
    mc_count = 0
    oe_count = 0
    
    for _, row in results_df.iterrows():
        if str(row['Answer']).isdigit():
            mc_count += 1
        else:
            oe_count += 1
    
    logger.info(f"Multiple choice answers: {mc_count}")
    logger.info(f"Open-ended answers: {oe_count}")
    logger.info(f"Total: {len(results_df)}")
    
    # 샘플 출력
    logger.info("\nFirst 5 results:")
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        logger.info(f"ID: {row['ID']}, Answer: {str(row['Answer'])[:50]}...")
    
    return output_file


if __name__ == "__main__":
    output_file = main()
    print(f"\n✅ Final submission saved to: {output_file}")