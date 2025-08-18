#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG v2.0 Final Submission Generator
전체 test.csv에 대한 예측 수행
"""

import os
import sys
import time
import logging
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# 경고 메시지 억제
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# RAG 시스템 임포트
from scripts.load_rag_v2 import RAGSystemV2

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


def setup_environment():
    """환경 설정"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Available: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        return True
    return False


def is_multiple_choice(text: str) -> bool:
    """객관식 문제 판별"""
    lines = text.strip().split('\n')
    options_count = 0
    
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and len(line) > 1:
            if line[1] in '.':
                options_count += 1
            elif len(line) > 2 and line[1:3] in ['. ', ') ']:
                options_count += 1
    
    return options_count >= 2


def extract_answer_from_text(text: str, question: str) -> str:
    """LLM 출력에서 답변 추출"""
    if is_multiple_choice(question):
        # 객관식: 숫자만 추출
        import re
        numbers = re.findall(r'\b([1-9]|[1-9][0-9])\b', text)
        if numbers:
            return numbers[0]
        return "1"
    else:
        # 주관식: 전체 텍스트 반환
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text if text else "답변을 생성할 수 없습니다."


def load_test_data(file_path: str) -> pd.DataFrame:
    """테스트 데이터 로드"""
    logger.info(f"Loading test data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} samples")
    return df


def initialize_rag_system():
    """RAG 시스템 초기화"""
    logger.info("Initializing RAG system v2.0...")
    
    # RAG 시스템 로드
    rag = RAGSystemV2()
    rag.load_all()
    
    # 하이브리드 검색기 생성
    retriever = rag.create_hybrid_retriever()
    
    logger.info("RAG system initialized successfully")
    return rag, retriever


def initialize_llm():
    """LLM 초기화 - Qwen2.5-7B-Instruct"""
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
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Model loaded successfully")
    return model, tokenizer


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
    """개선된 프롬프트 생성"""
    
    # 컨텍스트 통합
    context_text = "\n\n".join([f"[참고 {i+1}]\n{ctx[:800]}" for i, ctx in enumerate(contexts)])
    
    is_mc = is_multiple_choice(question)
    
    if is_mc:
        # 객관식 프롬프트
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
        # 주관식 프롬프트
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


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """LLM으로 답변 생성"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
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
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def process_batch(batch_df: pd.DataFrame, rag, retriever, model, tokenizer) -> List[Dict]:
    """배치 처리"""
    results = []
    
    for idx, row in batch_df.iterrows():
        question_id = row['ID']
        question = row['Question']
        
        try:
            # RAG 검색
            contexts = retrieve_context(retriever, question, k=5)
            
            # 프롬프트 생성
            prompt = create_improved_prompt(question, contexts)
            
            # 답변 생성
            raw_answer = generate_answer(model, tokenizer, prompt)
            answer = extract_answer_from_text(raw_answer, question)
            
            results.append({
                'ID': question_id,
                'Answer': answer
            })
            
            logger.info(f"Processed {question_id}: {answer[:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing {question_id}: {e}")
            # 에러 시 기본값
            if is_multiple_choice(question):
                answer = "1"
            else:
                answer = "답변 생성 실패"
            
            results.append({
                'ID': question_id,
                'Answer': answer
            })
    
    return results


def main():
    """메인 함수"""
    logger.info("="*60)
    logger.info("RAG v2.0 Final Submission Generation")
    logger.info("="*60)
    
    # 환경 설정
    has_gpu = setup_environment()
    if not has_gpu:
        logger.error("GPU not available! This will be very slow.")
        return
    
    # 테스트 데이터 로드
    test_df = load_test_data("test.csv")
    total_samples = len(test_df)
    
    # RAG 시스템 초기화
    rag, retriever = initialize_rag_system()
    
    # LLM 초기화
    model, tokenizer = initialize_llm()
    
    # 추론 실행
    logger.info("\n" + "="*60)
    logger.info(f"Starting inference for {total_samples} samples...")
    logger.info("="*60)
    
    start_time = time.time()
    batch_size = 10
    all_results = []
    
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        batch_df = test_df.iloc[i:batch_end]
        
        logger.info(f"\nProcessing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
        batch_results = process_batch(batch_df, rag, retriever, model, tokenizer)
        all_results.extend(batch_results)
        
        # 진행 상황 출력
        elapsed = time.time() - start_time
        progress = len(all_results) / total_samples
        eta = elapsed / progress - elapsed if progress > 0 else 0
        logger.info(f"Progress: {len(all_results)}/{total_samples} ({progress:.1%}) - ETA: {eta:.0f}s")
    
    total_time = time.time() - start_time
    
    # 결과 저장
    results_df = pd.DataFrame(all_results)
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
    logger.info("\nValidating results...")
    mc_count = sum(1 for _, row in results_df.iterrows() if row['Answer'].isdigit())
    oe_count = len(results_df) - mc_count
    logger.info(f"Multiple choice answers: {mc_count}")
    logger.info(f"Open-ended answers: {oe_count}")
    
    return output_file


if __name__ == "__main__":
    output_file = main()
    print(f"\n✅ Final submission saved to: {output_file}")