#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG v2.0 Final Submission Generator - BM25 Weight 0.7
키워드 기반 검색(BM25) 비중을 0.7로 높인 버전
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
        logging.FileHandler('final_submission_bm25_070.log', encoding='utf-8'),
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
    """RAG 시스템 초기화 - v2.0 (2300자 청킹, BM25 weight=0.7)"""
    logger.info("Initializing RAG system v2.0 with BM25 weight=0.7...")
    
    # RAG 시스템 로드
    from scripts.load_rag_v2 import RAGSystemV2
    rag = RAGSystemV2()
    rag.load_all()
    
    # 하이브리드 검색기 생성 (설정 파일에서 0.7/0.3 비율 사용)
    retriever = rag.create_hybrid_retriever()
    
    logger.info("RAG system initialized successfully")
    return rag, retriever


def initialize_llm():
    """LLM 초기화 - Qwen2.5-7B-Instruct 사용 (16-bit)"""
    logger.info("Initializing LLM with 16-bit quantization...")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Qwen2.5-7B-Instruct 모델 사용
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # 16-bit 모델 로드 (양자화 없음, 기본 float16)
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
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
    """객관식 문제 판별 - 테스트 스크립트와 동일"""
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
    """객관식/주관식 구분하여 개선된 프롬프트 생성 - 강화된 버전"""
    
    is_mc = is_multiple_choice(question)
    
    if is_mc:
        # 객관식 - 강화된 단일 숫자 출력 프롬프트
        max_choice = get_max_choice_number(question)
        
        # 참고 문서 섹션 - 상위 3개 사용
        context_section = ""
        if contexts and len(contexts) > 0:
            context_text = "\n\n".join(contexts[:3])
            context_section = f"""[참고 문서]
{context_text}

"""
        
        system_prompt = "당신은 한국 금융보안 분야의 전문가입니다. 한국어로만 답변하세요."
        
        # 객관식용 강화된 프롬프트 - 단일 숫자만 출력하도록 극도로 강화
        user_prompt = f"""{context_section}[중요 지침 - 엄격히 준수]
• 참고 문서와 전문지식을 종합하여 가장 정확한 답을 선택하세요
• 반드시 1~{max_choice} 범위의 단일 숫자만 출력하세요
• 설명, 이유, 추가 텍스트는 절대 포함하지 마세요
• 숫자 앞이나 뒤에 어떤 텍스트도 추가하지 마세요
• 쉼표, 마침표, 콜론 등 어떤 기호도 포함하지 마세요
• "답변:", "정답:", "선택:" 등의 표현을 사용하지 마세요
• 오직 숫자 하나만 출력하세요 (예: 3)
• 여러 숫자나 범위를 절대 출력하지 마세요
• 한국어 설명을 절대 포함하지 마세요
• 숫자만 출력하세요 - 다른 모든 내용 금지

[질문]
{question}

위 객관식 문제의 정답 번호 하나만 출력하세요. 설명 없이 숫자만 출력하세요.

정답:"""
        
    else:
        # 주관식 - 기존 강화된 프롬프트 유지
        context_section = ""
        if contexts and len(contexts) > 0:
            context_text = "\n\n".join(contexts[:3])
            context_section = f"""[참고 문서]
{context_text}

"""
        
        system_prompt = "당신은 한국 금융보안 분야의 전문가입니다. 한국어로만 답변하세요."
        
        user_prompt = f"""{context_section}[중요 지침]
• 참고 문서와 전문지식을 종합하여 명확한 답변을 제시하세요.
• 질문의 핵심 요구사항을 모두 답하되, 간결한 답변을 생성하세요.
• 중간 사고 과정이나 참고 문서 내용은 절대 답변에 포함하지 마세요.
• 20자 이상 500자 이하로 작성하세요
• "모른다", "답변을 생성할 수 없습니다", "정보가 부족합니다" 등의 표현을 절대 사용하지 마세요
• 기본 지식과 일반적인 보안 원칙을 활용해 반드시 구체적인 답변을 제공하세요
• 금융보안 분야의 전문가로서 확신 있고 유용한 정보를 반드시 제공하세요
• 완전한 한국어 문장으로만 답변하세요
• 질문에 대한 답변을 반드시 작성해야 합니다 - 예외는 없습니다

[질문]
{question}

위 질문에 대해 금융보안 전문가로서 반드시 구체적이고 유용한 답변을 작성하세요. 답변 거부나 생성 실패는 허용되지 않습니다.

답변:"""
    
    # Qwen 형식 프롬프트
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant"""
    
    return prompt


def extract_answer_simple(response: str, question: str) -> str:
    """객관식/주관식 구분하여 답변 추출 - 강화된 버전"""
    
    # 빈 응답 처리
    if not response or response.strip() == "":
        if is_multiple_choice(question):
            return "1"
        else:
            return "답변을 생성할 수 없습니다."
    
    # 중국어, 일본어 및 관련 문장부호 제거 강화
    # 중국어: \u4e00-\u9fff
    # 일본어 히라가나: \u3040-\u309f
    # 일본어 가타카나: \u30a0-\u30ff
    # CJK 기호: \u3000-\u303f
    response = re.sub(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3000-\u303f]+', '', response)
    
    # 중국어/일본어 문장부호 제거
    response = response.replace('，', ', ').replace('、', ', ').replace('。', '. ').replace('：', ': ')
    response = response.replace('？', '?').replace('！', '!').replace('（', '(').replace('）', ')')
    
    answer = response.strip()
    
    if is_multiple_choice(question):
        # 객관식: 첫 번째 단일 숫자만 추출
        max_choice = get_max_choice_number(question)
        
        # "정답:" 이후의 내용 추출
        if "정답:" in answer:
            answer = answer.split("정답:")[-1].strip()
        elif "답변:" in answer:
            answer = answer.split("답변:")[-1].strip()
        
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
        # 영어 단어 필터링 (일부 허용되는 단어 제외)
        allowed_english = ['API', 'CPU', 'GPU', 'URL', 'DNS', 'SQL', 'VPN', 'SSL', 'TLS', 'HTTP', 'HTTPS']
        words = response.split()
        filtered_words = []
        for word in words:
            # 영어 단어인지 확인 (알파벳만 포함)
            if re.match(r'^[A-Za-z]+$', word):
                # 허용된 단어인지 확인
                if word.upper() in allowed_english:
                    filtered_words.append(word)
                # 아니면 제거
            else:
                filtered_words.append(word)
        response = ' '.join(filtered_words)
        
        answer = response.strip()
        
        # "답변:" 이후의 내용 추출
        if "답변:" in answer:
            answer = answer.split("답변:")[-1].strip()
        
        # 프롬프트 유출 제거
        if answer.startswith("assistant"):
            answer = answer[9:].strip()
        elif answer.startswith("user"):
            answer = answer[4:].strip()
        
        # 최대 길이 제한 - 500자
        if len(answer) > 500:
            answer = answer[:500]
        
        return answer


def generate_answer(model, tokenizer, prompt: str, question: str, max_length: int = 512) -> str:
    """LLM으로 답변 생성 - 강화된 버전"""
    try:
        import torch
        
        # 토큰화
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 객관식/주관식에 따른 생성 파라미터 조정
        if is_multiple_choice(question):
            # 객관식: 매우 짧게, deterministic하게
            generation_config = {
                "max_new_tokens": 5,  # 극도로 짧게 - 숫자만 생성
                "temperature": 0.01,   # 거의 deterministic
                "top_p": 0.95,
                "do_sample": False,    # Greedy decoding
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.2  # 반복 방지
            }
        else:
            # 주관식: 기존 설정 유지
            generation_config = {
                "max_new_tokens": 512,
                "temperature": 0.05,  # 매우 낮춰서 안정적인 생성
                "top_p": 0.95,
                "do_sample": False,  # Greedy decoding
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3
            }
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
        
        # 디코딩
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 제거하고 답변만 추출
        if "<|im_start|>assistant" in full_response:
            parts = full_response.split("<|im_start|>assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
            else:
                response = full_response.strip()
        elif "정답:" in full_response:
            parts = full_response.split("정답:")
            if len(parts) > 1:
                response = parts[-1].strip()
            else:
                response = full_response.strip()
        elif "답변:" in full_response:
            parts = full_response.split("답변:")
            if len(parts) > 1:
                response = parts[-1].strip()
            else:
                response = full_response.strip()
        else:
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


def generate_answer_rag(model, tokenizer, question: str, retriever) -> str:
    """RAG를 사용한 답변 생성 - 테스트 스크립트와 동일"""
    
    # 1. RAG 검색 (리스트 형태로) - 상위 5개
    contexts = retrieve_context(retriever, question, k=5)
    logger.debug(f"Retrieved {len(contexts)} contexts")
    
    # 2. 프롬프트 생성 (개선된 버전)
    prompt = create_improved_prompt(question, contexts)
    
    # 3. 답변 생성
    answer = generate_answer(model, tokenizer, prompt, question)
    
    return answer


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
    logger.info("RAG v2.0 Final Submission Generation - BM25 Weight 0.7")
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
    logger.info("BM25 Weight: 0.7 (70% keyword-based)")
    logger.info("Vector Weight: 0.3 (30% vector-based)")
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
    output_file = f"submission_rag_v2_bm25_070_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    logger.info("\n" + "="*60)
    logger.info("Generation Complete!")
    logger.info("="*60)
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average time per sample: {total_time/total_samples:.2f}s")
    logger.info(f"BM25 Weight: 0.7, Vector Weight: 0.3")
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
    print(f"\n✅ Final submission with BM25 weight 0.7 saved to: {output_file}")