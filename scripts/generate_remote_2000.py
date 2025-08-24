#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5-14B 원격 서버용 2000개 질문 생성 버전
- 10배치 × 200개 = 총 2000개 질문
- 완화된 불용어 처리
- 파라미터: temperature=0.3, max_new_tokens=256, top_p=0.9
"""

import json
import random
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation_remote_2000.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 완화된 불용어 리스트 - 특수문자와 명백한 메타텍스트만
CRITICAL_STOPWORDS = {
    '자료 1', '자료 2', '자료 3', '위 내용', '위 자료', '이 문서', '이 텍스트',
    '답변:', '질문:', '예시:', '참고:', 'Q:', 'A:', '출처:',
    '###', '...', '---', '___', '***', '```',
    '연락처', '누리집', '웹사이트', '홈페이지', '발간', '페이지',
    '첫번째 자료', '두번째 자료', '세번째 자료'
}

def load_chunks(chunks_file: str) -> List[Dict]:
    """청크 데이터 로드"""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks

def sample_diverse_chunks(chunks: List[Dict], n_samples: int = 500) -> List[Dict]:
    """다양성 기반 청크 샘플링"""
    doc_groups = defaultdict(list)
    for chunk in chunks:
        doc_groups[chunk.get('source', 'unknown')].append(chunk)
    
    sampled = []
    docs = list(doc_groups.keys())
    
    # 각 문서에서 균등하게 샘플링
    while len(sampled) < n_samples and docs:
        for doc in docs[:]:
            if doc_groups[doc]:
                sampled.append(doc_groups[doc].pop(0))
                if len(sampled) >= n_samples:
                    break
            else:
                docs.remove(doc)
    
    if len(sampled) < n_samples:
        remaining = [c for c in chunks if c not in sampled]
        sampled.extend(random.sample(remaining, min(n_samples - len(sampled), len(remaining))))
    
    random.shuffle(sampled)
    logger.info(f"Sampled {len(sampled)} diverse chunks from {len(set(c.get('source') for c in sampled))} documents")
    return sampled

def create_single_question_prompt(chunk_content: str, question_type: str) -> str:
    """단일 질문 생성을 위한 프롬프트"""
    
    # 청크 내용 제한
    max_content_length = 1000
    if len(chunk_content) > max_content_length:
        chunk_content = chunk_content[:max_content_length]
    
    type_instructions = {
        'definition': "핵심 개념이나 용어의 정의",
        'process': "절차, 프로세스, 방법론",
        'regulation': "법률, 규정, 기준",
        'example': "구체적 사례나 실제 적용",
        'comparison': "차이점이나 비교",
        'application': "활용 방법이나 대응 방안"
    }
    
    # 타입별 좋은 예시
    type_examples = {
        'definition': "개인정보 보호위원회란 어떤 기구이며 주요 역할은 무엇인지 간략하게 설명하시오?",
        'process': "금융회사들이 오픈소스 소프트웨어를 활용하면서 라이선스 문제를 해결하기 위해 어떤 단계들을 거쳐야 하나요?",
        'regulation': "긴급 상황에서 개인 정보를 제한 없이 사용할 수 있는 조건과 그 내용을 설명하여라?",
        'example': "특정 고객의 거래 내역을 분석하는 과정에서 발생 가능한 개인정보 유출 위험이 어떤 것들이 있을까요?",
        'comparison': "TDES와 AES 암호화 알고리즘의 보안성과 성능 측면에서 어떤 차이가 있는지 설명하시오?",
        'application': "위치정보 이용에 따른 개인정보 노출 위험성을 어떻게 관리해야 하는지 설명하십시오?"
    }
    
    prompt = f"""<지침>
당신은 한국 금융보안원 시험 출제위원입니다. 다음 내용을 바탕으로 {type_instructions.get(question_type, '일반')}에 관한 질문을 하나만 작성하세요.

<좋은 질문 예시>
{type_examples.get(question_type, type_examples['definition'])}

<규칙>
- 반드시 한글로만 작성
- 질문 하나만 작성 (번호 없이)
- 완전한 한 문장으로 작성
- 물음표로 끝내기
- 구체적이고 명확한 질문
- 메타 텍스트나 참조 표현 금지

<참고 내용>
{chunk_content}

질문:"""
    
    return prompt

def contains_critical_stopwords(text: str) -> bool:
    """완화된 불용어 체크 - 특수문자와 명백한 메타텍스트만"""
    for stopword in CRITICAL_STOPWORDS:
        if stopword in text:
            return True
    return False

def clean_single_question(text: str) -> Optional[str]:
    """생성된 단일 질문 정제 (완화된 불용어 처리)"""
    
    # 첫 줄만 추출
    lines = text.strip().split('\n')
    if not lines:
        return None
    
    question = lines[0].strip()
    
    # 번호 및 레이블 제거
    question = re.sub(r'^\d+[\.\)]\\s*', '', question)
    question = re.sub(r'^질문[:：]\\s*', '', question)
    question = re.sub(r'^답변[:：]\\s*', '', question)
    question = re.sub(r'^Q\\d*[:：]\\s*', '', question)
    question = re.sub(r'^A\\d*[:：]\\s*', '', question)
    
    # 특수문자 정리 (물음표, 쉼표, 괄호는 유지)
    question = re.sub(r'[•·\\-\\*]+\\s*', '', question)
    question = re.sub(r'[\\[\\]<>{}]+', '', question)  # 괄호 ()는 제거하지 않음
    question = re.sub(r'#+\\s*', '', question)
    question = re.sub(r'_+\\s*', '', question)
    
    # 한글 포함 확인
    if not re.search(r'[가-힣]', question):
        return None
    
    # 외국어만으로 구성된 경우 제외
    if re.search(r'[\\u4e00-\\u9fff]', question):  # 중국어
        return None
    if re.search(r'[\\u3040-\\u309f\\u30a0-\\u30ff]', question):  # 일본어
        return None
    
    # 완화된 불용어 체크
    if contains_critical_stopwords(question):
        return None
    
    # 길이 체크
    if len(question) < 10 or len(question) > 300:
        return None
    
    # 물음표 추가
    if not question.endswith('?'):
        question += '?'
    
    # 추가 정제: 연속된 공백 제거
    question = ' '.join(question.split())
    
    return question

def generate_single_question(model, tokenizer, chunk: Dict, question_type: str, device: str) -> Optional[str]:
    """단일 질문 생성 (최적화된 파라미터)"""
    
    # 프롬프트 생성
    prompt = create_single_question_prompt(chunk['content'], question_type)
    
    # 토크나이즈
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
    
    # 모델의 디바이스 가져오기
    model_device = next(model.parameters()).device
    
    # 입력을 모델과 같은 디바이스로 이동
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    # 생성 (조정된 파라미터)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    # 디코딩
    generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    
    # 정제
    question = clean_single_question(generated)
    
    return question

def main():
    """메인 함수"""
    logger.info("="*60)
    logger.info("Starting Remote 2000 Questions Generation")
    logger.info("Target: 2000 questions (10 batches × 200 each)")
    logger.info("Parameters: temperature=0.3, max_new_tokens=256, top_p=0.9")
    logger.info("="*60)
    
    # GPU 확인
    if not torch.cuda.is_available():
        logger.error("GPU not available!")
        device = "cpu"
        dtype = torch.float32
    else:
        device = "cuda"
        dtype = torch.float16
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 모델 로드 - 14B
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    logger.info(f"Attempting to load: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",  # 명시적으로 GPU 지정
            trust_remote_code=True
        )
        logger.info("14B model loaded successfully with FP16")
        
    except Exception as e:
        logger.warning(f"Failed to load 14B model: {e}")
        logger.info("Using 7B model as fallback...")
        
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("7B model loaded as fallback")
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 청크 로드
    chunks = load_chunks("data/rag/chunks_2300.json")
    
    # 샘플링 (2000개 생성을 위해 더 많은 청크 준비)
    sampled_chunks = sample_diverse_chunks(chunks, n_samples=500)
    
    # 질문 유형 분포 (200개 배치용)
    question_types = {
        'definition': 40,    # 6 → 40 (10배)
        'process': 40,      # 6 → 40
        'regulation': 40,   # 6 → 40
        'example': 27,      # 4 → 27
        'comparison': 27,   # 4 → 27
        'application': 26   # 4 → 26
    }  # 총 200개
    
    # 질문 생성
    all_questions = []
    type_counts = defaultdict(int)
    chunk_idx = 0
    max_retries = 1000  # 200 → 1000으로 증가
    retry_count = 0
    
    logger.info("Generating 2000 questions with relaxed stopword filtering...")
    logger.info(f"Critical stopwords count: {len(CRITICAL_STOPWORDS)}")
    logger.info(f"Target per type: {question_types}")
    
    with tqdm(total=200, desc="Generating questions (Batch)") as pbar:
        for q_type, target_count in question_types.items():
            consecutive_failures = 0
            while type_counts[q_type] < target_count and retry_count < max_retries:
                if len(all_questions) >= 200:
                    break
                
                # 청크 선택
                chunk = sampled_chunks[chunk_idx % len(sampled_chunks)]
                chunk_idx += 1
                
                # 단일 질문 생성
                question = generate_single_question(model, tokenizer, chunk, q_type, device)
                
                if question:
                    # 중복 체크
                    if not any(q['question'] == question for q in all_questions):
                        all_questions.append({
                            'id': len(all_questions) + 1,
                            'question': question
                        })
                        
                        type_counts[q_type] += 1
                        pbar.update(1)
                        logger.info(f"Q{len(all_questions)} ({q_type}): {question[:60]}...")
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                else:
                    retry_count += 1
                    consecutive_failures += 1
                    if retry_count % 50 == 0:
                        logger.warning(f"Retry count: {retry_count}")
                
                # 연속 실패시 다음 타입으로
                if consecutive_failures > 20:  # 10 → 20으로 증가
                    logger.warning(f"Too many failures for {q_type}, moving to next type")
                    break
                
                # GPU 메모리 정리
                if device == "cuda" and len(all_questions) % 10 == 0:
                    torch.cuda.empty_cache()
    
    # 결과 저장 (간소화된 형식)
    output_dir = Path("data/synthetic_questions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"remote_2000_questions_{timestamp}.csv"
    
    # id와 question 열만 저장
    df = pd.DataFrame(all_questions)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    logger.info(f"\nSaved {len(all_questions)} questions to {output_file}")
    
    # 통계 출력
    logger.info("\n" + "="*60)
    logger.info("Generation Statistics:")
    logger.info(f"Total questions: {len(all_questions)}")
    logger.info(f"Total attempts: {chunk_idx}")
    if chunk_idx > 0:
        logger.info(f"Success rate: {len(all_questions)/chunk_idx*100:.1f}%")
    logger.info(f"Retry count: {retry_count}")
    logger.info("\nQuestion type distribution:")
    for q_type, count in type_counts.items():
        logger.info(f"  {q_type}: {count}")
    
    # 샘플 출력
    logger.info("\nSample questions:")
    for i in range(min(10, len(all_questions))):
        logger.info(f"  {i+1}. {all_questions[i]['question']}")
    
    logger.info("="*60)
    
    print(f"\nSuccessfully generated {len(all_questions)} questions!")
    print(f"Output saved to: {output_file}")
    if chunk_idx > 0:
        print(f"Success rate: {len(all_questions)/chunk_idx*100:.1f}%")

if __name__ == "__main__":
    main()