#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3000개 질문 대량 생성 스크립트
- 10개 배치 × 300개씩 생성
- 3가지 temperature 로테이션 (0.1, 0.3, 0.4)
- 6가지 질문 유형별 균등 분배
- 체크포인트 및 복구 기능
"""

import json
import random
import time
import os
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
        logging.FileHandler('bulk_generation_3000.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 완화된 불용어 리스트
CRITICAL_STOPWORDS = {
    '자료 1', '자료 2', '자료 3', '위 내용', '위 자료', '이 문서', '이 텍스트',
    '답변:', '질문:', '예시:', '참고:', 'Q:', 'A:', '출처:',
    '###', '...', '---', '___', '***', '```',
    '연락처', '누리집', '웹사이트', '홈페이지', '발간', '페이지',
    '첫번째 자료', '두번째 자료', '세번째 자료'
}

# Temperature 설정
TEMPERATURE_SETS = [
    {"temperature": 0.1, "top_p": 0.9},   # 매우 일관적
    {"temperature": 0.3, "top_p": 0.9},   # 기본값 (검증된)
    {"temperature": 0.4, "top_p": 0.9}    # 창의적
]

# 질문 유형별 목표 개수 (총 3000개)
QUESTION_TYPES = {
    'definition': 500,    # 개념/용어 정의
    'process': 500,       # 절차/방법론
    'regulation': 500,    # 법률/규정
    'example': 500,       # 구체적 사례
    'comparison': 500,    # 차이점/비교
    'application': 500    # 활용/대응방안
}

def load_chunks(chunks_file: str) -> List[Dict]:
    """청크 데이터 로드"""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks

def sample_diverse_chunks_for_batch(chunks: List[Dict], batch_num: int, total_batches: int = 10) -> List[Dict]:
    """배치별 다양성 기반 청크 샘플링 - 전체 청크 활용"""
    # 문서별 그룹핑
    doc_groups = defaultdict(list)
    for chunk in chunks:
        doc_groups[chunk.get('source', 'unknown')].append(chunk)
    
    # 배치별로 다른 청크 세트 사용
    chunk_per_batch = len(chunks) // total_batches
    start_idx = (batch_num - 1) * chunk_per_batch
    end_idx = min(batch_num * chunk_per_batch, len(chunks))
    
    # 배치 범위 내에서 샘플링
    batch_chunks = chunks[start_idx:end_idx]
    
    # 추가 다양성을 위해 셔플
    random.shuffle(batch_chunks)
    
    logger.info(f"Batch {batch_num}: Using chunks {start_idx}-{end_idx} ({len(batch_chunks)} chunks)")
    logger.info(f"Document sources: {len(set(c.get('source') for c in batch_chunks))}")
    
    return batch_chunks

def create_single_question_prompt(chunk_content: str, question_type: str) -> str:
    """단일 질문 생성을 위한 프롬프트"""
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
    """완화된 불용어 체크"""
    for stopword in CRITICAL_STOPWORDS:
        if stopword in text:
            return True
    return False

def clean_single_question(text: str) -> Optional[str]:
    """생성된 단일 질문 정제"""
    lines = text.strip().split('\n')
    if not lines:
        return None
    
    question = lines[0].strip()
    
    # 번호 및 레이블 제거
    question = re.sub(r'^\d+[\.\)]\s*', '', question)
    question = re.sub(r'^질문[:：]\s*', '', question)
    question = re.sub(r'^답변[:：]\s*', '', question)
    question = re.sub(r'^[QA]\d*[:：]\s*', '', question)
    
    # 특수문자 정리 (괄호는 유지)
    question = re.sub(r'[•·\-\*]+\s*', '', question)
    question = re.sub(r'[\[\]<>{}]+', '', question)
    question = re.sub(r'#+\s*', '', question)
    question = re.sub(r'_+\s*', '', question)
    
    # 한글 포함 확인
    if not re.search(r'[가-힣]', question):
        return None
    
    # 외국어 제외
    if re.search(r'[\u4e00-\u9fff]', question):  # 중국어
        return None
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', question):  # 일본어
        return None
    
    # 불용어 체크
    if contains_critical_stopwords(question):
        return None
    
    # 길이 체크
    if len(question) < 10 or len(question) > 300:
        return None
    
    # 물음표 추가
    if not question.endswith('?'):
        question += '?'
    
    # 공백 정리
    question = ' '.join(question.split())
    
    return question

def generate_single_question(model, tokenizer, chunk: Dict, question_type: str, temp_config: Dict) -> Optional[str]:
    """단일 질문 생성"""
    prompt = create_single_question_prompt(chunk['content'], question_type)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
    
    # 모델의 디바이스 확인
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=temp_config["temperature"],
            top_p=temp_config["top_p"],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    question = clean_single_question(generated)
    
    return question

def save_batch_results(batch_num: int, questions: List[Dict], output_dir: Path):
    """배치 결과 저장"""
    batch_file = output_dir / f"batch_{batch_num:03d}_questions_{len(questions)}.csv"
    df = pd.DataFrame(questions)
    df.to_csv(batch_file, index=False, encoding='utf-8-sig')
    logger.info(f"Saved batch {batch_num} to {batch_file}")

def save_checkpoint(batch_num: int, type_counts: Dict, all_questions: List[Dict], output_dir: Path):
    """체크포인트 저장"""
    checkpoint = {
        'batch_num': batch_num,
        'type_counts': dict(type_counts),
        'total_questions': len(all_questions),
        'timestamp': time.strftime('%Y%m%d_%H%M%S')
    }
    
    checkpoint_file = output_dir / f"checkpoint_batch_{batch_num:03d}.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

def load_checkpoint(output_dir: Path) -> Optional[Dict]:
    """가장 최근 체크포인트 로드"""
    checkpoint_files = list(output_dir.glob("checkpoint_batch_*.json"))
    if not checkpoint_files:
        return None
    
    # 가장 최근 체크포인트 찾기
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_checkpoint, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)
    
    logger.info(f"Loaded checkpoint: batch {checkpoint['batch_num']}, {checkpoint['total_questions']} questions")
    return checkpoint

def generate_batch(model, tokenizer, chunks: List[Dict], batch_num: int, target_per_type: int, 
                  temp_config: Dict, existing_questions: set) -> List[Dict]:
    """단일 배치 생성"""
    batch_questions = []
    type_counts = defaultdict(int)
    chunk_idx = 0
    max_attempts = 500  # 배치당 최대 시도
    attempts = 0
    
    logger.info(f"Batch {batch_num}: Starting with temperature={temp_config['temperature']}")
    
    # 각 타입별로 target_per_type개씩 생성
    for q_type in QUESTION_TYPES.keys():
        consecutive_failures = 0
        max_consecutive_failures = 20
        
        while type_counts[q_type] < target_per_type and attempts < max_attempts:
            if consecutive_failures > max_consecutive_failures:
                logger.warning(f"Batch {batch_num}: Too many consecutive failures for {q_type}, moving to next type")
                break
            
            # 청크 선택
            chunk = chunks[chunk_idx % len(chunks)]
            chunk_idx += 1
            attempts += 1
            
            # 질문 생성
            question = generate_single_question(model, tokenizer, chunk, q_type, temp_config)
            
            if question and question not in existing_questions:
                batch_questions.append({
                    'id': len(batch_questions) + 1,
                    'question': question,
                    'type': q_type,
                    'batch': batch_num,
                    'temperature': temp_config['temperature']
                })
                
                existing_questions.add(question)
                type_counts[q_type] += 1
                consecutive_failures = 0
                
                if len(batch_questions) % 10 == 0:
                    logger.info(f"Batch {batch_num}: {len(batch_questions)} questions generated")
            else:
                consecutive_failures += 1
            
            # GPU 메모리 정리
            if attempts % 20 == 0:
                torch.cuda.empty_cache()
    
    logger.info(f"Batch {batch_num} completed: {len(batch_questions)} questions, {attempts} attempts")
    logger.info(f"Type distribution: {dict(type_counts)}")
    
    return batch_questions

def main():
    """메인 함수"""
    logger.info("="*60)
    logger.info("Starting 3000 Questions Bulk Generation")
    logger.info("="*60)
    
    # GPU 확인
    if not torch.cuda.is_available():
        logger.error("GPU not available!")
        return
    
    device = "cuda"
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 모델 로드
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Model loaded successfully")
    
    # 청크 로드
    chunks = load_chunks("data/rag/chunks_2300.json")
    
    # 출력 디렉토리 생성
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"data/synthetic_questions/bulk_generation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 체크포인트 확인
    checkpoint = load_checkpoint(output_dir)
    start_batch = checkpoint['batch_num'] + 1 if checkpoint else 1
    all_questions = []
    existing_questions = set()
    total_type_counts = defaultdict(int)
    
    # 기존 결과 로드 (체크포인트가 있는 경우)
    if checkpoint:
        for batch_num in range(1, start_batch):
            batch_file = output_dir / f"batch_{batch_num:03d}_questions_*.csv"
            batch_files = list(output_dir.glob(f"batch_{batch_num:03d}_questions_*.csv"))
            if batch_files:
                df = pd.read_csv(batch_files[0])
                batch_questions = df.to_dict('records')
                all_questions.extend(batch_questions)
                for q in batch_questions:
                    existing_questions.add(q['question'])
                    if 'type' in q:
                        total_type_counts[q['type']] += 1
    
    # 배치 생성 (10개 배치)
    total_batches = 10
    target_per_batch_per_type = 50  # 각 배치에서 각 타입당 50개 (총 300개/배치)
    
    for batch_num in range(start_batch, total_batches + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing Batch {batch_num}/{total_batches}")
        logger.info(f"{'='*50}")
        
        # 배치별 청크 샘플링
        batch_chunks = sample_diverse_chunks_for_batch(chunks, batch_num, total_batches)
        
        # Temperature 설정 (로테이션)
        temp_config = TEMPERATURE_SETS[(batch_num - 1) % len(TEMPERATURE_SETS)]
        
        # 배치 생성
        batch_questions = generate_batch(
            model, tokenizer, batch_chunks, batch_num, 
            target_per_batch_per_type, temp_config, existing_questions
        )
        
        # 결과 저장
        save_batch_results(batch_num, batch_questions, output_dir)
        all_questions.extend(batch_questions)
        
        # 타입별 카운트 업데이트
        for q in batch_questions:
            if 'type' in q:
                total_type_counts[q['type']] += 1
        
        # 체크포인트 저장
        save_checkpoint(batch_num, total_type_counts, all_questions, output_dir)
        
        # 진행률 출력
        logger.info(f"Total progress: {len(all_questions)}/3000 ({len(all_questions)/3000*100:.1f}%)")
        logger.info(f"Current type distribution: {dict(total_type_counts)}")
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
    
    # 최종 통합 파일 생성
    combined_file = output_dir / "combined_3000_questions.csv"
    df_combined = pd.DataFrame(all_questions)
    df_combined['id'] = range(1, len(df_combined) + 1)  # ID 재부여
    df_combined.to_csv(combined_file, index=False, encoding='utf-8-sig')
    
    # 통계 보고서 생성
    report = {
        'generation_time': timestamp,
        'total_questions': len(all_questions),
        'total_batches': total_batches,
        'type_distribution': dict(total_type_counts),
        'temperature_sets': TEMPERATURE_SETS,
        'success_rate': f"{len(all_questions)/3000*100:.1f}%",
        'unique_questions': len(existing_questions),
        'duplicate_rate': f"{(len(all_questions) - len(existing_questions))/len(all_questions)*100:.1f}%"
    }
    
    report_file = output_dir / "generation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 최종 결과 출력
    logger.info("\n" + "="*60)
    logger.info("BULK GENERATION COMPLETED")
    logger.info("="*60)
    logger.info(f"Total questions generated: {len(all_questions)}")
    logger.info(f"Target: 3000, Success rate: {len(all_questions)/3000*100:.1f}%")
    logger.info(f"Type distribution: {dict(total_type_counts)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Combined file: {combined_file}")
    logger.info("="*60)

if __name__ == "__main__":
    main()