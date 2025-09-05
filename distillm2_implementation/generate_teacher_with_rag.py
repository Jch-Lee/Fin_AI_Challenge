#!/usr/bin/env python3
"""
Teacher Model (Qwen2.5-14B) Response Generation with RAG
교사 모델에만 RAG를 적용하여 고품질 답변 생성
"""

import json
import logging
import sys
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from kiwipiepy import Kiwi
import faiss
import pickle
from tqdm import tqdm
from datasets import Dataset

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TeacherWithRAG:
    """Teacher Model with RAG System"""
    
    def __init__(self, data_dir: str = "/workspace/data"):
        self.data_dir = Path(data_dir)
        self.llm = None
        self.tokenizer = None
        self.embedder = None
        self.chunks = None
        self.bm25_index = None
        self.faiss_index = None
        self.kiwi = Kiwi()
        
    def setup(self):
        """시스템 초기화"""
        logger.info("="*60)
        logger.info("Teacher Model with RAG 초기화 시작...")
        logger.info("Model: Qwen2.5-14B-Instruct")
        logger.info("RAG: BM25 + FAISS Hybrid")
        logger.info("="*60)
        
        # 1. RAG 데이터 로드
        logger.info("RAG 데이터 로드 중...")
        self.load_rag_data()
        
        # 2. 임베더 초기화
        logger.info("임베더 초기화 중...")
        self.embedder = SentenceTransformer("nlpai-lab/KURE-v1", device="cuda")
        
        # 3. Teacher Model 초기화
        logger.info("Qwen2.5-14B-Instruct 모델 초기화 중...")
        model_name = "Qwen/Qwen2.5-14B-Instruct"
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 모델 로드 (bf16으로 4xA100에서 실행)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Teacher Model 초기화 완료!")
        
    def load_rag_data(self):
        """RAG 데이터 로드"""
        rag_dir = self.data_dir / "rag"
        
        # 청크 로드
        chunks_path = rag_dir / "chunks_2300.json"
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        logger.info(f"청크 로드 완료: {len(self.chunks)}개")
        
        # FAISS 인덱스 로드
        self.faiss_index = faiss.read_index(str(rag_dir / "faiss_index_2300.index"))
        logger.info(f"FAISS 인덱스 로드 완료: {self.faiss_index.ntotal}개 벡터")
        
        # BM25 인덱스 로드
        with open(rag_dir / "bm25_index_2300.pkl", 'rb') as f:
            self.bm25_index = pickle.load(f)
        logger.info("BM25 인덱스 로드 완료")
    
    def is_multiple_choice(self, question: str) -> bool:
        """객관식 문제 여부 판단"""
        lines = question.split('\n')
        choices = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and len(line) > 1:
                if '.' in line[:3] or ')' in line[:3] or ' ' in line[:3]:
                    choices.append(line)
        return len(choices) >= 2
    
    def search_bm25(self, question: str, k: int = 3) -> List[str]:
        """BM25 검색"""
        try:
            # 질문 토큰화
            tokens = []
            for token in self.kiwi.tokenize(question):
                tokens.append(token.form)
            
            # 청크 컨텐츠 리스트 생성
            corpus = []
            for chunk in self.chunks:
                if isinstance(chunk, dict):
                    corpus.append(chunk.get('content', ''))
                else:
                    corpus.append(chunk)
            
            # 상위 k개 문서 검색
            top_docs = self.bm25_index.get_top_n(tokens, corpus, n=k)
            return top_docs
        except Exception as e:
            logger.error(f"BM25 검색 실패: {e}")
            return []
    
    def search_vector(self, question: str, k: int = 3) -> List[str]:
        """Vector 검색"""
        try:
            # 질문 임베딩
            query_embedding = self.embedder.encode([question])
            
            # FAISS 검색
            scores, indices = self.faiss_index.search(query_embedding, k)
            
            # 문서 추출
            contexts = []
            for idx in indices[0]:
                if 0 <= idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    if isinstance(chunk, dict):
                        contexts.append(chunk.get('content', ''))
                    else:
                        contexts.append(chunk)
            return contexts
        except Exception as e:
            logger.error(f"Vector 검색 실패: {e}")
            return []
    
    def retrieve_combined_contexts(self, question: str) -> List[str]:
        """BM25 상위 3개 + Vector 상위 3개 독립적으로 검색"""
        # BM25 상위 3개 검색
        bm25_contexts = self.search_bm25(question, k=3)
        
        # Vector 상위 3개 검색
        vector_contexts = self.search_vector(question, k=3)
        
        # 결합 (중복 허용)
        combined_contexts = bm25_contexts + vector_contexts
        
        return combined_contexts
    
    def create_prompt(self, question: str, contexts: List[str]) -> str:
        """Qwen 모델에 최적화된 프롬프트 생성"""
        is_mc = self.is_multiple_choice(question)
        
        # 참고 문서 포맷팅
        formatted_docs = ""
        for i, doc in enumerate(contexts, 1):
            if doc:
                formatted_docs += f"[참고문서 {i}]\n{doc[:600]}\n\n"
        
        if is_mc:
            # 객관식
            lines = question.split('\n')
            question_text = lines[0]
            choices = []
            for line in lines[1:]:
                line = line.strip()
                if line and line[0].isdigit():
                    choices.append(line)
            
            messages = [
                {"role": "system", "content": "당신은 한국 금융보안 전문가입니다. 참고 문서를 깊이 이해하고 정확한 답변을 제공하세요."},
                {"role": "user", "content": f"""참고 문서:
{formatted_docs}

문제: {question_text}

선택지:
{chr(10).join(choices)}

지침: 참고 문서를 바탕으로 가장 정확한 답을 선택하세요. 답변은 숫자만 출력하세요.

정답:"""}
            ]
        else:
            # 주관식
            messages = [
                {"role": "system", "content": "당신은 한국 금융보안 전문가입니다. 참고 문서를 바탕으로 핵심만 간결하게 답변하세요."},
                {"role": "user", "content": f"""참고 문서:
{formatted_docs}

질문: {question}

지침: 
- 참고 문서의 핵심 내용만 추출하여 답변하세요
- 2-3문장으로 간결하게 요약하세요
- 불필요한 설명은 제외하세요
- 정확한 사실만 포함하세요

답변:"""}
            ]
        
        # 채팅 템플릿 적용
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def generate_answer(self, prompt: str, question: str) -> str:
        """답변 생성"""
        try:
            # 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.llm.device)
            
            # 생성 파라미터
            is_mc = self.is_multiple_choice(question)
            
            with torch.no_grad():
                generated_ids = self.llm.generate(
                    **inputs,
                    max_new_tokens=32 if is_mc else 256,  # 더 짧게
                    temperature=0.3,  # 더 집중된 답변
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 생성된 토큰만 추출
            generated_ids = generated_ids[:, inputs.input_ids.shape[-1]:]
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return "답변을 생성할 수 없습니다."
    
    def process_questions(self, input_file: str, output_file: str):
        """질문 처리 및 답변 생성"""
        logger.info(f"질문 파일 로드: {input_file}")
        
        # CSV 로드
        df = pd.read_csv(input_file)
        logger.info(f"총 {len(df)}개 질문 로드")
        
        # 결과 저장용
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Teacher 답변 생성"):
            question = row['question']
            
            # RAG 검색
            contexts = self.retrieve_combined_contexts(question)
            
            # 프롬프트 생성
            prompt = self.create_prompt(question, contexts)
            
            # 답변 생성
            answer = self.generate_answer(prompt, question)
            
            results.append({
                'prompt': question,
                'chosen': answer,  # Teacher 답변
                'rejected': ""  # Student 답변은 나중에 채움
            })
            
            # 메모리 정리
            if (idx + 1) % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # 데이터셋 생성 및 저장
        dataset = Dataset.from_pandas(pd.DataFrame(results))
        dataset.save_to_disk(output_file)
        logger.info(f"Teacher 답변 저장 완료: {output_file}")
        
        return dataset

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Teacher Model with RAG')
    parser.add_argument('--input_file', type=str, default='/workspace/data/combined_3000_questions.csv')
    parser.add_argument('--output_dir', type=str, default='/workspace/data/teacher_responses')
    parser.add_argument('--data_dir', type=str, default='/workspace/data')
    parser.add_argument('--num_samples', type=int, default=None, help='처리할 샘플 수')
    args = parser.parse_args()
    
    # Teacher 모델 초기화
    teacher = TeacherWithRAG(data_dir=args.data_dir)
    teacher.setup()
    
    # 입력 파일 처리
    input_file = args.input_file
    if args.num_samples:
        # 일부만 처리
        df = pd.read_csv(input_file)
        df_subset = df.head(args.num_samples)
        temp_file = '/tmp/temp_questions.csv'
        df_subset.to_csv(temp_file, index=False)
        input_file = temp_file
    
    # Teacher 답변 생성
    teacher.process_questions(input_file, args.output_dir)

if __name__ == "__main__":
    main()