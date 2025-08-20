#!/usr/bin/env python3
"""
Gemma-Ko-7B + BM25+Vector Top3 방식 제출 파일 생성
모델 비교 실험: Qwen2.5-7B vs Gemma-Ko-7B
"""

import json
import logging
import sys
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from kiwipiepy import Kiwi
import faiss
import pickle
from tqdm import tqdm

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'gemma_submission_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GemmaCombinedTop3Predictor:
    """Gemma-Ko-7B + BM25+Vector Top3 예측기"""
    
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.embedder = None
        self.chunks = None
        self.bm25_index = None
        self.faiss_index = None
        self.kiwi = Kiwi()
        self.data_dir = project_root / "data" / "rag"
        
    def setup(self):
        """시스템 초기화"""
        logger.info("시스템 초기화 시작...")
        
        # 1. 데이터 로드
        logger.info("RAG 데이터 로드 중...")
        
        # 청크 로드
        with open(self.data_dir / "chunks_2300.json", 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        logger.info(f"청크 로드 완료: {len(self.chunks)}개")
        
        # FAISS 인덱스 로드
        self.faiss_index = faiss.read_index(str(self.data_dir / "faiss_index_2300.index"))
        logger.info("FAISS 인덱스 로드 완료")
        
        # BM25 인덱스 로드
        with open(self.data_dir / "bm25_index_2300.pkl", 'rb') as f:
            self.bm25_index = pickle.load(f)
        logger.info("BM25 인덱스 로드 완료")
        
        # 2. 임베더 초기화
        logger.info("임베더 초기화 중...")
        self.embedder = SentenceTransformer("nlpai-lab/KURE-v1", device="cuda")
        logger.info("임베더 초기화 완료")
        
        # 3. Gemma-Ko LLM 초기화
        logger.info("Gemma-Ko-7B 모델 초기화 중...")
        model_name = "beomi/gemma-ko-7b"
        
        # 4-bit 양자화 설정 (메모리 절약)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Gemma-Ko-7B 모델 초기화 완료 (4-bit quantized)")
        
        logger.info("시스템 초기화 완료!")
    
    def is_multiple_choice(self, question: str) -> bool:
        """객관식 문제 여부 판단"""
        lines = question.split('\n')
        choices = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and len(line) > 1:
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
    
    def create_prompt(self, question: str, contexts: List[str]) -> Tuple[str, str]:
        """Gemma 모델에 최적화된 프롬프트 생성"""
        is_mc = self.is_multiple_choice(question)
        
        if is_mc:
            # 객관식 - 극도로 제한적인 프롬프트
            lines = question.split('\n')
            question_text = lines[0]
            choices = []
            for line in lines[1:]:
                line = line.strip()
                if line and line[0].isdigit():
                    choices.append(line)
            
            max_choice = len(choices)
            
            formatted_docs = ""
            for i, doc in enumerate(contexts, 1):
                if doc:
                    doc_preview = doc[:300]
                    formatted_docs += f"\n[문서 {i}]:\n{doc_preview}\n"
            
            # Gemma는 system prompt를 지원하지 않으므로 user prompt에 통합
            prompt = f"""당신은 한국 금융보안 전문가입니다.

[참고 문서]
{formatted_docs}

[객관식 문제]
{question_text}

선택지:
{chr(10).join(choices)}

지침: 1~{max_choice} 중 하나의 숫자만 답하세요. 설명 없이 숫자만 출력하세요.

답:"""
            
            return "", prompt  # system prompt 비워둠
            
        else:
            # 주관식 - 구체적인 답변 요구
            formatted_docs = ""
            for i, doc in enumerate(contexts, 1):
                if doc:
                    doc_preview = doc[:400]
                    formatted_docs += f"\n[문서 {i}]:\n{doc_preview}\n"
            
            prompt = f"""당신은 한국 금융보안 전문가입니다.

[참고 문서]
{formatted_docs}

[주관식 문제]
{question}

지침: 참고 문서를 바탕으로 정확하고 간결한 답변을 작성하세요.

답변:"""
            
            return "", prompt  # system prompt 비워둠
    
    def generate_answer(self, system_prompt: str, user_prompt: str, question: str) -> str:
        """Gemma 모델을 사용한 답변 생성"""
        try:
            # Gemma는 채팅 템플릿을 사용하지 않고 직접 프롬프트 사용
            inputs = self.tokenizer(
                user_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.llm.device)
            
            with torch.no_grad():
                generated_ids = self.llm.generate(
                    **inputs,
                    max_new_tokens=64 if self.is_multiple_choice(question) else 512,
                    temperature=0.3,  # Gemma는 낮은 temperature가 더 안정적
                    top_p=0.8,
                    top_k=10,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1  # 반복 방지
                )
            
            # 생성된 토큰만 추출
            generated_ids = generated_ids[:, inputs.input_ids.shape[-1]:]
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # 답변 추출
            answer = self.extract_answer(response, question)
            
            return answer
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            
            if self.is_multiple_choice(question):
                return "1"
            else:
                return "답변을 생성할 수 없습니다."
    
    def extract_answer(self, text: str, question: str) -> str:
        """답변 추출"""
        if not text:
            return "1" if self.is_multiple_choice(question) else "답변 없음"
        
        # 중국어/일본어 문자 제거
        import re
        text = re.sub(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+', '', text)
        text = text.strip()
        
        if self.is_multiple_choice(question):
            # 숫자만 추출
            numbers = re.findall(r'\d+', text)
            if numbers:
                return numbers[0]
            return "1"
        else:
            # 주관식 답변 정리
            # "답:" 또는 "답변:" 이후 텍스트 추출
            if '답:' in text:
                text = text.split('답:', 1)[1].strip()
            elif '답변:' in text:
                text = text.split('답변:', 1)[1].strip()
            
            # 첫 500자만 반환
            text = text[:500]
            
            return text if text else "답변을 생성할 수 없습니다."
    
    def predict_all(self, test_file: str, output_file: str):
        """전체 테스트셋 예측"""
        logger.info(f"테스트 파일 로드: {test_file}")
        
        # 테스트 데이터 로드
        df = pd.read_csv(test_file)
        logger.info(f"총 {len(df)}개 문제 로드")
        
        # 문제 유형 통계
        mc_count = sum(1 for _, row in df.iterrows() if self.is_multiple_choice(row['Question']))
        desc_count = len(df) - mc_count
        logger.info(f"문제 유형: 객관식 {mc_count}개, 주관식 {desc_count}개")
        
        # 예측 수행
        predictions = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="예측 진행"):
            question_id = row['ID']
            question = row['Question']
            
            try:
                # 문서 검색 (BM25 3개 + Vector 3개)
                contexts = self.retrieve_combined_contexts(question)
                
                # 프롬프트 생성
                system_prompt, user_prompt = self.create_prompt(question, contexts)
                
                # 답변 생성
                answer = self.generate_answer(system_prompt, user_prompt, question)
                
                predictions.append({
                    'ID': question_id,
                    'Answer': answer
                })
                
                # 진행상황 로그 (100개마다)
                if (idx + 1) % 100 == 0:
                    logger.info(f"진행률: {idx + 1}/{len(df)} ({(idx + 1) / len(df) * 100:.1f}%)")
                    
                    # 메모리 정리
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"예측 실패 {question_id}: {e}")
                
                # 기본값 설정
                if self.is_multiple_choice(question):
                    answer = "1"
                else:
                    answer = "답변을 생성할 수 없습니다."
                
                predictions.append({
                    'ID': question_id,
                    'Answer': answer
                })
        
        # 결과 저장
        result_df = pd.DataFrame(predictions)
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"예측 완료! 결과 저장: {output_file}")
        
        # 통계 출력
        logger.info(f"총 예측 수: {len(predictions)}")
        
        return result_df

def main():
    """메인 실행 함수"""
    # 시작 시간 기록
    start_time = time.time()
    
    # 예측기 초기화
    predictor = GemmaCombinedTop3Predictor()
    predictor.setup()
    
    # 파일 경로 설정
    test_file = project_root / "test.csv"
    output_file = project_root / f"submission_gemma_combined_top3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # 예측 수행
    logger.info("="*60)
    logger.info("Gemma-Ko-7B + BM25+Vector Top3 예측 시작")
    logger.info("모델 비교 실험: Qwen2.5-7B (0.55) vs Gemma-Ko-7B")
    logger.info("="*60)
    
    result_df = predictor.predict_all(str(test_file), str(output_file))
    
    # 실행 시간 출력
    elapsed_time = time.time() - start_time
    logger.info(f"전체 실행 시간: {elapsed_time/60:.1f}분")
    
    # 최종 결과 요약
    logger.info("="*60)
    logger.info("예측 완료!")
    logger.info(f"제출 파일: {output_file}")
    logger.info(f"총 예측 수: {len(result_df)}")
    logger.info("이전 점수: Qwen2.5-7B = 0.55")
    logger.info("현재 실험: Gemma-Ko-7B = ?")
    logger.info("="*60)

if __name__ == "__main__":
    main()