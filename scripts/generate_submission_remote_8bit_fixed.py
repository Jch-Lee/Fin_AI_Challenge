#!/usr/bin/env python3
"""
Qwen2.5-7B + Updated DB 제출 파일 생성 - 8-bit 양자화 버전 (이미지 생성 방지)
대회 환경(RTX 4090 24GB)에 최적화
"""

import json
import logging
import sys
import time
import gc
import argparse
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
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'qwen_8bit_fixed_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QwenUpdatedDBPredictor:
    """Qwen2.5-7B + Updated DB (73 docs) 예측기 - 8-bit 버전 (이미지 생성 방지)"""
    
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
        logger.info("="*60)
        logger.info("시스템 초기화 시작...")
        logger.info("Updated DB: 73 documents, 8756 chunks")
        logger.info("8-bit 양자화 모드 (이미지 생성 방지)")
        logger.info("="*60)
        
        # 1. 데이터 로드
        logger.info("RAG 데이터 로드 중...")
        
        # 청크 로드
        chunks_path = self.data_dir / "chunks_2300.json"
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        logger.info(f"청크 로드 완료: {len(self.chunks)}개")
        
        # 메타데이터 확인
        metadata_path = self.data_dir / "metadata_2300.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"메타데이터: {metadata['total_documents']}개 문서, {metadata['statistics']['total_chunks']}개 청크")
        
        # FAISS 인덱스 로드
        self.faiss_index = faiss.read_index(str(self.data_dir / "faiss_index_2300.index"))
        logger.info(f"FAISS 인덱스 로드 완료: {self.faiss_index.ntotal}개 벡터")
        
        # BM25 인덱스 로드
        with open(self.data_dir / "bm25_index_2300.pkl", 'rb') as f:
            self.bm25_index = pickle.load(f)
        logger.info("BM25 인덱스 로드 완료")
        
        # 2. 임베더 초기화
        logger.info("임베더 초기화 중...")
        self.embedder = SentenceTransformer("nlpai-lab/KURE-v1", device="cuda")
        logger.info("KURE-v1 임베더 초기화 완료")
        
        # 3. Qwen2.5-7B LLM 초기화 (8-bit 양자화)
        logger.info("Qwen2.5-7B-Instruct 모델 초기화 중 (8-bit)...")
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        
        # 8-bit 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 8-bit 모델 로드
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Qwen2.5-7B-Instruct 모델 초기화 완료 (8-bit)")
        
        # GPU 정보 출력
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
        
        logger.info("시스템 초기화 완료!")
    
    def is_multiple_choice(self, question: str) -> bool:
        """객관식 문제 여부 판단"""
        lines = question.split('\n')
        choices = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and len(line) > 1:
                # 숫자로 시작하고 뒤에 내용이 있는 라인
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
        """Qwen 모델에 최적화된 프롬프트 생성 (이미지 생성 방지)"""
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
            
            # 참고 문서 포맷팅 (간결하게)
            formatted_docs = ""
            for i, doc in enumerate(contexts[:3], 1):  # 상위 3개만 사용
                if doc:
                    doc_preview = doc[:200]  # 200자로 제한
                    formatted_docs += f"[참고{i}] {doc_preview}\n"
            
            # Qwen 채팅 템플릿 사용
            messages = [
                {"role": "system", "content": "당신은 한국 금융보안 전문가입니다. 객관식 문제의 정답 번호만 답하세요."},
                {"role": "user", "content": f"""참고 자료:
{formatted_docs}

문제: {question_text}

선택지:
{chr(10).join(choices)}

지침: 1부터 {max_choice}까지 중 하나의 숫자만 답하세요. 설명 없이 숫자만 출력하세요.

정답 번호:"""}
            ]
            
        else:
            # 주관식 - 구체적인 답변 요구 (이미지 생성 방지 강화)
            # 참고 문서 포맷팅
            formatted_docs = ""
            for i, doc in enumerate(contexts, 1):
                if doc:
                    doc_preview = doc[:300]  # 300자로 제한
                    formatted_docs += f"[참고문서 {i}]\n{doc_preview}\n\n"
            
            messages = [
                {"role": "system", "content": "당신은 한국 금융보안 전문가입니다. 텍스트로만 답변하며, 이미지, 링크, URL, 마크다운 문법을 절대 사용하지 않습니다."},
                {"role": "user", "content": f"""참고 문서:
{formatted_docs}

질문: {question}

지침: 
- 참고 문서를 바탕으로 정확한 답변을 한국어로 작성하세요
- 순수 텍스트로만 답변하세요
- 이미지, URL, 링크, 마크다운 문법을 사용하지 마세요
- 그림이나 도표가 필요한 경우 텍스트로 설명하세요
- 핵심 내용을 2-3문장으로 설명하세요

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
        """Qwen 모델을 사용한 답변 생성 (이미지 토큰 차단)"""
        try:
            # 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.llm.device)
            
            # 차단할 토큰들 인코딩
            bad_tokens = ["![", "](", "http://", "https://", "www.", ".png", ".jpg", ".gif", ".jpeg", "placeholder"]
            bad_ids = []
            for token in bad_tokens:
                try:
                    ids = self.tokenizer.encode(token, add_special_tokens=False)
                    if ids:
                        bad_ids.append(ids)
                except:
                    pass
            
            # 생성 파라미터 (보수적 설정)
            is_mc = self.is_multiple_choice(question)
            
            with torch.no_grad():
                generated_ids = self.llm.generate(
                    **inputs,
                    max_new_tokens=32 if is_mc else 256,
                    temperature=0.05,  # 매우 낮게
                    top_p=0.9,  # 더 제한적
                    top_k=5,  # 더 작게
                    do_sample=False,  # 결정론적 생성
                    bad_words_ids=bad_ids if bad_ids else None,  # 차단 토큰
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 생성된 토큰만 추출
            generated_ids = generated_ids[:, inputs.input_ids.shape[-1]:]
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # 답변 추출 및 검증
            answer = self.extract_answer(response, question)
            
            return answer
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            
            if self.is_multiple_choice(question):
                return "1"
            else:
                return "답변을 생성할 수 없습니다."
    
    def extract_answer(self, text: str, question: str) -> str:
        """답변 추출 및 이미지 링크 제거"""
        if not text:
            return "1" if self.is_multiple_choice(question) else "답변 없음"
        
        # 이미지 마크다운 패턴 제거
        import re
        # 이미지 링크 패턴 제거
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        # URL 패턴 제거
        text = re.sub(r'https?://[^\s]+', '', text)
        # 중국어/일본어 문자 제거
        text = re.sub(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+', '', text)
        text = text.strip()
        
        # 이미지 관련 키워드가 포함된 경우 기본 답변
        if any(keyword in text.lower() for keyword in ['placeholder', 'image', 'img', 'png', 'jpg']):
            if self.is_multiple_choice(question):
                return "1"
            else:
                return "해당 질문에 대한 구체적인 답변은 참고 문서에서 찾을 수 없습니다."
        
        if self.is_multiple_choice(question):
            # 숫자만 추출
            numbers = re.findall(r'\d+', text)
            if numbers:
                # 첫 번째 숫자 반환
                return numbers[0]
            return "1"
        else:
            # 주관식 답변 정리
            # "답변:" 이후 텍스트 추출
            if '답변:' in text:
                text = text.split('답변:', 1)[1].strip()
            elif '답:' in text:
                text = text.split('답:', 1)[1].strip()
            
            # 불필요한 반복 제거
            lines = text.split('\n')
            if lines:
                text = lines[0].strip()
            
            # 최대 500자로 제한
            text = text[:500]
            
            return text if text else "답변을 생성할 수 없습니다."
    
    def predict_all(self, test_file: str, output_file: str, quick_test: bool = False):
        """전체 테스트셋 예측"""
        logger.info(f"테스트 파일 로드: {test_file}")
        
        # 테스트 데이터 로드
        df = pd.read_csv(test_file)
        
        if quick_test:
            df = df.head(10)  # 처음 10개만
            logger.info("빠른 테스트 모드: 10개 샘플만 처리")
        
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
                prompt = self.create_prompt(question, contexts)
                
                # 답변 생성
                answer = self.generate_answer(prompt, question)
                
                predictions.append({
                    'ID': question_id,
                    'Answer': answer
                })
                
                # 진행상황 로그 (50개마다)
                if (idx + 1) % 50 == 0:
                    logger.info(f"진행률: {idx + 1}/{len(df)} ({(idx + 1) / len(df) * 100:.1f}%)")
                
                # 메모리 정리 (100개마다)
                if (idx + 1) % 100 == 0:
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
        
        # 샘플 출력 (처음 5개)
        logger.info("\n처음 5개 예측 결과:")
        for i in range(min(5, len(predictions))):
            logger.info(f"  ID: {predictions[i]['ID']}, Answer: {predictions[i]['Answer'][:50]}...")
        
        return result_df

def main():
    """메인 실행 함수"""
    # 인자 파싱
    parser = argparse.ArgumentParser(description='Qwen2.5-7B + Updated DB 예측 (8-bit, 이미지 방지)')
    parser.add_argument('--quick-test', action='store_true', help='빠른 테스트 (10개만)')
    parser.add_argument('--test-file', type=str, default='test_full.csv', help='테스트 파일명')
    args = parser.parse_args()
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 예측기 초기화
    predictor = QwenUpdatedDBPredictor()
    predictor.setup()
    
    # 파일 경로 설정
    test_file = project_root / args.test_file
    
    if args.quick_test:
        output_file = project_root / f"quick_test_8bit_fixed_{timestamp}.csv"
    else:
        output_file = project_root / f"submission_8bit_fixed_{timestamp}.csv"
    
    # 예측 수행
    logger.info("="*60)
    logger.info("Qwen2.5-7B + Updated DB (73 docs) 예측 시작")
    logger.info("검색 방식: BM25+Vector Combined Top-3")
    logger.info("모델: 8-bit 양자화 (이미지 생성 방지)")
    logger.info("="*60)
    
    result_df = predictor.predict_all(str(test_file), str(output_file), args.quick_test)
    
    # 실행 시간 출력
    elapsed_time = time.time() - start_time
    logger.info(f"전체 실행 시간: {elapsed_time/60:.1f}분")
    
    # 최종 결과 요약
    logger.info("="*60)
    logger.info("예측 완료!")
    logger.info(f"제출 파일: {output_file}")
    logger.info(f"총 예측 수: {len(result_df)}")
    logger.info("="*60)

if __name__ == "__main__":
    main()