#!/usr/bin/env python3
"""
BM25 상위 3개 + Vector 상위 3개 결합 실험
각 검색 방법에서 독립적으로 상위 3개씩 선택하여 총 6개 문서로 답변 생성
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from packages.rag.rag_pipeline import RAGPipeline

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'combined_top3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CombinedTop3Experiment:
    """BM25 상위 3개 + Vector 상위 3개 결합 실험"""
    
    def __init__(self):
        self.rag_system = None
        self.llm = None
        self.results = []
        self.data_dir = project_root / "data" / "rag"
        self.embedder = None  # 임베더를 한 번만 초기화
        
    def setup(self):
        """시스템 초기화"""
        logger.info("RAG 시스템 초기화 중...")
        
        # 데이터 직접 로드
        import numpy as np
        import pickle
        import faiss
        
        # 청크 로드
        with open(self.data_dir / "chunks_2300.json", 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # 임베딩 로드
        self.embeddings = np.load(self.data_dir / "embeddings_2300.npy")
        
        # FAISS 인덱스 로드
        self.faiss_index = faiss.read_index(str(self.data_dir / "faiss_index_2300.index"))
        
        # BM25 인덱스 로드
        with open(self.data_dir / "bm25_index_2300.pkl", 'rb') as f:
            self.bm25_index = pickle.load(f)
        
        logger.info(f"데이터 로드 완료: {len(self.chunks)} chunks")
        
        # 임베더 초기화 (한 번만)
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("nlpai-lab/KURE-v1", device="cuda")
        logger.info("임베더 초기화 완료")
        
        logger.info("RAG 초기화 완료")
        
        # LLM 초기화 (16-bit)
        logger.info("LLM 초기화 중 (16-bit)...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("LLM 초기화 완료")
    
    def load_test_questions(self, num_questions: int = 20) -> List[Dict]:
        """테스트 문제 로드"""
        test_file = project_root / "test.csv"
        questions = []
        
        import pandas as pd
        df = pd.read_csv(test_file)
        
        # 처음 20개 문제 사용
        for idx, row in df.head(num_questions).iterrows():
            questions.append({
                'ID': row['ID'],
                'Question': row['Question']
            })
        
        # 문제 유형 분류
        mc_count = 0
        desc_count = 0
        for q in questions:
            if self.is_multiple_choice(q['Question']):
                mc_count += 1
            else:
                desc_count += 1
        
        logger.info(f"로드된 문제: 객관식 {mc_count}개, 주관식 {desc_count}개")
        return questions
    
    def is_multiple_choice(self, question: str) -> bool:
        """객관식 문제 여부 판단"""
        lines = question.split('\n')
        choices = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and len(line) > 1:
                choices.append(line)
        return len(choices) >= 2
    
    def retrieve_combined_contexts(self, question: str) -> Tuple[List[str], Dict]:
        """BM25 상위 3개 + Vector 상위 3개 독립적으로 검색"""
        try:
            # BM25 상위 3개 검색
            bm25_contexts = self.search_bm25(question, k=3)
            
            # Vector 상위 3개 검색
            vector_contexts = self.search_vector(question, k=3)
            
            # 결합 (중복 허용)
            combined_contexts = bm25_contexts + vector_contexts
            
            # 메타데이터 생성
            metadata = {
                'bm25_docs': len(bm25_contexts),
                'vector_docs': len(vector_contexts),
                'total_docs': len(combined_contexts),
                'bm25_titles': self.extract_titles(bm25_contexts),
                'vector_titles': self.extract_titles(vector_contexts)
            }
            
            return combined_contexts, metadata
            
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return [], {}
    
    def search_bm25(self, question: str, k: int = 3) -> List[str]:
        """BM25 검색"""
        try:
            # Kiwi tokenizer를 사용한 한국어 토큰화
            from kiwipiepy import Kiwi
            kiwi = Kiwi()
            
            # 질문 토큰화
            tokens = []
            for token in kiwi.tokenize(question):
                tokens.append(token.form)
            
            # BM25 검색 (get_top_n 메서드 사용)
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
            # 질문 임베딩 (사전 초기화된 임베더 사용)
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
    
    def extract_titles(self, contexts: List[str]) -> List[str]:
        """문서 제목 추출"""
        titles = []
        for ctx in contexts:
            if ctx:
                lines = ctx.split('\n')
                title = "제목 없음"
                for line in lines:
                    if line.strip():
                        if line.startswith("##"):
                            title = line[2:].strip()[:50]
                        elif line.startswith("#"):
                            title = line[1:].strip()[:50]
                        else:
                            title = line.strip()[:50]
                        break
                titles.append(title)
        return titles
    
    def create_improved_prompt(self, question: str, contexts: List[str]) -> str:
        """개선된 프롬프트 생성 (generate_final_submission_bm25_070.py와 동일)"""
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
                    formatted_docs += f"\n[문서 {i}]\n{doc_preview}\n"
            
            system_prompt = """당신은 한국 금융보안 분야의 최고 전문가입니다.
주어진 참고 문서를 바탕으로 정확한 답변을 제공하세요.

핵심 규칙:
- 객관식은 반드시 숫자만 출력
- 주관식은 간결하고 정확한 설명 제공"""
            
            user_prompt = f"""[중요 지침 - 엄격히 준수]
• 반드시 1~{max_choice} 범위의 단일 숫자만 출력하세요
• 설명, 이유, 추가 텍스트는 절대 포함하지 마세요
• 숫자만 출력하세요 - 다른 모든 내용 금지

[참고 문서]
{formatted_docs}

[객관식 문제]
{question_text}

선택지:
{chr(10).join(choices)}

답변 (1~{max_choice} 중 하나의 숫자만):"""
            
        else:
            # 주관식 - 구체적인 답변 요구
            formatted_docs = ""
            for i, doc in enumerate(contexts, 1):
                if doc:
                    doc_preview = doc[:400]
                    formatted_docs += f"\n[문서 {i}]\n{doc_preview}\n"
            
            system_prompt = """당신은 한국 금융보안 분야의 최고 전문가입니다.
주어진 참고 문서를 바탕으로 정확한 답변을 제공하세요.

핵심 규칙:
- 객관식은 반드시 숫자만 출력
- 주관식은 간결하고 정확한 설명 제공"""
            
            user_prompt = f"""[중요 지침]
• 구체적이고 전문적인 답변을 작성하세요
• 핵심 내용을 명확히 설명하세요
• 한국어로 자연스럽게 작성하세요

[참고 문서]
{formatted_docs}

[주관식 문제]
{question}

답변:"""
        
        return system_prompt, user_prompt
    
    def generate_answer(self, system_prompt: str, user_prompt: str, question: str) -> str:
        """LLM을 사용한 답변 생성"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)
            
            import torch
            with torch.no_grad():
                generated_ids = self.llm.generate(
                    **model_inputs,
                    max_new_tokens=64 if self.is_multiple_choice(question) else 512,
                    temperature=0.1,
                    top_p=0.9,
                    top_k=10,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 답변 추출
            answer = self.extract_answer_simple(response, question)
            
            return answer
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            
            if self.is_multiple_choice(question):
                return "1"
            else:
                return "답변 생성 중 오류가 발생했습니다."
    
    def extract_answer_simple(self, text: str, question: str) -> str:
        """답변 추출 (개선된 버전)"""
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
            if ':' in text:
                text = text.split(':', 1)[1].strip()
            
            # 첫 500자만 반환
            text = text[:500]
            
            return text if text else "답변을 생성할 수 없습니다."
    
    def run_experiment(self):
        """실험 실행"""
        logger.info("\n" + "="*60)
        logger.info("BM25 상위 3개 + Vector 상위 3개 결합 실험 시작")
        logger.info("="*60)
        
        # 시스템 초기화
        self.setup()
        
        # 테스트 문제 로드
        questions = self.load_test_questions(20)
        
        # 각 문제 처리
        for idx, q in enumerate(questions, 1):
            logger.info(f"\n[{idx}/{len(questions)}] {q['ID']} 처리 중...")
            start_time = time.time()
            
            # BM25 + Vector 결합 검색
            contexts, metadata = self.retrieve_combined_contexts(q['Question'])
            
            # 프롬프트 생성
            system_prompt, user_prompt = self.create_improved_prompt(q['Question'], contexts)
            
            # 답변 생성
            answer = self.generate_answer(system_prompt, user_prompt, q['Question'])
            
            # 결과 저장
            result = {
                'question_id': q['ID'],
                'question_type': 'multiple_choice' if self.is_multiple_choice(q['Question']) else 'descriptive',
                'question_full': q['Question'],
                'answer': answer,
                'metadata': metadata,
                'processing_time': f"{time.time() - start_time:.1f}초"
            }
            
            self.results.append(result)
            
            logger.info(f"  답변: {answer[:50]}...")
            logger.info(f"  BM25 문서: {metadata.get('bm25_docs', 0)}개")
            logger.info(f"  Vector 문서: {metadata.get('vector_docs', 0)}개")
            logger.info(f"  처리 시간: {result['processing_time']}")
        
        # 결과 저장
        self.save_results()
    
    def save_results(self):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 통계 생성
        mc_results = [r for r in self.results if r['question_type'] == 'multiple_choice']
        desc_results = [r for r in self.results if r['question_type'] == 'descriptive']
        
        output = {
            'test_info': {
                'timestamp': timestamp,
                'test_type': 'BM25 Top3 + Vector Top3 Combined',
                'total_questions': len(self.results),
                'mc_questions': len(mc_results),
                'desc_questions': len(desc_results),
                'method': 'Independent top-3 from each method (6 docs total)'
            },
            'results': self.results,
            'summary': {
                'total_processing_time': sum(float(r['processing_time'][:-1]) for r in self.results),
                'avg_processing_time': sum(float(r['processing_time'][:-1]) for r in self.results) / len(self.results)
            }
        }
        
        output_file = f"combined_top3_results_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n결과 저장 완료: {output_file}")
        
        # 요약 출력
        print("\n" + "="*60)
        print("BM25 상위 3개 + Vector 상위 3개 결합 실험 완료")
        print("="*60)
        print(f"\n총 문제 수: {len(self.results)}")
        print(f"객관식: {len(mc_results)}개")
        print(f"주관식: {len(desc_results)}개")
        print(f"평균 처리 시간: {output['summary']['avg_processing_time']:.1f}초")
        print("\n" + "="*60)
        print("상세 결과는 JSON 파일을 확인하세요.")
        print("="*60)

if __name__ == "__main__":
    import torch
    
    experiment = CombinedTop3Experiment()
    experiment.run_experiment()