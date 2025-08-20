#!/usr/bin/env python3
"""
문서 활용도 분석 스크립트
각 청크의 검색 빈도와 활용 패턴을 분석하여 벡터DB 최적화 정보 제공
"""

import json
import logging
import sys
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
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

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'document_usage_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentUsageAnalyzer:
    """문서 활용도 분석기"""
    
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.embedder = None
        self.chunks = None
        self.bm25_index = None
        self.faiss_index = None
        self.kiwi = Kiwi()
        self.data_dir = project_root / "data" / "rag"
        
        # 활용도 추적 변수
        self.bm25_usage = defaultdict(int)  # 청크 인덱스 -> BM25 검색 횟수
        self.vector_usage = defaultdict(int)  # 청크 인덱스 -> Vector 검색 횟수
        self.total_usage = defaultdict(int)  # 청크 인덱스 -> 전체 사용 횟수
        self.question_types = defaultdict(list)  # 청크 인덱스 -> 문제 유형 리스트
        self.retrieval_scores = defaultdict(list)  # 청크 인덱스 -> 검색 점수 리스트
        
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
        
        # 3. LLM 초기화
        logger.info("LLM 초기화 중 (16-bit)...")
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
    
    def search_bm25_with_tracking(self, question: str, k: int = 3) -> Tuple[List[str], List[int]]:
        """BM25 검색 (인덱스 추적 포함)"""
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
            
            # 상위 k개 문서 검색 (인덱스 포함)
            top_docs = self.bm25_index.get_top_n(tokens, corpus, n=k)
            
            # 인덱스 찾기
            indices = []
            for doc in top_docs:
                for i, chunk_content in enumerate(corpus):
                    if chunk_content == doc:
                        indices.append(i)
                        # BM25 사용 횟수 증가
                        self.bm25_usage[i] += 1
                        break
            
            return top_docs, indices
            
        except Exception as e:
            logger.error(f"BM25 검색 실패: {e}")
            return [], []
    
    def search_vector_with_tracking(self, question: str, k: int = 3) -> Tuple[List[str], List[int], np.ndarray]:
        """Vector 검색 (인덱스와 점수 추적 포함)"""
        try:
            # 질문 임베딩
            query_embedding = self.embedder.encode([question])
            
            # FAISS 검색
            scores, indices = self.faiss_index.search(query_embedding, k)
            
            # 문서 추출
            contexts = []
            valid_indices = []
            valid_scores = []
            
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    if isinstance(chunk, dict):
                        contexts.append(chunk.get('content', ''))
                    else:
                        contexts.append(chunk)
                    
                    valid_indices.append(idx)
                    valid_scores.append(float(score))
                    
                    # Vector 사용 횟수 증가
                    self.vector_usage[idx] += 1
                    # 검색 점수 기록
                    self.retrieval_scores[idx].append(float(score))
            
            return contexts, valid_indices, np.array(valid_scores)
            
        except Exception as e:
            logger.error(f"Vector 검색 실패: {e}")
            return [], [], np.array([])
    
    def retrieve_combined_contexts_with_tracking(self, question: str, question_type: str) -> Tuple[List[str], Dict]:
        """BM25 상위 3개 + Vector 상위 3개 독립적으로 검색 (추적 포함)"""
        # BM25 상위 3개 검색
        bm25_contexts, bm25_indices = self.search_bm25_with_tracking(question, k=3)
        
        # Vector 상위 3개 검색
        vector_contexts, vector_indices, vector_scores = self.search_vector_with_tracking(question, k=3)
        
        # 결합 (중복 허용)
        combined_contexts = bm25_contexts + vector_contexts
        
        # 전체 사용 횟수 업데이트
        all_indices = bm25_indices + vector_indices
        for idx in all_indices:
            self.total_usage[idx] += 1
            self.question_types[idx].append(question_type)
        
        # 메타데이터 생성
        metadata = {
            'bm25_indices': bm25_indices,
            'vector_indices': vector_indices,
            'vector_scores': vector_scores.tolist() if len(vector_scores) > 0 else [],
            'question_type': question_type
        }
        
        return combined_contexts, metadata
    
    def create_prompt(self, question: str, contexts: List[str]) -> Tuple[str, str]:
        """프롬프트 생성"""
        is_mc = self.is_multiple_choice(question)
        
        if is_mc:
            # 객관식 프롬프트
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
            # 주관식 프롬프트
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
            
            with torch.no_grad():
                generated_ids = self.llm.generate(
                    **model_inputs,
                    max_new_tokens=64 if self.is_multiple_choice(question) else 1024,
                    temperature=0.1,
                    top_p=0.90,
                    top_k=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
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
            if ':' in text:
                text = text.split(':', 1)[1].strip()
            
            # 첫 500자만 반환
            text = text[:500]
            
            return text if text else "답변을 생성할 수 없습니다."
    
    def analyze_and_predict(self, test_file: str, output_file: str):
        """전체 테스트셋 예측 및 활용도 분석"""
        logger.info(f"테스트 파일 로드: {test_file}")
        
        # 테스트 데이터 로드
        df = pd.read_csv(test_file)
        logger.info(f"총 {len(df)}개 문제 로드")
        
        # 문제 유형 통계
        mc_count = sum(1 for _, row in df.iterrows() if self.is_multiple_choice(row['Question']))
        desc_count = len(df) - mc_count
        logger.info(f"문제 유형: 객관식 {mc_count}개, 주관식 {desc_count}개")
        
        # 예측 수행 및 메타데이터 수집
        predictions = []
        retrieval_metadata = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="예측 및 분석 진행"):
            question_id = row['ID']
            question = row['Question']
            question_type = "MC" if self.is_multiple_choice(question) else "DESC"
            
            try:
                # 문서 검색 (BM25 3개 + Vector 3개) with 추적
                contexts, metadata = self.retrieve_combined_contexts_with_tracking(question, question_type)
                metadata['question_id'] = question_id
                retrieval_metadata.append(metadata)
                
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
        
        # 활용도 분석 결과 저장
        self.save_usage_analysis(retrieval_metadata)
        
        return result_df
    
    def save_usage_analysis(self, retrieval_metadata):
        """활용도 분석 결과 저장"""
        logger.info("활용도 분석 결과 생성 중...")
        
        # 1. 청크별 활용도 통계
        chunk_stats = []
        for i in range(len(self.chunks)):
            chunk_content = ""
            if isinstance(self.chunks[i], dict):
                chunk_content = self.chunks[i].get('content', '')[:100]  # 처음 100자
            else:
                chunk_content = self.chunks[i][:100]
            
            stats = {
                'chunk_id': i,
                'content_preview': chunk_content,
                'bm25_count': self.bm25_usage[i],
                'vector_count': self.vector_usage[i],
                'total_count': self.total_usage[i],
                'mc_count': self.question_types[i].count('MC'),
                'desc_count': self.question_types[i].count('DESC'),
                'avg_vector_score': np.mean(self.retrieval_scores[i]) if self.retrieval_scores[i] else 0
            }
            chunk_stats.append(stats)
        
        # DataFrame 생성 및 저장
        stats_df = pd.DataFrame(chunk_stats)
        stats_df = stats_df.sort_values('total_count', ascending=False)
        
        # 활용도 통계 파일 저장
        stats_file = project_root / f"document_usage_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        logger.info(f"활용도 통계 저장: {stats_file}")
        
        # 2. 요약 리포트 생성
        report = []
        report.append("=" * 60)
        report.append("문서 활용도 분석 리포트")
        report.append("=" * 60)
        report.append("")
        
        # 전체 통계
        report.append("## 전체 통계")
        report.append(f"- 총 청크 수: {len(self.chunks)}")
        report.append(f"- 사용된 청크 수: {len([i for i in range(len(self.chunks)) if self.total_usage[i] > 0])}")
        report.append(f"- 미사용 청크 수: {len([i for i in range(len(self.chunks)) if self.total_usage[i] == 0])}")
        report.append(f"- 평균 사용 횟수: {np.mean(list(self.total_usage.values())):.2f}")
        report.append("")
        
        # 상위 10개 청크
        report.append("## 가장 많이 사용된 청크 TOP 10")
        top_10 = stats_df.head(10)
        for idx, row in top_10.iterrows():
            report.append(f"{row['chunk_id']:4d} | 총 {row['total_count']:3d}회 | BM25: {row['bm25_count']:3d} | Vector: {row['vector_count']:3d} | {row['content_preview'][:50]}...")
        report.append("")
        
        # 미사용 청크 분석
        unused = stats_df[stats_df['total_count'] == 0]
        report.append(f"## 미사용 청크 분석")
        report.append(f"- 미사용 청크 수: {len(unused)} / {len(self.chunks)} ({len(unused)/len(self.chunks)*100:.1f}%)")
        report.append(f"- 미사용 청크 ID 샘플: {unused['chunk_id'].head(20).tolist()}")
        report.append("")
        
        # 검색 방법별 통계
        report.append("## 검색 방법별 통계")
        report.append(f"- BM25 전용 사용: {len([i for i in range(len(self.chunks)) if self.bm25_usage[i] > 0 and self.vector_usage[i] == 0])}")
        report.append(f"- Vector 전용 사용: {len([i for i in range(len(self.chunks)) if self.vector_usage[i] > 0 and self.bm25_usage[i] == 0])}")
        report.append(f"- 둘 다 사용: {len([i for i in range(len(self.chunks)) if self.bm25_usage[i] > 0 and self.vector_usage[i] > 0])}")
        report.append("")
        
        # 문제 유형별 통계
        report.append("## 문제 유형별 선호도")
        mc_preferred = stats_df[stats_df['mc_count'] > stats_df['desc_count']]
        desc_preferred = stats_df[stats_df['desc_count'] > stats_df['mc_count']]
        report.append(f"- 객관식 선호 청크: {len(mc_preferred)}")
        report.append(f"- 주관식 선호 청크: {len(desc_preferred)}")
        report.append("")
        
        # 권장사항
        report.append("## 권장사항")
        report.append("### 제거 대상")
        report.append(f"- 미사용 청크 {len(unused)}개 제거 검토")
        report.append("")
        
        report.append("### 품질 개선 대상")
        low_score = stats_df[(stats_df['total_count'] > 0) & (stats_df['avg_vector_score'] < 0.5)]
        report.append(f"- 낮은 유사도 점수 청크 {len(low_score)}개 내용 개선 필요")
        report.append("")
        
        report.append("### 증강 대상")
        high_usage = stats_df[stats_df['total_count'] > 10]
        report.append(f"- 고빈도 사용 청크 {len(high_usage)}개 유사 문서 추가 검토")
        
        # 리포트 저장
        report_file = project_root / f"document_usage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        logger.info(f"분석 리포트 저장: {report_file}")
        
        # 메타데이터 JSON 저장
        metadata_file = project_root / f"retrieval_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_questions': len(retrieval_metadata),
                'chunk_usage': dict(self.total_usage),
                'bm25_usage': dict(self.bm25_usage),
                'vector_usage': dict(self.vector_usage),
                'retrieval_details': retrieval_metadata[:100]  # 처음 100개만 샘플로
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"메타데이터 저장: {metadata_file}")

def main():
    """메인 실행 함수"""
    # 시작 시간 기록
    start_time = time.time()
    
    # 분석기 초기화
    analyzer = DocumentUsageAnalyzer()
    analyzer.setup()
    
    # 파일 경로 설정
    test_file = project_root / "test.csv"
    output_file = project_root / f"submission_with_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # 예측 및 분석 수행
    logger.info("="*60)
    logger.info("문서 활용도 분석을 포함한 예측 시작")
    logger.info("="*60)
    
    result_df = analyzer.analyze_and_predict(str(test_file), str(output_file))
    
    # 실행 시간 출력
    elapsed_time = time.time() - start_time
    logger.info(f"전체 실행 시간: {elapsed_time/60:.1f}분")
    
    # 최종 결과 요약
    logger.info("="*60)
    logger.info("분석 완료!")
    logger.info(f"제출 파일: {output_file}")
    logger.info(f"총 예측 수: {len(result_df)}")
    logger.info("활용도 분석 파일이 생성되었습니다.")
    logger.info("="*60)

if __name__ == "__main__":
    main()