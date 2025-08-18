#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG + LLM 추론 파이프라인
금융보안 AI 대회 submission 생성
"""

import os
import sys
import json
import time
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.rag.rag_pipeline import RAGPipeline
from packages.rag.embeddings.kure_embedder import KUREEmbedder
from packages.llm.qwen_quantized import QuantizedQwenLLM
from packages.llm.prompt_templates import FinancePromptTemplate, PromptOptimizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """추론 설정"""
    # 경로
    test_csv_path: str = "data/competition/test.csv"
    submission_path: str = "submissions/submission.csv"
    knowledge_base_path: str = "data/rag/knowledge_base_fixed"  # 수정된 KB 경로
    cache_dir: str = "data/cache/inference"
    
    # RAG 설정
    rag_top_k_initial: int = 30
    rag_top_k_final: int = 5
    enable_reranking: bool = True
    
    # LLM 설정
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    quantization_type: str = "4bit"
    max_new_tokens: int = 256
    temperature: float = 0.3
    
    # 배치 설정
    batch_size: int = 8
    max_memory_gb: float = 22.0
    
    # 타임아웃 설정
    timeout_per_question: float = 30.0  # 초
    
    # 캐싱 설정
    enable_cache: bool = True
    cache_ttl: int = 3600  # 초


class RAGInferencePipeline:
    """RAG 기반 추론 파이프라인"""
    
    def __init__(self, config: InferenceConfig):
        """
        Args:
            config: 추론 설정
        """
        self.config = config
        self.cache = {}
        self.statistics = {
            "total_questions": 0,
            "processed": 0,
            "cache_hits": 0,
            "rag_failures": 0,
            "llm_failures": 0,
            "timeouts": 0,
            "start_time": None,
            "end_time": None
        }
        
        # 디렉토리 생성
        Path(config.submission_path).parent.mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # 컴포넌트 초기화
        self._initialize_components()
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        logger.info("Initializing components...")
        
        # 1. Embedder 초기화
        logger.info("Loading KURE embedder...")
        self.embedder = KUREEmbedder(
            model_name="nlpai-lab/KURE-v1",
            batch_size=32,
            show_progress=False
        )
        
        # 2. RAG 파이프라인 초기화
        logger.info("Loading RAG pipeline with Vector Retriever + Reranking...")
        self.rag_pipeline = RAGPipeline(
            embedder=self.embedder,
            retriever_type="vector",  # Vector retriever 사용 (HybridRetriever 문제로 임시 변경)
            knowledge_base_path=self.config.knowledge_base_path if Path(self.config.knowledge_base_path).exists() else None,
            enable_reranking=self.config.enable_reranking,
            initial_retrieve_k=self.config.rag_top_k_initial,  # 30개 초기 검색
            final_k=self.config.rag_top_k_final  # 5개 최종 선택
        )
        
        # 3. LLM 초기화
        logger.info(f"Loading LLM: {self.config.model_id}")
        self.llm = QuantizedQwenLLM(
            model_id=self.config.model_id,
            quantization_type=self.config.quantization_type,
            cache_dir="./models"
        )
        
        # 4. 프롬프트 템플릿 초기화
        self.prompt_template = FinancePromptTemplate()
        self.prompt_optimizer = PromptOptimizer()
        
        logger.info("All components initialized successfully!")
        logger.info(f"GPU Memory: {self.llm.get_memory_footprint()}")
    
    def is_multiple_choice(self, question_text: str) -> Tuple[bool, int]:
        """
        객관식 여부 판단 및 선택지 개수 반환
        
        Args:
            question_text: 질문 텍스트
            
        Returns:
            (객관식 여부, 선택지 개수)
        """
        lines = question_text.strip().split("\n")
        option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
        return option_count >= 2, option_count
    
    def extract_answer_from_generation(self, generated_text: str, question: str) -> str:
        """
        생성된 텍스트에서 답변 추출
        
        Args:
            generated_text: 생성된 텍스트
            question: 원본 질문
            
        Returns:
            추출된 답변
        """
        # "답변:" 이후 텍스트 추출
        if "답변:" in generated_text:
            text = generated_text.split("답변:")[-1].strip()
        else:
            text = generated_text.strip()
        
        # 빈 응답 처리
        if not text:
            return "미응답"
        
        # 객관식 처리
        is_mc, option_count = self.is_multiple_choice(question)
        if is_mc:
            # 숫자만 추출
            match = re.match(r"\D*([1-9][0-9]?)", text)
            if match:
                answer_num = int(match.group(1))
                # 선택지 범위 검증 (1부터 option_count까지)
                if 1 <= answer_num <= option_count:
                    return str(answer_num)
                else:
                    # 범위를 벗어난 경우 1 반환
                    logger.warning(f"Answer {answer_num} out of range [1, {option_count}], defaulting to 1")
                    return "1"
            else:
                return "1"  # 기본값
        else:
            # 주관식: 전체 텍스트 반환
            return text
    
    def retrieve_context(self, question: str) -> List[str]:
        """
        RAG를 통한 컨텍스트 검색
        
        Args:
            question: 질문
            
        Returns:
            검색된 컨텍스트 리스트
        """
        try:
            # 검색 수행 - RAGPipeline.retrieve는 query_embedding을 내부에서 생성함
            # reranking이 활성화되어 있으면 initial_k(30) → final_k(5) 파이프라인 실행
            results = self.rag_pipeline.retrieve(
                query=question,
                top_k=self.config.rag_top_k_final,  # 최종 문서 개수 (5개)
                use_reranking=self.config.enable_reranking  # reranking 사용 여부
            )
            
            # 컨텍스트 추출
            contexts = [doc.get('content', '') for doc in results if doc.get('content')]
            
            # 로그 추가 - reranking 적용 여부 확인
            if self.config.enable_reranking:
                logger.debug(f"Retrieved {len(contexts)} documents after reranking (30→5)")
            else:
                logger.debug(f"Retrieved {len(contexts)} documents without reranking")
            
            return contexts[:self.config.rag_top_k_final]
            
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            self.statistics["rag_failures"] += 1
            return []
    
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """
        LLM을 통한 답변 생성
        
        Args:
            question: 질문
            contexts: RAG 컨텍스트
            
        Returns:
            생성된 답변
        """
        try:
            # 프롬프트 생성
            prompts = self.prompt_template.create_prompt(
                question=question,
                contexts=contexts if contexts else ["관련 문서를 찾을 수 없습니다."],
                include_citations=False  # 대회 제출용이므로 인용 제외
            )
            
            # 컨텍스트 최적화 (토큰 제한)
            optimized_contexts = self.prompt_optimizer.truncate_context(
                contexts,
                max_tokens=1500
            )
            
            # 최종 프롬프트
            is_mc, option_count = self.is_multiple_choice(question)
            if is_mc:
                # 객관식 전용 프롬프트 (선택지 개수 명시)
                final_prompt = f"""당신은 금융보안 전문가입니다.

참고 문서:
{chr(10).join(optimized_contexts) if optimized_contexts else "관련 문서 없음"}

다음 객관식 질문에 대해 정답 번호만 출력하세요.
주의: 선택지는 1번부터 {option_count}번까지 있습니다. 반드시 이 범위 내의 번호를 선택하세요.

질문: {question}

답변:"""
            else:
                # 주관식 프롬프트
                final_prompt = prompts["user"]
            
            # LLM 생성
            response = self.llm.generate_optimized(
                prompt=final_prompt,
                max_new_tokens=64 if is_mc else self.config.max_new_tokens,
                temperature=self.config.temperature,
                use_cache=True
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            self.statistics["llm_failures"] += 1
            
            # 폴백 응답
            is_mc, _ = self.is_multiple_choice(question)
            if is_mc:
                return "1"
            else:
                return "답변 생성 실패"
    
    def process_single_question(self, row: pd.Series) -> Dict[str, str]:
        """
        단일 질문 처리
        
        Args:
            row: DataFrame row (ID, Question)
            
        Returns:
            결과 딕셔너리
        """
        question_id = row['ID']
        question = row['Question']
        
        # 캐시 확인
        if self.config.enable_cache and question in self.cache:
            self.statistics["cache_hits"] += 1
            return {
                "ID": question_id,
                "Answer": self.cache[question]
            }
        
        try:
            # 1. RAG 검색
            contexts = self.retrieve_context(question)
            
            # 2. LLM 생성
            generated = self.generate_answer(question, contexts)
            
            # 3. 답변 추출
            answer = self.extract_answer_from_generation(generated, question)
            
            # 4. 캐싱
            if self.config.enable_cache:
                self.cache[question] = answer
            
            return {
                "ID": question_id,
                "Answer": answer
            }
            
        except Exception as e:
            logger.error(f"Failed to process question {question_id}: {e}")
            
            # 폴백 답변
            is_mc, _ = self.is_multiple_choice(question)
            if is_mc:
                answer = "1"
            else:
                answer = "처리 실패"
            
            return {
                "ID": question_id,
                "Answer": answer
            }
    
    def process_batch(self, batch_df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        배치 처리
        
        Args:
            batch_df: 배치 DataFrame
            
        Returns:
            결과 리스트
        """
        results = []
        
        for _, row in batch_df.iterrows():
            result = self.process_single_question(row)
            results.append(result)
            self.statistics["processed"] += 1
        
        return results
    
    def run(self, test_path: Optional[str] = None, sample_size: Optional[int] = None):
        """
        전체 추론 실행
        
        Args:
            test_path: 테스트 CSV 경로 (None이면 config 사용)
            sample_size: 샘플 크기 (테스트용)
        """
        # 통계 초기화
        self.statistics["start_time"] = datetime.now()
        
        # 데이터 로드
        test_path = test_path or self.config.test_csv_path
        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path, encoding='utf-8')
        
        # 샘플링 (테스트용)
        if sample_size:
            test_df = test_df.head(sample_size)
            logger.info(f"Using sample of {sample_size} questions")
        
        self.statistics["total_questions"] = len(test_df)
        logger.info(f"Total questions: {len(test_df)}")
        
        # 결과 저장용
        all_results = []
        
        # 배치 처리
        batch_size = self.config.batch_size
        num_batches = (len(test_df) + batch_size - 1) // batch_size
        
        with tqdm(total=len(test_df), desc="Processing questions") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(test_df))
                
                batch_df = test_df.iloc[start_idx:end_idx]
                
                # 배치 처리
                batch_results = self.process_batch(batch_df)
                all_results.extend(batch_results)
                
                # 진행 상황 업데이트
                pbar.update(len(batch_df))
                
                # 메모리 상태 체크
                if batch_idx % 10 == 0:
                    memory_status = self.llm.get_memory_footprint()
                    logger.info(f"Batch {batch_idx}/{num_batches}, Memory: {memory_status}")
        
        # 결과 저장
        self._save_results(all_results)
        
        # 통계 출력
        self.statistics["end_time"] = datetime.now()
        self._print_statistics()
    
    def _save_results(self, results: List[Dict[str, str]]):
        """결과 저장"""
        # DataFrame 생성
        submission_df = pd.DataFrame(results)
        
        # ID 순서대로 정렬
        submission_df = submission_df.sort_values('ID')
        
        # CSV 저장
        submission_path = self.config.submission_path
        submission_df.to_csv(submission_path, index=False, encoding='utf-8-sig')
        logger.info(f"Submission saved to {submission_path}")
        
        # 백업 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = submission_path.replace('.csv', f'_{timestamp}.csv')
        submission_df.to_csv(backup_path, index=False, encoding='utf-8-sig')
        logger.info(f"Backup saved to {backup_path}")
        
        # 캐시 저장
        if self.config.enable_cache and self.cache:
            cache_path = Path(self.config.cache_dir) / f"cache_{timestamp}.json"
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Cache saved to {cache_path}")
    
    def _print_statistics(self):
        """통계 출력"""
        duration = (self.statistics["end_time"] - self.statistics["start_time"]).total_seconds()
        
        print("\n" + "="*60)
        print(" 추론 완료 통계")
        print("="*60)
        print(f"총 질문 수: {self.statistics['total_questions']}")
        print(f"처리 완료: {self.statistics['processed']}")
        print(f"캐시 히트: {self.statistics['cache_hits']}")
        print(f"RAG 실패: {self.statistics['rag_failures']}")
        print(f"LLM 실패: {self.statistics['llm_failures']}")
        print(f"타임아웃: {self.statistics['timeouts']}")
        print(f"소요 시간: {duration:.2f}초 ({duration/60:.2f}분)")
        print(f"평균 처리 시간: {duration/self.statistics['processed']:.2f}초/질문")
        print("="*60)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG+LLM 추론 파이프라인")
    parser.add_argument("--test-csv", type=str, help="테스트 CSV 경로")
    parser.add_argument("--submission-path", type=str, help="제출 파일 경로")
    parser.add_argument("--knowledge-base", type=str, help="지식베이스 경로")
    parser.add_argument("--sample-size", type=int, help="샘플 크기 (테스트용)")
    parser.add_argument("--batch-size", type=int, default=8, help="배치 크기")
    parser.add_argument("--no-cache", action="store_true", help="캐시 비활성화")
    parser.add_argument("--no-reranking", action="store_true", help="재순위화 비활성화")
    
    args = parser.parse_args()
    
    # 설정 생성
    config = InferenceConfig()
    
    if args.test_csv:
        config.test_csv_path = args.test_csv
    if args.submission_path:
        config.submission_path = args.submission_path
    if args.knowledge_base:
        config.knowledge_base_path = args.knowledge_base
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.no_cache:
        config.enable_cache = False
    if args.no_reranking:
        config.enable_reranking = False
    
    # 파이프라인 실행
    pipeline = RAGInferencePipeline(config)
    pipeline.run(sample_size=args.sample_size)


if __name__ == "__main__":
    main()