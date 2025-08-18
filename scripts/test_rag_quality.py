#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 파이프라인 및 답변 품질 종합 테스트
test.csv의 샘플 질문으로 전체 파이프라인 검증
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import pandas as pd
import torch
from colorama import init, Fore, Style

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.rag.rag_pipeline import RAGPipeline
from packages.rag.embeddings.kure_embedder import KUREEmbedder
from packages.llm.qwen_quantized import QuantizedQwenLLM
from packages.llm.prompt_templates import FinancePromptTemplate

# colorama 초기화
init(autoreset=True)

def print_section(title: str):
    """섹션 구분선 출력"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}{title}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

def print_success(msg: str):
    """성공 메시지"""
    print(f"{Fore.GREEN}[OK] {msg}{Style.RESET_ALL}")

def print_warning(msg: str):
    """경고 메시지"""
    print(f"{Fore.YELLOW}[WARN] {msg}{Style.RESET_ALL}")

def print_error(msg: str):
    """에러 메시지"""
    print(f"{Fore.RED}[ERROR] {msg}{Style.RESET_ALL}")

def test_rag_retrieval(rag_pipeline: RAGPipeline, question: str) -> Tuple[List[Dict], float]:
    """
    RAG 검색 테스트 및 상세 분석
    """
    print(f"\n{Fore.MAGENTA}[RAG 검색 테스트]{Style.RESET_ALL}")
    print(f"질문: {question[:100]}...")
    
    start_time = time.time()
    
    # Reranking 활성화 상태로 검색
    results_with_rerank = rag_pipeline.retrieve(
        query=question,
        top_k=5,
        use_reranking=True
    )
    
    # Reranking 비활성화 상태로 검색 (비교용)
    results_without_rerank = rag_pipeline.retrieve(
        query=question,
        top_k=5,
        use_reranking=False
    )
    
    retrieval_time = time.time() - start_time
    
    print(f"\n검색 시간: {retrieval_time:.2f}초")
    print(f"Reranking 적용: {len(results_with_rerank)}개 문서")
    print(f"Reranking 미적용: {len(results_without_rerank)}개 문서")
    
    # 상위 문서 내용 출력
    print(f"\n{Fore.YELLOW}[Reranking 적용 후 상위 3개 문서]{Style.RESET_ALL}")
    for i, doc in enumerate(results_with_rerank[:3], 1):
        content = doc.get('content', '')[:200]
        score = doc.get('score', 0.0)
        print(f"\n문서 {i} (점수: {score:.4f}):")
        print(f"{content}...")
    
    return results_with_rerank, retrieval_time

def generate_answer_with_llm(
    llm: QuantizedQwenLLM,
    question: str,
    contexts: List[str],
    is_multiple_choice: bool
) -> Tuple[str, float]:
    """
    LLM을 통한 답변 생성
    """
    print(f"\n{Fore.MAGENTA}[LLM 답변 생성]{Style.RESET_ALL}")
    
    start_time = time.time()
    
    # 프롬프트 생성
    if is_multiple_choice:
        prompt = f"""당신은 금융보안 전문가입니다.

참고 문서:
{chr(10).join(contexts[:3]) if contexts else "관련 문서 없음"}

다음 객관식 질문에 대해 정답 번호만 출력하세요.

질문: {question}

답변:"""
    else:
        prompt = f"""당신은 금융보안 전문가입니다.

참고 문서:
{chr(10).join(contexts[:3]) if contexts else "관련 문서 없음"}

다음 질문에 대해 정확하고 간략한 답변을 한국어로 작성하세요.

질문: {question}

답변:"""
    
    # LLM 생성
    try:
        response = llm.generate_optimized(
            prompt=prompt,
            max_new_tokens=64 if is_multiple_choice else 256,
            temperature=0.3,
            use_cache=False
        )
        generation_time = time.time() - start_time
        
        print(f"생성 시간: {generation_time:.2f}초")
        print(f"답변: {Fore.GREEN}{response}{Style.RESET_ALL}")
        
        return response, generation_time
        
    except Exception as e:
        print_error(f"LLM 생성 실패: {e}")
        return "생성 실패", 0.0

def evaluate_answer_quality(question: str, answer: str, contexts: List[str]) -> Dict:
    """
    답변 품질 평가
    """
    print(f"\n{Fore.MAGENTA}[답변 품질 평가]{Style.RESET_ALL}")
    
    evaluation = {
        "answer_length": len(answer),
        "has_number": any(c.isdigit() for c in answer),
        "is_korean": any('\uac00' <= c <= '\ud7a3' for c in answer),
        "context_relevance": 0.0,
        "completeness": "unknown"
    }
    
    # 객관식 답변 평가
    if "번" in question or any(f"{i}." in question for i in range(1, 10)):
        evaluation["question_type"] = "multiple_choice"
        # 숫자가 포함되어 있는지 확인
        if evaluation["has_number"]:
            evaluation["completeness"] = "valid"
            print_success("객관식 답변 형식 올바름")
        else:
            evaluation["completeness"] = "invalid"
            print_warning("객관식 답변에 숫자 없음")
    else:
        evaluation["question_type"] = "open_ended"
        # 최소 길이 체크
        if evaluation["answer_length"] > 10:
            evaluation["completeness"] = "valid"
            print_success("주관식 답변 길이 적절")
        else:
            evaluation["completeness"] = "too_short"
            print_warning("주관식 답변 너무 짧음")
    
    # 컨텍스트 관련성 체크 (간단한 키워드 매칭)
    if contexts:
        answer_lower = answer.lower()
        matching_keywords = 0
        total_keywords = 0
        
        for context in contexts[:3]:
            # 컨텍스트에서 주요 단어 추출 (간단한 방법)
            context_words = set(context.split())
            answer_words = set(answer.split())
            
            if context_words and answer_words:
                matching = len(context_words & answer_words)
                matching_keywords += matching
                total_keywords += len(context_words)
        
        if total_keywords > 0:
            evaluation["context_relevance"] = matching_keywords / total_keywords
            print(f"컨텍스트 관련성: {evaluation['context_relevance']:.2%}")
    
    return evaluation

def main():
    """메인 테스트 함수"""
    print_section("RAG 파이프라인 품질 테스트 시작")
    
    # GPU 확인
    if torch.cuda.is_available():
        print_success(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print_warning("GPU 사용 불가, CPU 모드로 실행")
    
    # 1. 테스트 데이터 로드
    print_section("1. 테스트 데이터 로드")
    test_csv_path = "data/competition/test.csv"
    
    if not Path(test_csv_path).exists():
        print_error(f"{test_csv_path} 파일이 없습니다")
        return
    
    df = pd.read_csv(test_csv_path)
    print_success(f"테스트 데이터 로드 완료: {len(df)}개 질문")
    
    # 샘플 선택 (다양한 유형의 질문 선택)
    sample_indices = [0, 10, 50, 100, 200]  # 5개 샘플
    sample_df = df.iloc[sample_indices]
    
    print(f"\n테스트할 샘플 질문 ID: {sample_df['ID'].tolist()}")
    
    # 2. RAG 파이프라인 초기화
    print_section("2. RAG 파이프라인 초기화")
    
    try:
        # Embedder 초기화
        print("KURE Embedder 로딩...")
        embedder = KUREEmbedder(
            model_name="nlpai-lab/KURE-v1",
            batch_size=32,
            show_progress=False
        )
        print_success("Embedder 초기화 완료")
        
        # RAG Pipeline 초기화
        print("RAG Pipeline 로딩...")
        rag_pipeline = RAGPipeline(
            embedder=embedder,
            retriever_type="hybrid",
            knowledge_base_path="data/rag/knowledge_base",
            enable_reranking=True,  # Reranking 활성화
            initial_retrieve_k=30,   # 초기 검색 30개
            final_k=5                # 최종 5개
        )
        print_success("RAG Pipeline 초기화 완료")
        
        # LLM 초기화
        print("Qwen2.5-7B-Instruct 로딩...")
        llm = QuantizedQwenLLM(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            quantization_type="4bit",
            cache_dir="./models"
        )
        print_success("LLM 초기화 완료")
        print(f"GPU 메모리 사용량: {llm.get_memory_footprint()}")
        
    except Exception as e:
        print_error(f"초기화 실패: {e}")
        return
    
    # 3. 샘플 질문 테스트
    print_section("3. 샘플 질문 테스트 시작")
    
    results = []
    total_time = 0
    
    for idx, row in sample_df.iterrows():
        question_id = row['ID']
        question = row['Question']
        
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"테스트 {len(results)+1}/5 - ID: {question_id}")
        print(f"{'='*60}{Style.RESET_ALL}")
        
        # 객관식 여부 판단
        is_mc = "번" in question or any(f"{i}." in question for i in range(1, 10))
        question_type = "객관식" if is_mc else "주관식"
        print(f"질문 유형: {question_type}")
        
        # RAG 검색
        retrieved_docs, retrieval_time = test_rag_retrieval(rag_pipeline, question)
        contexts = [doc.get('content', '') for doc in retrieved_docs]
        
        # LLM 답변 생성
        answer, generation_time = generate_answer_with_llm(llm, question, contexts, is_mc)
        
        # 품질 평가
        evaluation = evaluate_answer_quality(question, answer, contexts)
        
        # 결과 저장
        result = {
            "id": question_id,
            "question_type": question_type,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": retrieval_time + generation_time,
            "answer": answer,
            "answer_length": len(answer),
            "num_contexts": len(contexts),
            "evaluation": evaluation
        }
        results.append(result)
        total_time += result["total_time"]
        
        print(f"\n총 처리 시간: {result['total_time']:.2f}초")
    
    # 4. 전체 결과 요약
    print_section("4. 테스트 결과 요약")
    
    print(f"\n{Fore.YELLOW}[성능 메트릭]{Style.RESET_ALL}")
    avg_retrieval = sum(r["retrieval_time"] for r in results) / len(results)
    avg_generation = sum(r["generation_time"] for r in results) / len(results)
    avg_total = sum(r["total_time"] for r in results) / len(results)
    
    print(f"평균 검색 시간: {avg_retrieval:.2f}초")
    print(f"평균 생성 시간: {avg_generation:.2f}초")
    print(f"평균 총 시간: {avg_total:.2f}초")
    print(f"총 테스트 시간: {total_time:.2f}초")
    
    print(f"\n{Fore.YELLOW}[답변 품질]{Style.RESET_ALL}")
    valid_answers = sum(1 for r in results if r["evaluation"]["completeness"] == "valid")
    print(f"유효한 답변: {valid_answers}/{len(results)}")
    
    mc_results = [r for r in results if r["question_type"] == "객관식"]
    oe_results = [r for r in results if r["question_type"] == "주관식"]
    
    if mc_results:
        print(f"객관식 답변: {len(mc_results)}개")
        mc_valid = sum(1 for r in mc_results if r["evaluation"]["completeness"] == "valid")
        print(f"  - 유효: {mc_valid}/{len(mc_results)}")
    
    if oe_results:
        print(f"주관식 답변: {len(oe_results)}개")
        oe_valid = sum(1 for r in oe_results if r["evaluation"]["completeness"] == "valid")
        print(f"  - 유효: {oe_valid}/{len(oe_results)}")
        avg_length = sum(r["answer_length"] for r in oe_results) / len(oe_results)
        print(f"  - 평균 길이: {avg_length:.1f}자")
    
    # 5. 상세 결과 저장
    print_section("5. 결과 저장")
    
    # JSON으로 저장
    output_path = "test_results/rag_quality_test_results.json"
    Path("test_results").mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "test_time": datetime.now().isoformat(),
            "num_samples": len(results),
            "avg_metrics": {
                "retrieval_time": avg_retrieval,
                "generation_time": avg_generation,
                "total_time": avg_total
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print_success(f"결과 저장 완료: {output_path}")
    
    # 예상 전체 실행 시간 계산
    total_questions = len(df)
    estimated_time = avg_total * total_questions
    print(f"\n{Fore.CYAN}[전체 데이터셋 예상 시간]{Style.RESET_ALL}")
    print(f"총 {total_questions}개 질문 처리 예상 시간: {estimated_time/60:.1f}분")
    
    if estimated_time > 4.5 * 3600:  # 4.5시간 제한
        print_warning("예상 시간이 대회 제한 시간(4.5시간)을 초과합니다!")
        print_warning("최적화가 필요합니다 (배치 처리, 캐싱 등)")
    else:
        print_success(f"대회 제한 시간 내 처리 가능 (여유: {(4.5*3600 - estimated_time)/60:.1f}분)")

if __name__ == "__main__":
    main()