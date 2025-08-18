#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10개 문서 + 개선된 프롬프트 테스트
원격 서버에서 10개 질문 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import time
from datetime import datetime
from scripts.simple_improved_prompt import SimpleImprovedPrompt
from scripts.improved_answer_extraction import ImprovedAnswerExtractor

from packages.rag.rag_pipeline import RAGPipeline
from packages.rag.embeddings.kure_embedder import KUREEmbedder
from packages.llm.qwen_quantized import QuantizedQwenLLM

def test_10_questions():
    """10개 질문 테스트"""
    
    print("="*80)
    print("10개 문서 + 개선된 프롬프트 테스트")
    print("="*80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 테스트 데이터 로드 (10개만)
    print("\n[데이터 로드]")
    test_df = pd.read_csv("data/competition/test.csv")
    test_sample = test_df.head(10)  # 첫 10개 질문
    print(f"테스트 질문 수: {len(test_sample)}개")
    
    # 2. 컴포넌트 초기화
    print("\n[시스템 초기화]")
    
    # Embedder
    print("  - KURE Embedder 초기화...")
    embedder = KUREEmbedder(
        model_name="nlpai-lab/KURE-v1",
        batch_size=32,
        show_progress=False
    )
    
    # RAG Pipeline - 10개 문서, 리랭킹 없음
    print("  - RAG Pipeline 초기화 (10개 문서, 리랭킹 없음)...")
    rag_pipeline = RAGPipeline(
        embedder=embedder,
        retriever_type="hybrid",
        knowledge_base_path="data/rag/knowledge_base_fixed",
        enable_reranking=False,  # 리랭킹 비활성화
        initial_retrieve_k=10,    # 10개 검색
        final_k=10                # 10개 모두 사용
    )
    
    # LLM (8-bit)
    print("  - Qwen2.5-7B-Instruct (8-bit) 로드...")
    llm = QuantizedQwenLLM(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        quantization_type="8bit",
        cache_dir="./models"
    )
    memory_info = llm.get_memory_footprint()
    print(f"  - GPU 메모리: {memory_info}")
    
    # 프롬프트 및 추출기
    prompt_generator = SimpleImprovedPrompt()
    extractor = ImprovedAnswerExtractor()
    
    # 3. 질문 처리
    print("\n[테스트 시작]")
    print("-"*80)
    
    results = []
    start_time = time.time()
    
    for idx, row in test_sample.iterrows():
        question_id = row['ID']
        question = row['Question']
        
        print(f"\n질문 {idx+1}/10: {question_id}")
        print(f"내용: {question[:100]}...")
        
        question_start = time.time()
        
        try:
            # RAG 검색 - 10개 문서
            contexts_list = rag_pipeline.retrieve(
                query=question,
                top_k=10,
                use_reranking=False
            )
            
            # 점수 0.7 이상인 문서만 필터링
            filtered_docs = []
            if contexts_list:
                for doc in contexts_list:
                    score = doc.get('score', 0)
                    if score >= 0.7:
                        filtered_docs.append(doc.get('content', ''))
                    else:
                        print(f"    [필터링] 점수 {score:.3f} < 0.7, 제외")
            
            contexts = filtered_docs
            print(f"  검색된 문서: 10개 중 {len(contexts)}개 사용 (점수 >= 0.7)")
            
            # 객관식 여부 판단
            is_mc = any(f"{i}." in question or f"{i} " in question for i in range(1, 10))
            question_type = "객관식" if is_mc else "주관식"
            print(f"  질문 유형: {question_type}")
            
            # 프롬프트 생성
            prompt = prompt_generator.create_improved_prompt(
                question=question,
                contexts=contexts,
                is_mc=is_mc
            )
            print(f"  프롬프트 길이: {len(prompt)}자")
            
            # LLM 생성
            response = llm.generate_optimized(
                prompt=prompt,
                max_new_tokens=128 if is_mc else 256,
                temperature=0.1,
                use_cache=False
            )
            
            # 답변 추출
            if is_mc:
                max_choice = extractor.extract_choice_range(question)
                answer = extractor.extract_answer_from_response(
                    response=response,
                    max_choice=max_choice,
                    question=question
                )
                print(f"  선택지 범위: 1-{max_choice}")
            else:
                answer = extractor.extract_open_ended_answer(response)
            
            # 답변 검증
            if is_mc and len(answer) > 2:
                answer = "1"
            elif not is_mc and len(answer) > 500:
                answer = answer[:500]
            
            question_time = time.time() - question_start
            
            print(f"  최종 답변: {answer[:100] if len(answer) > 100 else answer}")
            print(f"  처리 시간: {question_time:.2f}초")
            
        except Exception as e:
            print(f"  [ERROR] 처리 실패: {e}")
            answer = "1" if is_mc else "처리 실패"
        
        results.append({
            "ID": question_id,
            "Answer": answer,
            "Type": question_type,
            "Time": f"{question_time:.2f}s"
        })
    
    total_time = time.time() - start_time
    
    # 4. 결과 요약
    print("\n" + "="*80)
    print("테스트 결과 요약")
    print("="*80)
    
    result_df = pd.DataFrame(results)
    
    print(f"\n총 처리 시간: {total_time:.1f}초")
    print(f"평균 처리 시간: {total_time/10:.1f}초/질문")
    
    print("\n[질문 유형별 통계]")
    type_counts = result_df['Type'].value_counts()
    for q_type, count in type_counts.items():
        print(f"  {q_type}: {count}개")
    
    print("\n[답변 결과]")
    for _, row in result_df.iterrows():
        answer_preview = str(row['Answer'])[:50] + "..." if len(str(row['Answer'])) > 50 else str(row['Answer'])
        print(f"  {row['ID']} ({row['Type']}): {answer_preview} [{row['Time']}]")
    
    # 5. 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results/test_10docs_improved_{timestamp}.csv"
    result_df[['ID', 'Answer']].to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n결과 저장: {output_file}")
    
    print(f"\n완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    test_10_questions()