#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10개 문서 참고 버전 - 리랭커 없이 전체 활용
추론 시간 단축을 위한 최적화
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

def validate_answer_length(answer: str, is_mc: bool) -> str:
    """답변 길이 검증"""
    if is_mc:
        if len(answer) > 2:
            return "1"
        return answer
    else:
        if len(answer) > 500:
            return answer[:500]
        return answer

def main():
    """메인 실행 함수 - 10개 문서 버전"""
    
    print("제출파일 생성 시작 (10개 문서 참고):", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    start_time = time.time()
    
    # 1. 데이터 로드
    print("테스트 데이터 로드 중...")
    test_df = pd.read_csv("data/competition/test.csv")
    total_questions = len(test_df)
    print(f"총 {total_questions}개 질문")
    
    # 2. 컴포넌트 초기화
    print("컴포넌트 초기화 중...")
    
    # Embedder
    embedder = KUREEmbedder(
        model_name="nlpai-lab/KURE-v1",
        batch_size=32,
        show_progress=False
    )
    print("Embedder 완료")
    
    # RAG Pipeline - 10개 문서 검색, 리랭킹 없음
    rag_pipeline = RAGPipeline(
        embedder=embedder,
        retriever_type="hybrid",
        knowledge_base_path="data/rag/knowledge_base_fixed",
        enable_reranking=False,  # 리랭킹 비활성화
        initial_retrieve_k=10,    # 10개 검색
        final_k=10                # 10개 모두 사용
    )
    print("RAG Pipeline 완료 (10개 문서 검색)")
    
    # LLM (8-bit)
    llm = QuantizedQwenLLM(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        quantization_type="8bit",
        cache_dir="./models"
    )
    print("LLM 로드 완료")
    
    # 프롬프트 및 추출기
    prompt_generator = SimpleImprovedPrompt()
    extractor = ImprovedAnswerExtractor()
    
    # 3. 질문 처리
    print(f"질문 처리 시작... (총 {total_questions}개)")
    
    results = []
    failed_count = 0
    
    for idx, row in test_df.iterrows():
        question_id = row['ID']
        question = row['Question']
        
        current_num = idx + 1
        
        # 진행률 출력 (매 50개마다)
        if current_num % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / current_num
            remaining = (total_questions - current_num) * avg_time
            print(f"진행: {current_num}/{total_questions} ({current_num/total_questions*100:.1f}%) - 예상 남은 시간: {remaining/60:.0f}분")
        
        try:
            # RAG 검색 - 10개 문서 검색
            contexts_list = rag_pipeline.retrieve(
                query=question,
                top_k=10,  # 10개 문서 검색
                use_reranking=False  # 리랭킹 없음
            )
            
            # 점수 0.7 이상인 문서만 사용
            filtered_docs = []
            if contexts_list:
                for doc in contexts_list:
                    score = doc.get('score', 0)
                    if score >= 0.7:
                        filtered_docs.append(doc.get('content', ''))
            
            contexts = filtered_docs
            
            # 객관식 여부 판단
            is_mc = any(f"{i}." in question or f"{i} " in question for i in range(1, 10))
            
            # 프롬프트 생성 (10개 문서 모두 포함)
            prompt = prompt_generator.create_improved_prompt(
                question=question,
                contexts=contexts,  # 10개 문서 모두 전달
                is_mc=is_mc
            )
            
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
            else:
                answer = extractor.extract_open_ended_answer(response)
            
            # 답변 검증
            answer = validate_answer_length(answer, is_mc)
            
        except Exception as e:
            answer = "1" if is_mc else "처리 실패"
            failed_count += 1
        
        results.append({
            "ID": question_id,
            "Answer": answer
        })
        
        # 메모리 정리 (매 100개마다)
        if current_num % 100 == 0:
            import torch
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    
    # 4. 결과 저장
    print("결과 저장 중...")
    
    submission_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"submission_8bit_10docs_{timestamp}.csv"
    
    submission_df.to_csv(submission_path, index=False, encoding='utf-8-sig')
    
    # 백업 저장
    backup_path = f"test_results/submission_8bit_10docs_backup_{timestamp}.csv"
    submission_df.to_csv(backup_path, index=False, encoding='utf-8-sig')
    
    # 5. 최종 결과
    print(f"\n완료! 총 처리시간: {total_time/60:.1f}분")
    print(f"처리 실패: {failed_count}개")
    print(f"제출파일: {submission_path}")
    print(f"백업파일: {backup_path}")
    
    # 파일 검증
    verify_df = pd.read_csv(submission_path)
    print(f"검증: {len(verify_df)}행, ID범위: {verify_df['ID'].min()}-{verify_df['ID'].max()}")
    
    # 샘플 출력
    print("\n샘플 답변:")
    for i in range(min(5, len(verify_df))):
        row = verify_df.iloc[i]
        print(f"  {row['ID']}: {row['Answer']}")
    
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()