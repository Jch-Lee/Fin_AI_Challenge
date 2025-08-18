#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
개선된 프롬프트를 적용한 하이브리드 RAG 파이프라인 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from scripts.simple_improved_prompt import SimpleImprovedPrompt
from scripts.improved_answer_extraction import ImprovedAnswerExtractor

# 로깅 설정
logging.basicConfig(level=logging.INFO)

from packages.rag.rag_pipeline import RAGPipeline
from packages.rag.embeddings.kure_embedder import KUREEmbedder
from packages.llm.qwen_quantized import QuantizedQwenLLM

def test_improved_pipeline():
    """개선된 프롬프트를 사용한 파이프라인 테스트"""
    
    print("="*60)
    print("개선된 프롬프트 하이브리드 RAG 파이프라인 테스트")
    print("="*60)
    
    # 1. test.csv의 처음 3개 질문 로드
    test_df = pd.read_csv("data/competition/test.csv")
    first_three = test_df.head(3)
    
    print("\n[테스트 질문]")
    for idx, row in first_three.iterrows():
        print(f"{row['ID']}: {row['Question'][:80]}...")
    
    # 2. 컴포넌트 초기화
    print("\n" + "="*60)
    print("파이프라인 초기화...")
    print("="*60)
    
    # Embedder
    print("\n1. KURE Embedder 초기화...")
    embedder = KUREEmbedder(
        model_name="nlpai-lab/KURE-v1",
        batch_size=32,
        show_progress=False
    )
    print("   [OK] Embedder 로드 완료")
    
    # RAG Pipeline (하이브리드)
    print("\n2. 하이브리드 RAG Pipeline 초기화...")
    rag_pipeline = RAGPipeline(
        embedder=embedder,
        retriever_type="hybrid",
        knowledge_base_path="data/rag/knowledge_base_fixed",
        enable_reranking=False,
        initial_retrieve_k=10,
        final_k=5
    )
    print("   [OK] 하이브리드 RAG Pipeline 초기화 완료")
    
    # LLM
    print("\n3. Qwen2.5-7B-Instruct 초기화...")
    llm = QuantizedQwenLLM(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        quantization_type="4bit",
        cache_dir="./models"
    )
    print("   [OK] LLM 로드 완료")
    print(f"   메모리 사용량: {llm.get_memory_footprint()}")
    
    # 3. 개선된 프롬프트로 예측 실행
    print("\n" + "="*60)
    print("개선된 프롬프트로 예측...")
    print("="*60)
    
    results = []
    prompt_generator = SimpleImprovedPrompt()
    
    for idx, row in first_three.iterrows():
        question_id = row['ID']
        question = row['Question']
        
        print(f"\n처리 중: {question_id}")
        print("-" * 40)
        
        # RAG 검색
        print("1. 하이브리드 RAG 검색 중...")
        try:
            contexts_list = rag_pipeline.retrieve(
                query=question,
                top_k=5,
                use_reranking=False
            )
            print(f"   - 검색된 문서: {len(contexts_list)}개")
            
            if contexts_list:
                contexts = [doc.get('content', '') for doc in contexts_list]
                print(f"   - 첫 번째 문서 점수: {contexts_list[0].get('score', 0):.4f}")
            else:
                contexts = []
                print("   - 검색 결과 없음")
                
        except Exception as e:
            print(f"   [ERROR] RAG 검색 실패: {e}")
            contexts = []
        
        # 객관식 여부 판단
        is_mc = any(f"{i}." in question or f"{i} " in question for i in range(1, 10))
        
        # 개선된 프롬프트 생성
        print("2. 개선된 프롬프트 생성...")
        
        # Chain-of-Thought 버전 사용
        prompt = prompt_generator.create_improved_prompt(
            question=question,
            contexts=contexts,
            is_mc=is_mc
        )
        
        print(f"   - 프롬프트 타입: {'객관식' if is_mc else '주관식'}")
        print(f"   - CoT 추론 단계 포함: Yes")
        print(f"   - 참고문서 + 전문지식 균형: Yes")
        
        # LLM 생성
        print("3. LLM 답변 생성 중...")
        try:
            response = llm.generate_optimized(
                prompt=prompt,
                max_new_tokens=128 if is_mc else 256,  # CoT를 위해 좀 더 길게
                temperature=0.1,
                use_cache=False
            )
            
            # 개선된 답변 추출 로직 사용
            extractor = ImprovedAnswerExtractor()
            
            if is_mc:
                # 객관식: 선택지 범위 확인 후 답변 추출
                max_choice = extractor.extract_choice_range(question)
                answer = extractor.extract_answer_from_response(
                    response=response,
                    max_choice=max_choice,
                    question=question
                )
                print(f"   - 선택지 범위: 1-{max_choice}")
                print(f"   - CoT 추론: {response[:100]}...")
                print(f"   - 추출된 답: {answer}")
            else:
                # 주관식: 정제된 답변 추출
                answer = extractor.extract_open_ended_answer(response)
                print(f"   - 답변 길이: {len(answer)}자")
            
        except Exception as e:
            print(f"   [ERROR] LLM 생성 실패: {e}")
            answer = "1" if is_mc else "생성 실패"
        
        results.append({
            "ID": question_id,
            "Answer": answer,
            "Type": "MC" if is_mc else "OE"
        })
        
        print(f"   - 최종 답변: {answer[:100]}...")
    
    # 4. 결과 비교 (이전 방식 vs 개선된 방식)
    print("\n" + "="*60)
    print("결과 요약")
    print("="*60)
    
    print("\n[개선된 프롬프트 특징]")
    print("✅ 참고 문서와 전문지식의 균형적 활용")
    print("✅ Chain-of-Thought 추론 과정 포함")
    print("✅ 한국 금융 규제 및 보안 표준 고려 명시")
    print("✅ 참고자료 부족 시 전문지식으로 보완")
    
    print("\n[예측 결과]")
    for result in results:
        print(f"{result['ID']} ({result['Type']}): {result['Answer'][:50]}...")
    
    # 5. 결과 저장
    result_dir = Path("test_results")
    result_dir.mkdir(exist_ok=True)
    
    result_df = pd.DataFrame([{"ID": r["ID"], "Answer": r["Answer"]} for r in results])
    result_path = result_dir / "improved_prompt_results.csv"
    result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    
    print(f"\n결과 저장: {result_path}")
    
    print("\n" + "="*60)
    print("✅ 개선된 프롬프트 테스트 완료!")
    print("="*60)

if __name__ == "__main__":
    test_improved_pipeline()