#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
하이브리드 RAG 파이프라인 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

from packages.rag.rag_pipeline import RAGPipeline
from packages.rag.embeddings.kure_embedder import KUREEmbedder
from packages.llm.qwen_quantized import QuantizedQwenLLM

def test_hybrid_pipeline():
    """하이브리드 RAG 파이프라인 테스트"""
    
    print("="*60)
    print("하이브리드 RAG 파이프라인 테스트")
    print("="*60)
    
    # 1. test.csv의 처음 3개 질문 로드
    test_df = pd.read_csv("data/competition/test.csv")
    first_three = test_df.head(3)
    
    print("\n[테스트 질문]")
    for idx, row in first_three.iterrows():
        print(f"{row['ID']}: {row['Question'][:80]}...")
    
    # 2. RAG Pipeline 초기화 (하이브리드 모드)
    print("\n" + "="*60)
    print("하이브리드 RAG 파이프라인 초기화...")
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
        retriever_type="hybrid",  # 하이브리드 모드
        knowledge_base_path="data/rag/knowledge_base_fixed",
        enable_reranking=False,  # 일단 리랭킹 비활성화
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
    
    # 3. 예측 실행
    print("\n" + "="*60)
    print("하이브리드 검색 및 예측...")
    print("="*60)
    
    results = []
    
    for idx, row in first_three.iterrows():
        question_id = row['ID']
        question = row['Question']
        
        print(f"\n처리 중: {question_id}")
        print("-" * 40)
        
        # RAG 검색 (하이브리드)
        print("1. 하이브리드 RAG 검색 중...")
        try:
            contexts_list = rag_pipeline.retrieve(
                query=question,
                top_k=5,
                use_reranking=False
            )
            print(f"   - 검색된 문서: {len(contexts_list)}개")
            
            if contexts_list:
                # 첫 번째 문서 정보
                first_doc = contexts_list[0]
                print(f"   - 첫 번째 문서 점수: {first_doc.get('score', 0):.4f}")
                
                # 하이브리드 점수 확인 (HybridRetriever가 반환하는 경우)
                if 'bm25_score' in first_doc:
                    print(f"     * BM25 점수: {first_doc.get('bm25_score', 0):.4f}")
                if 'vector_score' in first_doc:
                    print(f"     * Vector 점수: {first_doc.get('vector_score', 0):.4f}")
                
                print(f"   - 첫 번째 문서 내용: {first_doc.get('content', '')[:100]}...")
                
                contexts = [doc.get('content', '') for doc in contexts_list]
            else:
                contexts = []
                print("   [WARN] 검색 결과 없음")
                
        except Exception as e:
            print(f"   [ERROR] RAG 검색 실패: {e}")
            contexts = []
        
        # 객관식 여부 판단
        is_mc = any(f"{i}." in question or f"{i} " in question for i in range(1, 10))
        
        # 프롬프트 생성
        if is_mc:
            if contexts:
                context_text = "\n\n".join(contexts[:3])
                prompt = f"""당신은 금융보안 전문가입니다.

[참고 문서]
{context_text}

[질문]
{question}

위 객관식 문제의 정답 번호만 출력하세요. 숫자만 답하세요.

답변:"""
            else:
                prompt = f"""당신은 금융보안 전문가입니다.

[질문]
{question}

위 객관식 문제의 정답 번호만 출력하세요. 숫자만 답하세요.

답변:"""
        else:
            # 주관식
            if contexts:
                context_text = "\n\n".join(contexts[:3])
                prompt = f"""당신은 금융보안 전문가입니다.

[참고 문서]
{context_text}

[질문]
{question}

간결하고 정확하게 답변하세요.

답변:"""
            else:
                prompt = f"""당신은 금융보안 전문가입니다.

[질문]
{question}

간결하고 정확하게 답변하세요.

답변:"""
        
        # LLM 생성
        print("2. LLM 답변 생성 중...")
        try:
            response = llm.generate_optimized(
                prompt=prompt,
                max_new_tokens=64 if is_mc else 256,
                temperature=0.1,
                use_cache=False
            )
            print(f"   - 생성된 답변: {response[:100]}")
            
            # 답변 추출
            if is_mc:
                import re
                match = re.search(r'[1-9]', response)
                if match:
                    answer = match.group(0)
                else:
                    answer = "1"
            else:
                answer = response.strip()
            
        except Exception as e:
            print(f"   [ERROR] LLM 생성 실패: {e}")
            answer = "1" if is_mc else "생성 실패"
        
        results.append({
            "ID": question_id,
            "Answer": answer,
            "Type": "MC" if is_mc else "OE"
        })
        
        print(f"   - 최종 답변: {answer}")
    
    # 4. 결과 출력
    print("\n" + "="*60)
    print("하이브리드 RAG 결과")
    print("="*60)
    
    for result in results:
        print(f"\n{result['ID']} ({result['Type']}): {result['Answer']}")
    
    print("\n" + "="*60)
    print("✅ 하이브리드 RAG 파이프라인 테스트 완료!")
    print("="*60)

if __name__ == "__main__":
    test_hybrid_pipeline()