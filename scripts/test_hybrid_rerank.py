#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HybridRetriever와 30→5 리랭킹 파이프라인 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.rag.rag_pipeline import RAGPipeline
from packages.rag.embeddings.kure_embedder import KUREEmbedder

def test_hybrid_reranking():
    """HybridRetriever와 리랭킹 파이프라인 테스트"""
    
    print("="*60)
    print("HybridRetriever & 30→5 리랭킹 파이프라인 테스트")
    print("="*60)
    
    # 1. Embedder 초기화
    print("\n1. KURE Embedder 로딩...")
    embedder = KUREEmbedder(
        model_name="nlpai-lab/KURE-v1",
        batch_size=32,
        show_progress=False
    )
    print("   [OK] Embedder 로드 완료")
    
    # 2. RAG Pipeline 초기화 (HybridRetriever + Reranking)
    print("\n2. RAG Pipeline 초기화...")
    print("   - Retriever: HybridRetriever")
    print("   - Knowledge Base: data/rag/knowledge_base_fixed")
    print("   - Reranking: Enabled (30→5)")
    
    rag_pipeline = RAGPipeline(
        embedder=embedder,
        retriever_type="hybrid",  # HybridRetriever 사용
        knowledge_base_path="data/rag/knowledge_base_fixed",
        enable_reranking=True,
        initial_retrieve_k=30,  # 초기 30개 검색
        final_k=5  # 최종 5개 선택
    )
    print("   [OK] RAG Pipeline 초기화 완료")
    
    # 3. 테스트 질문
    test_questions = [
        "개인정보보호법에서 정의하는 개인정보란 무엇인가?",
        "금융회사의 정보보안 관리체계는 어떻게 구성되어야 하는가?",
        "제로트러스트 아키텍처의 핵심 원칙은 무엇인가?"
    ]
    
    print("\n3. 검색 및 리랭킹 테스트")
    print("-"*60)
    
    for idx, question in enumerate(test_questions, 1):
        print(f"\n테스트 {idx}: {question[:50]}...")
        
        try:
            # Step 1: 초기 검색 (30개, reranking 없이)
            print("\n  [Step 1] 초기 검색 (30개)")
            results_initial = rag_pipeline.retrieve(
                query=question,
                top_k=30,
                use_reranking=False
            )
            print(f"    - 검색된 문서: {len(results_initial)}개")
            
            if len(results_initial) > 0:
                scores = [doc.get('score', 0) for doc in results_initial]
                print(f"    - 점수 범위: {min(scores):.4f} ~ {max(scores):.4f}")
                print(f"    - 평균 점수: {sum(scores)/len(scores):.4f}")
            
            # Step 2: 리랭킹 적용 (30→5)
            print("\n  [Step 2] 리랭킹 적용 (30→5)")
            results_reranked = rag_pipeline.retrieve(
                query=question,
                top_k=5,
                use_reranking=True
            )
            print(f"    - 최종 문서: {len(results_reranked)}개")
            
            if len(results_reranked) > 0:
                print("\n  [결과 비교]")
                print("    초기 Top 3:")
                for i, doc in enumerate(results_initial[:3], 1):
                    content = doc.get('content', '')[:100]
                    print(f"      {i}. Score: {doc.get('score', 0):.4f} | {content}...")
                
                print("\n    리랭킹 후 Top 3:")
                for i, doc in enumerate(results_reranked[:3], 1):
                    content = doc.get('content', '')[:100]
                    rerank_score = doc.get('rerank_score', doc.get('score', 0))
                    print(f"      {i}. Score: {rerank_score:.4f} | {content}...")
                
                # 순서 변경 확인
                initial_ids = [doc.get('id', doc.get('chunk_id', i)) for i, doc in enumerate(results_initial[:5])]
                reranked_ids = [doc.get('id', doc.get('chunk_id', i)) for i, doc in enumerate(results_reranked)]
                
                if initial_ids != reranked_ids[:len(initial_ids)]:
                    print("\n    [OK] 리랭킹으로 문서 순서가 변경되었습니다!")
                else:
                    print("\n    [WARN] 리랭킹 후에도 순서가 동일합니다.")
            
        except Exception as e:
            print(f"    [ERROR] 에러 발생: {e}")
        
        print("-"*60)
    
    # 4. 파이프라인 검증 결과
    print("\n" + "="*60)
    print("파이프라인 검증 결과")
    print("="*60)
    
    if hasattr(rag_pipeline, 'retriever'):
        retriever = rag_pipeline.retriever
        print(f"[OK] Retriever 타입: {type(retriever).__name__}")
        
        if hasattr(retriever, 'base_retriever'):
            print(f"[OK] Base Retriever: {type(retriever.base_retriever).__name__}")
        
        if hasattr(retriever, 'reranker') and retriever.reranker:
            print(f"[OK] Reranker: {type(retriever.reranker).__name__}")
            print(f"[OK] 리랭킹 파이프라인: 30개 초기 검색 → 5개 최종 선택")
        else:
            print("[WARN] Reranker가 활성화되지 않았습니다")
    
    print("\n[DONE] HybridRetriever와 30→5 리랭킹 파이프라인 테스트 완료!")

if __name__ == "__main__":
    test_hybrid_reranking()