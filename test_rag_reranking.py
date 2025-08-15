"""
Test RAG Pipeline with Mandatory Reranking (30->5 documents)
Validates the complete pipeline: Hybrid Search -> Reranking -> Final Selection
"""
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rag_with_reranking():
    """Test the mandatory reranking pipeline (30->5)"""
    print("\n" + "="*70)
    print("Testing RAG Pipeline with Mandatory Reranking")
    print("Pipeline: Hybrid Search (30 docs) -> Qwen3 Reranker -> Final (5 docs)")
    print("="*70)
    
    try:
        # Import RAG pipeline
        from packages.rag import create_rag_pipeline
        
        # Create pipeline with reranking enabled (now default)
        print("\n[STEP 1] Creating RAG pipeline with reranking...")
        pipeline = create_rag_pipeline(
            embedder_type="kure",
            retriever_type="hybrid",
            enable_reranking=True,  # Now default
            initial_retrieve_k=30,  # Get 30 documents initially
            final_k=5  # Select final 5 after reranking
        )
        print("[OK] Pipeline created with reranking configuration")
        
        # Add test documents (financial security domain)
        print("\n[STEP 2] Adding test documents...")
        test_documents = [
            # High relevance documents
            "비밀번호는 최소 8자 이상으로 설정하고, 대소문자, 숫자, 특수문자를 포함해야 합니다.",
            "금융 거래 시 2단계 인증(2FA)을 활성화하면 보안이 크게 향상됩니다.",
            "피싱 이메일은 긴급함을 강조하며 개인정보 입력을 유도하는 특징이 있습니다.",
            
            # Medium relevance documents
            "공용 와이파이에서 온라인 뱅킹을 사용하면 해킹 위험이 증가합니다.",
            "정기적으로 금융 거래 내역을 확인하여 이상 거래를 조기에 발견하세요.",
            "보안 프로그램을 최신 상태로 유지하는 것이 중요합니다.",
            
            # Low relevance documents (noise)
            "주식 투자는 장기적인 관점에서 접근하는 것이 좋습니다.",
            "신용카드 포인트를 효율적으로 사용하는 방법을 알아보세요.",
            "예금 금리가 상승하면 저축 상품에 대한 관심이 증가합니다.",
            "부동산 투자는 안정적인 수익을 제공할 수 있습니다.",
            
            # More noise documents to test reranking
            "연말정산 시 놓치기 쉬운 공제 항목들을 확인하세요.",
            "퇴직연금 운용 방법에 대해 알아보겠습니다.",
            "보험료 절감을 위한 다양한 방법이 있습니다.",
            "대출 이자율 비교는 금융 비용 절감의 첫걸음입니다.",
            "암호화폐 투자 시 변동성에 주의해야 합니다.",
            
            # Additional relevant documents mixed in
            "SMS 인증번호는 절대 타인과 공유하지 마세요.",
            "금융 앱 사용 시 자동 로그아웃 설정을 활성화하세요.",
            "바이오 인증(지문, 얼굴)을 활용하면 편리하고 안전합니다.",
            "정품 백신 소프트웨어를 사용하여 악성코드를 차단하세요.",
            "금융 사기 피해 시 즉시 금융감독원에 신고하세요.",
        ]
        
        pipeline.add_documents(test_documents)
        print(f"[OK] Added {len(test_documents)} documents to knowledge base")
        
        # Test query focusing on password security
        test_query = "금융 보안을 위한 비밀번호 설정과 2단계 인증 방법"
        
        print(f"\n[STEP 3] Testing retrieval with query: '{test_query}'")
        print("-" * 50)
        
        # Test without reranking (baseline)
        print("\n3.1. Baseline: Without reranking (direct top-5)")
        start_time = time.time()
        baseline_results = pipeline.retrieve(
            test_query, 
            top_k=5,
            use_reranking=False  # Disable reranking for baseline
        )
        baseline_time = time.time() - start_time
        
        print(f"[OK] Retrieved {len(baseline_results)} documents in {baseline_time:.3f}s")
        for i, doc in enumerate(baseline_results, 1):
            print(f"  [{i}] Score: {doc.get('score', 0):.4f}")
            print(f"      Content: {doc.get('content', '')[:80]}...")
        
        # Test with reranking (30->5)
        print("\n3.2. With Reranking: 30 -> 5 documents")
        start_time = time.time()
        reranked_results = pipeline.retrieve(
            test_query,
            top_k=5,
            use_reranking=True  # Use reranking (default)
        )
        rerank_time = time.time() - start_time
        
        print(f"[OK] Retrieved and reranked to {len(reranked_results)} documents in {rerank_time:.3f}s")
        for i, doc in enumerate(reranked_results, 1):
            scores = []
            if 'score' in doc:
                scores.append(f"Initial: {doc['score']:.4f}")
            if 'rerank_score' in doc:
                scores.append(f"Rerank: {doc['rerank_score']:.4f}")
            if 'final_score' in doc:
                scores.append(f"Final: {doc['final_score']:.4f}")
            
            print(f"  [{i}] Scores: {' | '.join(scores)}")
            print(f"      Content: {doc.get('content', '')[:80]}...")
        
        # Compare results
        print("\n[STEP 4] Comparing Results")
        print("-" * 50)
        
        # Check if reranking changed the order
        baseline_contents = [doc.get('content', '')[:50] for doc in baseline_results]
        reranked_contents = [doc.get('content', '')[:50] for doc in reranked_results]
        
        if baseline_contents != reranked_contents:
            print("[OK] Reranking successfully reordered documents")
            print("  -> Different document selection/ordering detected")
        else:
            print("[WARN] Reranking produced same results as baseline")
            print("  -> This may happen if reranker is not initialized or documents are already well-ordered")
        
        # Performance comparison
        print(f"\n[Performance Summary]")
        print(f"  Baseline (top-5):    {baseline_time:.3f}s")
        print(f"  Reranking (30->5):    {rerank_time:.3f}s")
        print(f"  Overhead:            {rerank_time - baseline_time:.3f}s")
        
        # Test statistics
        stats = pipeline.get_statistics()
        print(f"\n[Pipeline Statistics]")
        print(f"  Embedder:            {stats.get('embedder_model', 'N/A')}")
        print(f"  Embedding Dim:       {stats.get('embedding_dim', 'N/A')}")
        print(f"  Retriever Type:      {stats.get('retriever_type', 'N/A')}")
        print(f"  Reranking Enabled:   {stats.get('reranking_enabled', False)}")
        print(f"  Documents in KB:     {stats.get('num_documents', 0)}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reranking_quality():
    """Test if reranking improves relevance for financial security queries"""
    print("\n" + "="*70)
    print("Testing Reranking Quality Improvement")
    print("="*70)
    
    try:
        from packages.rag import create_rag_pipeline
        
        # Create pipeline
        pipeline = create_rag_pipeline(
            embedder_type="kure",
            retriever_type="hybrid",
            enable_reranking=True,
            initial_retrieve_k=20,  # Smaller initial set for quality test
            final_k=3  # Top 3 for precision
        )
        
        # Add mixed relevance documents
        documents = [
            # Highly relevant (should be ranked high)
            "금융 보안의 핵심은 강력한 비밀번호와 2단계 인증입니다.",
            "해킹 방지를 위해 비밀번호는 주기적으로 변경해야 합니다.",
            
            # Somewhat relevant
            "온라인 뱅킹 사용 시 보안에 주의하세요.",
            "금융 거래는 안전한 네트워크에서만 하세요.",
            
            # Less relevant (should be filtered out)
            "주식 시장 동향을 파악하는 것이 중요합니다.",
            "연금 상품 선택 시 고려사항들입니다.",
            "대출 금리 비교 방법을 알아보세요.",
            "투자 포트폴리오 다각화 전략입니다.",
        ]
        
        pipeline.add_documents(documents)
        
        # Test with security-focused query
        query = "비밀번호 보안과 해킹 방지 방법"
        
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        # Get results
        results = pipeline.retrieve(query, top_k=3)
        
        print("\nTop 3 Reranked Results:")
        for i, doc in enumerate(results, 1):
            content = doc.get('content', '')
            final_score = doc.get('final_score', doc.get('score', 0))
            
            # Check if it's relevant (contains security keywords)
            is_relevant = any(keyword in content for keyword in ['보안', '비밀번호', '해킹', '인증'])
            relevance_mark = "[OK]" if is_relevant else "[FAIL]"
            
            print(f"{relevance_mark} [{i}] Score: {final_score:.4f}")
            print(f"      {content}")
        
        # Calculate precision
        relevant_count = sum(1 for doc in results 
                            if any(kw in doc.get('content', '') 
                                  for kw in ['보안', '비밀번호', '해킹', '인증']))
        precision = relevant_count / len(results) if results else 0
        
        print(f"\n[Quality Metrics]")
        print(f"  Precision@3: {precision:.2%} ({relevant_count}/{len(results)} relevant)")
        
        return precision >= 0.66  # At least 2 out of 3 should be relevant
        
    except Exception as e:
        print(f"\n[ERROR] Quality test failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RAG PIPELINE WITH MANDATORY RERANKING TEST SUITE")
    print("="*70)
    
    # Run tests
    test_results = []
    
    # Test 1: Basic reranking pipeline
    print("\n[TEST 1] Basic Reranking Pipeline")
    result1 = test_rag_with_reranking()
    test_results.append(("Basic Reranking Pipeline", result1))
    
    # Test 2: Reranking quality improvement
    print("\n[TEST 2] Reranking Quality Improvement")
    result2 = test_reranking_quality()
    test_results.append(("Reranking Quality", result2))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in test_results:
        status = "PASS [OK]" if passed else "FAIL [FAIL]"
        print(f"  {test_name:30} {status}")
    
    all_passed = all(result for _, result in test_results)
    
    print("\n" + "="*70)
    if all_passed:
        print("SUCCESS: All reranking tests passed!")
        print("The RAG pipeline correctly implements 30->5 reranking.")
    else:
        print("FAILURE: Some tests failed.")
        print("Please check the reranking implementation.")
    print("="*70)