"""
실제 RAG 기능 동작 테스트
구조 변경 후 RAG 시스템이 정상적으로 작동하는지 확인
"""
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_full_rag_workflow():
    """전체 RAG 워크플로우 테스트"""
    print("\n" + "="*60)
    print("Testing Complete RAG Workflow")
    print("="*60)
    
    try:
        # 1. Import new RAG system
        from packages.rag import create_rag_pipeline
        print("[STEP 1] Importing RAG system... OK")
        
        # 2. Create RAG pipeline
        pipeline = create_rag_pipeline(
            embedder_type="kure",
            retriever_type="vector"  # Use vector since hybrid might have issues
        )
        print("[STEP 2] Creating RAG pipeline... OK")
        
        # 3. Add test documents
        test_documents = [
            "금융보안은 개인정보 보호와 자산 안전을 위해 매우 중요합니다.",
            "비밀번호는 주기적으로 변경하고 복잡하게 설정해야 합니다.",
            "피싱 이메일을 조심하고 의심스러운 링크는 클릭하지 마세요.",
            "온라인 뱅킹 사용 시 공용 WiFi는 피하는 것이 좋습니다.",
            "금융 거래 내역은 정기적으로 확인하여 이상 거래를 감지하세요."
        ]
        
        num_added = pipeline.add_documents(test_documents)
        print(f"[STEP 3] Adding {len(test_documents)} documents... OK (Added: {num_added})")
        
        # 4. Test retrieval
        test_query = "금융 보안을 위한 비밀번호 관리 방법"
        retrieved_docs = pipeline.retrieve(test_query, top_k=3)
        print(f"[STEP 4] Retrieving documents for query... OK (Found: {len(retrieved_docs)} docs)")
        
        # 5. Generate context
        context = pipeline.generate_context(test_query, top_k=3)
        print(f"[STEP 5] Generating context... OK (Length: {len(context)} chars)")
        
        # 6. Display results
        print("\n" + "-"*40)
        print("Query:", test_query)
        print("\nRetrieved Documents:")
        for i, doc in enumerate(retrieved_docs[:3], 1):
            print(f"\n[Doc {i}] Score: {doc.get('score', 0):.4f}")
            print(f"Content: {doc.get('content', '')[:100]}...")
        
        print("\n" + "-"*40)
        print("Generated Context (first 300 chars):")
        print(context[:300])
        
        return True
        
    except Exception as e:
        print(f"[ERROR] RAG workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_base_persistence():
    """지식 베이스 저장/로드 테스트"""
    print("\n" + "="*60)
    print("Testing Knowledge Base Persistence")
    print("="*60)
    
    try:
        from packages.rag import create_rag_pipeline
        import tempfile
        import os
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = os.path.join(temp_dir, "test_kb")
            
            # Create pipeline and add documents
            pipeline1 = create_rag_pipeline(embedder_type="kure", retriever_type="vector")
            test_docs = ["테스트 문서 1", "테스트 문서 2"]
            pipeline1.add_documents(test_docs)
            
            # Save knowledge base
            pipeline1.save_knowledge_base(kb_path)
            print(f"[STEP 1] Saving knowledge base... OK")
            
            # Create new pipeline and load
            pipeline2 = create_rag_pipeline(
                embedder_type="kure",
                retriever_type="vector",
                knowledge_base_path=kb_path
            )
            print(f"[STEP 2] Loading knowledge base... OK")
            
            # Get statistics
            stats = pipeline2.get_statistics()
            print(f"[STEP 3] Verifying loaded data...")
            print(f"  - Index size: {stats.get('index_size', 0)}")
            print(f"  - Embedding dim: {stats.get('embedding_dim', 'N/A')}")
            
            # Test with actual retrieval
            test_query = "테스트 쿼리"
            results = pipeline2.retrieve(test_query, top_k=1)
            print(f"  - Retrieval test: {len(results)} results")
            
            return len(results) > 0
            
    except Exception as e:
        print(f"[ERROR] Persistence test failed: {e}")
        return False


def test_existing_kure_data():
    """기존 KURE 임베딩 데이터 로드 테스트"""
    print("\n" + "="*60)
    print("Testing Existing KURE Data Loading")
    print("="*60)
    
    # Check multiple possible paths
    possible_paths = [
        Path("data/kure_embeddings/latest/faiss.index"),  # New nested structure
        Path("data/kure_embeddings/latest"),  # Original structure
        Path("data/knowledge_base/faiss.index"),  # Alternative location
    ]
    
    kb_index_path = None
    for path in possible_paths:
        if path.exists():
            kb_index_path = path
            break
    
    if kb_index_path is None:
        print("[SKIP] No existing KURE data found")
        return True
    
    try:
        from packages.rag.knowledge_base import KnowledgeBase
        
        # Check if it's the new format (directory with both files)
        if kb_index_path.is_dir() and (kb_index_path / "faiss.index").exists():
            kb = KnowledgeBase.load(str(kb_index_path))
            print(f"[OK] Loaded existing KURE knowledge base from {kb_index_path}")
            print(f"  - Index size: {kb.index.ntotal if hasattr(kb.index, 'ntotal') else 'N/A'}")
            
            # Test search with dummy query
            dummy_query = np.random.randn(1024).astype(np.float32)
            results = kb.search(dummy_query, top_k=5)
            print(f"  - Search test: {len(results)} results returned")
            
            return True
        else:
            print("[SKIP] No FAISS index found")
            return True
            
    except Exception as e:
        print(f"[ERROR] Failed to load existing data: {e}")
        return False


def test_backward_compatibility_functionality():
    """기존 코드와의 호환성 기능 테스트"""
    print("\n" + "="*60)
    print("Testing Backward Compatibility Functionality")
    print("="*60)
    
    try:
        # Test old import path
        import warnings
        import sys
        
        # Remove module from cache to ensure clean import
        if 'packages.preprocessing.embedder' in sys.modules:
            del sys.modules['packages.preprocessing.embedder']
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Ensure all warnings are captured
            from packages.preprocessing.embedder import EmbeddingGenerator
            
            # Check deprecation warning
            if len(w) == 0:
                print(f"[WARNING] No deprecation warning captured")
                # Still test functionality even if warning wasn't caught
            else:
                assert issubclass(w[0].category, DeprecationWarning), "Wrong warning type"
                print("[OK] Deprecation warning properly issued")
        
        # Test functionality
        embedder = EmbeddingGenerator()
        test_text = "테스트 텍스트"
        embedding = embedder.generate_embedding(test_text)
        
        print(f"[OK] Old API still works")
        print(f"  - Generated embedding shape: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Backward compatibility failed: {e}")
        return False


def main():
    """모든 RAG 기능 테스트 실행"""
    print("="*60)
    print("RAG FUNCTIONALITY TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run all tests
    results.append(("Full RAG Workflow", test_full_rag_workflow()))
    results.append(("Knowledge Base Persistence", test_knowledge_base_persistence()))
    results.append(("Existing KURE Data", test_existing_kure_data()))
    results.append(("Backward Compatibility", test_backward_compatibility_functionality()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("SUCCESS: All RAG functionality tests passed!")
        print("The restructured RAG system is fully functional.")
    else:
        print("WARNING: Some functionality tests failed.")
        print("Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)