"""
Test script for the new RAG architecture
Verifies that the restructured system works correctly
"""
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_new_imports():
    """Test that new import structure works"""
    print("\n=== Testing New Import Structure ===")
    
    try:
        # Test core RAG imports
        from packages.rag import (
            RAGPipeline,
            create_rag_pipeline,
            KnowledgeBase
        )
        print("[OK] Core RAG imports successful")
        
        # Test embeddings imports
        from packages.rag.embeddings import (
            KUREEmbedder,
            E5Embedder,
            BaseEmbedder
        )
        print("[OK] Embeddings imports successful")
        
        # Test retrieval imports
        from packages.rag.retrieval import (
            VectorRetriever,
            BM25Retriever,
            HybridRetriever
        )
        print("[OK] Retrieval imports successful")
        
        return True
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility layer"""
    print("\n=== Testing Backward Compatibility ===")
    
    try:
        # Test old import path (should show deprecation warning)
        from packages.preprocessing.embedder import EmbeddingGenerator, TextEmbedder
        print("[OK] Backward compatibility imports work (check for deprecation warnings above)")
        return True
    except ImportError as e:
        print(f"[ERROR] Backward compatibility broken: {e}")
        return False


def test_pipeline_creation():
    """Test creating a RAG pipeline"""
    print("\n=== Testing Pipeline Creation ===")
    
    try:
        from packages.rag import create_rag_pipeline
        
        # Create pipeline with KURE embedder
        pipeline = create_rag_pipeline(
            embedder_type="kure",
            retriever_type="hybrid"
        )
        
        # Get statistics
        stats = pipeline.get_statistics()
        print(f"[OK] Pipeline created successfully")
        print(f"   - Embedder: {stats['embedder_model']}")
        print(f"   - Embedding dim: {stats['embedding_dim']}")
        print(f"   - Retriever type: {stats['retriever_type']}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Pipeline creation failed: {e}")
        return False


def test_embedder_functionality():
    """Test that embedders work correctly"""
    print("\n=== Testing Embedder Functionality ===")
    
    try:
        from packages.rag.embeddings import KUREEmbedder
        
        # Create embedder
        embedder = KUREEmbedder()
        
        # Test embedding generation
        test_text = "금융보안은 매우 중요합니다."
        embedding = embedder.embed(test_text)
        
        print(f"[OK] Embedder works correctly")
        print(f"   - Input: {test_text}")
        print(f"   - Embedding shape: {embedding.shape}")
        print(f"   - Embedding dim: {embedder.get_embedding_dim()}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Embedder test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Testing New RAG Architecture")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Import Structure", test_new_imports()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("Pipeline Creation", test_pipeline_creation()))
    results.append(("Embedder Functionality", test_embedder_functionality()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("SUCCESS: All tests passed! The new RAG architecture is working correctly.")
    else:
        print("WARNING: Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)