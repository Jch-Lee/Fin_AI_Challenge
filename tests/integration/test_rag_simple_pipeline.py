#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 파이프라인 간단 검증
각 컴포넌트가 정상 작동하는지 빠르게 확인
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """1. Import 테스트"""
    print("1. Import 테스트...")
    try:
        from packages.rag import RAGPipeline
        from packages.rag.embeddings import KUREEmbedder
        from packages.preprocessing.chunker import DocumentChunker
        print("   [O] 모든 모듈 import 성공")
        return True
    except Exception as e:
        print(f"   [X] Import 실패: {e}")
        return False

def test_embedder():
    """2. Embedder 테스트"""
    print("\n2. KURE Embedder 테스트...")
    try:
        from packages.rag.embeddings import KUREEmbedder
        
        embedder = KUREEmbedder()
        test_text = "금융 AI 시스템의 보안"
        embedding = embedder.embed(test_text)
        
        print(f"   [O] 임베딩 생성 성공")
        print(f"      - 모델: {embedder.model_name}")
        print(f"      - 차원: {embedding.shape}")
        return True
    except Exception as e:
        print(f"   [X] Embedder 실패: {e}")
        return False

def test_chunker():
    """3. Chunker 테스트"""
    print("\n3. Document Chunker 테스트...")
    try:
        from packages.preprocessing.chunker import DocumentChunker
        
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        test_text = "테스트 문서입니다. " * 100  # 긴 텍스트 생성
        chunks = chunker.chunk_document(test_text)
        
        print(f"   [O] 청킹 성공")
        print(f"      - 청크 수: {len(chunks)}")
        print(f"      - 평균 길이: {sum(len(c.content) for c in chunks) / len(chunks):.1f}")
        return True
    except Exception as e:
        print(f"   [X] Chunker 실패: {e}")
        return False

def test_rag_pipeline():
    """4. RAG Pipeline 통합 테스트"""
    print("\n4. RAG Pipeline 테스트...")
    try:
        from packages.rag import RAGPipeline
        
        # 파이프라인 생성
        pipeline = RAGPipeline(
            embedder=None,  # 기본 KURE 사용
            retriever_type="vector",  # 간단한 벡터 검색
            knowledge_base_path=None
        )
        
        # 문서 추가
        test_docs = [
            "금융 AI 시스템의 보안은 매우 중요합니다.",
            "AI 모델의 적대적 공격을 방지해야 합니다.",
            "데이터 프라이버시를 보호해야 합니다."
        ]
        
        num_added = pipeline.add_documents(test_docs)
        print(f"   [O] 문서 추가 성공: {num_added}개")
        
        # 검색 테스트
        query = "AI 보안"
        results = pipeline.retrieve(query, top_k=2)
        print(f"   [O] 검색 성공: {len(results)}개 결과")
        
        # 컨텍스트 생성
        context = pipeline.generate_context(query, top_k=2)
        print(f"   [O] 컨텍스트 생성 성공: {len(context)}자")
        
        # 통계
        stats = pipeline.get_statistics()
        print(f"   [O] 통계 수집 성공")
        print(f"      - 임베딩 모델: {stats['embedder_model']}")
        print(f"      - 문서 수: {stats['num_documents']}")
        
        return True
    except Exception as e:
        print(f"   [X] Pipeline 실패: {e}")
        return False

def test_hybrid_retriever():
    """5. Hybrid Retriever 테스트 (선택적)"""
    print("\n5. Hybrid Retriever 테스트...")
    try:
        from packages.rag.retrieval import HybridRetriever
        from packages.rag import RAGPipeline
        
        # 하이브리드 파이프라인
        pipeline = RAGPipeline(retriever_type="hybrid")
        
        # 문서 추가
        test_docs = ["테스트 문서 1", "테스트 문서 2"]
        pipeline.add_documents(test_docs)
        
        # 검색
        results = pipeline.retrieve("테스트", top_k=1)
        
        print(f"   [O] Hybrid Retriever 성공")
        return True
    except Exception as e:
        print(f"   [!] Hybrid Retriever 미지원 또는 오류: {e}")
        return True  # 선택적 기능이므로 실패해도 OK

def main():
    """메인 테스트 실행"""
    print("="*60)
    print(" RAG 파이프라인 간단 검증")
    print("="*60)
    
    tests = [
        ("Import", test_imports),
        ("Embedder", test_embedder),
        ("Chunker", test_chunker),
        ("Pipeline", test_rag_pipeline),
        ("Hybrid", test_hybrid_retriever)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n테스트 {name} 실행 중 오류: {e}")
            results[name] = False
    
    # 결과 요약
    print("\n" + "="*60)
    print(" 테스트 결과 요약")
    print("="*60)
    
    for name, success in results.items():
        status = "[O]" if success else "[X]"
        print(f" {status} {name}")
    
    # 전체 성공 여부
    core_tests = ["Import", "Embedder", "Chunker", "Pipeline"]
    core_success = all(results.get(name, False) for name in core_tests)
    
    if core_success:
        print("\n[SUCCESS] 핵심 RAG 파이프라인이 정상 작동합니다!")
    else:
        print("\n[WARNING] 일부 컴포넌트에 문제가 있습니다.")
    
    return core_success

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # 로그 레벨 조정
    
    success = main()
    sys.exit(0 if success else 1)