#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 시스템 마이그레이션 테스트
BM25 0.7 파이프라인이 메인으로 잘 통합되었는지 확인
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

def test_migration():
    """마이그레이션 테스트"""
    
    print("="*60)
    print("RAG System Migration Test")
    print("="*60)
    
    # 1. 기존 방식 (RAGSystemV2)
    print("\n1. Testing RAGSystemV2 (old way)...")
    try:
        from scripts.load_rag_v2 import RAGSystemV2
        rag_v2 = RAGSystemV2()
        rag_v2.load_all()
        stats = rag_v2.get_statistics()
        print(f"   OK: RAGSystemV2 loaded: {stats.get('total_chunks', 0)} chunks")
        
        retriever_v2 = rag_v2.create_hybrid_retriever()
        print(f"   OK: Hybrid retriever created")
    except Exception as e:
        print(f"   X Failed: {e}")
    
    # 2. 새로운 통합 방식 (packages.rag)
    print("\n2. Testing integrated RAGPipeline (new way)...")
    try:
        from packages.rag import create_rag_pipeline, load_rag_v2_pipeline
        
        # 기본 파이프라인 생성 (설정에서 읽기)
        pipeline = create_rag_pipeline()
        print(f"   OK: Pipeline created with reranking={pipeline.enable_reranking}")
        
        # V2 데이터 로드
        pipeline_v2 = load_rag_v2_pipeline()
        stats = pipeline_v2.get_statistics()
        print(f"   OK: V2 data loaded: {stats.get('num_documents', 0)} documents")
        print(f"   OK: Reranking: {stats.get('reranking_enabled', False)}")
    except Exception as e:
        print(f"   X Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 설정 파일 확인
    print("\n3. Checking configuration...")
    try:
        import yaml
        with open("configs/rag_config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        search_config = config.get('search', {})
        print(f"   OK: BM25 weight: {search_config.get('bm25_weight', 'Not set')}")
        print(f"   OK: Vector weight: {search_config.get('vector_weight', 'Not set')}")
        print(f"   OK: Reranking enabled: {search_config.get('enable_reranking', 'Not set')}")
    except Exception as e:
        print(f"   ✗ Failed to read config: {e}")
    
    # 4. 호환성 테스트
    print("\n4. Testing backward compatibility...")
    try:
        # 기존 코드가 여전히 작동하는지 확인
        from packages.rag import RAGPipeline
        
        # 명시적으로 reranking 비활성화
        pipeline_no_rerank = RAGPipeline(enable_reranking=False)
        print(f"   OK: Pipeline with explicit disable: reranking={pipeline_no_rerank.enable_reranking}")
        
        # 명시적으로 reranking 활성화
        pipeline_with_rerank = RAGPipeline(enable_reranking=True)
        print(f"   OK: Pipeline with explicit enable: reranking={pipeline_with_rerank.enable_reranking}")
        
        # 설정에서 읽기 (None)
        pipeline_from_config = RAGPipeline(enable_reranking=None)
        print(f"   OK: Pipeline from config: reranking={pipeline_from_config.enable_reranking}")
    except Exception as e:
        print(f"   X Failed: {e}")
    
    print("\n" + "="*60)
    print("Migration test complete!")
    print("="*60)

if __name__ == "__main__":
    test_migration()