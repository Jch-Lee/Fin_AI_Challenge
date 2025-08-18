#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HybridRetriever 수정된 초기화 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

from packages.rag.retrieval.hybrid_retriever import HybridRetriever
from packages.rag.retrieval.bm25_retriever import BM25Retriever
from packages.rag.embeddings.kure_embedder import KUREEmbedder
from packages.rag.knowledge_base import KnowledgeBase

def test_hybrid_fixed():
    """수정된 HybridRetriever 테스트"""
    
    print("="*60)
    print("HybridRetriever 수정 테스트")
    print("="*60)
    
    # 1. Embedder 초기화
    print("\n1. Embedder 초기화...")
    try:
        embedder = KUREEmbedder(
            model_name="nlpai-lab/KURE-v1",
            batch_size=32,
            show_progress=False
        )
        print("   [OK] Embedder 초기화 성공")
    except Exception as e:
        print(f"   [ERROR] Embedder 초기화 실패: {e}")
        return
    
    # 2. Knowledge Base 로드
    print("\n2. Knowledge Base 로드...")
    kb_path = "data/rag/knowledge_base_fixed"
    try:
        kb = KnowledgeBase.load(kb_path)
        print(f"   [OK] Knowledge Base 로드 성공: {kb.doc_count} documents")
    except Exception as e:
        print(f"   [ERROR] Knowledge Base 로드 실패: {e}")
        return
    
    # 3. BM25 Retriever 올바른 초기화
    print("\n3. BM25 Retriever 초기화 (수정됨)...")
    bm25_retriever = None
    try:
        # chunks.json에서 문서 로드
        import json
        chunks_path = Path("data/rag/chunks.json")
        if chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # 문서 텍스트와 ID 추출
            documents = []
            doc_ids = []
            metadata_list = []
            
            for i, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    content = chunk.get('content', chunk.get('text', ''))
                    doc_id = chunk.get('id', chunk.get('chunk_id', f'chunk_{i}'))
                    meta = {
                        'chunk_index': chunk.get('chunk_index', i),
                        'doc_id': chunk.get('doc_id', 'unknown'),
                        'source': chunk.get('source', '')
                    }
                    documents.append(content)
                    doc_ids.append(doc_id)
                    metadata_list.append(meta)
                else:
                    documents.append(str(chunk))
                    doc_ids.append(f"doc_{i}")
                    metadata_list.append({})
            
            print(f"   로드된 문서: {len(documents)}개")
            
            # BM25 초기화 (올바른 방법)
            bm25_retriever = BM25Retriever(
                tokenizer_method="simple"  # konlpy 없으면 simple 사용
            )
            
            # 인덱스 구축
            bm25_retriever.build_index(
                documents=documents,
                doc_ids=doc_ids,
                metadata=metadata_list
            )
            print("   [OK] BM25 Retriever 초기화 및 인덱스 구축 성공")
            
        else:
            print("   [WARN] chunks.json 파일 없음")
    except Exception as e:
        print(f"   [ERROR] BM25 Retriever 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. HybridRetriever 초기화
    print("\n4. HybridRetriever 초기화...")
    try:
        hybrid_retriever = HybridRetriever(
            bm25_retriever=bm25_retriever,
            vector_retriever=kb,
            embedder=embedder,
            bm25_weight=0.3,
            vector_weight=0.7,
            normalization_method="min_max"
        )
        print("   [OK] HybridRetriever 초기화 성공")
        
        # 통계 출력
        stats = hybrid_retriever.get_stats()
        print(f"   - BM25 가중치: {stats['weights']['bm25_weight']:.1f}")
        print(f"   - Vector 가중치: {stats['weights']['vector_weight']:.1f}")
        print(f"   - BM25 사용 가능: {stats['components']['bm25_available']}")
        print(f"   - Vector 사용 가능: {stats['components']['vector_available']}")
        
    except Exception as e:
        print(f"   [ERROR] HybridRetriever 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 테스트 검색
    print("\n5. 테스트 검색...")
    test_queries = [
        "금융보안 관리체계",
        "개인정보 보호법",
        "재해복구 계획"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        try:
            # search 메서드 테스트
            results = hybrid_retriever.search(query=query, k=3)
            print(f"   [OK] search() 메서드: {len(results)}개 결과")
            
            if results:
                result = results[0]
                print(f"   첫 번째 결과:")
                print(f"     - Hybrid Score: {result.hybrid_score:.4f}")
                print(f"     - BM25 Score: {result.bm25_score:.4f}")
                print(f"     - Vector Score: {result.vector_score:.4f}")
                print(f"     - Methods: {result.retrieval_methods}")
                print(f"     - Content: {result.content[:100]}...")
                
        except Exception as e:
            print(f"   [ERROR] 검색 실패: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("테스트 완료")
    print("="*60)

if __name__ == "__main__":
    test_hybrid_fixed()