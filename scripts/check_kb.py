#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Base 상태 확인
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.rag.knowledge_base import KnowledgeBase
import numpy as np

def check_knowledge_base():
    """Knowledge Base 상태 확인"""
    
    print("="*60)
    print("Knowledge Base 상태 확인")
    print("="*60)
    
    # 1. knowledge_base_fixed 확인
    kb_path = "data/rag/knowledge_base_fixed"
    print(f"\n1. KB 경로: {kb_path}")
    
    if not Path(kb_path).exists():
        print("   [ERROR] Knowledge Base 디렉토리가 없습니다!")
        return
    
    # 2. KB 로드
    print("\n2. Knowledge Base 로드 중...")
    kb = KnowledgeBase.load(kb_path)
    
    # 3. 통계 출력
    print("\n3. Knowledge Base 통계:")
    print(f"   - 문서 수: {kb.doc_count}")
    print(f"   - 실제 documents 리스트 크기: {len(kb.documents)}")
    print(f"   - 실제 metadata 리스트 크기: {len(kb.metadata)}")
    print(f"   - FAISS 인덱스 크기: {kb.index.ntotal if kb.index else 0}")
    print(f"   - 임베딩 차원: {kb.embedding_dim}")
    print(f"   - 인덱스 타입: {kb.index_type}")
    
    # 4. 샘플 문서 확인
    if kb.documents:
        print("\n4. 샘플 문서 (처음 3개):")
        for i in range(min(3, len(kb.documents))):
            doc = kb.documents[i]
            if isinstance(doc, str):
                print(f"   [{i}] {doc[:100]}...")
            else:
                print(f"   [{i}] {str(doc)[:100]}...")
    else:
        print("\n4. [ERROR] 문서가 비어있습니다!")
    
    # 5. 테스트 검색
    if kb.index and kb.index.ntotal > 0:
        print("\n5. 테스트 검색:")
        # 랜덤 임베딩으로 테스트
        test_embedding = np.random.randn(kb.embedding_dim).astype('float32')
        results = kb.search(test_embedding, k=3)
        
        if results:
            print(f"   - 검색 결과: {len(results)}개")
            for i, result in enumerate(results[:2], 1):
                if isinstance(result, dict):
                    print(f"   [{i}] Score: {result.get('score', 0):.4f}")
                    print(f"       Content: {result.get('content', '')[:100]}...")
        else:
            print("   [ERROR] 검색 결과가 없습니다!")
    
    # 6. HybridRetriever 테스트를 위한 BM25 인덱스 확인
    bm25_path = Path("data/rag/bm25_index.pkl")
    if bm25_path.exists():
        print("\n6. BM25 인덱스: [OK] 존재함")
    else:
        print("\n6. BM25 인덱스: [WARN] 없음 (HybridRetriever가 제대로 작동하지 않을 수 있음)")

if __name__ == "__main__":
    check_knowledge_base()