#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Base 재구축 스크립트
chunks.json과 embeddings.npy를 사용하여 올바른 KB 생성
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.rag.knowledge_base import KnowledgeBase

def rebuild_knowledge_base():
    """Knowledge Base를 chunks.json과 embeddings.npy로부터 재구축"""
    
    print("Knowledge Base 재구축 시작...")
    
    # 1. 데이터 로드
    print("1. 데이터 로드 중...")
    chunks_path = Path("data/rag/chunks.json")
    embeddings_path = Path("data/rag/embeddings.npy")
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"   - 로드된 청크: {len(chunks)}개")
    
    embeddings = np.load(embeddings_path)
    print(f"   - 로드된 임베딩: {embeddings.shape}")
    
    # 2. Knowledge Base 생성
    print("\n2. Knowledge Base 생성...")
    kb = KnowledgeBase(
        embedding_dim=1024,
        index_type="IVF",
        nlist=100,
        nprobe=10
    )
    
    # 3. 문서와 메타데이터 준비
    print("\n3. 문서 처리 중...")
    documents = []
    metadata_list = []
    
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            # content 추출
            content = chunk.get('content', chunk.get('text', ''))
            
            # 메타데이터 준비
            meta = {
                'chunk_id': chunk.get('id', f'chunk_{i}'),
                'doc_id': chunk.get('doc_id', 'unknown'),
                'chunk_index': chunk.get('chunk_index', i),
                'source': chunk.get('source', ''),
            }
            
            # 추가 메타데이터가 있으면 포함
            if 'metadata' in chunk:
                meta.update(chunk['metadata'])
            
            documents.append(content)
            metadata_list.append(meta)
        else:
            # dict가 아닌 경우 문자열로 변환
            documents.append(str(chunk))
            metadata_list.append({'chunk_id': f'chunk_{i}'})
        
        if (i + 1) % 1000 == 0:
            print(f"   - {i + 1}/{len(chunks)} 처리 완료")
    
    print(f"   - 총 {len(documents)}개 문서 준비 완료")
    
    # 4. Knowledge Base에 추가
    print("\n4. Knowledge Base에 문서 추가 중...")
    
    # FAISS 인덱스에 임베딩 추가
    if kb.index_type == "IVF":
        # IVF 인덱스는 학습이 필요
        print("   - IVF 인덱스 학습 중...")
        kb.index.train(embeddings)
    
    kb.index.add(embeddings)
    print(f"   - {kb.index.ntotal}개 벡터 추가 완료")
    
    # 문서와 메타데이터 저장
    kb.documents = documents
    kb.metadata = metadata_list
    kb.doc_count = len(documents)
    
    # 5. 저장
    print("\n5. Knowledge Base 저장 중...")
    save_dir = Path("data/rag/knowledge_base_fixed")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # pickle 형식으로 저장 (새 형식)
    data = {
        'documents': kb.documents,
        'metadata': kb.metadata,
        'doc_count': kb.doc_count,
        'embedding_dim': kb.embedding_dim,
        'index_type': kb.index_type,
        'nlist': kb.nlist,
        'nprobe': kb.nprobe
    }
    
    with open(save_dir / "knowledge_base.pkl", 'wb') as f:
        pickle.dump(data, f)
    print(f"   - knowledge_base.pkl 저장 완료")
    
    # FAISS 인덱스 저장
    import faiss
    faiss.write_index(kb.index, str(save_dir / "faiss.index"))
    print(f"   - faiss.index 저장 완료")
    
    # 메타데이터 저장
    with open(save_dir / "metadata.json", 'w', encoding='utf-8') as f:
        original_metadata = {
            "num_chunks": len(chunks),
            "embedding_dim": 1024,
            "embedder": "nlpai-lab/KURE-v1",
            "index_type": "IVF",
            "nlist": 100,
            "nprobe": 10
        }
        json.dump(original_metadata, f, ensure_ascii=False, indent=2)
    print(f"   - metadata.json 저장 완료")
    
    # 6. 검증
    print("\n6. Knowledge Base 검증...")
    
    # 다시 로드
    kb_loaded = KnowledgeBase.load(str(save_dir))
    print(f"   - 로드된 문서 수: {kb_loaded.doc_count}")
    print(f"   - FAISS 인덱스 크기: {kb_loaded.index.ntotal}")
    
    # 테스트 검색
    test_embedding = embeddings[0]
    results = kb_loaded.search(test_embedding, top_k=3)
    print(f"\n   테스트 검색 결과: {len(results)}개")
    for i, result in enumerate(results[:2], 1):
        if isinstance(result, dict):
            print(f"   [{i}] Score: {result.get('score', 0):.4f}")
            print(f"       Content: {result.get('content', '')[:100]}...")
        else:
            print(f"   [{i}] Score: {result.score:.4f}")
            print(f"       Content: {result.content[:100]}...")
    
    print("\n✅ Knowledge Base 재구축 완료!")
    print(f"   저장 위치: {save_dir}")
    return str(save_dir)

if __name__ == "__main__":
    rebuild_knowledge_base()