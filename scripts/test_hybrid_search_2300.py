#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
하이브리드 RAG 검색 테스트 (2300자 청킹 버전)
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 모듈 임포트
from packages.rag.embeddings.kure_embedder import KUREEmbedder
from packages.rag.retrieval.bm25_retriever import BM25Retriever
from packages.rag.retrieval.hybrid_retriever import HybridRetriever

# FAISS 임포트
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available")


class SimpleVectorRetriever:
    """간단한 벡터 검색기"""
    
    def __init__(self, faiss_index, chunks):
        self.index = faiss_index
        self.chunks = chunks
        self.documents = [c['content'] for c in chunks]
        self.metadata = [c.get('metadata', {}) for c in chunks]
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """벡터 검색"""
        # 정규화
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # 검색
        scores, indices = self.index.search(query_embedding, k)
        
        # 결과 반환
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.chunks):
                results.append({
                    'chunk_id': self.chunks[idx].get('id', f'chunk_{idx}'),
                    'content': self.chunks[idx]['content'],
                    'score': float(score),
                    'metadata': self.chunks[idx].get('metadata', {})
                })
        
        return results
    
    def get_stats(self):
        return {'num_vectors': self.index.ntotal}


def load_rag_components(rag_dir: str = "data/rag"):
    """RAG 컴포넌트 로드"""
    rag_path = Path(rag_dir)
    
    print("Loading RAG components...")
    
    # 1. 청크 로드
    chunks_file = rag_path / "chunks_2300.json"
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"  - Loaded {len(chunks)} chunks")
    
    # 2. FAISS 인덱스 로드
    if FAISS_AVAILABLE:
        faiss_file = rag_path / "faiss_index_2300.index"
        faiss_index = faiss.read_index(str(faiss_file))
        print(f"  - Loaded FAISS index with {faiss_index.ntotal} vectors")
    else:
        faiss_index = None
        print("  - FAISS not available")
    
    # 3. BM25 인덱스 로드
    bm25_file = rag_path / "bm25_index_2300.pkl"
    with open(bm25_file, 'rb') as f:
        bm25_index = pickle.load(f)
    print(f"  - Loaded BM25 index")
    
    # 4. 메타데이터 로드
    metadata_file = rag_path / "metadata_2300.json"
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return chunks, faiss_index, bm25_index, metadata


def test_hybrid_search():
    """하이브리드 검색 테스트"""
    
    # 1. 컴포넌트 로드
    chunks, faiss_index, bm25_index, metadata = load_rag_components()
    
    print("\n" + "="*60)
    print("RAG System Statistics:")
    print(f"  Total chunks: {metadata['statistics']['total_chunks']}")
    print(f"  Regular chunks: {metadata['statistics']['regular_chunks']}")
    print(f"  Boundary chunks: {metadata['statistics']['boundary_chunks']}")
    print(f"  Avg chunk size: {metadata['statistics']['avg_chunk_size']:.1f} chars")
    print(f"  Embedder: {metadata['embedder']}")
    print("="*60)
    
    # 2. 검색 시스템 초기화
    print("\nInitializing search components...")
    
    # 임베더 초기화
    embedder = KUREEmbedder(show_progress=False)
    
    # BM25 검색기 초기화
    bm25_retriever = BM25Retriever()
    # 청크 데이터로 재구성
    doc_ids = [c['id'] for c in chunks]
    contents = [c['content'] for c in chunks]
    metadata_list = [c.get('metadata', {}) for c in chunks]
    bm25_retriever.build_index(contents, doc_ids, metadata_list)
    
    # 벡터 검색기 초기화
    vector_retriever = SimpleVectorRetriever(faiss_index, chunks)
    
    # 하이브리드 검색기 초기화 (5:5 가중치)
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        embedder=embedder,
        bm25_weight=0.5,  # 50%
        vector_weight=0.5,  # 50%
        normalization_method="min_max"
    )
    
    print("  - Embedder initialized")
    print("  - BM25 retriever initialized")
    print("  - Vector retriever initialized")
    print("  - Hybrid retriever initialized (BM25:Vector = 50:50)")
    
    # 3. 테스트 쿼리
    test_queries = [
        "개인정보 보호법의 주요 내용은?",
        "금융 보안에서 중요한 보안 조치는?",
        "AI 시스템의 개인정보 처리 원칙",
        "랜섬웨어 대응 방안",
        "ISMS-P 인증 기준"
    ]
    
    print("\n" + "="*60)
    print("Testing Hybrid Search")
    print("="*60)
    
    for query_idx, query in enumerate(test_queries, 1):
        print(f"\n[Query {query_idx}] {query}")
        print("-" * 50)
        
        # 하이브리드 검색
        results = hybrid_retriever.search(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Score: {result.hybrid_score:.4f} (BM25: {result.bm25_score:.4f}, Vector: {result.vector_score:.4f})")
            print(f"    Methods: {', '.join(result.retrieval_methods)}")
            print(f"    Content preview: {result.content[:200]}...")
            if result.metadata:
                source = result.metadata.get('source_file', 'unknown')
                print(f"    Source: {source}")
    
    # 4. 검색 통계
    print("\n" + "="*60)
    print("Search Statistics")
    print("="*60)
    
    stats = hybrid_retriever.get_stats()
    print(f"  Weights: BM25={stats['weights']['bm25_weight']:.1%}, Vector={stats['weights']['vector_weight']:.1%}")
    print(f"  Normalization: {stats['normalization_method']}")
    print(f"  BM25 available: {stats['components']['bm25_available']}")
    print(f"  Vector available: {stats['components']['vector_available']}")


if __name__ == "__main__":
    test_hybrid_search()