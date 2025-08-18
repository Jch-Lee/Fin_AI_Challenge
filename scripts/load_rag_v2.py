#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG v2.0 (2300자 청킹) 로더
모든 RAG 관련 스크립트에서 이 모듈을 import하여 사용
"""

import sys
import json
import pickle
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# FAISS 임포트
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available")


class RAGSystemV2:
    """RAG 시스템 v2.0 로더"""
    
    def __init__(self, config_path: str = "configs/rag_config.yaml"):
        """
        Args:
            config_path: RAG 설정 파일 경로
        """
        # 설정 로드
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            # 기본 설정
            self.config = {
                'version': '2300',
                'files': {
                    'chunks': 'chunks_2300.json',
                    'embeddings': 'embeddings_2300.npy',
                    'faiss_index': 'faiss_index_2300.index',
                    'bm25_index': 'bm25_index_2300.pkl',
                    'metadata': 'metadata_2300.json'
                },
                'search': {
                    'bm25_weight': 0.5,
                    'vector_weight': 0.5,
                    'normalization_method': 'min_max'
                }
            }
        
        self.version = self.config['version']
        self.rag_dir = Path("data/rag")
        
        # 데이터 초기화
        self.chunks = None
        self.embeddings = None
        self.faiss_index = None
        self.bm25_index = None
        self.metadata = None
        
        print(f"RAG System v2.0 initialized (version: {self.version})")
    
    def load_all(self):
        """모든 RAG 컴포넌트 로드"""
        self.load_chunks()
        self.load_embeddings()
        self.load_faiss_index()
        self.load_bm25_index()
        self.load_metadata()
        
        print(f"All components loaded successfully")
        return self
    
    def load_chunks(self):
        """청크 데이터 로드"""
        chunks_file = self.rag_dir / self.config['files']['chunks']
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print(f"  - Loaded {len(self.chunks)} chunks from {chunks_file.name}")
        return self.chunks
    
    def load_embeddings(self):
        """임베딩 로드"""
        embeddings_file = self.rag_dir / self.config['files']['embeddings']
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        self.embeddings = np.load(embeddings_file)
        print(f"  - Loaded embeddings with shape {self.embeddings.shape} from {embeddings_file.name}")
        return self.embeddings
    
    def load_faiss_index(self):
        """FAISS 인덱스 로드"""
        if not FAISS_AVAILABLE:
            print("  - FAISS not available, skipping index load")
            return None
        
        faiss_file = self.rag_dir / self.config['files']['faiss_index']
        if not faiss_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_file}")
        
        self.faiss_index = faiss.read_index(str(faiss_file))
        print(f"  - Loaded FAISS index with {self.faiss_index.ntotal} vectors from {faiss_file.name}")
        return self.faiss_index
    
    def load_bm25_index(self):
        """BM25 인덱스 로드"""
        bm25_file = self.rag_dir / self.config['files']['bm25_index']
        if not bm25_file.exists():
            raise FileNotFoundError(f"BM25 index not found: {bm25_file}")
        
        with open(bm25_file, 'rb') as f:
            self.bm25_index = pickle.load(f)
        
        print(f"  - Loaded BM25 index from {bm25_file.name}")
        return self.bm25_index
    
    def load_metadata(self):
        """메타데이터 로드"""
        metadata_file = self.rag_dir / self.config['files']['metadata']
        if not metadata_file.exists():
            print(f"  - Metadata file not found: {metadata_file}, skipping")
            return None
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"  - Loaded metadata from {metadata_file.name}")
        return self.metadata
    
    def get_search_config(self) -> Dict[str, Any]:
        """검색 설정 반환"""
        return self.config.get('search', {})
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        if self.metadata:
            return self.metadata.get('statistics', {})
        
        # 메타데이터가 없으면 직접 계산
        stats = {}
        if self.chunks:
            stats['total_chunks'] = len(self.chunks)
            
            # 청크 타입 분류
            regular_chunks = sum(1 for c in self.chunks 
                               if not c.get('metadata', {}).get('is_boundary_chunk', False))
            boundary_chunks = len(self.chunks) - regular_chunks
            
            stats['regular_chunks'] = regular_chunks
            stats['boundary_chunks'] = boundary_chunks
            
            # 청크 크기 통계
            sizes = [len(c.get('content', '')) for c in self.chunks]
            if sizes:
                stats['avg_chunk_size'] = sum(sizes) / len(sizes)
                stats['min_chunk_size'] = min(sizes)
                stats['max_chunk_size'] = max(sizes)
                stats['total_characters'] = sum(sizes)
        
        if self.embeddings is not None:
            stats['embedding_dim'] = self.embeddings.shape[1]
            stats['total_embeddings'] = self.embeddings.shape[0]
        
        if self.faiss_index is not None:
            stats['faiss_vectors'] = self.faiss_index.ntotal
        
        return stats
    
    def create_hybrid_retriever(self):
        """하이브리드 검색기 생성"""
        from packages.rag.retrieval.hybrid_retriever import HybridRetriever
        from packages.rag.retrieval.bm25_retriever import BM25Retriever
        from packages.rag.embeddings.kure_embedder import KUREEmbedder
        
        # 임베더 초기화
        embedder = KUREEmbedder(show_progress=False)
        
        # BM25 검색기 초기화
        bm25_retriever = BM25Retriever()
        if self.chunks:
            doc_ids = [c.get('id', f'chunk_{i}') for i, c in enumerate(self.chunks)]
            contents = [c['content'] for c in self.chunks]
            # 메타데이터에 source 정보 추가
            metadata_list = []
            for c in self.chunks:
                meta = c.get('metadata', {}).copy()
                # source가 메타데이터에 없으면 청크 레벨에서 가져오기
                if 'source' not in meta and 'source' in c:
                    meta['source'] = c['source']
                # source_file이 있으면 source로도 저장
                if 'source_file' in meta and 'source' not in meta:
                    meta['source'] = meta['source_file']
                metadata_list.append(meta)
            bm25_retriever.build_index(contents, doc_ids, metadata_list)
        
        # 벡터 검색기 초기화 (간단한 래퍼)
        class SimpleVectorRetriever:
            def __init__(self, faiss_index, chunks):
                self.index = faiss_index
                self.chunks = chunks
                self.documents = [c['content'] for c in chunks]
                self.metadata = [c.get('metadata', {}) for c in chunks]
            
            def search(self, query_embedding: np.ndarray, k: int = 5):
                query_embedding = query_embedding.reshape(1, -1)
                faiss.normalize_L2(query_embedding)
                scores, indices = self.index.search(query_embedding, k)
                
                results = []
                for idx, score in zip(indices[0], scores[0]):
                    if idx < len(self.chunks):
                        chunk = self.chunks[idx]
                        metadata = chunk.get('metadata', {}).copy()
                        
                        # source 정보 추가
                        if 'source' not in metadata:
                            if 'source' in chunk:
                                metadata['source'] = chunk['source']
                            elif 'source_file' in metadata:
                                metadata['source'] = metadata['source_file']
                        
                        results.append({
                            'chunk_id': chunk.get('id', f'chunk_{idx}'),
                            'content': chunk['content'],
                            'score': float(score),
                            'metadata': metadata
                        })
                return results
            
            def get_stats(self):
                return {'num_vectors': self.index.ntotal}
        
        vector_retriever = SimpleVectorRetriever(self.faiss_index, self.chunks)
        
        # 하이브리드 검색기 생성
        search_config = self.get_search_config()
        hybrid_retriever = HybridRetriever(
            bm25_retriever=bm25_retriever,
            vector_retriever=vector_retriever,
            embedder=embedder,
            bm25_weight=search_config.get('bm25_weight', 0.5),
            vector_weight=search_config.get('vector_weight', 0.5),
            normalization_method=search_config.get('normalization_method', 'min_max')
        )
        
        print(f"Created hybrid retriever with weights: BM25={search_config.get('bm25_weight', 0.5):.1%}, Vector={search_config.get('vector_weight', 0.5):.1%}")
        
        return hybrid_retriever
    
    def print_summary(self):
        """시스템 요약 출력"""
        print("\n" + "="*60)
        print(f"RAG System v2.0 Summary (Version: {self.version})")
        print("="*60)
        
        stats = self.get_statistics()
        if stats:
            print(f"Total chunks: {stats.get('total_chunks', 'N/A')}")
            print(f"  - Regular: {stats.get('regular_chunks', 'N/A')}")
            print(f"  - Boundary: {stats.get('boundary_chunks', 'N/A')}")
            print(f"Average chunk size: {stats.get('avg_chunk_size', 0):.1f} chars")
            print(f"Embedding dimension: {stats.get('embedding_dim', 'N/A')}")
        
        search_config = self.get_search_config()
        print(f"\nSearch weights:")
        print(f"  - BM25: {search_config.get('bm25_weight', 0.5):.1%}")
        print(f"  - Vector: {search_config.get('vector_weight', 0.5):.1%}")
        print(f"  - Normalization: {search_config.get('normalization_method', 'min_max')}")
        print("="*60)


# 편의 함수
def load_rag_system() -> RAGSystemV2:
    """RAG 시스템 로드 (간단한 래퍼)"""
    rag = RAGSystemV2()
    rag.load_all()
    return rag


if __name__ == "__main__":
    # 테스트
    print("Loading RAG System v2.0...")
    rag = load_rag_system()
    rag.print_summary()
    
    # 하이브리드 검색기 생성 테스트
    print("\nCreating hybrid retriever...")
    retriever = rag.create_hybrid_retriever()
    
    print("\nSystem ready for use!")