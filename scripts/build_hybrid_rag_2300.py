#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
하이브리드 RAG 시스템 구축 스크립트 (2300자 청킹)
전체 문서를 청킹하고 벡터 DB 구축
"""

import os
import sys
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
from tqdm import tqdm
import numpy as np
import torch

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 모듈 임포트
from packages.preprocessing.hybrid_chunker_2300 import HybridChunker2300, HybridChunkConfig
from packages.rag.embeddings.kure_embedder import KUREEmbedder
from packages.preprocessing.chunker import DocumentChunk

# FAISS 임포트
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")

# BM25 임포트
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank_bm25 not available. Install with: pip install rank-bm25")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybrid_rag_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HybridRAGBuilder:
    """하이브리드 RAG 시스템 구축기"""
    
    def __init__(self,
                 input_dir: str = "data/processed",
                 output_dir: str = "data/rag",
                 chunk_size: int = 2300,
                 chunk_overlap: int = 200,
                 boundary_overlap: int = 300,
                 batch_size: int = 32):
        """
        Args:
            input_dir: 처리된 텍스트 파일들이 있는 디렉토리
            output_dir: 결과를 저장할 디렉토리
            chunk_size: 청크 크기
            chunk_overlap: 일반 오버랩
            boundary_overlap: 섹션 경계 오버랩
            batch_size: 임베딩 배치 크기
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 청킹 설정
        self.chunk_config = HybridChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            boundary_overlap=boundary_overlap
        )
        
        self.batch_size = batch_size
        
        # GPU 사용 가능 여부 확인
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # 컴포넌트 초기화
        self.chunker = HybridChunker2300(config=self.chunk_config)
        self.embedder = None  # 나중에 초기화
        
    def load_documents(self) -> List[Dict[str, Any]]:
        """모든 텍스트 문서 로드"""
        documents = []
        txt_files = list(self.input_dir.glob("*.txt"))
        
        logger.info(f"Found {len(txt_files)} text files to process")
        
        for file_path in tqdm(txt_files, desc="Loading documents"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if content.strip():
                    documents.append({
                        'content': content,
                        'filename': file_path.name,
                        'metadata': {
                            'source': str(file_path),
                            'filename': file_path.name,
                            'file_size': len(content)
                        }
                    })
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def chunk_all_documents(self, documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """모든 문서를 청킹"""
        logger.info("Starting document chunking...")
        
        all_chunks = []
        total_start = time.time()
        
        for doc_idx, doc in enumerate(tqdm(documents, desc="Chunking documents")):
            start_time = time.time()
            
            # 문서별 청킹
            chunks = self.chunker.chunk_document(
                doc['content'],
                doc['metadata']
            )
            
            # 파일명 정보 추가
            for chunk in chunks:
                chunk.metadata['source_file'] = doc['filename']
                chunk.metadata['doc_index'] = doc_idx
            
            all_chunks.extend(chunks)
            
            elapsed = time.time() - start_time
            logger.debug(f"Document {doc_idx+1}/{len(documents)}: "
                        f"{len(chunks)} chunks in {elapsed:.2f}s")
        
        total_elapsed = time.time() - total_start
        
        # 통계 출력
        stats = self.chunker.get_statistics(all_chunks)
        logger.info(f"Chunking completed in {total_elapsed:.2f}s")
        logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
        
        return all_chunks
    
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """청크들에 대한 임베딩 생성"""
        logger.info("Initializing embedder...")
        
        # 임베더 초기화
        self.embedder = KUREEmbedder(
            device=self.device,
            batch_size=self.batch_size,
            show_progress=True
        )
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # 청크 텍스트 추출
        texts = [chunk.content for chunk in chunks]
        
        # 임베딩 생성 - embed_batch 메서드 사용
        start_time = time.time()
        embeddings = self.embedder.embed_batch(texts, batch_size=self.batch_size)
        elapsed = time.time() - start_time
        
        logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
        logger.info(f"Embedding shape: {embeddings.shape}")
        
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> Optional[faiss.Index]:
        """FAISS 인덱스 구축"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping index creation")
            return None
        
        logger.info("Building FAISS index...")
        
        # 차원 확인
        dimension = embeddings.shape[1]
        logger.info(f"Embedding dimension: {dimension}")
        
        # 인덱스 생성 (내적 유사도 사용)
        index = faiss.IndexFlatIP(dimension)
        
        # 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(embeddings)
        
        # 벡터 추가
        index.add(embeddings)
        
        logger.info(f"FAISS index created with {index.ntotal} vectors")
        
        return index
    
    def build_bm25_index(self, chunks: List[DocumentChunk]) -> Optional[BM25Okapi]:
        """BM25 인덱스 구축"""
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available, skipping index creation")
            return None
        
        logger.info("Building BM25 index...")
        
        # 토큰화 (간단한 공백 분리)
        tokenized_chunks = []
        for chunk in chunks:
            # 한국어 처리를 위한 간단한 토큰화
            tokens = chunk.content.split()
            tokenized_chunks.append(tokens)
        
        # BM25 인덱스 생성
        bm25_index = BM25Okapi(tokenized_chunks)
        
        logger.info(f"BM25 index created for {len(chunks)} documents")
        
        return bm25_index
    
    def save_results(self, 
                    chunks: List[DocumentChunk],
                    embeddings: np.ndarray,
                    faiss_index: Optional[faiss.Index],
                    bm25_index: Optional[BM25Okapi]):
        """결과 저장"""
        logger.info("Saving results...")
        
        # 1. 청크 저장 (JSON)
        chunks_file = self.output_dir / "chunks_2300.json"
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                'id': f"{chunk.doc_id}_chunk_{chunk.chunk_index}",
                'content': chunk.content,
                'source': chunk.metadata.get('source_file', 'unknown'),
                'doc_id': chunk.doc_id,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata
            })
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved chunks to {chunks_file}")
        
        # 2. 임베딩 저장 (NumPy)
        embeddings_file = self.output_dir / "embeddings_2300.npy"
        np.save(embeddings_file, embeddings)
        logger.info(f"Saved embeddings to {embeddings_file}")
        
        # 3. FAISS 인덱스 저장
        if faiss_index:
            faiss_file = self.output_dir / "faiss_index_2300.index"
            faiss.write_index(faiss_index, str(faiss_file))
            logger.info(f"Saved FAISS index to {faiss_file}")
        
        # 4. BM25 인덱스 저장
        if bm25_index:
            bm25_file = self.output_dir / "bm25_index_2300.pkl"
            with open(bm25_file, 'wb') as f:
                pickle.dump(bm25_index, f)
            logger.info(f"Saved BM25 index to {bm25_file}")
        
        # 5. 메타데이터 저장
        metadata_file = self.output_dir / "metadata_2300.json"
        metadata = {
            'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'chunk_size': self.chunk_config.chunk_size,
                'chunk_overlap': self.chunk_config.chunk_overlap,
                'boundary_overlap': self.chunk_config.boundary_overlap
            },
            'statistics': self.chunker.get_statistics(chunks),
            'embedder': 'KURE-v1',
            'device': self.device,
            'total_documents': len(set(c.metadata.get('source_file') for c in chunks))
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")
    
    def build(self):
        """전체 빌드 프로세스 실행"""
        logger.info("=" * 60)
        logger.info("Starting Hybrid RAG System Build (2300 chars)")
        logger.info("=" * 60)
        
        total_start = time.time()
        
        try:
            # 1. 문서 로드
            logger.info("\n[Step 1/5] Loading documents...")
            documents = self.load_documents()
            
            # 2. 청킹
            logger.info("\n[Step 2/5] Chunking documents...")
            chunks = self.chunk_all_documents(documents)
            
            # 3. 임베딩 생성
            logger.info("\n[Step 3/5] Generating embeddings...")
            embeddings = self.generate_embeddings(chunks)
            
            # 4. 인덱스 구축
            logger.info("\n[Step 4/5] Building indices...")
            faiss_index = self.build_faiss_index(embeddings)
            bm25_index = self.build_bm25_index(chunks)
            
            # 5. 저장
            logger.info("\n[Step 5/5] Saving results...")
            self.save_results(chunks, embeddings, faiss_index, bm25_index)
            
            total_elapsed = time.time() - total_start
            
            logger.info("\n" + "=" * 60)
            logger.info("Build completed successfully!")
            logger.info(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Build failed: {e}", exc_info=True)
            raise


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Hybrid RAG System with 2300-char chunks")
    parser.add_argument("--input-dir", default="data/processed", help="Input directory")
    parser.add_argument("--output-dir", default="data/rag", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=2300, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--boundary-overlap", type=int, default=300, help="Boundary overlap")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    
    args = parser.parse_args()
    
    # 빌더 생성 및 실행
    builder = HybridRAGBuilder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        boundary_overlap=args.boundary_overlap,
        batch_size=args.batch_size
    )
    
    builder.build()


if __name__ == "__main__":
    main()