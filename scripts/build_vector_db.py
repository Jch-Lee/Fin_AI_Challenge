#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
벡터 데이터베이스 구축 스크립트
처리된 텍스트 파일들을 청킹하고 임베딩을 생성하여 벡터DB 구축
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.rag.embeddings.kure_embedder import KUREEmbedder

# 청커 임포트
try:
    from packages.preprocessing.hierarchical_chunker import HierarchicalMarkdownChunker
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HierarchicalMarkdownChunker = None
    HIERARCHICAL_AVAILABLE = False
    
try:
    from packages.preprocessing.document_chunker import DocumentChunker
except ImportError:
    DocumentChunker = None

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorDBBuilder:
    """벡터 데이터베이스 구축기"""
    
    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 embedder_name: str = "kure-v1",
                 batch_size: int = 32):
        """
        Args:
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
            embedder_name: 임베더 이름
            batch_size: 배치 크기
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        # 청커 초기화 (사용 가능한 청커 선택)
        if HIERARCHICAL_AVAILABLE and HierarchicalMarkdownChunker is not None:
            try:
                # HierarchicalMarkdownChunker는 별도의 chunk_size 파라미터를 받지 않음
                # 대신 내부적으로 계층별 설정을 사용
                self.chunker = HierarchicalMarkdownChunker(
                    use_chunk_cleaner=True,
                    enable_semantic=False
                )
                logger.info("Using HierarchicalMarkdownChunker with hierarchical sizing")
                self.chunker_type = "hierarchical"
            except Exception as e:
                logger.warning(f"Failed to init HierarchicalMarkdownChunker: {e}")
                self.chunker = None
                self.chunker_type = None
        elif DocumentChunker is not None:
            try:
                self.chunker = DocumentChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                logger.info("Using DocumentChunker")
                self.chunker_type = "document"
            except Exception as e:
                logger.warning(f"Failed to init DocumentChunker: {e}")
                self.chunker = None
                self.chunker_type = None
        
        # 모든 청커가 실패하면 기본 청킹 사용
        if not hasattr(self, 'chunker') or self.chunker is None:
            logger.warning("Using basic text splitting")
            self.chunker = None
            self.chunker_type = "basic"
        
        # 임베더 초기화
        if embedder_name == "kure-v1":
            self.embedder = KUREEmbedder(
                model_name="nlpai-lab/KURE-v1",
                batch_size=batch_size,
                show_progress=True
            )
        else:
            raise ValueError(f"Unknown embedder: {embedder_name}")
        
        logger.info(f"VectorDBBuilder initialized with {embedder_name}")
    
    def load_text_files(self, input_dir: str) -> List[Dict[str, str]]:
        """
        텍스트 파일들을 로드
        
        Args:
            input_dir: 입력 디렉토리
            
        Returns:
            문서 리스트
        """
        documents = []
        input_path = Path(input_dir)
        
        # txt 파일들 찾기
        txt_files = list(input_path.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} text files")
        
        for file_path in tqdm(txt_files, desc="Loading text files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 문서 정보
                doc = {
                    'id': file_path.stem,
                    'source': str(file_path),
                    'content': content,
                    'metadata': {
                        'filename': file_path.name,
                        'size': len(content),
                        'type': 'text'
                    }
                }
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def _basic_chunking(self, text: str) -> List[str]:
        """기본 텍스트 청킹"""
        chunks = []
        text_length = len(text)
        
        for i in range(0, text_length, self.chunk_size - self.chunk_overlap):
            end = min(i + self.chunk_size, text_length)
            chunk = text[i:end]
            chunks.append(chunk)
            
            if end >= text_length:
                break
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict]:
        """
        문서들을 청킹
        
        Args:
            documents: 문서 리스트
            
        Returns:
            청크 리스트
        """
        all_chunks = []
        
        for doc in tqdm(documents, desc="Chunking documents"):
            try:
                # 청킹 수행
                if self.chunker_type == "hierarchical":
                    # HierarchicalMarkdownChunker 사용
                    chunk_objects = self.chunker.chunk_document(
                        doc['content'],
                        metadata={'source': doc['source'], 'filename': doc['metadata']['filename']}
                    )
                    # DocumentChunk 객체를 문자열 리스트로 변환
                    chunks = []
                    for chunk_obj in chunk_objects:
                        chunks.append({
                            'content': chunk_obj.content,
                            'metadata': chunk_obj.metadata
                        })
                elif self.chunker is not None and hasattr(self.chunker, 'chunk_text'):
                    # 기본 청커
                    chunks = self.chunker.chunk_text(
                        doc['content'],
                        source_id=doc['id']
                    )
                    # 문자열을 딕셔너리로 변환
                    chunks = [{'content': c, 'metadata': {}} for c in chunks]
                else:
                    # 기본 텍스트 분할
                    chunk_texts = self._basic_chunking(doc['content'])
                    chunks = [{'content': c, 'metadata': {}} for c in chunk_texts]
                
                # 메타데이터 추가
                for i, chunk in enumerate(chunks):
                    # chunk는 이미 딕셔너리 형태
                    content = chunk['content']
                    chunk_metadata = chunk.get('metadata', {})
                    
                    chunk_dict = {
                        'id': f"{doc['id']}_chunk_{i}",
                        'content': content,
                        'source': doc['source'],
                        'doc_id': doc['id'],
                        'chunk_index': i,
                        'metadata': {
                            **doc['metadata'],
                            **chunk_metadata,  # 청커에서 제공한 메타데이터 병합
                            'chunk_size': len(content)
                        }
                    }
                    
                    all_chunks.append(chunk_dict)
                    
            except Exception as e:
                logger.error(f"Failed to chunk document {doc['id']}: {e}")
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        청크들의 임베딩 생성
        
        Args:
            chunks: 청크 리스트
            
        Returns:
            임베딩 배열
        """
        # 텍스트 추출
        texts = [chunk['content'] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        
        # 배치 단위로 임베딩 생성
        embeddings = self.embedder.embed_batch(
            texts,
            batch_size=self.batch_size
        )
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def save_vector_db(self, 
                       chunks: List[Dict],
                       embeddings: np.ndarray,
                       output_dir: str):
        """
        벡터 데이터베이스 저장
        
        Args:
            chunks: 청크 리스트
            embeddings: 임베딩 배열
            output_dir: 출력 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 청크 정보 저장 (JSON)
        chunks_path = output_path / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved chunks to {chunks_path}")
        
        # 2. 임베딩 저장 (NumPy)
        embeddings_path = output_path / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        logger.info(f"Saved embeddings to {embeddings_path}")
        
        # 3. 메타데이터 저장
        metadata = {
            'num_chunks': len(chunks),
            'embedding_dim': embeddings.shape[1],
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedder': self.embedder.model_name,
            'documents': list(set(chunk['doc_id'] for chunk in chunks))
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # 4. 통계 출력
        self._print_statistics(chunks, embeddings)
    
    def _print_statistics(self, chunks: List[Dict], embeddings: np.ndarray):
        """통계 출력"""
        print("\n" + "="*60)
        print(" 벡터 데이터베이스 구축 완료")
        print("="*60)
        
        # 기본 통계
        print(f"총 청크 수: {len(chunks)}")
        print(f"임베딩 차원: {embeddings.shape[1]}")
        print(f"총 문서 수: {len(set(chunk['doc_id'] for chunk in chunks))}")
        
        # 청크 크기 통계
        chunk_sizes = [chunk['metadata']['chunk_size'] for chunk in chunks]
        print(f"\n청크 크기 통계:")
        print(f"  평균: {np.mean(chunk_sizes):.1f}자")
        print(f"  최소: {np.min(chunk_sizes)}자")
        print(f"  최대: {np.max(chunk_sizes)}자")
        
        # 문서별 청크 수
        doc_chunk_counts = {}
        for chunk in chunks:
            doc_id = chunk['doc_id']
            doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1
        
        print(f"\n문서별 청크 수:")
        print(f"  평균: {np.mean(list(doc_chunk_counts.values())):.1f}개")
        print(f"  최소: {np.min(list(doc_chunk_counts.values()))}개")
        print(f"  최대: {np.max(list(doc_chunk_counts.values()))}개")
        
        print("="*60)
    
    def build(self, input_dir: str, output_dir: str):
        """
        전체 벡터DB 구축 프로세스
        
        Args:
            input_dir: 입력 디렉토리
            output_dir: 출력 디렉토리
        """
        logger.info("Starting vector database building process")
        
        # 1. 텍스트 파일 로드
        logger.info("Step 1: Loading text files")
        documents = self.load_text_files(input_dir)
        
        if not documents:
            logger.error("No documents found!")
            return
        
        # 2. 문서 청킹
        logger.info("Step 2: Chunking documents")
        chunks = self.chunk_documents(documents)
        
        if not chunks:
            logger.error("No chunks created!")
            return
        
        # 3. 임베딩 생성
        logger.info("Step 3: Generating embeddings")
        embeddings = self.generate_embeddings(chunks)
        
        # 4. 저장
        logger.info("Step 4: Saving vector database")
        self.save_vector_db(chunks, embeddings, output_dir)
        
        logger.info("Vector database building completed successfully!")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="벡터 데이터베이스 구축")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="입력 텍스트 파일 디렉토리")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="출력 벡터DB 디렉토리")
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="청크 크기")
    parser.add_argument("--chunk-overlap", type=int, default=50,
                       help="청크 오버랩")
    parser.add_argument("--embedder", type=str, default="kure-v1",
                       choices=["kure-v1"],
                       help="임베더 선택")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="배치 크기")
    
    args = parser.parse_args()
    
    # 빌더 생성
    builder = VectorDBBuilder(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedder_name=args.embedder,
        batch_size=args.batch_size
    )
    
    # 벡터DB 구축
    builder.build(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()