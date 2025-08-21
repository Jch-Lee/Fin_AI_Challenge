#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터DB 증분 업데이트 스크립트 (VL 모델 기반)
새로운 문서를 VL 모델로 처리하고 기존 벡터DB에 추가
"""

import os
import sys
import json
import pickle
import time
import subprocess
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

# VL 모델 기반 텍스트 추출
try:
    from packages.vision.vision_extraction import VisionTextExtractor
    VL_AVAILABLE = True
except ImportError:
    VL_AVAILABLE = False
    print("ERROR: VisionTextExtractor not available. Please check packages/vision/vision_extraction.py")
    sys.exit(1)

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

# Kiwi 임포트 (한국어 토크나이저)
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False
    print("Warning: kiwipiepy not available. Install with: pip install kiwipiepy")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('incremental_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IncrementalRAGUpdater:
    """증분 RAG 업데이트 클래스 (VL 모델 기반)"""
    
    def __init__(self,
                 raw_dir: str = "data/raw",
                 processed_dir: str = "data/processed",
                 rag_dir: str = "data/rag",
                 chunk_size: int = 2300,
                 chunk_overlap: int = 200,
                 boundary_overlap: int = 300,
                 batch_size: int = 32):
        """
        Args:
            raw_dir: 원본 PDF 파일 디렉토리
            processed_dir: 처리된 텍스트 파일 디렉토리
            rag_dir: RAG 데이터 디렉토리
            chunk_size: 청크 크기
            chunk_overlap: 일반 오버랩
            boundary_overlap: 섹션 경계 오버랩
            batch_size: 임베딩 배치 크기
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.rag_dir = Path(rag_dir)
        
        # 디렉토리 생성
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.rag_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        if self.device == 'cpu':
            logger.warning("WARNING: CPU detected. VL model requires GPU for optimal performance!")
        
        # 컴포넌트 초기화
        self.chunker = HybridChunker2300(config=self.chunk_config)
        self.embedder = None
        self.kiwi = Kiwi() if KIWI_AVAILABLE else None
        
        # VL 모델 텍스트 추출기 초기화
        if VL_AVAILABLE:
            logger.info("Initializing VisionTextExtractor...")
            self.vision_extractor = VisionTextExtractor(device=self.device)
        else:
            logger.error("VisionTextExtractor not available. Cannot proceed without VL model.")
            raise RuntimeError("VL model is required but not available")
        
    def find_new_documents(self) -> List[Path]:
        """새로운 문서 찾기"""
        logger.info("Finding new documents...")
        
        # raw 폴더의 PDF 파일들
        raw_pdfs = list(self.raw_dir.glob("*.pdf"))
        raw_stems = {pdf.stem for pdf in raw_pdfs}
        
        # processed 폴더의 TXT 파일들
        processed_txts = list(self.processed_dir.glob("*.txt"))
        processed_stems = {txt.stem for txt in processed_txts}
        
        # 새로운 파일 찾기
        new_stems = raw_stems - processed_stems
        
        # 새로운 PDF 파일 경로 리스트 생성
        new_files = []
        for pdf in raw_pdfs:
            if pdf.stem in new_stems:
                new_files.append(pdf)
        
        new_files.sort(key=lambda x: x.name)
        
        logger.info(f"Found {len(new_files)} new documents")
        return new_files
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """VL 모델을 사용한 PDF 텍스트 추출"""
        logger.info(f"Starting VL model text extraction for: {pdf_path.name}")
        
        try:
            # VL 모델로 텍스트 추출
            extracted_text = self.vision_extractor.extract_text_from_pdf(
                pdf_path=str(pdf_path),
                save_output=False  # 메모리에서만 처리
            )
            
            if not extracted_text or extracted_text.strip() == "":
                # VL 모델 추출 실패 - 사용자에게 보고
                error_msg = f"VL model failed to extract text from {pdf_path.name}"
                logger.error(error_msg)
                self.report_vl_error(pdf_path, "Empty extraction result")
                return ""
            
            logger.info(f"Successfully extracted {len(extracted_text)} characters from {pdf_path.name}")
            return extracted_text
            
        except torch.cuda.OutOfMemoryError as e:
            # GPU 메모리 부족
            error_msg = f"CUDA Out of Memory while processing {pdf_path.name}"
            logger.error(error_msg)
            logger.error("Attempting to clear GPU cache and retry with smaller batch...")
            
            # GPU 캐시 정리
            torch.cuda.empty_cache()
            
            # 재시도 (페이지별 처리)
            try:
                logger.info("Retrying with page-by-page processing...")
                extracted_text = self.vision_extractor.extract_text_from_pdf(
                    pdf_path=str(pdf_path),
                    save_output=False,
                    batch_size=1  # 페이지별 처리
                )
                
                if extracted_text:
                    logger.info(f"Retry successful: extracted {len(extracted_text)} characters")
                    return extracted_text
                else:
                    self.report_vl_error(pdf_path, "CUDA OOM - Retry failed")
                    return ""
                    
            except Exception as retry_error:
                error_msg = f"Retry failed for {pdf_path.name}: {retry_error}"
                logger.error(error_msg)
                self.report_vl_error(pdf_path, f"CUDA OOM - Retry error: {retry_error}")
                return ""
                
        except Exception as e:
            # 기타 VL 모델 오류
            error_msg = f"VL model error for {pdf_path.name}: {str(e)}"
            logger.error(error_msg)
            self.report_vl_error(pdf_path, str(e))
            return ""
    
    def report_vl_error(self, pdf_path: Path, error_details: str):
        """VL 모델 오류를 사용자에게 보고"""
        error_report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'file': pdf_path.name,
            'error': error_details,
            'device': self.device,
            'gpu_memory': None
        }
        
        # GPU 메모리 상태 확인
        if torch.cuda.is_available():
            error_report['gpu_memory'] = {
                'allocated': f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                'reserved': f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
                'total': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            }
        
        # 오류 보고서 저장
        error_file = self.rag_dir / "vl_extraction_errors.json"
        existing_errors = []
        if error_file.exists():
            with open(error_file, 'r', encoding='utf-8') as f:
                existing_errors = json.load(f)
        
        existing_errors.append(error_report)
        
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(existing_errors, f, ensure_ascii=False, indent=2)
        
        # 콘솔에 경고 출력
        logger.error("="*60)
        logger.error("VL MODEL ERROR - USER ACTION REQUIRED")
        logger.error("="*60)
        logger.error(f"File: {pdf_path.name}")
        logger.error(f"Error: {error_details}")
        if error_report['gpu_memory']:
            logger.error(f"GPU Memory: {error_report['gpu_memory']}")
        logger.error("Please check vl_extraction_errors.json for details")
        logger.error("="*60)
    
    def process_new_documents(self, new_files: List[Path]) -> List[Dict[str, Any]]:
        """새로운 문서들을 처리하여 텍스트 추출 및 저장"""
        logger.info(f"Processing {len(new_files)} new documents...")
        
        processed_docs = []
        
        for pdf_path in tqdm(new_files, desc="Extracting text from PDFs"):
            # PDF에서 텍스트 추출
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                logger.warning(f"No text extracted from {pdf_path.name}")
                continue
            
            # 텍스트 파일로 저장
            txt_path = self.processed_dir / f"{pdf_path.stem}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"Saved text to {txt_path.name}")
            
            # 문서 정보 저장
            processed_docs.append({
                'content': text,
                'filename': txt_path.name,
                'source_pdf': pdf_path.name,
                'metadata': {
                    'source': str(txt_path),
                    'filename': txt_path.name,
                    'file_size': len(text),
                    'source_pdf': pdf_path.name
                }
            })
        
        logger.info(f"Processed {len(processed_docs)} documents")
        return processed_docs
    
    def load_existing_data(self) -> Tuple[List[Dict], np.ndarray, Optional[faiss.Index], Optional[BM25Okapi]]:
        """기존 데이터 로드"""
        logger.info("Loading existing data...")
        
        # 기존 청크 로드
        chunks_file = self.rag_dir / "chunks_2300.json"
        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                existing_chunks = json.load(f)
            logger.info(f"Loaded {len(existing_chunks)} existing chunks")
        else:
            existing_chunks = []
            logger.info("No existing chunks found, starting fresh")
        
        # 기존 임베딩 로드
        embeddings_file = self.rag_dir / "embeddings_2300.npy"
        if embeddings_file.exists():
            existing_embeddings = np.load(embeddings_file)
            logger.info(f"Loaded existing embeddings: {existing_embeddings.shape}")
        else:
            existing_embeddings = None
            logger.info("No existing embeddings found")
        
        # 기존 FAISS 인덱스 로드
        faiss_file = self.rag_dir / "faiss_index_2300.index"
        if FAISS_AVAILABLE and faiss_file.exists():
            existing_faiss = faiss.read_index(str(faiss_file))
            logger.info(f"Loaded FAISS index with {existing_faiss.ntotal} vectors")
        else:
            existing_faiss = None
            logger.info("No existing FAISS index found")
        
        # 기존 BM25 인덱스 로드
        bm25_file = self.rag_dir / "bm25_index_2300.pkl"
        if BM25_AVAILABLE and bm25_file.exists():
            with open(bm25_file, 'rb') as f:
                existing_bm25 = pickle.load(f)
            logger.info("Loaded existing BM25 index")
        else:
            existing_bm25 = None
            logger.info("No existing BM25 index found")
        
        return existing_chunks, existing_embeddings, existing_faiss, existing_bm25
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """문서들을 청킹"""
        logger.info("Chunking documents...")
        
        all_chunks = []
        
        for doc_idx, doc in enumerate(tqdm(documents, desc="Chunking documents")):
            # 문서별 청킹
            chunks = self.chunker.chunk_document(
                doc['content'],
                doc['metadata']
            )
            
            # 파일명 정보 추가
            for chunk in chunks:
                chunk.metadata['source_file'] = doc['filename']
                chunk.metadata['doc_index'] = doc_idx
                chunk.metadata['incremental_update'] = True  # 증분 업데이트 표시
            
            all_chunks.extend(chunks)
        
        logger.info(f"Generated {len(all_chunks)} chunks from new documents")
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
        
        # 임베딩 생성
        start_time = time.time()
        embeddings = self.embedder.embed_batch(texts, batch_size=self.batch_size)
        elapsed = time.time() - start_time
        
        logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
        logger.info(f"Embedding shape: {embeddings.shape}")
        
        return embeddings
    
    def update_faiss_index(self, existing_index: Optional[faiss.Index], 
                          new_embeddings: np.ndarray) -> faiss.Index:
        """FAISS 인덱스 업데이트"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available")
            return None
        
        logger.info("Updating FAISS index...")
        
        # 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(new_embeddings)
        
        if existing_index is None:
            # 새 인덱스 생성
            dimension = new_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(new_embeddings)
            logger.info(f"Created new FAISS index with {index.ntotal} vectors")
        else:
            # 기존 인덱스에 추가
            existing_index.add(new_embeddings)
            index = existing_index
            logger.info(f"Updated FAISS index to {index.ntotal} vectors")
        
        return index
    
    def update_bm25_index(self, existing_chunks: List, new_chunks: List[DocumentChunk]) -> BM25Okapi:
        """BM25 인덱스 업데이트"""
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available")
            return None
        
        logger.info("Updating BM25 index...")
        
        # 모든 청크 결합
        all_chunks_content = []
        
        # 기존 청크 처리
        for chunk in existing_chunks:
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
            else:
                content = str(chunk)
            all_chunks_content.append(content)
        
        # 새 청크 처리
        for chunk in new_chunks:
            all_chunks_content.append(chunk.content)
        
        # 토큰화
        tokenized_chunks = []
        if self.kiwi:
            # Kiwi를 사용한 한국어 토큰화
            for content in tqdm(all_chunks_content, desc="Tokenizing for BM25"):
                tokens = []
                for token in self.kiwi.tokenize(content):
                    tokens.append(token.form)
                tokenized_chunks.append(tokens)
        else:
            # 간단한 공백 분리
            for content in all_chunks_content:
                tokens = content.split()
                tokenized_chunks.append(tokens)
        
        # BM25 인덱스 생성
        bm25_index = BM25Okapi(tokenized_chunks)
        
        logger.info(f"Created BM25 index for {len(all_chunks_content)} documents")
        
        return bm25_index
    
    def save_updated_data(self, 
                         all_chunks: List,
                         all_embeddings: np.ndarray,
                         faiss_index: Optional[faiss.Index],
                         bm25_index: Optional[BM25Okapi]):
        """업데이트된 데이터 저장"""
        logger.info("Saving updated data...")
        
        # 1. 청크 저장 (JSON)
        chunks_file = self.rag_dir / "chunks_2300.json"
        chunks_data = []
        
        for idx, chunk in enumerate(all_chunks):
            if isinstance(chunk, dict):
                # 기존 청크
                chunks_data.append(chunk)
            else:
                # 새 청크 (DocumentChunk 객체)
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
        logger.info(f"Saved {len(chunks_data)} chunks to {chunks_file}")
        
        # 2. 임베딩 저장 (NumPy)
        embeddings_file = self.rag_dir / "embeddings_2300.npy"
        np.save(embeddings_file, all_embeddings)
        logger.info(f"Saved embeddings to {embeddings_file}: {all_embeddings.shape}")
        
        # 3. FAISS 인덱스 저장
        if faiss_index is not None:
            faiss_file = self.rag_dir / "faiss_index_2300.index"
            faiss.write_index(faiss_index, str(faiss_file))
            logger.info(f"Saved FAISS index to {faiss_file}")
        
        # 4. BM25 인덱스 저장
        if bm25_index is not None:
            bm25_file = self.rag_dir / "bm25_index_2300.pkl"
            with open(bm25_file, 'wb') as f:
                pickle.dump(bm25_index, f)
            logger.info(f"Saved BM25 index to {bm25_file}")
        
        # 5. 메타데이터 저장
        metadata_file = self.rag_dir / "metadata.json"
        metadata = {
            'total_chunks': len(chunks_data),
            'total_documents': len(set(chunk.get('source', '') for chunk in chunks_data)),
            'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
            'chunk_size': self.chunk_config.chunk_size,
            'chunk_overlap': self.chunk_config.chunk_overlap,
            'boundary_overlap': self.chunk_config.boundary_overlap,
            'embedding_dimension': all_embeddings.shape[1] if all_embeddings is not None else 0
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")
    
    def run_incremental_update(self):
        """증분 업데이트 실행 (VL 모델 기반)"""
        logger.info("="*60)
        logger.info("Starting Incremental RAG Update (VL Model Based)")
        logger.info("="*60)
        
        # VL 모델 상태 확인
        logger.info("Checking VL model status...")
        if not VL_AVAILABLE:
            logger.error("FATAL: VL model is not available. Cannot proceed.")
            logger.error("Please ensure packages/vision/vision_extraction.py is properly installed.")
            return
        
        # GPU 메모리 상태 확인
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {gpu_props.name}")
            logger.info(f"GPU Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
            
            if gpu_props.total_memory < 16 * 1024**3:
                logger.warning("WARNING: GPU memory < 16GB. May encounter OOM errors.")
                logger.warning("Consider processing documents in smaller batches.")
        else:
            logger.warning("WARNING: No GPU detected. VL model will run very slowly on CPU.")
        
        # 1. 새로운 문서 찾기
        new_files = self.find_new_documents()
        
        if not new_files:
            logger.info("No new documents to process")
            return
        
        logger.info(f"Found {len(new_files)} new documents:")
        for idx, file in enumerate(new_files, 1):
            logger.info(f"  {idx}. {file.name}")
        
        # 2. 새로운 문서 처리 (PDF -> TXT)
        new_documents = self.process_new_documents(new_files)
        
        if not new_documents:
            logger.error("No documents were successfully processed")
            return
        
        # 3. 기존 데이터 로드
        existing_chunks, existing_embeddings, existing_faiss, existing_bm25 = self.load_existing_data()
        
        # 4. 새 문서 청킹
        new_chunks = self.chunk_documents(new_documents)
        
        # 5. 새 청크에 대한 임베딩 생성
        new_embeddings = self.generate_embeddings(new_chunks)
        
        # 6. 데이터 결합
        # 청크 결합
        all_chunks = existing_chunks + [
            {
                'id': f"{chunk.doc_id}_chunk_{chunk.chunk_index}",
                'content': chunk.content,
                'source': chunk.metadata.get('source_file', 'unknown'),
                'doc_id': chunk.doc_id,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata
            } for chunk in new_chunks
        ]
        
        # 임베딩 결합
        if existing_embeddings is not None:
            all_embeddings = np.vstack([existing_embeddings, new_embeddings])
        else:
            all_embeddings = new_embeddings
        
        logger.info(f"Combined data: {len(all_chunks)} chunks, {all_embeddings.shape} embeddings")
        
        # 7. 인덱스 업데이트
        updated_faiss = self.update_faiss_index(existing_faiss, new_embeddings)
        updated_bm25 = self.update_bm25_index(existing_chunks, new_chunks)
        
        # 8. 업데이트된 데이터 저장
        self.save_updated_data(all_chunks, all_embeddings, updated_faiss, updated_bm25)
        
        # 9. VL 모델 오류 확인
        error_file = self.rag_dir / "vl_extraction_errors.json"
        if error_file.exists():
            with open(error_file, 'r', encoding='utf-8') as f:
                errors = json.load(f)
            if errors:
                logger.warning(f"WARNING: {len(errors)} VL extraction errors occurred")
                logger.warning("Please check vl_extraction_errors.json for details")
        
        logger.info("="*60)
        logger.info("Incremental Update Complete! (VL Model Based)")
        logger.info(f"Added {len(new_chunks)} new chunks from {len(new_documents)} documents")
        logger.info(f"Total chunks: {len(all_chunks)}")
        logger.info("="*60)


def main():
    """메인 실행 함수"""
    updater = IncrementalRAGUpdater()
    updater.run_incremental_update()


if __name__ == "__main__":
    main()