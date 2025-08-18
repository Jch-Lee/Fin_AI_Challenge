#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
전체 PDF 문서 처리 스크립트
59개 PDF 파일을 처리하여 RAG 시스템 구축
"""

import sys
import os
import time
import json
import logging
import gc
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PDFBatchProcessor:
    """PDF 배치 처리 클래스"""
    
    def __init__(self, 
                 use_vision: bool = False,
                 batch_size: int = 5,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """
        Args:
            use_vision: Vision V2 모델 사용 여부
            batch_size: 한 번에 처리할 PDF 수
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
        """
        self.use_vision = use_vision
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 디렉토리 생성
        self.output_dir = Path("data/knowledge_base")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # 컴포넌트 초기화
        self._init_components()
        
    def _init_components(self):
        """컴포넌트 초기화"""
        # PDF 프로세서
        if self.use_vision:
            try:
                from packages.preprocessing.pdf_processor_vision import VisionPDFProcessor
                self.pdf_processor = VisionPDFProcessor()
                logger.info("Using Vision V2 PDF processor")
            except Exception as e:
                logger.warning(f"Vision V2 not available: {e}, falling back to PyMuPDF")
                self.use_vision = False
        
        if not self.use_vision:
            try:
                from packages.preprocessing.pdf_processor_traditional import TraditionalPDFProcessor
                self.pdf_processor = TraditionalPDFProcessor()
                logger.info("Using PyMuPDF PDF processor")
            except:
                # 직접 PyMuPDF 사용
                import fitz
                self.pdf_processor = None
                logger.info("Using direct PyMuPDF")
        
        # 청킹
        try:
            from packages.preprocessing.hierarchical_chunker import HierarchicalMarkdownChunker
            self.chunker = HierarchicalMarkdownChunker(
                use_chunk_cleaner=True,
                enable_semantic=False
            )
            logger.info("Using HierarchicalMarkdownChunker")
        except:
            from packages.preprocessing.chunker import DocumentChunker
            self.chunker = DocumentChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info("Using basic DocumentChunker")
        
        # 임베딩
        from packages.rag.embeddings import KUREEmbedder
        self.embedder = KUREEmbedder()
        logger.info(f"Using {self.embedder.model_name} embedder")
        
        # 지식베이스
        from packages.rag.knowledge_base import KnowledgeBase
        self.knowledge_base = KnowledgeBase(
            embedding_dim=self.embedder.embedding_dim
        )
        logger.info("Knowledge base initialized")
    
    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """단일 PDF 처리"""
        # 이미 처리된 파일 체크
        processed_file = self.processed_dir / f"{pdf_path.stem}.txt"
        if processed_file.exists():
            logger.info(f"Skipping already processed: {pdf_path.name}")
            return {
                'file': pdf_path.name,
                'status': 'skipped',
                'chunks': 0,
                'embeddings': 0,
                'error': None
            }
        
        result = {
            'file': pdf_path.name,
            'status': 'processing',
            'chunks': 0,
            'embeddings': 0,
            'error': None
        }
        
        try:
            # 1. PDF 텍스트 추출
            if self.pdf_processor:
                pdf_result = self.pdf_processor.extract_pdf(str(pdf_path))
                text = pdf_result.text
            else:
                # 직접 PyMuPDF 사용
                import fitz
                doc = fitz.open(str(pdf_path))
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
            
            if not text or len(text) < 100:
                result['status'] = 'empty'
                result['error'] = 'No text extracted'
                return result
            
            # 2. 텍스트 청킹
            chunks = self.chunker.chunk_document(
                text,
                metadata={
                    'source': str(pdf_path),
                    'doc_id': pdf_path.stem,
                    'file_name': pdf_path.name
                }
            )
            result['chunks'] = len(chunks)
            
            # 3. 임베딩 생성
            chunk_texts = [c.content for c in chunks]
            embeddings = self.embedder.embed_batch(chunk_texts, batch_size=32)
            result['embeddings'] = len(embeddings)
            
            # 4. 지식베이스에 추가
            documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc = {
                    'id': f"{pdf_path.stem}_{i}",
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'embedding': embedding
                }
                documents.append(doc)
            
            self.knowledge_base.add_documents(embeddings, documents)
            
            # 처리된 텍스트 저장
            processed_file = self.processed_dir / f"{pdf_path.stem}.txt"
            with open(processed_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            result['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def process_all_pdfs(self, pdf_dir: Path) -> Dict[str, Any]:
        """모든 PDF 처리"""
        pdf_files = list(pdf_dir.glob("*.pdf"))
        total_files = len(pdf_files)
        
        logger.info(f"Found {total_files} PDF files to process")
        
        results = {
            'total': total_files,
            'success': 0,
            'failed': 0,
            'empty': 0,
            'total_chunks': 0,
            'total_embeddings': 0,
            'processing_time': 0,
            'files': []
        }
        
        start_time = time.time()
        
        # 배치 처리
        for i in range(0, total_files, self.batch_size):
            batch = pdf_files[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_files + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            for pdf_path in batch:
                logger.info(f"Processing {i+1}/{total_files}: {pdf_path.name}")
                
                file_result = self.process_pdf(pdf_path)
                results['files'].append(file_result)
                
                if file_result['status'] == 'success':
                    results['success'] += 1
                    results['total_chunks'] += file_result['chunks']
                    results['total_embeddings'] += file_result['embeddings']
                elif file_result['status'] == 'skipped':
                    results['success'] += 1  # 이미 성공적으로 처리된 파일
                    logger.info(f"Already processed: {pdf_path.name}")
                elif file_result['status'] == 'empty':
                    results['empty'] += 1
                else:
                    results['failed'] += 1
                
                # 진행 상황 출력
                progress = (i + 1) / total_files * 100
                logger.info(f"Progress: {progress:.1f}% ({results['success']} success, {results['failed']} failed)")
            
            # 메모리 정리
            gc.collect()
            
            # 중간 저장 (5 배치마다)
            if batch_num % 5 == 0:
                self.save_checkpoint(results)
        
        results['processing_time'] = time.time() - start_time
        
        # 최종 저장
        self.save_final_results(results)
        
        return results
    
    def save_checkpoint(self, results: Dict[str, Any]):
        """중간 체크포인트 저장"""
        checkpoint_file = self.output_dir / "checkpoint.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 지식베이스 저장
        kb_file = self.output_dir / "knowledge_base_checkpoint.pkl"
        self.knowledge_base.save(str(kb_file))
        
        logger.info(f"Checkpoint saved: {results['success']} files processed")
    
    def save_final_results(self, results: Dict[str, Any]):
        """최종 결과 저장"""
        # 결과 요약 저장
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 지식베이스 저장
        kb_file = self.output_dir / "knowledge_base.pkl"
        self.knowledge_base.save(str(kb_file))
        
        # FAISS 인덱스 별도 저장
        faiss_file = self.output_dir / "faiss_index.bin"
        import faiss
        faiss.write_index(self.knowledge_base.index, str(faiss_file))
        
        logger.info(f"Final results saved to {self.output_dir}")
        logger.info(f"Total processing time: {results['processing_time']:.2f} seconds")
        logger.info(f"Success rate: {results['success']}/{results['total']} ({results['success']/results['total']*100:.1f}%)")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process all PDF files for RAG system")
    parser.add_argument('--pdf-dir', type=str, default='data/raw', help='PDF directory')
    parser.add_argument('--use-vision', action='store_true', help='Use Vision V2 model')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size')
    parser.add_argument('--chunk-size', type=int, default=512, help='Chunk size')
    parser.add_argument('--test', action='store_true', help='Test mode (process only 3 files)')
    
    args = parser.parse_args()
    
    # 프로세서 초기화
    processor = PDFBatchProcessor(
        use_vision=args.use_vision,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )
    
    # PDF 디렉토리
    pdf_dir = Path(args.pdf_dir)
    
    if args.test:
        # 테스트 모드: 처음 3개 파일만
        pdf_files = list(pdf_dir.glob("*.pdf"))[:3]
        test_dir = Path("test_pdfs")
        test_dir.mkdir(exist_ok=True)
        
        # 테스트 파일 복사
        import shutil
        for pdf in pdf_files:
            shutil.copy(pdf, test_dir / pdf.name)
        
        results = processor.process_all_pdfs(test_dir)
    else:
        # 전체 처리
        results = processor.process_all_pdfs(pdf_dir)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print(" Processing Complete")
    print("=" * 60)
    print(f"Total files: {results['total']}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Empty: {results['empty']}")
    print(f"Total chunks: {results['total_chunks']}")
    print(f"Total embeddings: {results['total_embeddings']}")
    print(f"Processing time: {results['processing_time']:.2f} seconds")
    print("=" * 60)
    
    return results['failed'] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)