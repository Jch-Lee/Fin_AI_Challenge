"""
배치 PDF 처리 파이프라인
- Vision V2 기반 PDF 파싱
- Kiwi 기반 텍스트 전처리 (사용 가능한 경우)
- 메모리 효율적인 대용량 처리
"""

import os
import gc
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)

# 조건부 imports
from . import TextProcessor, KIWI_AVAILABLE, LANGCHAIN_AVAILABLE
from .chunker import DocumentChunker
if LANGCHAIN_AVAILABLE:
    from .hierarchical_chunker import HierarchicalMarkdownChunker
from .embedder import EmbeddingGenerator
from ..rag.knowledge_base import KnowledgeBase

# Vision 파서 import
try:
    from ..parsers.vision_v2_parser import VisionV2Parser
    VISION_AVAILABLE = True
except ImportError:
    logger.warning("Vision V2 parser not available")
    VISION_AVAILABLE = False
    from ..parsers.pdf_parser import PyMuPDFParser as VisionV2Parser


@dataclass
class ProcessingResult:
    """처리 결과 데이터 클래스"""
    file_path: str
    status: str  # 'success', 'partial', 'failed'
    chunks_created: int
    embeddings_created: int
    error_message: Optional[str] = None
    processing_time: float = 0.0


class BatchPDFProcessor:
    """
    대규모 PDF 배치 처리 파이프라인
    - data/raw의 PDF를 벡터 DB로 변환
    - 메모리 효율적인 점진적 처리
    """
    
    def __init__(self, 
                 use_kiwi: bool = True,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 batch_size: int = 10,
                 checkpoint_interval: int = 100):
        """
        Args:
            use_kiwi: Kiwi 텍스트 프로세서 사용 여부
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
            batch_size: 한 번에 처리할 PDF 수
            checkpoint_interval: 체크포인트 저장 간격
        """
        # Vision 파서 초기화
        self.vision_parser = VisionV2Parser()
        
        # 텍스트 프로세서 선택
        if use_kiwi and KIWI_AVAILABLE:
            logger.info("Using KiwiTextProcessor for text processing")
            self.text_processor = TextProcessor(
                num_workers=4,  # 병렬 처리
                apply_spacing=True,
                apply_normalization=True
            )
            self.using_kiwi = True
        else:
            if use_kiwi and not KIWI_AVAILABLE:
                logger.warning("Kiwi requested but not available, using basic processor")
            logger.info("Using basic KoreanEnglishTextProcessor")
            self.text_processor = TextProcessor()
            self.using_kiwi = False
        
        # 청킹 및 임베딩
        # Vision V2는 마크다운을 생성하므로 계층적 청킹 우선 사용
        if LANGCHAIN_AVAILABLE and VISION_AVAILABLE:
            logger.info("Using HierarchicalMarkdownChunker for Vision V2 markdown output")
            self.chunker = HierarchicalMarkdownChunker(
                use_chunk_cleaner=True,
                enable_semantic=False
            )
        else:
            logger.info(f"Using basic DocumentChunker (langchain: {LANGCHAIN_AVAILABLE}, vision: {VISION_AVAILABLE})")
            self.chunker = DocumentChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        self.embedder = EmbeddingGenerator()
        
        # 지식 베이스
        self.knowledge_base = KnowledgeBase()
        
        # 배치 처리 설정
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        
        # 처리 통계
        self.stats = {
            'total_files': 0,
            'success': 0,
            'partial': 0,
            'failed': 0,
            'total_chunks': 0,
            'total_embeddings': 0
        }
    
    def process_pdf_directory(self, 
                            pdf_dir: str = "data/raw",
                            output_dir: str = "data/processed",
                            resume_from: Optional[str] = None) -> List[ProcessingResult]:
        """
        디렉토리의 모든 PDF 처리
        
        Args:
            pdf_dir: PDF 파일 디렉토리
            output_dir: 처리 결과 저장 디렉토리
            resume_from: 재개할 파일 경로 (체크포인트)
        """
        # 디렉토리 확인
        pdf_path = Path(pdf_dir)
        if not pdf_path.exists():
            raise ValueError(f"PDF directory not found: {pdf_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # PDF 파일 목록
        pdf_files = sorted(pdf_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # 재개 지점 찾기
        start_idx = 0
        if resume_from:
            for i, pdf_file in enumerate(pdf_files):
                if str(pdf_file) == resume_from:
                    start_idx = i + 1
                    logger.info(f"Resuming from file {start_idx}/{len(pdf_files)}")
                    break
        
        # 배치 처리
        results = []
        for batch_start in range(start_idx, len(pdf_files), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(pdf_files))
            batch_files = pdf_files[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start+1}-{batch_end}/{len(pdf_files)}")
            
            # 배치 처리
            batch_results = self._process_batch(batch_files)
            results.extend(batch_results)
            
            # 체크포인트 저장
            if (batch_end % self.checkpoint_interval) == 0:
                self._save_checkpoint(output_dir, batch_end, results)
            
            # 메모리 정리
            self._cleanup_memory()
        
        # 최종 체크포인트
        self._save_checkpoint(output_dir, len(pdf_files), results)
        
        # 통계 출력
        self._print_stats()
        
        return results
    
    def process_single_pdf(self, pdf_path: str) -> ProcessingResult:
        """
        단일 PDF 파일 처리
        
        Args:
            pdf_path: PDF 파일 경로
        """
        import time
        start_time = time.time()
        
        try:
            # 1. PDF 파싱 (Vision V2)
            logger.info(f"Parsing PDF: {pdf_path}")
            parsed_content = self.vision_parser.parse_pdf(pdf_path)
            
            if not parsed_content or not parsed_content.get('text'):
                raise ValueError("No text extracted from PDF")
            
            # 2. 텍스트 전처리
            logger.info("Processing text...")
            processed_text = self.text_processor.process(
                parsed_content['text'],
                preserve_paragraphs=True,
                financial_processing=True
            )
            
            # 3. 청킹
            logger.info("Creating chunks...")
            chunks = self.chunker.chunk_text(processed_text)
            
            if not chunks:
                raise ValueError("No chunks created from text")
            
            # 4. 임베딩 생성
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedder.embed_batch(chunks)
            
            # 5. 지식 베이스에 추가
            doc_ids = [f"{Path(pdf_path).stem}_chunk_{i}" for i in range(len(chunks))]
            self.knowledge_base.add_documents(chunks, embeddings, doc_ids)
            
            # 통계 업데이트
            self.stats['success'] += 1
            self.stats['total_chunks'] += len(chunks)
            self.stats['total_embeddings'] += len(embeddings)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                file_path=pdf_path,
                status='success',
                chunks_created=len(chunks),
                embeddings_created=len(embeddings),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            self.stats['failed'] += 1
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                file_path=pdf_path,
                status='failed',
                chunks_created=0,
                embeddings_created=0,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _process_batch(self, pdf_files: List[Path]) -> List[ProcessingResult]:
        """배치 처리"""
        results = []
        
        for pdf_file in pdf_files:
            self.stats['total_files'] += 1
            result = self.process_single_pdf(str(pdf_file))
            results.append(result)
            
            # 진행 상황 출력
            if result.status == 'success':
                logger.info(f"✅ Processed: {pdf_file.name} "
                          f"({result.chunks_created} chunks, "
                          f"{result.processing_time:.1f}s)")
            else:
                logger.error(f"❌ Failed: {pdf_file.name} - {result.error_message}")
        
        return results
    
    def _cleanup_memory(self):
        """메모리 정리"""
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Vision 모델 주기적 언로드 (메모리 절약)
        if hasattr(self.vision_parser, 'unload_model'):
            if self.stats['total_files'] % 20 == 0:
                logger.info("Unloading Vision model to free memory...")
                self.vision_parser.unload_model()
    
    def _save_checkpoint(self, output_dir: str, processed_count: int, results: List[ProcessingResult]):
        """체크포인트 저장"""
        import json
        
        checkpoint_path = Path(output_dir) / f"checkpoint_{processed_count}.json"
        
        checkpoint_data = {
            'processed_count': processed_count,
            'stats': self.stats,
            'using_kiwi': self.using_kiwi,
            'results': [
                {
                    'file_path': r.file_path,
                    'status': r.status,
                    'chunks_created': r.chunks_created,
                    'embeddings_created': r.embeddings_created,
                    'error_message': r.error_message,
                    'processing_time': r.processing_time
                }
                for r in results
            ]
        }
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # 지식 베이스 저장
        if hasattr(self.knowledge_base, 'save'):
            kb_path = Path(output_dir) / f"knowledge_base_{processed_count}.pkl"
            self.knowledge_base.save(str(kb_path))
    
    def _print_stats(self):
        """통계 출력"""
        logger.info("\n" + "="*50)
        logger.info("📊 Processing Statistics")
        logger.info("="*50)
        logger.info(f"Total Files: {self.stats['total_files']}")
        logger.info(f"Success: {self.stats['success']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Total Chunks: {self.stats['total_chunks']}")
        logger.info(f"Total Embeddings: {self.stats['total_embeddings']}")
        
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['success'] / self.stats['total_files']) * 100
            avg_chunks = self.stats['total_chunks'] / max(self.stats['success'], 1)
            logger.info(f"Success Rate: {success_rate:.1f}%")
            logger.info(f"Avg Chunks per File: {avg_chunks:.1f}")
        
        logger.info(f"Text Processor: {'Kiwi' if self.using_kiwi else 'Basic'}")
        logger.info("="*50)


# 스크립트로 직접 실행
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch PDF Processing Pipeline")
    parser.add_argument("--pdf-dir", default="data/raw", help="PDF directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--no-kiwi", action="store_true", help="Disable Kiwi processor")
    parser.add_argument("--resume", help="Resume from checkpoint file")
    
    args = parser.parse_args()
    
    # 프로세서 초기화
    processor = BatchPDFProcessor(
        use_kiwi=not args.no_kiwi,
        batch_size=args.batch_size
    )
    
    # 처리 실행
    results = processor.process_pdf_directory(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        resume_from=args.resume
    )
    
    print(f"\n✅ Processing complete: {len(results)} files processed")