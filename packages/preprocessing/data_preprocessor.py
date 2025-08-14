"""
Data Preprocessor Component
Architecture.md 기준 DataPreprocessor 인터페이스 구현
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pymupdf  # PyMuPDF
import pymupdf4llm  # PyMuPDF4LLM for better extraction
from bs4 import BeautifulSoup
import re
import os
import json
from datetime import datetime
import logging
from .text_cleaner import TextCleaner
from .pdf_processor_traditional import AdvancedPDFProcessor
from .pdf_processor_vision import VisionPDFProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """처리된 문서 데이터 모델"""
    doc_id: str
    source_path: str
    content: str
    metadata: Dict[str, Any]
    chunks: Optional[List[str]] = None
    processed_at: Optional[str] = None


class IDataPreprocessor(ABC):
    """데이터 전처리 인터페이스 (Architecture.md 정의)"""
    
    @abstractmethod
    def process_pdf(self, file_path: str) -> ProcessedDocument:
        """PDF 파일 처리"""
        pass
    
    @abstractmethod
    def process_html(self, file_path: str) -> ProcessedDocument:
        """HTML 파일 처리"""
        pass
    
    @abstractmethod
    def process_text(self, file_path: str) -> ProcessedDocument:
        """텍스트 파일 처리"""
        pass
    
    @abstractmethod
    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        pass


class DataPreprocessor(IDataPreprocessor):
    """
    Epic 1.2 데이터 전처리 컴포넌트 구현
    - Vision-Language 모델을 이용한 고품질 PDF 파싱 (메인)
    - PyMuPDF를 이용한 PDF 파싱 (fallback)
    - BeautifulSoup4를 이용한 HTML 파싱
    - 국영문 혼합 텍스트 정제
    """
    
    def __init__(self, output_dir: str = "data/processed", 
                 use_text_cleaner: bool = True,
                 use_vision_pdf: bool = True,
                 use_advanced_pdf: bool = True):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/metadata", exist_ok=True)
        self.text_cleaner = TextCleaner(aggressive=False) if use_text_cleaner else None
        
        # Vision PDF 프로세서 (메인)
        self.use_vision_pdf = use_vision_pdf
        self.vision_processor = None
        if use_vision_pdf:
            try:
                self.vision_processor = VisionPDFProcessor(
                    use_markdown=True,
                    extract_tables=True,
                    preserve_layout=True
                )
                if not self.vision_processor.is_available():
                    logger.warning("Vision PDF processor not available (GPU/dependencies missing)")
                    self.vision_processor = None
                    self.use_vision_pdf = False
            except Exception as e:
                logger.warning(f"Failed to initialize Vision PDF processor: {e}")
                self.vision_processor = None
                self.use_vision_pdf = False
        
        # Traditional PDF 프로세서 (fallback)
        self.use_advanced_pdf = use_advanced_pdf
        self.pdf_processor = None
        if use_advanced_pdf:
            try:
                self.pdf_processor = AdvancedPDFProcessor(
                    use_markdown=True,
                    extract_tables=True,
                    preserve_layout=True
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Traditional PDF processor: {e}")
                self.pdf_processor = None
                self.use_advanced_pdf = False
        
    def process_pdf(self, file_path: str) -> ProcessedDocument:
        """
        PDF 파일에서 텍스트와 표 추출
        Pipeline.md 1.2.3 요구사항 구현
        Vision 모델 우선 → Traditional PDF 프로세서 → 기본 추출 순서
        """
        logger.info(f"Processing PDF: {file_path}")
        
        combined_text = None
        metadata = {}
        extraction_method = "basic_pymupdf"
        
        # 1순위: Vision PDF 프로세서 (41.2% 개선 검증됨)
        if self.use_vision_pdf and self.vision_processor:
            try:
                logger.info("Attempting Vision PDF extraction (primary method)...")
                result = self.vision_processor.extract_pdf(file_path)
                
                # 텍스트 선택 (Markdown 우선)
                combined_text = result.markdown if result.markdown else result.text
                
                # 메타데이터 구성
                metadata = result.metadata.copy()
                metadata["tables"] = result.tables
                metadata["toc"] = result.toc
                metadata["extraction_method"] = "vision_v2"
                extraction_method = "vision_v2"
                
                logger.info(f"Vision extraction successful: {len(combined_text):,} characters")
                
            except Exception as e:
                logger.warning(f"Vision PDF processing failed: {e}")
                logger.info("Falling back to Traditional PDF processor...")
                combined_text = None
        
        # 2순위: Traditional PDF 프로세서 (fallback)
        if combined_text is None and self.use_advanced_pdf and self.pdf_processor:
            try:
                logger.info("Attempting Traditional PDF extraction (fallback method)...")
                result = self.pdf_processor.extract_pdf(file_path)
                
                # 텍스트 선택 (Markdown 우선)
                combined_text = result.markdown if result.markdown else result.text
                
                # 메타데이터 구성
                metadata = result.metadata.copy()
                metadata["tables"] = result.tables
                metadata["toc"] = result.toc
                metadata["extraction_method"] = "pymupdf4llm_fallback"
                extraction_method = "pymupdf4llm_fallback"
                
                logger.info(f"Traditional extraction successful: {len(combined_text):,} characters")
                
            except Exception as e:
                logger.warning(f"Traditional PDF processing failed: {e}")
                logger.info("Falling back to basic PDF extraction...")
                combined_text = None
        
        # 3순위: 기본 PDF 추출 (최종 fallback)
        if combined_text is None:
            logger.info("Using basic PDF extraction (final fallback)...")
            combined_text, metadata = self._basic_pdf_extraction(file_path)
            extraction_method = "basic_pymupdf_final_fallback"
        
        # 텍스트 정제
        if self.text_cleaner:
            cleaned_text = self.text_cleaner.clean_text(combined_text)
        else:
            cleaned_text = self.clean_text(combined_text)
        
        # ProcessedDocument 생성
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # 추출 방법 로그
        logger.info(f"PDF extraction completed using: {extraction_method}")
        
        processed_doc = ProcessedDocument(
            doc_id=doc_id,
            source_path=file_path,
            content=cleaned_text,
            metadata=metadata,
            processed_at=datetime.now().isoformat()
        )
        
        # 메타데이터 저장
        self._save_metadata(processed_doc)
        
        return processed_doc
    
    def process_html(self, file_path: str) -> ProcessedDocument:
        """
        HTML 파일 파싱 및 정제
        Pipeline.md 1.2.4 요구사항 구현
        """
        logger.info(f"Processing HTML: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 스크립트와 스타일 제거
        for script in soup(["script", "style"]):
            script.decompose()
        
        # 텍스트 추출
        text = soup.get_text()
        
        # 메타데이터 추출
        metadata = {
            "title": soup.title.string if soup.title else "",
            "meta_description": "",
            "meta_keywords": "",
            "links": [],
            "headings": []
        }
        
        # 메타 태그 정보 추출
        for meta in soup.find_all('meta'):
            if meta.get('name') == 'description':
                metadata["meta_description"] = meta.get('content', '')
            elif meta.get('name') == 'keywords':
                metadata["meta_keywords"] = meta.get('content', '')
        
        # 헤딩 추출
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            metadata["headings"].append({
                "level": heading.name,
                "text": heading.get_text().strip()
            })
        
        # 링크 추출
        for link in soup.find_all('a', href=True):
            metadata["links"].append(link['href'])
        
        # 텍스트 정제
        cleaned_text = self.clean_text(text)
        
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        processed_doc = ProcessedDocument(
            doc_id=doc_id,
            source_path=file_path,
            content=cleaned_text,
            metadata=metadata,
            processed_at=datetime.now().isoformat()
        )
        
        self._save_metadata(processed_doc)
        
        return processed_doc
    
    def process_text(self, file_path: str) -> ProcessedDocument:
        """일반 텍스트 파일 처리"""
        logger.info(f"Processing text file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned_text = self.clean_text(content)
        
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        processed_doc = ProcessedDocument(
            doc_id=doc_id,
            source_path=file_path,
            content=cleaned_text,
            metadata={"file_size": os.path.getsize(file_path)},
            processed_at=datetime.now().isoformat()
        )
        
        self._save_metadata(processed_doc)
        
        return processed_doc
    
    def clean_text(self, text: str) -> str:
        """
        국영문 혼합 텍스트 정제
        Pipeline.md 1.2.5 KoreanEnglishTextProcessor 구현
        """
        # 기존 text_processor.py의 로직 통합
        # 금융 관련 특수문자 정규화
        text = re.sub(r'％', '%', text)
        text = re.sub(r'＄', '$', text)
        text = re.sub(r'￦', '원', text)
        
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        
        # 불필요한 줄바꿈 제거
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 특수문자 주변 공백 정리
        text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
        
        # 괄호 주변 공백 정리
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def _basic_pdf_extraction(self, file_path: str) -> tuple:
        """기본 PDF 추출 (fallback)"""
        doc = pymupdf.open(file_path)
        full_text = []
        metadata = {
            "page_count": len(doc),
            "file_size": os.path.getsize(file_path),
            "tables": [],
            "title": doc.metadata.get('title', ''),
            "author": doc.metadata.get('author', ''),
            "subject": doc.metadata.get('subject', ''),
            "keywords": doc.metadata.get('keywords', ''),
            "extraction_method": "basic_pymupdf"
        }
        
        for page_num, page in enumerate(doc):
            # 텍스트 추출
            text = page.get_text()
            full_text.append(f"\n[Page {page_num + 1}]\n{text}")
            
            # 표 추출 시도
            tables = page.find_tables()
            if tables:
                for table_idx, table in enumerate(tables):
                    try:
                        table_data = table.extract()
                        metadata["tables"].append({
                            "page": page_num + 1,
                            "table_index": table_idx,
                            "rows": len(table_data),
                            "data": table_data
                        })
                    except Exception as e:
                        logger.warning(f"Table extraction failed on page {page_num + 1}: {e}")
        
        doc.close()
        combined_text = "\n".join(full_text)
        return combined_text, metadata
    
    def _save_metadata(self, doc: ProcessedDocument):
        """처리된 문서의 메타데이터 저장"""
        metadata_path = os.path.join(
            self.output_dir, 
            "metadata", 
            f"{doc.doc_id}_metadata.json"
        )
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "doc_id": doc.doc_id,
                "source_path": doc.source_path,
                "processed_at": doc.processed_at,
                "metadata": doc.metadata,
                "content_length": len(doc.content)
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metadata saved: {metadata_path}")
    
    def process_directory(self, directory: str) -> List[ProcessedDocument]:
        """디렉토리 내 모든 문서 처리"""
        processed_docs = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.endswith('.pdf'):
                        doc = self.process_pdf(file_path)
                        processed_docs.append(doc)
                    elif file.endswith('.html'):
                        doc = self.process_html(file_path)
                        processed_docs.append(doc)
                    elif file.endswith('.txt'):
                        doc = self.process_text(file_path)
                        processed_docs.append(doc)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        
        return processed_docs


if __name__ == "__main__":
    # 테스트 실행
    preprocessor = DataPreprocessor()
    
    # PDF 파일 처리 테스트
    pdf_path = "금융분야 AI 보안 가이드라인.pdf"
    if os.path.exists(pdf_path):
        result = preprocessor.process_pdf(pdf_path)
        print(f"Processed PDF: {result.doc_id}")
        print(f"Content length: {len(result.content)} chars")
        print(f"Page count: {result.metadata.get('page_count')}")
        print(f"Tables found: {len(result.metadata.get('tables', []))}")