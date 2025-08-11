#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced PDF Processor using PyMuPDF4LLM
레이아웃 보존 및 구조화된 텍스트 추출
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pymupdf
import pymupdf4llm
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class PDFExtractionResult:
    """PDF 추출 결과"""
    text: str
    markdown: str
    metadata: Dict[str, Any]
    tables: List[Dict[str, Any]]
    toc: List[Tuple[int, str, int]]  # (level, title, page)
    page_texts: List[str]
    
    
class AdvancedPDFProcessor:
    """PyMuPDF4LLM을 사용한 고급 PDF 처리기"""
    
    def __init__(self, 
                 use_markdown: bool = True,
                 extract_tables: bool = True,
                 extract_images: bool = False,
                 preserve_layout: bool = True):
        """
        Args:
            use_markdown: Markdown 형식으로 변환
            extract_tables: 표 추출 여부
            extract_images: 이미지 추출 여부 (현재는 False 권장)
            preserve_layout: 레이아웃 보존 여부
        """
        self.use_markdown = use_markdown
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.preserve_layout = preserve_layout
        
    def extract_pdf(self, pdf_path: str) -> PDFExtractionResult:
        """
        PDF에서 구조화된 텍스트 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            PDFExtractionResult 객체
        """
        logger.info(f"Processing PDF with PyMuPDF4LLM: {pdf_path}")
        
        # 파일 존재 확인
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # 기본 메타데이터
        metadata = {
            'file_path': pdf_path,
            'file_name': os.path.basename(pdf_path),
            'file_size': os.path.getsize(pdf_path),
        }
        
        # PyMuPDF로 문서 열기
        doc = pymupdf.open(pdf_path)
        
        # 문서 메타데이터 추가
        metadata.update({
            'page_count': len(doc),
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'keywords': doc.metadata.get('keywords', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
        })
        
        # 목차 추출
        toc = self._extract_toc(doc)
        
        # 페이지별 텍스트 추출
        page_texts = []
        tables = []
        
        for page_num, page in enumerate(doc):
            # 페이지 텍스트 추출
            page_text = page.get_text()
            page_texts.append(page_text)
            
            # 표 추출
            if self.extract_tables:
                page_tables = self._extract_page_tables(page, page_num)
                tables.extend(page_tables)
        
        doc.close()
        
        # PyMuPDF4LLM으로 Markdown 변환
        markdown_text = ""
        plain_text = ""
        
        if self.use_markdown:
            try:
                # PyMuPDF4LLM의 to_markdown 함수 사용
                markdown_result = pymupdf4llm.to_markdown(
                    pdf_path,
                    page_chunks=True,  # 페이지 단위로 청킹
                    write_images=self.extract_images,
                    image_path="images/",
                    image_format="png",
                    dpi=150
                )
                
                # 결과가 리스트인 경우 처리
                if isinstance(markdown_result, list):
                    # 페이지별 마크다운을 하나로 합침
                    markdown_text = "\n\n---\n\n".join([page.get('text', '') for page in markdown_result if isinstance(page, dict)])
                else:
                    markdown_text = str(markdown_result)
                
                # Markdown에서 일반 텍스트 추출
                plain_text = self._markdown_to_plain_text(markdown_text)
                
            except Exception as e:
                logger.warning(f"Markdown conversion failed: {e}")
                # Fallback to plain text
                plain_text = "\n\n".join(page_texts)
                markdown_text = plain_text
        else:
            plain_text = "\n\n".join(page_texts)
            markdown_text = plain_text
        
        # 텍스트 후처리
        plain_text = self._post_process_text(plain_text)
        markdown_text = self._post_process_markdown(markdown_text)
        
        return PDFExtractionResult(
            text=plain_text,
            markdown=markdown_text,
            metadata=metadata,
            tables=tables,
            toc=toc,
            page_texts=page_texts
        )
    
    def _extract_toc(self, doc: pymupdf.Document) -> List[Tuple[int, str, int]]:
        """목차 추출"""
        toc = []
        try:
            raw_toc = doc.get_toc()
            for item in raw_toc:
                level, title, page = item[0], item[1], item[2]
                toc.append((level, title, page))
        except Exception as e:
            logger.debug(f"TOC extraction failed: {e}")
        return toc
    
    def _extract_page_tables(self, page: pymupdf.Page, page_num: int) -> List[Dict[str, Any]]:
        """페이지에서 표 추출"""
        tables = []
        try:
            # PyMuPDF의 find_tables 메서드 사용
            page_tables = page.find_tables()
            for idx, table in enumerate(page_tables):
                try:
                    table_data = table.extract()
                    if table_data:
                        tables.append({
                            'page': page_num + 1,
                            'table_index': idx,
                            'rows': len(table_data),
                            'cols': len(table_data[0]) if table_data else 0,
                            'data': table_data
                        })
                except Exception as e:
                    logger.debug(f"Table extraction failed: {e}")
        except Exception as e:
            logger.debug(f"No tables found on page {page_num + 1}: {e}")
        return tables
    
    def _markdown_to_plain_text(self, markdown: str) -> str:
        """Markdown을 일반 텍스트로 변환"""
        # 제목 마커 제거
        text = re.sub(r'^#{1,6}\s+', '', markdown, flags=re.MULTILINE)
        
        # 굵은 글씨, 이탤릭 제거
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # 링크 제거
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # 코드 블록 제거
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # 리스트 마커 정리
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # 수평선 제거
        text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _post_process_text(self, text: str) -> str:
        """텍스트 후처리"""
        # 과도한 공백 제거
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # 페이지 마커 정리
        text = re.sub(r'\[Page \d+\]', '', text)
        text = re.sub(r'Page \d+ of \d+', '', text)
        
        return text.strip()
    
    def _post_process_markdown(self, markdown: str) -> str:
        """Markdown 후처리"""
        # 빈 제목 제거
        markdown = re.sub(r'^#{1,6}\s*$', '', markdown, flags=re.MULTILINE)
        
        # 과도한 줄바꿈 제거
        markdown = re.sub(r'\n{4,}', '\n\n\n', markdown)
        
        return markdown.strip()
    
    def extract_specific_sections(self, 
                                 pdf_path: str, 
                                 section_titles: List[str]) -> Dict[str, str]:
        """
        특정 섹션만 추출
        
        Args:
            pdf_path: PDF 파일 경로
            section_titles: 추출할 섹션 제목 리스트
            
        Returns:
            섹션별 텍스트 딕셔너리
        """
        result = self.extract_pdf(pdf_path)
        sections = {}
        
        # Markdown 텍스트에서 섹션 추출
        markdown = result.markdown
        
        for title in section_titles:
            # 섹션 찾기 (대소문자 무시)
            pattern = rf'(?i)#{1,6}\s*{re.escape(title)}.*?(?=#{1,6}|\Z)'
            match = re.search(pattern, markdown, re.DOTALL)
            
            if match:
                section_text = match.group(0)
                # 일반 텍스트로 변환
                section_text = self._markdown_to_plain_text(section_text)
                sections[title] = section_text
            else:
                logger.warning(f"Section not found: {title}")
                sections[title] = ""
        
        return sections
    
    def extract_with_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        구조화된 형태로 PDF 추출
        
        Returns:
            계층적 구조의 문서 딕셔너리
        """
        result = self.extract_pdf(pdf_path)
        
        # 목차를 기반으로 구조화
        structured = {
            'metadata': result.metadata,
            'toc': [],
            'content': {},
            'tables': result.tables
        }
        
        # 목차 구조화
        for level, title, page in result.toc:
            structured['toc'].append({
                'level': level,
                'title': title,
                'page': page
            })
        
        # 섹션별 컨텐츠 구조화
        if result.toc:
            for i, (level, title, start_page) in enumerate(result.toc):
                # 다음 섹션의 시작 페이지 찾기
                if i + 1 < len(result.toc):
                    end_page = result.toc[i + 1][2]
                else:
                    end_page = len(result.page_texts)
                
                # 해당 섹션의 텍스트 추출
                section_text = "\n".join(
                    result.page_texts[start_page-1:end_page-1]
                )
                
                structured['content'][title] = {
                    'level': level,
                    'pages': f"{start_page}-{end_page}",
                    'text': section_text
                }
        else:
            # 목차가 없으면 전체 텍스트
            structured['content']['full_text'] = result.text
        
        return structured


# 사용 예시
if __name__ == "__main__":
    processor = AdvancedPDFProcessor(
        use_markdown=True,
        extract_tables=True,
        preserve_layout=True
    )
    
    # 테스트 PDF
    pdf_path = "금융분야 AI 보안 가이드라인.pdf"
    
    if os.path.exists(pdf_path):
        print("PDF 처리 중...")
        result = processor.extract_pdf(pdf_path)
        
        print(f"\n=== PDF 메타데이터 ===")
        print(f"페이지 수: {result.metadata['page_count']}")
        print(f"제목: {result.metadata['title']}")
        print(f"저자: {result.metadata['author']}")
        
        print(f"\n=== 목차 (TOC) ===")
        for level, title, page in result.toc[:5]:
            indent = "  " * (level - 1)
            print(f"{indent}{title} ... {page}")
        
        print(f"\n=== 추출된 표 ===")
        print(f"총 {len(result.tables)}개 표 발견")
        
        print(f"\n=== 텍스트 샘플 ===")
        print("Plain Text (처음 500자):")
        print(result.text[:500])
        
        print("\n\nMarkdown (처음 500자):")
        print(result.markdown[:500])
    else:
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")