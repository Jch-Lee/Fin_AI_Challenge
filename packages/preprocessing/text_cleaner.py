#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TextCleaner: 텍스트 정제 및 메타데이터 제거 모듈

금융 문서의 메타데이터, 특수문자, 노이즈를 제거하여
LLM이 깨끗한 텍스트로 학습할 수 있도록 전처리
"""

import re
import unicodedata
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    """텍스트 정제 및 메타데이터 제거 클래스"""
    
    def __init__(self, aggressive: bool = False):
        """
        Args:
            aggressive: True일 경우 더 공격적인 정제 수행
        """
        self.aggressive = aggressive
        
        # 제거할 반복 기호 패턴
        self.repetitive_patterns = [
            r'[·․．\.\-_]{3,}',  # 점선 리더 (목차에서 자주 사용)
            r'[\=]{3,}',         # 연속된 등호
            r'[\*]{3,}',         # 연속된 별표
            r'[\#]{3,}',         # 연속된 샵
            r'[\~]{3,}',         # 연속된 물결
            r'[\s]{5,}',         # 과도한 공백
        ]
        
        # 메타데이터 패턴
        self.metadata_patterns = [
            # 페이지 번호 패턴
            r'\[\s*Page\s*\d+\s*\]',
            r'\[\s*페이지\s*\d+\s*\]',
            r'^\s*\d+\s*$',  # 줄 전체가 숫자만
            r'^\s*-\s*\d+\s*-\s*$',  # - 1 - 형태
            r'페이지\s*\d+\s*/\s*\d+',  # 페이지 1/10 형태
            
            # 목차 패턴
            r'^목\s*차$',
            r'^차\s*례$',
            r'^CONTENTS?$',
            r'^Table\s+of\s+Contents?$',
            
            # 머리말/꼬리말
            r'^부\s*록$',
            r'^참\s*고\s*문\s*헌$',
            r'^색\s*인$',
            r'^INDEX$',
        ]
        
        # 특수 공백 문자들
        self.special_spaces = {
            '\u200b': '',  # Zero-width space
            '\u00a0': ' ',  # Non-breaking space  
            '\u3000': ' ',  # Ideographic space
            '\u2009': ' ',  # Thin space
            '\u2003': ' ',  # Em space
            '\u2002': ' ',  # En space
            '\ufeff': '',  # Zero-width no-break space
        }
    
    def clean_text(self, text: str) -> str:
        """
        텍스트 전체 정제 파이프라인
        
        Args:
            text: 정제할 텍스트
            
        Returns:
            정제된 텍스트
        """
        if not text:
            return ""
        
        # 1단계: 유니코드 정규화
        text = self._normalize_unicode(text)
        
        # 2단계: 특수 공백 치환
        text = self._replace_special_spaces(text)
        
        # 3단계: 반복 기호 제거
        text = self._remove_repetitive_symbols(text)
        
        # 4단계: 메타데이터 제거
        text = self._remove_metadata(text)
        
        # 5단계: 목차 점선 제거
        text = self._remove_toc_leaders(text)
        
        # 6단계: 불필요한 공백 정리
        text = self._clean_whitespace(text)
        
        # 7단계: 한글 자소 분리 수정 (옵션)
        if self.aggressive:
            text = self._fix_korean_jamo(text)
        
        # 8단계: 최종 트리밍
        text = text.strip()
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """유니코드 정규화 (NFC)"""
        return unicodedata.normalize('NFC', text)
    
    def _replace_special_spaces(self, text: str) -> str:
        """특수 공백 문자를 일반 공백으로 치환"""
        for special, normal in self.special_spaces.items():
            text = text.replace(special, normal)
        return text
    
    def _remove_repetitive_symbols(self, text: str) -> str:
        """반복되는 기호 제거"""
        for pattern in self.repetitive_patterns:
            text = re.sub(pattern, ' ', text)
        return text
    
    def _remove_metadata(self, text: str) -> str:
        """메타데이터 패턴 제거"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 메타데이터 패턴 체크
            is_metadata = False
            for pattern in self.metadata_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    is_metadata = True
                    break
            
            if not is_metadata:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _remove_toc_leaders(self, text: str) -> str:
        """목차의 점선 리더 제거 (페이지 번호 연결선)"""
        # 패턴: 텍스트...........숫자
        pattern = r'([^\.\s]+)\s*\.{2,}\s*(\d+)'
        text = re.sub(pattern, r'\1 \2', text)
        
        # 패턴: 텍스트 ···········숫자
        pattern = r'([^·\s]+)\s*·{2,}\s*(\d+)'
        text = re.sub(pattern, r'\1 \2', text)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """불필요한 공백 정리"""
        # 연속된 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        
        # 연속된 줄바꿈을 최대 2개로
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 각 줄의 앞뒤 공백 제거
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def _fix_korean_jamo(self, text: str) -> str:
        """한글 자소 분리 문제 수정 (aggressive 모드)"""
        # 자소가 분리된 패턴 찾기 (예: ㄱ ㅏ ㅇ)
        jamo_pattern = r'([ㄱ-ㅎ])\s+([ㅏ-ㅣ])'
        text = re.sub(jamo_pattern, r'\1\2', text)
        
        return text
    
    def clean_for_chunking(self, text: str) -> str:
        """
        청킹을 위한 특별 정제
        청킹 시 문맥을 해치지 않으면서 노이즈 제거
        """
        # 기본 정제
        text = self.clean_text(text)
        
        # 청킹을 위한 추가 처리
        # 문단 구분 보존
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text
    
    def clean_for_llm(self, text: str) -> str:
        """
        LLM 입력을 위한 특별 정제
        LLM이 혼동하지 않도록 더 공격적으로 정제
        """
        # 공격적 모드로 정제
        old_aggressive = self.aggressive
        self.aggressive = True
        text = self.clean_text(text)
        self.aggressive = old_aggressive
        
        # LLM을 위한 추가 처리
        # 번호 매기기 정리
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[가-하]\.\s+', '', text, flags=re.MULTILINE)
        
        # 괄호 안 참조 제거 (옵션)
        if self.aggressive:
            text = re.sub(r'\([^)]*\d+[^)]*\)', '', text)  # (참조: 123) 형태 제거
        
        return text
    
    def extract_clean_sentences(self, text: str) -> List[str]:
        """
        깨끗한 문장 단위로 추출
        
        Returns:
            정제된 문장 리스트
        """
        # 기본 정제
        text = self.clean_text(text)
        
        # 문장 분리 (한국어 문장 종결 고려)
        sentences = re.split(r'[.!?。]\s+', text)
        
        # 각 문장 정제
        clean_sentences = []
        for sent in sentences:
            sent = sent.strip()
            # 최소 길이 체크 (10자 이상)
            if len(sent) >= 10:
                clean_sentences.append(sent)
        
        return clean_sentences
    
    def is_likely_metadata(self, text: str) -> bool:
        """
        텍스트가 메타데이터일 가능성 체크
        
        Returns:
            메타데이터일 가능성이 높으면 True
        """
        text = text.strip()
        
        # 너무 짧은 텍스트
        if len(text) < 5:
            return True
        
        # 숫자만 있는 경우
        if text.isdigit():
            return True
        
        # 특수문자 비율이 높은 경우
        special_chars = len(re.findall(r'[^\w\s가-힣a-zA-Z0-9]', text))
        if special_chars / max(len(text), 1) > 0.5:
            return True
        
        # 메타데이터 패턴 매칭
        for pattern in self.metadata_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def get_cleaning_stats(self, original: str, cleaned: str) -> dict:
        """
        정제 통계 반환
        
        Returns:
            정제 전후 비교 통계
        """
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'reduction_rate': 1 - (len(cleaned) / max(len(original), 1)),
            'original_lines': original.count('\n'),
            'cleaned_lines': cleaned.count('\n'),
            'removed_chars': len(original) - len(cleaned)
        }


class ChunkCleaner(TextCleaner):
    """청크 단위 텍스트 정제 특화 클래스"""
    
    def clean_chunk(self, chunk: str) -> Optional[str]:
        """
        개별 청크 정제
        
        Args:
            chunk: 정제할 청크
            
        Returns:
            정제된 청크 또는 None (제거해야 할 경우)
        """
        # 메타데이터 청크는 제거
        if self.is_likely_metadata(chunk):
            logger.debug(f"메타데이터 청크 제거: {chunk[:50]}...")
            return None
        
        # 기본 정제
        cleaned = self.clean_for_chunking(chunk)
        
        # 너무 짧아진 청크는 제거
        if len(cleaned) < 50:
            return None
        
        return cleaned
    
    def clean_chunks(self, chunks: List[str]) -> List[str]:
        """
        청크 리스트 일괄 정제
        
        Args:
            chunks: 청크 리스트
            
        Returns:
            정제된 청크 리스트
        """
        cleaned_chunks = []
        
        for chunk in chunks:
            cleaned = self.clean_chunk(chunk)
            if cleaned:
                cleaned_chunks.append(cleaned)
        
        logger.info(f"청크 정제: {len(chunks)} -> {len(cleaned_chunks)} 개")
        
        return cleaned_chunks


# 사용 예시
if __name__ == "__main__":
    # 테스트
    cleaner = TextCleaner(aggressive=False)
    
    # 문제가 되는 텍스트 예시
    problem_text = """
    목 차
    
    제1장 서론 ················································· 1
    제2장 보안 요구사항 ···································· 15
    
    [Page 1]
    
    1. 개요
    
    금융분야 AI 보안 가이드라인은...
    
    ···············································8 2···············································
    
    참고문헌
    """
    
    cleaned = cleaner.clean_text(problem_text)
    print("원본:")
    print(problem_text)
    print("\n정제 후:")
    print(cleaned)
    
    # 통계
    stats = cleaner.get_cleaning_stats(problem_text, cleaned)
    print("\n정제 통계:")
    for key, value in stats.items():
        print(f"  {key}: {value}")