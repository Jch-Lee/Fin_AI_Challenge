#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
다양성 최대화 청크 샘플링 모듈
"""

import random
import logging
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DiversitySampler:
    """다양성을 최대화하는 청크 샘플러"""
    
    def __init__(self, chunks: List[Dict]):
        """
        Args:
            chunks: 전체 청크 리스트
        """
        self.chunks = chunks
        self.chunks_by_document = self._group_by_document()
        self.chunks_by_length = self._group_by_length()
        
        logger.info(f"Initialized sampler with {len(chunks)} chunks from {len(self.chunks_by_document)} documents")
    
    def _group_by_document(self) -> Dict[str, List[Dict]]:
        """문서별로 청크 그룹화"""
        grouped = defaultdict(list)
        for chunk in self.chunks:
            source = chunk.get('source', 'unknown')
            grouped[source].append(chunk)
        return dict(grouped)
    
    def _group_by_length(self) -> Dict[str, List[Dict]]:
        """길이별로 청크 그룹화"""
        grouped = {
            'short': [],    # 50-500자
            'medium': [],   # 500-1500자
            'long': []      # 1500-2300자
        }
        
        for chunk in self.chunks:
            content_length = len(chunk.get('content', ''))
            if content_length < 500:
                grouped['short'].append(chunk)
            elif content_length < 1500:
                grouped['medium'].append(chunk)
            else:
                grouped['long'].append(chunk)
        
        return grouped
    
    def sample_diverse_chunks(self, 
                            n_samples: int = 1000,
                            min_per_document: int = 10) -> List[Dict]:
        """
        다양성을 최대화하여 청크 샘플링
        
        Args:
            n_samples: 샘플링할 청크 수
            min_per_document: 문서당 최소 샘플 수
            
        Returns:
            샘플링된 청크 리스트
        """
        logger.info(f"Sampling {n_samples} diverse chunks...")
        
        sampled_chunks = []
        used_chunk_ids = set()
        
        # 1. 문서별 균등 샘플링 (최소 보장)
        logger.info("Step 1: Document-wise sampling...")
        for doc_name, doc_chunks in tqdm(self.chunks_by_document.items(), desc="Documents"):
            # 각 문서에서 최소 개수만큼 샘플링
            n_from_doc = min(min_per_document, len(doc_chunks))
            
            # 문서 내에서 위치 다양성 고려
            doc_samples = self._sample_from_document(doc_chunks, n_from_doc)
            
            for chunk in doc_samples:
                if chunk['id'] not in used_chunk_ids:
                    sampled_chunks.append(chunk)
                    used_chunk_ids.add(chunk['id'])
        
        logger.info(f"Sampled {len(sampled_chunks)} chunks from document-wise sampling")
        
        # 2. 길이 다양성 기반 추가 샘플링
        remaining_samples = n_samples - len(sampled_chunks)
        if remaining_samples > 0:
            logger.info(f"Step 2: Length-based sampling for {remaining_samples} more chunks...")
            
            # 각 길이 그룹에서 균등하게 샘플링
            samples_per_length = remaining_samples // 3
            
            for length_category, length_chunks in self.chunks_by_length.items():
                available_chunks = [c for c in length_chunks if c['id'] not in used_chunk_ids]
                
                if available_chunks:
                    n_to_sample = min(samples_per_length, len(available_chunks))
                    length_samples = random.sample(available_chunks, n_to_sample)
                    
                    for chunk in length_samples:
                        if chunk['id'] not in used_chunk_ids:
                            sampled_chunks.append(chunk)
                            used_chunk_ids.add(chunk['id'])
        
        # 3. 내용 다양성 기반 필터링
        logger.info("Step 3: Content diversity filtering...")
        sampled_chunks = self._filter_similar_chunks(sampled_chunks)
        
        # 4. 최종 샘플 수 조정
        if len(sampled_chunks) > n_samples:
            sampled_chunks = random.sample(sampled_chunks, n_samples)
        elif len(sampled_chunks) < n_samples:
            # 부족한 경우 랜덤 추가
            remaining = n_samples - len(sampled_chunks)
            available = [c for c in self.chunks if c['id'] not in used_chunk_ids]
            if available:
                additional = random.sample(available, min(remaining, len(available)))
                sampled_chunks.extend(additional)
        
        logger.info(f"Final sample size: {len(sampled_chunks)} chunks")
        
        # 샘플링 통계
        self._print_sampling_statistics(sampled_chunks)
        
        return sampled_chunks
    
    def _sample_from_document(self, doc_chunks: List[Dict], n_samples: int) -> List[Dict]:
        """
        문서 내에서 위치 다양성을 고려한 샘플링
        
        Args:
            doc_chunks: 문서의 청크들
            n_samples: 샘플링할 개수
            
        Returns:
            샘플링된 청크들
        """
        if len(doc_chunks) <= n_samples:
            return doc_chunks
        
        # 청크 인덱스로 정렬
        sorted_chunks = sorted(doc_chunks, key=lambda x: x.get('chunk_index', 0))
        
        # 문서를 3개 섹션으로 나누어 샘플링 (시작, 중간, 끝)
        sections = [
            sorted_chunks[:len(sorted_chunks)//3],  # 시작 부분
            sorted_chunks[len(sorted_chunks)//3:2*len(sorted_chunks)//3],  # 중간 부분
            sorted_chunks[2*len(sorted_chunks)//3:]  # 끝 부분
        ]
        
        samples = []
        samples_per_section = n_samples // 3
        
        for section in sections:
            if section:
                n_from_section = min(samples_per_section, len(section))
                samples.extend(random.sample(section, n_from_section))
        
        # 부족한 경우 랜덤 추가
        if len(samples) < n_samples:
            remaining = n_samples - len(samples)
            unused = [c for c in doc_chunks if c not in samples]
            if unused:
                samples.extend(random.sample(unused, min(remaining, len(unused))))
        
        return samples[:n_samples]
    
    def _filter_similar_chunks(self, chunks: List[Dict], threshold: float = 0.95) -> List[Dict]:
        """
        유사한 청크 필터링 (간단한 텍스트 유사도 기반)
        
        Args:
            chunks: 청크 리스트
            threshold: 유사도 임계값
            
        Returns:
            필터링된 청크 리스트
        """
        filtered = []
        seen_contents = set()
        
        for chunk in chunks:
            content = chunk.get('content', '')
            
            # 간단한 중복 검사 (처음 100자 기준)
            content_key = content[:100].strip()
            
            if content_key and content_key not in seen_contents:
                filtered.append(chunk)
                seen_contents.add(content_key)
            elif not content_key:
                # 빈 내용이 아닌 경우만 추가
                if content.strip():
                    filtered.append(chunk)
        
        return filtered
    
    def _print_sampling_statistics(self, sampled_chunks: List[Dict]):
        """샘플링 통계 출력"""
        # 문서별 분포
        doc_distribution = defaultdict(int)
        for chunk in sampled_chunks:
            doc_distribution[chunk.get('source', 'unknown')] += 1
        
        # 길이별 분포
        length_distribution = {'short': 0, 'medium': 0, 'long': 0}
        for chunk in sampled_chunks:
            content_length = len(chunk.get('content', ''))
            if content_length < 500:
                length_distribution['short'] += 1
            elif content_length < 1500:
                length_distribution['medium'] += 1
            else:
                length_distribution['long'] += 1
        
        logger.info("\n" + "=" * 50)
        logger.info("Sampling Statistics:")
        logger.info(f"  Total sampled chunks: {len(sampled_chunks)}")
        logger.info(f"  Number of documents covered: {len(doc_distribution)}")
        logger.info(f"  Average chunks per document: {len(sampled_chunks) / len(doc_distribution):.1f}")
        logger.info(f"  Length distribution:")
        logger.info(f"    - Short (< 500): {length_distribution['short']}")
        logger.info(f"    - Medium (500-1500): {length_distribution['medium']}")
        logger.info(f"    - Long (> 1500): {length_distribution['long']}")
        logger.info("=" * 50 + "\n")
    
    def get_document_coverage(self, sampled_chunks: List[Dict]) -> float:
        """문서 커버리지 계산"""
        sampled_docs = set(chunk.get('source', 'unknown') for chunk in sampled_chunks)
        total_docs = len(self.chunks_by_document)
        return len(sampled_docs) / total_docs if total_docs > 0 else 0
    
    def get_length_distribution(self, sampled_chunks: List[Dict]) -> Dict[str, float]:
        """길이 분포 계산"""
        distribution = {'short': 0, 'medium': 0, 'long': 0}
        
        for chunk in sampled_chunks:
            content_length = len(chunk.get('content', ''))
            if content_length < 500:
                distribution['short'] += 1
            elif content_length < 1500:
                distribution['medium'] += 1
            else:
                distribution['long'] += 1
        
        total = len(sampled_chunks)
        if total > 0:
            for key in distribution:
                distribution[key] = distribution[key] / total
        
        return distribution