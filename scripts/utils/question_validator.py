#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
질문 품질 검증 및 다양성 평가 모듈
"""

import re
import logging
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
import numpy as np
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class QuestionValidator:
    """질문 품질 검증 및 다양성 평가"""
    
    def __init__(self):
        """검증기 초기화"""
        self.min_length = 10  # 최소 질문 길이
        self.max_length = 500  # 최대 질문 길이
        self.similarity_threshold = 0.85  # 유사도 임계값
        
    def validate_questions(self, questions: List[Dict]) -> List[Dict]:
        """
        질문 검증 및 필터링
        
        Args:
            questions: 생성된 질문 리스트
            
        Returns:
            검증된 질문 리스트
        """
        logger.info(f"Validating {len(questions)} questions...")
        
        validated = []
        rejected = []
        seen_questions = set()
        
        for q in questions:
            question_text = q.get('question', '')
            
            # 1. 형식 검증
            if not self._validate_format(question_text):
                rejected.append({'question': question_text, 'reason': 'format'})
                continue
            
            # 2. 길이 검증
            if not self._validate_length(question_text):
                rejected.append({'question': question_text, 'reason': 'length'})
                continue
            
            # 3. 한국어 검증
            if not self._validate_korean(question_text):
                rejected.append({'question': question_text, 'reason': 'not_korean'})
                continue
            
            # 4. 중복 검사
            if self._is_duplicate(question_text, seen_questions):
                rejected.append({'question': question_text, 'reason': 'duplicate'})
                continue
            
            # 5. 유사도 검사
            if self._is_too_similar(question_text, validated):
                rejected.append({'question': question_text, 'reason': 'too_similar'})
                continue
            
            # 검증 통과
            validated.append(q)
            seen_questions.add(question_text)
        
        logger.info(f"Validation complete: {len(validated)} passed, {len(rejected)} rejected")
        
        if rejected:
            rejection_reasons = Counter(r['reason'] for r in rejected)
            logger.info(f"Rejection reasons: {dict(rejection_reasons)}")
        
        return validated
    
    def _validate_format(self, question: str) -> bool:
        """형식 검증 (한 문장, 물음표로 끝남)"""
        if not question:
            return False
        
        # 물음표로 끝나는지 확인
        if not question.strip().endswith('?'):
            return False
        
        # 너무 많은 문장이 아닌지 확인 (대략적)
        # 한국어는 마침표가 없을 수 있으므로 느슨하게 체크
        if question.count('.') > 3:
            return False
        
        return True
    
    def _validate_length(self, question: str) -> bool:
        """길이 검증"""
        length = len(question.strip())
        return self.min_length <= length <= self.max_length
    
    def _validate_korean(self, question: str) -> bool:
        """한국어 포함 여부 검증"""
        # 한글이 포함되어 있는지 확인
        korean_pattern = re.compile('[가-힣]+')
        return bool(korean_pattern.search(question))
    
    def _is_duplicate(self, question: str, seen_questions: Set[str]) -> bool:
        """정확한 중복 검사"""
        normalized = question.strip().lower()
        return normalized in seen_questions
    
    def _is_too_similar(self, question: str, validated_questions: List[Dict], 
                        threshold: float = None) -> bool:
        """유사도 기반 중복 검사"""
        if threshold is None:
            threshold = self.similarity_threshold
        
        for vq in validated_questions:
            existing_question = vq.get('question', '')
            similarity = self._calculate_similarity(question, existing_question)
            
            if similarity > threshold:
                return True
        
        return False
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간 유사도 계산 (편집 거리 기반)"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def calculate_diversity_metrics(self, questions: List[Dict]) -> Dict:
        """
        다양성 메트릭 계산
        
        Args:
            questions: 검증된 질문 리스트
            
        Returns:
            다양성 메트릭 딕셔너리
        """
        if not questions:
            return {
                'total_questions': 0,
                'document_coverage': 0,
                'question_type_distribution': {},
                'avg_pairwise_similarity': 0,
                'unique_keywords_ratio': 0,
                'question_type_entropy': 0
            }
        
        # 1. 문서 커버리지
        unique_sources = set(q.get('source_file', 'unknown') for q in questions)
        
        # 2. 질문 유형 분포
        type_distribution = Counter(q.get('question_type', 'unknown') for q in questions)
        
        # 3. 평균 쌍별 유사도
        similarities = []
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                sim = self._calculate_similarity(
                    questions[i].get('question', ''),
                    questions[j].get('question', '')
                )
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        
        # 4. 고유 키워드 비율
        all_keywords = []
        unique_keywords = set()
        
        for q in questions:
            question_text = q.get('question', '')
            # 간단한 토큰화 (공백 기준)
            keywords = [w for w in question_text.split() if len(w) > 2]
            all_keywords.extend(keywords)
            unique_keywords.update(keywords)
        
        unique_ratio = len(unique_keywords) / len(all_keywords) if all_keywords else 0
        
        # 5. 질문 유형 엔트로피
        type_entropy = self._calculate_entropy(list(type_distribution.values()))
        
        metrics = {
            'total_questions': len(questions),
            'document_coverage': len(unique_sources),
            'question_type_distribution': dict(type_distribution),
            'avg_pairwise_similarity': round(avg_similarity, 3),
            'unique_keywords_ratio': round(unique_ratio, 3),
            'question_type_entropy': round(type_entropy, 3),
            'diversity_score': self._calculate_diversity_score(
                document_coverage=len(unique_sources),
                avg_similarity=avg_similarity,
                unique_ratio=unique_ratio,
                type_entropy=type_entropy
            )
        }
        
        return metrics
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """엔트로피 계산"""
        if not values or sum(values) == 0:
            return 0
        
        total = sum(values)
        probabilities = [v / total for v in values]
        
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_diversity_score(self, document_coverage: int, avg_similarity: float,
                                  unique_ratio: float, type_entropy: float) -> float:
        """
        종합 다양성 점수 계산 (0-100)
        
        Args:
            document_coverage: 문서 커버리지
            avg_similarity: 평균 유사도 (낮을수록 좋음)
            unique_ratio: 고유 키워드 비율
            type_entropy: 질문 유형 엔트로피
            
        Returns:
            다양성 점수 (0-100)
        """
        # 각 요소를 0-1로 정규화
        doc_score = min(document_coverage / 30, 1.0)  # 30개 문서 이상이면 만점
        sim_score = 1.0 - avg_similarity  # 유사도가 낮을수록 높은 점수
        keyword_score = unique_ratio
        entropy_score = min(type_entropy / 2.5, 1.0)  # 엔트로피 2.5 이상이면 만점
        
        # 가중 평균
        weights = {
            'document': 0.25,
            'similarity': 0.30,
            'keywords': 0.20,
            'entropy': 0.25
        }
        
        total_score = (
            doc_score * weights['document'] +
            sim_score * weights['similarity'] +
            keyword_score * weights['keywords'] +
            entropy_score * weights['entropy']
        ) * 100
        
        return round(total_score, 1)
    
    def generate_validation_report(self, questions: List[Dict]) -> str:
        """
        검증 리포트 생성
        
        Args:
            questions: 검증된 질문 리스트
            
        Returns:
            리포트 문자열
        """
        metrics = self.calculate_diversity_metrics(questions)
        
        report = []
        report.append("=" * 60)
        report.append("Question Validation and Diversity Report")
        report.append("=" * 60)
        report.append(f"\n총 질문 수: {metrics['total_questions']}")
        report.append(f"문서 커버리지: {metrics['document_coverage']} documents")
        
        report.append("\n질문 유형 분포:")
        for qtype, count in metrics['question_type_distribution'].items():
            report.append(f"  - {qtype}: {count}")
        
        report.append(f"\n다양성 메트릭:")
        report.append(f"  - 평균 유사도: {metrics['avg_pairwise_similarity']:.3f}")
        report.append(f"  - 고유 키워드 비율: {metrics['unique_keywords_ratio']:.3f}")
        report.append(f"  - 질문 유형 엔트로피: {metrics['question_type_entropy']:.3f}")
        
        report.append(f"\n종합 다양성 점수: {metrics['diversity_score']}/100")
        
        # 평가
        if metrics['diversity_score'] >= 80:
            report.append("평가: 우수한 다양성")
        elif metrics['diversity_score'] >= 60:
            report.append("평가: 양호한 다양성")
        elif metrics['diversity_score'] >= 40:
            report.append("평가: 보통 수준의 다양성")
        else:
            report.append("평가: 다양성 개선 필요")
        
        report.append("=" * 60)
        
        return "\n".join(report)