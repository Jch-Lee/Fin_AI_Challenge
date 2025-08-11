#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quality Checker for RAG System
RAG 시스템 품질 검증 도구
"""

import re
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import Counter
import difflib

logger = logging.getLogger(__name__)


class QualityIssue(Enum):
    """품질 문제 유형"""
    # 텍스트 품질
    ENCODING_ERROR = "encoding_error"           # 인코딩 오류 (깨진 문자)
    METADATA_CONTAMINATION = "metadata"         # 메타데이터 오염
    REPETITIVE_PATTERN = "repetitive"          # 반복 패턴
    TRUNCATED_TEXT = "truncated"               # 잘린 텍스트
    EXCESSIVE_WHITESPACE = "whitespace"        # 과도한 공백
    
    # 답변 품질
    EMPTY_ANSWER = "empty_answer"              # 빈 답변
    TOO_SHORT = "too_short"                    # 너무 짧은 답변
    TOO_LONG = "too_long"                      # 너무 긴 답변
    OFF_TOPIC = "off_topic"                    # 주제 벗어남
    HALLUCINATION = "hallucination"            # 환각 (없는 정보)
    
    # 형식 문제
    WRONG_FORMAT = "wrong_format"              # 잘못된 형식
    INVALID_MC_ANSWER = "invalid_mc"           # 잘못된 객관식 답변
    INCOMPLETE_ANSWER = "incomplete"           # 불완전한 답변
    
    # 언어 문제
    LANGUAGE_MIX = "language_mix"              # 언어 혼재
    GRAMMAR_ERROR = "grammar_error"            # 문법 오류


@dataclass
class QualityReport:
    """품질 검사 보고서"""
    score: float                               # 전체 점수 (0-100)
    issues: List[QualityIssue]                # 발견된 문제들
    details: Dict[str, Any]                   # 상세 정보
    recommendations: List[str]                # 개선 권고사항
    passed: bool                               # 품질 기준 통과 여부
    
    def to_dict(self) -> Dict:
        return {
            "score": float(self.score),
            "issues": [issue.value for issue in self.issues],
            "details": self._convert_details(self.details),
            "recommendations": self.recommendations,
            "passed": bool(self.passed)
        }
    
    def _convert_details(self, details: Dict) -> Dict:
        """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
        result = {}
        for key, value in details.items():
            if isinstance(value, dict):
                result[key] = self._convert_details(value)
            elif isinstance(value, (np.bool_, bool)):
                result[key] = bool(value)
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                result[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                result[key] = float(value)
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result


class TextQualityChecker:
    """텍스트 품질 검사기"""
    
    def __init__(self):
        # 문제 패턴들
        self.problematic_patterns = {
            "encoding_error": [
                r'[\ufffd]',                   # 대체 문자
                r'[^\x00-\x7F\u0080-\uFFFF]',  # 비정상 유니코드
                r'â€™|â€"|â€˜|â€œ',            # 잘못된 인코딩
            ],
            "metadata": [
                r'\[Page\s+\d+\]',
                r'목\s*차',
                r'참\s*고\s*문\s*헌',
                r'^-\s*\d+\s*-\s*$',
                r'\.{5,}',                     # 점선
                r'·{3,}',                      # 중간점 반복
            ],
            "repetitive": [
                r'(.)\1{10,}',                 # 같은 문자 10회 이상
                r'(\S+\s+)\1{5,}',             # 같은 단어 5회 이상
            ],
            "whitespace": [
                r'\s{5,}',                     # 연속 공백 5개 이상
                r'\n{4,}',                     # 연속 줄바꿈 4개 이상
            ]
        }
        
        # 금융 도메인 키워드
        self.financial_keywords = [
            "금융", "AI", "보안", "데이터", "모델", "시스템", "규제", "가이드라인",
            "리스크", "암호화", "인증", "감사", "컴플라이언스", "프라이버시",
            "알고리즘", "학습", "추론", "편향", "공격", "방어"
        ]
    
    def check_text(self, text: str) -> Tuple[List[QualityIssue], Dict[str, Any]]:
        """
        텍스트 품질 검사
        
        Returns:
            (발견된 문제들, 상세 정보)
        """
        issues = []
        details = {}
        
        # 1. 인코딩 오류 검사
        encoding_errors = self._check_encoding_errors(text)
        if encoding_errors:
            issues.append(QualityIssue.ENCODING_ERROR)
            details["encoding_errors"] = encoding_errors
        
        # 2. 메타데이터 오염 검사
        metadata_count = self._check_metadata_contamination(text)
        if metadata_count > 0:
            issues.append(QualityIssue.METADATA_CONTAMINATION)
            details["metadata_patterns"] = metadata_count
        
        # 3. 반복 패턴 검사
        repetitive_count = self._check_repetitive_patterns(text)
        if repetitive_count > 0:
            issues.append(QualityIssue.REPETITIVE_PATTERN)
            details["repetitive_patterns"] = repetitive_count
        
        # 4. 공백 문제 검사
        whitespace_issues = self._check_whitespace_issues(text)
        if whitespace_issues > 0:
            issues.append(QualityIssue.EXCESSIVE_WHITESPACE)
            details["whitespace_issues"] = whitespace_issues
        
        # 5. 텍스트 길이 검사
        if len(text.strip()) < 10:
            issues.append(QualityIssue.TRUNCATED_TEXT)
            details["text_length"] = len(text.strip())
        
        return issues, details
    
    def _check_encoding_errors(self, text: str) -> int:
        """인코딩 오류 검사"""
        error_count = 0
        for pattern in self.problematic_patterns["encoding_error"]:
            matches = re.findall(pattern, text)
            error_count += len(matches)
        return error_count
    
    def _check_metadata_contamination(self, text: str) -> int:
        """메타데이터 오염 검사"""
        contamination_count = 0
        for pattern in self.problematic_patterns["metadata"]:
            matches = re.findall(pattern, text, re.MULTILINE)
            contamination_count += len(matches)
        return contamination_count
    
    def _check_repetitive_patterns(self, text: str) -> int:
        """반복 패턴 검사"""
        repetitive_count = 0
        for pattern in self.problematic_patterns["repetitive"]:
            matches = re.findall(pattern, text)
            repetitive_count += len(matches)
        return repetitive_count
    
    def _check_whitespace_issues(self, text: str) -> int:
        """공백 문제 검사"""
        whitespace_count = 0
        for pattern in self.problematic_patterns["whitespace"]:
            matches = re.findall(pattern, text)
            whitespace_count += len(matches)
        return whitespace_count
    
    def calculate_text_quality_score(self, text: str) -> float:
        """
        텍스트 품질 점수 계산 (0-100)
        """
        issues, details = self.check_text(text)
        
        # 기본 점수 100점에서 차감
        score = 100.0
        
        # 문제별 감점
        deductions = {
            QualityIssue.ENCODING_ERROR: 30,
            QualityIssue.METADATA_CONTAMINATION: 20,
            QualityIssue.REPETITIVE_PATTERN: 15,
            QualityIssue.EXCESSIVE_WHITESPACE: 10,
            QualityIssue.TRUNCATED_TEXT: 25,
        }
        
        for issue in issues:
            if issue in deductions:
                score -= deductions[issue]
        
        # 추가 세부 감점
        if "encoding_errors" in details:
            score -= min(details["encoding_errors"] * 2, 20)
        if "metadata_patterns" in details:
            score -= min(details["metadata_patterns"] * 1, 10)
        
        return max(score, 0.0)


class AnswerQualityChecker:
    """답변 품질 검사기"""
    
    def __init__(self):
        self.text_checker = TextQualityChecker()
        
        # 답변 길이 기준
        self.length_criteria = {
            "multiple_choice": (1, 2),      # 1-2 문자
            "descriptive": (50, 2000),      # 50-2000 문자
            "definition": (30, 500),         # 30-500 문자
            "comparison": (100, 1500),      # 100-1500 문자
            "process": (100, 2000),         # 100-2000 문자
        }
    
    def check_answer(self, 
                     answer: str,
                     question: str,
                     question_type: str = "descriptive",
                     context: Optional[str] = None) -> Tuple[List[QualityIssue], Dict[str, Any]]:
        """
        답변 품질 검사
        
        Args:
            answer: 생성된 답변
            question: 원본 질문
            question_type: 질문 유형
            context: 사용된 컨텍스트
            
        Returns:
            (발견된 문제들, 상세 정보)
        """
        issues = []
        details = {}
        
        # 1. 빈 답변 검사
        if not answer or not answer.strip():
            issues.append(QualityIssue.EMPTY_ANSWER)
            return issues, details
        
        # 2. 답변 길이 검사
        length_issues = self._check_answer_length(answer, question_type)
        issues.extend(length_issues)
        details["answer_length"] = len(answer)
        
        # 3. 객관식 답변 형식 검사
        if question_type == "multiple_choice":
            if not self._is_valid_mc_answer(answer):
                issues.append(QualityIssue.INVALID_MC_ANSWER)
        
        # 4. 주제 관련성 검사
        relevance_score = self._check_relevance(answer, question, context)
        if relevance_score < 0.3:
            issues.append(QualityIssue.OFF_TOPIC)
        details["relevance_score"] = relevance_score
        
        # 5. 완전성 검사
        if self._is_incomplete(answer):
            issues.append(QualityIssue.INCOMPLETE_ANSWER)
        
        # 6. 언어 혼재 검사
        if self._has_language_mix(answer):
            issues.append(QualityIssue.LANGUAGE_MIX)
        
        return issues, details
    
    def _check_answer_length(self, answer: str, question_type: str) -> List[QualityIssue]:
        """답변 길이 검사"""
        issues = []
        
        if question_type in self.length_criteria:
            min_len, max_len = self.length_criteria[question_type]
            answer_len = len(answer.strip())
            
            if answer_len < min_len:
                issues.append(QualityIssue.TOO_SHORT)
            elif answer_len > max_len:
                issues.append(QualityIssue.TOO_LONG)
        
        return issues
    
    def _is_valid_mc_answer(self, answer: str) -> bool:
        """객관식 답변 유효성 검사"""
        # 숫자만 있는지 확인
        answer = answer.strip()
        if re.match(r'^\d{1,2}$', answer):
            num = int(answer)
            return 1 <= num <= 99
        return False
    
    def _check_relevance(self, answer: str, question: str, context: Optional[str]) -> float:
        """주제 관련성 점수 계산 (0-1)"""
        # 질문의 키워드 추출
        question_keywords = self._extract_keywords(question)
        
        # 답변에서 키워드 매칭
        answer_lower = answer.lower()
        matched_keywords = sum(1 for kw in question_keywords if kw in answer_lower)
        
        if not question_keywords:
            return 0.5
        
        relevance = matched_keywords / len(question_keywords)
        
        # 컨텍스트와의 유사도도 고려
        if context:
            context_similarity = self._calculate_similarity(answer, context)
            relevance = (relevance + context_similarity) / 2
        
        return min(relevance, 1.0)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 방법 필요)
        text = text.lower()
        
        # 금융 도메인 키워드 우선
        keywords = []
        for kw in self.text_checker.financial_keywords:
            if kw in text:
                keywords.append(kw)
        
        # 명사 추출 (간단한 휴리스틱)
        words = re.findall(r'\b[가-힣]{2,}\b', text)
        for word in words[:5]:  # 상위 5개
            if word not in keywords:
                keywords.append(word)
        
        return keywords
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산"""
        # 간단한 자카드 유사도
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _is_incomplete(self, answer: str) -> bool:
        """불완전한 답변 검사"""
        incomplete_patterns = [
            r'\.{3,}$',                    # ... 로 끝남
            r'등등$',
            r'기타$',
            r'미완성',
            r'작성 중',
            r'\[.*?\]$',                   # [TODO] 등
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, answer):
                return True
        
        return False
    
    def _has_language_mix(self, answer: str) -> bool:
        """언어 혼재 검사"""
        has_korean = bool(re.search(r'[가-힣]', answer))
        has_english = bool(re.search(r'[a-zA-Z]{3,}', answer))
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', answer))
        has_japanese = bool(re.search(r'[\u3040-\u309f\u30a0-\u30ff]', answer))
        
        # 한국어와 영어 혼재는 허용 (전문용어 때문)
        language_count = sum([has_chinese, has_japanese])
        
        return language_count > 0


class RAGQualityChecker:
    """RAG 시스템 전체 품질 검사기"""
    
    def __init__(self):
        self.text_checker = TextQualityChecker()
        self.answer_checker = AnswerQualityChecker()
        
        # 품질 기준 점수
        self.quality_thresholds = {
            "excellent": 90,
            "good": 70,
            "acceptable": 50,
            "poor": 30
        }
    
    def check_pipeline_quality(self,
                              input_text: str,
                              processed_text: str,
                              chunks: List[str],
                              retrieved_contexts: List[str],
                              generated_answer: str,
                              question: str,
                              question_type: str = "descriptive") -> QualityReport:
        """
        전체 파이프라인 품질 검사
        
        Args:
            input_text: 원본 입력 텍스트
            processed_text: 전처리된 텍스트
            chunks: 청킹된 텍스트들
            retrieved_contexts: 검색된 컨텍스트들
            generated_answer: 생성된 답변
            question: 질문
            question_type: 질문 유형
            
        Returns:
            QualityReport 객체
        """
        all_issues = []
        all_details = {}
        scores = {}
        
        # 1. 입력 텍스트 품질
        if input_text:
            input_issues, input_details = self.text_checker.check_text(input_text)
            all_issues.extend(input_issues)
            all_details["input"] = input_details
            scores["input"] = self.text_checker.calculate_text_quality_score(input_text)
        
        # 2. 전처리 텍스트 품질
        if processed_text:
            processed_issues, processed_details = self.text_checker.check_text(processed_text)
            all_issues.extend(processed_issues)
            all_details["processed"] = processed_details
            scores["processed"] = self.text_checker.calculate_text_quality_score(processed_text)
        
        # 3. 청킹 품질
        if chunks:
            chunk_scores = []
            for i, chunk in enumerate(chunks[:5]):  # 샘플 5개만 검사
                chunk_score = self.text_checker.calculate_text_quality_score(chunk)
                chunk_scores.append(chunk_score)
            scores["chunks"] = np.mean(chunk_scores) if chunk_scores else 0
            all_details["chunk_count"] = len(chunks)
        
        # 4. 검색 컨텍스트 품질
        if retrieved_contexts:
            context_scores = []
            for context in retrieved_contexts[:3]:  # 상위 3개만 검사
                context_score = self.text_checker.calculate_text_quality_score(context)
                context_scores.append(context_score)
            scores["contexts"] = np.mean(context_scores) if context_scores else 0
        
        # 5. 답변 품질
        answer_issues, answer_details = self.answer_checker.check_answer(
            generated_answer, question, question_type, 
            "\n".join(retrieved_contexts) if retrieved_contexts else None
        )
        all_issues.extend(answer_issues)
        all_details["answer"] = answer_details
        
        # 답변 품질 점수
        answer_score = 100.0
        answer_deductions = {
            QualityIssue.EMPTY_ANSWER: 50,
            QualityIssue.TOO_SHORT: 20,
            QualityIssue.TOO_LONG: 10,
            QualityIssue.OFF_TOPIC: 30,
            QualityIssue.INVALID_MC_ANSWER: 40,
            QualityIssue.INCOMPLETE_ANSWER: 25,
        }
        
        for issue in answer_issues:
            if issue in answer_deductions:
                answer_score -= answer_deductions[issue]
        
        scores["answer"] = max(answer_score, 0.0)
        
        # 전체 점수 계산 (가중 평균)
        weights = {
            "input": 0.1,
            "processed": 0.15,
            "chunks": 0.15,
            "contexts": 0.25,
            "answer": 0.35
        }
        
        total_score = sum(scores.get(k, 0) * weights[k] for k in weights)
        
        # 품질 수준 결정
        quality_level = self._determine_quality_level(total_score)
        
        # 개선 권고사항 생성
        recommendations = self._generate_recommendations(all_issues, scores)
        
        # 통과 여부
        passed = total_score >= self.quality_thresholds["acceptable"]
        
        return QualityReport(
            score=total_score,
            issues=list(set(all_issues)),  # 중복 제거
            details={
                "scores": scores,
                "quality_level": quality_level,
                **all_details
            },
            recommendations=recommendations,
            passed=passed
        )
    
    def _determine_quality_level(self, score: float) -> str:
        """품질 수준 결정"""
        if score >= self.quality_thresholds["excellent"]:
            return "excellent"
        elif score >= self.quality_thresholds["good"]:
            return "good"
        elif score >= self.quality_thresholds["acceptable"]:
            return "acceptable"
        elif score >= self.quality_thresholds["poor"]:
            return "poor"
        else:
            return "very_poor"
    
    def _generate_recommendations(self, issues: List[QualityIssue], scores: Dict[str, float]) -> List[str]:
        """개선 권고사항 생성"""
        recommendations = []
        
        # 문제별 권고사항
        issue_recommendations = {
            QualityIssue.ENCODING_ERROR: "UTF-8 인코딩 설정을 확인하고 파일 재처리",
            QualityIssue.METADATA_CONTAMINATION: "TextCleaner의 메타데이터 제거 기능 강화",
            QualityIssue.REPETITIVE_PATTERN: "반복 패턴 제거 필터 적용",
            QualityIssue.EXCESSIVE_WHITESPACE: "공백 정규화 처리 강화",
            QualityIssue.EMPTY_ANSWER: "모델 생성 파라미터 조정 (temperature, max_tokens)",
            QualityIssue.TOO_SHORT: "프롬프트에 상세 답변 요구사항 추가",
            QualityIssue.OFF_TOPIC: "RAG 검색 정확도 개선 및 컨텍스트 품질 향상",
            QualityIssue.INVALID_MC_ANSWER: "객관식 답변 추출 로직 개선",
        }
        
        for issue in set(issues):
            if issue in issue_recommendations:
                recommendations.append(issue_recommendations[issue])
        
        # 점수 기반 권고사항
        if scores.get("processed", 100) < 70:
            recommendations.append("PDF 처리 파이프라인 개선 필요")
        
        if scores.get("contexts", 100) < 70:
            recommendations.append("RAG 검색 품질 개선 필요 (임베딩 모델 변경 고려)")
        
        if scores.get("answer", 100) < 70:
            recommendations.append("답변 생성 모델 파인튜닝 또는 프롬프트 엔지니어링 필요")
        
        return recommendations
    
    def generate_report(self, report: QualityReport, output_path: Optional[str] = None) -> str:
        """
        품질 보고서 생성
        
        Args:
            report: QualityReport 객체
            output_path: 저장할 파일 경로 (선택사항)
            
        Returns:
            보고서 텍스트
        """
        report_text = f"""
========================================
       RAG 시스템 품질 검사 보고서
========================================

전체 품질 점수: {report.score:.1f}/100
품질 수준: {report.details.get('quality_level', 'N/A')}
통과 여부: {'통과' if report.passed else '미통과'}

----------------------------------------
세부 점수
----------------------------------------
"""
        
        if "scores" in report.details:
            for component, score in report.details["scores"].items():
                report_text += f"  - {component:15s}: {score:6.1f}/100\n"
        
        report_text += f"""
----------------------------------------
발견된 문제 ({len(report.issues)}건)
----------------------------------------
"""
        
        for issue in report.issues:
            report_text += f"  - {issue.value}\n"
        
        report_text += f"""
----------------------------------------
개선 권고사항
----------------------------------------
"""
        
        for i, rec in enumerate(report.recommendations, 1):
            report_text += f"  {i}. {rec}\n"
        
        report_text += """
========================================
"""
        
        # 파일로 저장
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
                # JSON 형식으로도 저장
                json_path = output_path.replace('.txt', '.json')
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump(report.to_dict(), jf, ensure_ascii=False, indent=2)
        
        return report_text


# 사용 예시
if __name__ == "__main__":
    # 품질 검사기 초기화
    checker = RAGQualityChecker()
    
    # 테스트 데이터
    test_input = "금융분야 AI 보안 가이드라인···············8 2···············"
    test_processed = "금융분야 AI 보안 가이드라인"
    test_chunks = ["금융 AI 시스템은 보안이 중요합니다.", "데이터 보호가 필수입니다."]
    test_contexts = ["AI 보안은 다층적 접근이 필요합니다.", "금융 데이터는 암호화해야 합니다."]
    test_answer = "3"
    test_question = "다음 중 맞는 것은?\n1. A\n2. B\n3. C\n4. D"
    
    # 품질 검사 실행
    report = checker.check_pipeline_quality(
        input_text=test_input,
        processed_text=test_processed,
        chunks=test_chunks,
        retrieved_contexts=test_contexts,
        generated_answer=test_answer,
        question=test_question,
        question_type="multiple_choice"
    )
    
    # 보고서 출력
    report_text = checker.generate_report(report)
    print(report_text)