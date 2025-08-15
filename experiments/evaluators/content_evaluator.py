#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
콘텐츠 품질 평가 모듈
VL 모델과 기존 방법의 텍스트 추출 품질을 평가
"""

import re
import json
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import difflib
from collections import Counter
import numpy as np


@dataclass
class ContentQualityMetrics:
    """콘텐츠 품질 메트릭"""
    # 기본 통계
    total_chars: int
    total_words: int
    total_sentences: int
    unique_words: int
    
    # 도메인 특화 메트릭
    financial_terms_count: int
    regulatory_terms_count: int
    technical_terms_count: int
    
    # 시각적 콘텐츠 관련
    visual_references: int
    table_indicators: int
    chart_indicators: int
    diagram_indicators: int
    
    # 구조적 요소
    headers_detected: int
    lists_detected: int
    numbered_items: int
    
    # 품질 지표
    readability_score: float
    coherence_score: float
    completeness_score: float
    
    def to_dict(self) -> Dict:
        return {
            "basic_stats": {
                "total_chars": self.total_chars,
                "total_words": self.total_words,
                "total_sentences": self.total_sentences,
                "unique_words": self.unique_words,
                "vocabulary_diversity": self.unique_words / max(self.total_words, 1)
            },
            "domain_coverage": {
                "financial_terms": self.financial_terms_count,
                "regulatory_terms": self.regulatory_terms_count,
                "technical_terms": self.technical_terms_count
            },
            "visual_content": {
                "visual_references": self.visual_references,
                "table_indicators": self.table_indicators,
                "chart_indicators": self.chart_indicators,
                "diagram_indicators": self.diagram_indicators
            },
            "structure": {
                "headers_detected": self.headers_detected,
                "lists_detected": self.lists_detected,
                "numbered_items": self.numbered_items
            },
            "quality_scores": {
                "readability": self.readability_score,
                "coherence": self.coherence_score,
                "completeness": self.completeness_score
            }
        }


class ContentQualityEvaluator:
    """콘텐츠 품질 평가기"""
    
    def __init__(self):
        self.financial_terms = self._load_financial_terms()
        self.regulatory_terms = self._load_regulatory_terms()
        self.technical_terms = self._load_technical_terms()
        self.visual_keywords = self._load_visual_keywords()
    
    def _load_financial_terms(self) -> Set[str]:
        """금융 관련 용어 로드"""
        terms = {
            # 기본 금융 용어
            "금융", "은행", "투자", "자산", "부채", "자본", "수익", "손실",
            "리스크", "위험", "수익률", "금리", "이자", "대출", "예금",
            "보험", "연금", "펀드", "주식", "채권", "파생상품",
            
            # 핀테크 용어
            "핀테크", "디지털뱅킹", "모바일뱅킹", "전자결제", "암호화폐",
            "블록체인", "디지털자산", "가상화폐", "디지털화폐",
            
            # 규제 관련
            "금융위원회", "금융감독원", "바젤", "IFRS", "회계기준",
            "자기자본", "유동성", "신용위험", "시장위험", "운영위험",
            
            # 영어 용어
            "fintech", "banking", "investment", "asset", "liability",
            "risk", "compliance", "regulation", "aml", "kyc"
        }
        return {term.lower() for term in terms}
    
    def _load_regulatory_terms(self) -> Set[str]:
        """규제 관련 용어 로드"""
        terms = {
            # 규제 기관
            "금융위원회", "금융감독원", "한국은행", "예금보험공사",
            "금융정보분석원", "금융보안원",
            
            # 규제 용어
            "규제", "컴플라이언스", "준수", "감독", "검사", "제재",
            "법령", "규정", "지침", "가이드라인", "기준", "표준",
            "인가", "허가", "신고", "보고", "공시", "공개",
            
            # 보안 관련
            "보안", "개인정보", "정보보호", "사이버보안", "데이터보호",
            "암호화", "인증", "접근제어", "감사", "모니터링",
            
            # 영어 용어
            "compliance", "regulation", "supervision", "security",
            "privacy", "gdpr", "sox", "basel", "mifid"
        }
        return {term.lower() for term in terms}
    
    def _load_technical_terms(self) -> Set[str]:
        """기술 관련 용어 로드"""
        terms = {
            # AI/ML 용어
            "인공지능", "머신러닝", "딥러닝", "신경망", "알고리즘",
            "모델", "학습", "훈련", "예측", "분류", "회귀",
            "자연어처리", "컴퓨터비전", "데이터마이닝",
            
            # 데이터 관련
            "빅데이터", "데이터베이스", "데이터웨어하우스", "ETL",
            "데이터레이크", "데이터파이프라인", "스트리밍",
            
            # 클라우드/인프라
            "클라우드", "서버", "네트워크", "API", "마이크로서비스",
            "컨테이너", "도커", "쿠버네티스", "DevOps",
            
            # 영어 용어
            "ai", "ml", "nlp", "api", "cloud", "server",
            "algorithm", "model", "training", "prediction"
        }
        return {term.lower() for term in terms}
    
    def _load_visual_keywords(self) -> Dict[str, Set[str]]:
        """시각적 콘텐츠 키워드 로드"""
        return {
            "charts": {
                "차트", "그래프", "도표", "그림", "figure", "chart",
                "graph", "plot", "막대그래프", "선그래프", "원그래프",
                "히스토그램", "산점도", "박스플롯"
            },
            "tables": {
                "표", "테이블", "목록", "리스트", "table", "list",
                "행", "열", "셀", "row", "column", "cell"
            },
            "diagrams": {
                "다이어그램", "구조도", "흐름도", "조직도", "시스템도",
                "아키텍처", "diagram", "flowchart", "structure",
                "architecture", "workflow"
            }
        }
    
    def evaluate_content_quality(self, text: str, method_name: str = "") -> ContentQualityMetrics:
        """텍스트의 콘텐츠 품질 평가"""
        
        # 기본 통계 계산
        basic_stats = self._calculate_basic_stats(text)
        
        # 도메인 특화 용어 계산
        domain_stats = self._calculate_domain_coverage(text)
        
        # 시각적 콘텐츠 관련 계산
        visual_stats = self._calculate_visual_content(text)
        
        # 구조적 요소 계산
        structure_stats = self._calculate_structural_elements(text)
        
        # 품질 점수 계산
        quality_scores = self._calculate_quality_scores(text, basic_stats)
        
        return ContentQualityMetrics(
            # 기본 통계
            total_chars=basic_stats["chars"],
            total_words=basic_stats["words"],
            total_sentences=basic_stats["sentences"],
            unique_words=basic_stats["unique_words"],
            
            # 도메인 특화
            financial_terms_count=domain_stats["financial"],
            regulatory_terms_count=domain_stats["regulatory"],
            technical_terms_count=domain_stats["technical"],
            
            # 시각적 콘텐츠
            visual_references=visual_stats["visual_refs"],
            table_indicators=visual_stats["tables"],
            chart_indicators=visual_stats["charts"],
            diagram_indicators=visual_stats["diagrams"],
            
            # 구조적 요소
            headers_detected=structure_stats["headers"],
            lists_detected=structure_stats["lists"],
            numbered_items=structure_stats["numbered"],
            
            # 품질 지표
            readability_score=quality_scores["readability"],
            coherence_score=quality_scores["coherence"],
            completeness_score=quality_scores["completeness"]
        )
    
    def _calculate_basic_stats(self, text: str) -> Dict[str, int]:
        """기본 텍스트 통계 계산"""
        chars = len(text)
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        unique_words = len(set(word.lower().strip('.,!?;:"()[]{}') for word in words))
        
        return {
            "chars": chars,
            "words": len(words),
            "sentences": len(sentences),
            "unique_words": unique_words
        }
    
    def _calculate_domain_coverage(self, text: str) -> Dict[str, int]:
        """도메인 특화 용어 커버리지 계산"""
        text_lower = text.lower()
        
        financial_count = sum(1 for term in self.financial_terms if term in text_lower)
        regulatory_count = sum(1 for term in self.regulatory_terms if term in text_lower)
        technical_count = sum(1 for term in self.technical_terms if term in text_lower)
        
        return {
            "financial": financial_count,
            "regulatory": regulatory_count,
            "technical": technical_count
        }
    
    def _calculate_visual_content(self, text: str) -> Dict[str, int]:
        """시각적 콘텐츠 관련 지표 계산"""
        text_lower = text.lower()
        
        visual_refs = 0
        tables = 0
        charts = 0
        diagrams = 0
        
        for category, keywords in self.visual_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            
            if category == "charts":
                charts = count
            elif category == "tables":
                tables = count
            elif category == "diagrams":
                diagrams = count
            
            visual_refs += count
        
        return {
            "visual_refs": visual_refs,
            "tables": tables,
            "charts": charts,
            "diagrams": diagrams
        }
    
    def _calculate_structural_elements(self, text: str) -> Dict[str, int]:
        """구조적 요소 계산"""
        
        # 헤더 감지 (markdown style이나 번호 패턴)
        headers = len(re.findall(r'^#{1,6}\s+.+$|^\d+\.?\s+[가-힣A-Za-z].+$', text, re.MULTILINE))
        
        # 리스트 감지
        lists = len(re.findall(r'^\s*[-*+•]\s+.+$|^\s*\d+\.\s+.+$', text, re.MULTILINE))
        
        # 번호가 매겨진 항목
        numbered = len(re.findall(r'^\s*\d+\.\s+.+$|^\s*\(\d+\)\s+.+$', text, re.MULTILINE))
        
        return {
            "headers": headers,
            "lists": lists,
            "numbered": numbered
        }
    
    def _calculate_quality_scores(self, text: str, basic_stats: Dict[str, int]) -> Dict[str, float]:
        """품질 점수 계산"""
        
        # 가독성 점수 (단어 길이와 문장 길이 기반)
        readability = self._calculate_readability(text, basic_stats)
        
        # 일관성 점수 (반복 패턴과 구조 기반)
        coherence = self._calculate_coherence(text)
        
        # 완성도 점수 (콘텐츠 풍부함 기반)
        completeness = self._calculate_completeness(text, basic_stats)
        
        return {
            "readability": max(0.0, min(1.0, readability)),
            "coherence": max(0.0, min(1.0, coherence)),
            "completeness": max(0.0, min(1.0, completeness))
        }
    
    def _calculate_readability(self, text: str, basic_stats: Dict[str, int]) -> float:
        """가독성 점수 계산 (한국어 적응)"""
        if basic_stats["words"] == 0 or basic_stats["sentences"] == 0:
            return 0.0
        
        # 평균 단어 길이
        avg_word_length = basic_stats["chars"] / basic_stats["words"]
        
        # 평균 문장 길이 (단어 수)
        avg_sentence_length = basic_stats["words"] / basic_stats["sentences"]
        
        # 어휘 다양성
        vocab_diversity = basic_stats["unique_words"] / basic_stats["words"]
        
        # 한국어 특성을 고려한 가독성 점수
        # 너무 긴 단어나 문장은 가독성을 떨어뜨림
        word_score = max(0, 1 - (avg_word_length - 3) / 10)  # 3글자 기준
        sentence_score = max(0, 1 - (avg_sentence_length - 15) / 30)  # 15단어 기준
        diversity_score = min(1, vocab_diversity * 2)  # 다양성 보너스
        
        return (word_score + sentence_score + diversity_score) / 3
    
    def _calculate_coherence(self, text: str) -> float:
        """일관성 점수 계산"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
        
        if len(sentences) < 2:
            return 0.5
        
        # 문장 간 어휘 중복도 계산
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            
            if len(words1) > 0 and len(words2) > 0:
                overlap = len(words1 & words2) / len(words1 | words2)
                coherence_scores.append(overlap)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _calculate_completeness(self, text: str, basic_stats: Dict[str, int]) -> float:
        """완성도 점수 계산"""
        
        # 텍스트 길이 점수
        length_score = min(1.0, basic_stats["chars"] / 10000)  # 10,000자 기준
        
        # 구조적 완성도 (헤더, 리스트 등의 존재)
        has_headers = bool(re.search(r'^#{1,6}\s+.+$|^\d+\.?\s+[가-힣A-Za-z].+$', text, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[-*+•]\s+.+$|^\s*\d+\.\s+.+$', text, re.MULTILINE))
        has_paragraphs = len(text.split('\n\n')) > 3
        
        structure_score = sum([has_headers, has_lists, has_paragraphs]) / 3
        
        # 도메인 커버리지 점수
        domain_coverage = self._calculate_domain_coverage(text)
        domain_score = min(1.0, sum(domain_coverage.values()) / 20)  # 20개 용어 기준
        
        return (length_score + structure_score + domain_score) / 3
    
    def compare_content_quality(self, 
                              traditional_text: str, 
                              vl_text: str) -> Dict[str, any]:
        """두 추출 방법의 콘텐츠 품질 비교"""
        
        # 각각의 품질 평가
        traditional_metrics = self.evaluate_content_quality(traditional_text, "traditional")
        vl_metrics = self.evaluate_content_quality(vl_text, "vl")
        
        # 비교 분석
        comparison = {
            "traditional_quality": traditional_metrics.to_dict(),
            "vl_quality": vl_metrics.to_dict(),
            "improvements": self._calculate_improvements(traditional_metrics, vl_metrics),
            "content_overlap": self._calculate_content_overlap(traditional_text, vl_text),
            "unique_content": self._identify_unique_content(traditional_text, vl_text),
            "overall_assessment": self._assess_overall_quality(traditional_metrics, vl_metrics)
        }
        
        return comparison
    
    def _calculate_improvements(self, traditional: ContentQualityMetrics, vl: ContentQualityMetrics) -> Dict[str, float]:
        """개선 정도 계산"""
        def safe_percentage(new_val, old_val):
            if old_val == 0:
                return 100.0 if new_val > 0 else 0.0
            return ((new_val - old_val) / old_val) * 100
        
        return {
            "content_volume": {
                "chars_improvement": safe_percentage(vl.total_chars, traditional.total_chars),
                "words_improvement": safe_percentage(vl.total_words, traditional.total_words),
                "sentences_improvement": safe_percentage(vl.total_sentences, traditional.total_sentences)
            },
            "domain_coverage": {
                "financial_improvement": safe_percentage(vl.financial_terms_count, traditional.financial_terms_count),
                "regulatory_improvement": safe_percentage(vl.regulatory_terms_count, traditional.regulatory_terms_count),
                "technical_improvement": safe_percentage(vl.technical_terms_count, traditional.technical_terms_count)
            },
            "visual_content": {
                "visual_refs_improvement": safe_percentage(vl.visual_references, traditional.visual_references),
                "charts_improvement": safe_percentage(vl.chart_indicators, traditional.chart_indicators),
                "tables_improvement": safe_percentage(vl.table_indicators, traditional.table_indicators)
            },
            "quality_scores": {
                "readability_improvement": vl.readability_score - traditional.readability_score,
                "coherence_improvement": vl.coherence_score - traditional.coherence_score,
                "completeness_improvement": vl.completeness_score - traditional.completeness_score
            }
        }
    
    def _calculate_content_overlap(self, text1: str, text2: str) -> Dict[str, float]:
        """콘텐츠 중복도 계산"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return {"overlap_ratio": 0.0, "unique_to_text1": 0.0, "unique_to_text2": 0.0}
        
        overlap = words1 & words2
        unique_to_1 = words1 - words2
        unique_to_2 = words2 - words1
        
        total_unique = len(words1 | words2)
        
        return {
            "overlap_ratio": len(overlap) / total_unique,
            "unique_to_text1": len(unique_to_1) / total_unique,
            "unique_to_text2": len(unique_to_2) / total_unique,
            "jaccard_similarity": len(overlap) / len(words1 | words2)
        }
    
    def _identify_unique_content(self, traditional_text: str, vl_text: str) -> Dict[str, List[str]]:
        """각 방법의 고유 콘텐츠 식별"""
        
        # 문장 단위로 분리
        trad_sentences = set(re.split(r'[.!?]+', traditional_text))
        vl_sentences = set(re.split(r'[.!?]+', vl_text))
        
        # 정리
        trad_sentences = {s.strip() for s in trad_sentences if s.strip() and len(s) > 10}
        vl_sentences = {s.strip() for s in vl_sentences if s.strip() and len(s) > 10}
        
        # 고유 콘텐츠 식별
        unique_to_traditional = list(trad_sentences - vl_sentences)[:10]  # 상위 10개
        unique_to_vl = list(vl_sentences - trad_sentences)[:10]  # 상위 10개
        
        return {
            "unique_to_traditional": unique_to_traditional,
            "unique_to_vl": unique_to_vl
        }
    
    def _assess_overall_quality(self, traditional: ContentQualityMetrics, vl: ContentQualityMetrics) -> Dict[str, any]:
        """전체적인 품질 평가"""
        
        # 각 방법의 종합 점수 계산
        def calculate_overall_score(metrics: ContentQualityMetrics) -> float:
            # 가중 평균 (완성도를 가장 중요하게 고려)
            return (
                metrics.completeness_score * 0.4 +
                metrics.coherence_score * 0.3 +
                metrics.readability_score * 0.2 +
                min(1.0, (metrics.financial_terms_count + metrics.regulatory_terms_count) / 50) * 0.1
            )
        
        trad_score = calculate_overall_score(traditional)
        vl_score = calculate_overall_score(vl)
        
        # 추천사항 결정
        score_diff = vl_score - trad_score
        
        if score_diff > 0.2:
            recommendation = "VL_STRONGLY_RECOMMENDED"
            reason = "VL 방법이 품질 측면에서 상당한 개선을 보임"
        elif score_diff > 0.1:
            recommendation = "VL_RECOMMENDED"
            reason = "VL 방법이 품질 개선을 보임"
        elif score_diff > -0.1:
            recommendation = "COMPARABLE"
            reason = "두 방법의 품질이 비슷함"
        else:
            recommendation = "TRADITIONAL_PREFERRED"
            reason = "기존 방법이 품질 측면에서 더 나음"
        
        return {
            "traditional_overall_score": trad_score,
            "vl_overall_score": vl_score,
            "score_difference": score_diff,
            "recommendation": recommendation,
            "reason": reason,
            "quality_summary": {
                "content_richness": "VL" if vl.total_chars > traditional.total_chars * 1.1 else "Traditional",
                "domain_coverage": "VL" if (vl.financial_terms_count + vl.regulatory_terms_count) > (traditional.financial_terms_count + traditional.regulatory_terms_count) else "Traditional",
                "structural_quality": "VL" if (vl.headers_detected + vl.lists_detected) > (traditional.headers_detected + traditional.lists_detected) else "Traditional",
                "visual_content_handling": "VL" if vl.visual_references > traditional.visual_references else "Traditional"
            }
        }


def main():
    """콘텐츠 품질 평가기 테스트"""
    
    evaluator = ContentQualityEvaluator()
    
    # 샘플 텍스트
    sample_traditional = """
    금융분야 AI 보안 가이드라인
    1. 개요
    본 가이드라인은 금융회사의 AI 시스템 보안을 위한 기준을 제시한다.
    
    2. 보안 요구사항
    - 데이터 암호화
    - 접근 제어
    - 감사 추적
    """
    
    sample_vl = """
    금융분야 AI 보안 가이드라인
    
    1. 개요
    본 가이드라인은 금융회사의 AI 시스템 보안을 위한 기준을 제시한다.
    인공지능 기술의 발전에 따라 금융권에서의 AI 활용이 증가하고 있으며,
    이에 따른 보안 위험 관리가 필요하다.
    
    [Figure 1: AI 보안 위험 매트릭스]
    위험도: 높음 - 데이터 유출, 모델 조작
    위험도: 중간 - 편향성 문제, 설명가능성 부족
    위험도: 낮음 - 성능 저하, 호환성 문제
    
    2. 보안 요구사항
    - 데이터 암호화: AES-256 이상 암호화 적용
    - 접근 제어: 역할 기반 접근 제어(RBAC) 구현
    - 감사 추적: 모든 AI 모델 접근 기록 보관
    
    [Table 1: 보안 통제 매트릭스]
    통제 항목 | 적용 수준 | 검증 방법
    암호화 | 필수 | 기술 검토
    접근제어 | 필수 | 정책 검토  
    모니터링 | 권장 | 프로세스 검토
    """
    
    # 품질 평가 실행
    print("=== 콘텐츠 품질 평가 테스트 ===\n")
    
    comparison = evaluator.compare_content_quality(sample_traditional, sample_vl)
    
    # 결과 출력
    print("📊 품질 비교 결과:")
    print(f"- 기존 방법 종합 점수: {comparison['overall_assessment']['traditional_overall_score']:.3f}")
    print(f"- VL 방법 종합 점수: {comparison['overall_assessment']['vl_overall_score']:.3f}")
    print(f"- 점수 차이: {comparison['overall_assessment']['score_difference']:+.3f}")
    print(f"- 추천사항: {comparison['overall_assessment']['recommendation']}")
    print(f"- 사유: {comparison['overall_assessment']['reason']}")
    
    print("\n📈 개선 정도:")
    improvements = comparison['improvements']
    print(f"- 문자 수 증가: {improvements['content_volume']['chars_improvement']:+.1f}%")
    print(f"- 금융 용어 증가: {improvements['domain_coverage']['financial_improvement']:+.1f}%")
    print(f"- 시각적 참조 증가: {improvements['visual_content']['visual_refs_improvement']:+.1f}%")
    
    print(f"\n📋 JSON 결과 저장...")
    with open("content_quality_test.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print("완료!")


if __name__ == "__main__":
    main()