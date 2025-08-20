#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG System Detailed Diagnostic Test
상세한 사고 과정과 판단 근거를 포함한 진단 테스트
"""

import os
import sys
import json
import logging
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import time

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetailedDiagnosticTester:
    """상세 진단 테스터"""
    
    def __init__(self):
        self.rag_system = None
        self.retriever = None
        self.model = None
        self.tokenizer = None
        self.results = []
        
    def initialize_rag(self):
        """RAG 시스템 초기화"""
        logger.info("RAG 시스템 초기화 중...")
        
        from scripts.load_rag_v2 import RAGSystemV2
        
        self.rag_system = RAGSystemV2()
        self.rag_system.load_all()
        self.retriever = self.rag_system.create_hybrid_retriever()
        
        logger.info(f"RAG 초기화 완료: BM25={self.retriever.bm25_weight:.1%}, Vector={self.retriever.vector_weight:.1%}")
    
    def initialize_llm(self):
        """LLM 초기화"""
        logger.info("LLM 초기화 중...")
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("LLM 초기화 완료")
    
    def create_detailed_prompt(self, question: str, contexts: List[str], scores: List[Dict]) -> str:
        """상세 진단 프롬프트 생성"""
        
        # 문서 포맷팅
        docs_text = ""
        for i, (ctx, score) in enumerate(zip(contexts, scores), 1):
            docs_text += f"\n[문서 {i}]\n"
            docs_text += f"내용: {ctx[:300]}...\n"
            docs_text += f"BM25 점수: {score['bm25_score']:.4f}\n"
            docs_text += f"Vector 점수: {score['vector_score']:.4f}\n"
            docs_text += f"Hybrid 점수: {score['hybrid_score']:.4f}\n"
            docs_text += f"검색 방법: {', '.join(score['retrieval_methods'])}\n"
            docs_text += "-" * 50
        
        prompt = f"""당신은 금융보안 전문가입니다. 다음 문제를 해결하는 과정을 상세히 설명해주세요.

문제: {question}

검색된 관련 문서 5개:
{docs_text}

다음 형식으로 답변해주세요:

1. 사고 과정:
[문제를 이해하고 해결하는 사고 과정을 단계별로 설명]

2. 문서 분석:
[각 문서의 유용성과 관련성 평가]
[BM25 점수가 높은 문서 vs Vector 점수가 높은 문서의 차이점]

3. 판단 근거:
[최종 답변을 도출한 핵심 근거]
[어떤 문서가 가장 유용했는지와 그 이유]

4. 점수 해석:
[BM25, Vector, Hybrid 점수의 의미와 중요도]
[어떤 검색 방식이 이 문제에 더 적합했는지]

5. 최종 답변:
[명확한 답변 제시]
"""
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """LLM으로 답변 생성"""
        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=0.3,
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 응답 추출
            if "1. 사고 과정:" in response:
                response = response.split("1. 사고 과정:")[1]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"생성 오류: {e}")
            return "답변 생성 실패"
    
    def analyze_question(self, question_id: str, question: str, question_type: str) -> Dict:
        """단일 문제 상세 분석"""
        logger.info(f"\n분석 중: {question_id} ({question_type})")
        
        # 문서 검색
        results = self.retriever.search(question, k=5)
        
        contexts = []
        scores = []
        
        for result in results:
            contexts.append(getattr(result, 'content', ''))
            scores.append({
                'bm25_score': float(getattr(result, 'bm25_score', 0)),
                'vector_score': float(getattr(result, 'vector_score', 0)),
                'hybrid_score': float(getattr(result, 'hybrid_score', 0)),
                'retrieval_methods': getattr(result, 'retrieval_methods', [])
            })
        
        # 상세 프롬프트 생성 및 답변
        prompt = self.create_detailed_prompt(question, contexts, scores)
        detailed_response = self.generate_answer(prompt)
        
        # 점수 분석
        avg_bm25 = np.mean([s['bm25_score'] for s in scores])
        avg_vector = np.mean([s['vector_score'] for s in scores])
        ratio = avg_bm25 / (avg_vector + 1e-10)
        
        # 결과 구성
        result = {
            'question_id': question_id,
            'question_type': question_type,
            'question': question,
            'retrieved_documents': [
                {
                    'doc_id': i,
                    'content_preview': ctx[:200] + '...' if len(ctx) > 200 else ctx,
                    'bm25_score': score['bm25_score'],
                    'vector_score': score['vector_score'],
                    'hybrid_score': score['hybrid_score'],
                    'retrieval_methods': score['retrieval_methods']
                }
                for i, (ctx, score) in enumerate(zip(contexts, scores), 1)
            ],
            'score_analysis': {
                'avg_bm25': float(avg_bm25),
                'avg_vector': float(avg_vector),
                'bm25_vector_ratio': float(ratio),
                'dominance': 'BM25' if ratio > 1.5 else 'Vector' if ratio < 0.67 else 'Balanced'
            },
            'llm_analysis': {
                'full_response': detailed_response,
                'thought_process': self.extract_section(detailed_response, "사고 과정"),
                'document_analysis': self.extract_section(detailed_response, "문서 분석"),
                'judgment_basis': self.extract_section(detailed_response, "판단 근거"),
                'score_interpretation': self.extract_section(detailed_response, "점수 해석"),
                'final_answer': self.extract_section(detailed_response, "최종 답변")
            },
            'pipeline_conclusion': {
                'hybrid_weight': f"BM25={self.retriever.bm25_weight:.1%}, Vector={self.retriever.vector_weight:.1%}",
                'top_document_used': scores[0]['retrieval_methods'][0] if scores and scores[0]['retrieval_methods'] else 'unknown',
                'effectiveness': self.evaluate_effectiveness(ratio, question_type)
            }
        }
        
        return result
    
    def extract_section(self, text: str, section_name: str) -> str:
        """텍스트에서 특정 섹션 추출"""
        try:
            if section_name in text:
                parts = text.split(section_name + ":")
                if len(parts) > 1:
                    section_text = parts[1].split("\n\n")[0].strip()
                    return section_text[:500]  # 최대 500자
        except:
            pass
        return "추출 실패"
    
    def evaluate_effectiveness(self, ratio: float, question_type: str) -> str:
        """파이프라인 효과성 평가"""
        if ratio > 2.0:
            return "BM25가 매우 효과적 - 키워드 매칭이 핵심"
        elif ratio > 1.5:
            return "BM25가 더 효과적 - 키워드 중심 검색 유리"
        elif ratio < 0.5:
            return "Vector가 매우 효과적 - 의미적 유사성이 핵심"
        elif ratio < 0.67:
            return "Vector가 더 효과적 - 의미 기반 검색 유리"
        else:
            return "균형잡힌 효과 - 하이브리드 방식이 적절"
    
    def run_test(self, num_questions: int = 5):
        """진단 테스트 실행"""
        # 테스트 문제 준비
        test_questions = [
            {"ID": "TEST_000", "Type": "multiple_choice", 
             "Question": "금융산업의 이해와 관련하여 금융투자업의 구분에 해당하지 않는 것은?\n1 소비자금융업\n2 투자자문업\n3 투자매매업\n4 투자중개업\n5 보험중개업"},
            {"ID": "TEST_001", "Type": "multiple_choice",
             "Question": "위험 관리 계획 수립 시 고려해야 할 요소로 적절하지 않은 것은?\n1 수행인력\n2 위험 수용\n3 위험 대응 전략 선정\n4 대상\n5 기간"},
            {"ID": "TEST_002", "Type": "multiple_choice",
             "Question": "관리체계 수립 및 운영'의 '정책 수립' 단계에서 가장 중요한 요소는 무엇인가?\n1 정보보호 및 개인정보보호 정책의 제·개정\n2 경영진의 참여\n3 최고책임자의 지정\n4 자원 할당\n5 내부 감사 절차의 수립"},
            {"ID": "TEST_004", "Type": "descriptive",
             "Question": "트로이 목마(Trojan) 기반 원격제어 악성코드(RAT)의 특징과 주요 탐지 지표를 설명하세요."},
            {"ID": "TEST_007", "Type": "descriptive",
             "Question": "전자금융거래법에 따라 이용자가 금융 분쟁조정을 신청할 수 있는 기관을 기술하세요."}
        ]
        
        # 선택된 문제만 실행
        test_questions = test_questions[:num_questions]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"상세 진단 테스트 시작 ({len(test_questions)}문제)")
        logger.info(f"{'='*60}")
        
        for idx, q in enumerate(test_questions, 1):
            logger.info(f"\n[{idx}/{len(test_questions)}] 처리 중...")
            
            start_time = time.time()
            result = self.analyze_question(q['ID'], q['Question'], q['Type'])
            elapsed = time.time() - start_time
            
            result['processing_time'] = f"{elapsed:.1f}초"
            self.results.append(result)
            
            logger.info(f"  처리 시간: {elapsed:.1f}초")
            logger.info(f"  점수 우세: {result['score_analysis']['dominance']}")
            logger.info(f"  효과성: {result['pipeline_conclusion']['effectiveness']}")
        
        # 결과 저장
        self.save_results()
    
    def save_results(self):
        """결과를 단일 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"detailed_diagnostic_results_{timestamp}.json"
        
        # 요약 통계
        summary = {
            'test_info': {
                'timestamp': timestamp,
                'total_questions': len(self.results),
                'rag_config': {
                    'bm25_weight': float(self.retriever.bm25_weight),
                    'vector_weight': float(self.retriever.vector_weight),
                    'chunk_count': 8270
                }
            },
            'overall_statistics': {
                'dominance_distribution': self.calculate_dominance_stats(),
                'average_scores': self.calculate_average_scores(),
                'effectiveness_summary': self.summarize_effectiveness()
            },
            'detailed_results': self.results
        }
        
        # JSON 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n결과 저장 완료: {output_file}")
        
        # 간단한 요약 출력
        self.print_summary()
    
    def calculate_dominance_stats(self) -> Dict:
        """우세 패턴 통계"""
        dominance_counts = {'BM25': 0, 'Vector': 0, 'Balanced': 0}
        for r in self.results:
            dom = r['score_analysis']['dominance']
            dominance_counts[dom] += 1
        
        total = len(self.results)
        return {
            'BM25': f"{dominance_counts['BM25']}/{total} ({dominance_counts['BM25']/total*100:.0f}%)",
            'Vector': f"{dominance_counts['Vector']}/{total} ({dominance_counts['Vector']/total*100:.0f}%)",
            'Balanced': f"{dominance_counts['Balanced']}/{total} ({dominance_counts['Balanced']/total*100:.0f}%)"
        }
    
    def calculate_average_scores(self) -> Dict:
        """평균 점수 계산"""
        avg_bm25 = np.mean([r['score_analysis']['avg_bm25'] for r in self.results])
        avg_vector = np.mean([r['score_analysis']['avg_vector'] for r in self.results])
        
        return {
            'avg_bm25': float(avg_bm25),
            'avg_vector': float(avg_vector),
            'overall_ratio': float(avg_bm25 / (avg_vector + 1e-10))
        }
    
    def summarize_effectiveness(self) -> List[str]:
        """효과성 요약"""
        effectiveness = [r['pipeline_conclusion']['effectiveness'] for r in self.results]
        return list(set(effectiveness))  # 중복 제거
    
    def print_summary(self):
        """요약 출력"""
        print("\n" + "="*60)
        print("상세 진단 테스트 완료")
        print("="*60)
        
        stats = self.calculate_dominance_stats()
        print("\n점수 우세 분포:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        avg_scores = self.calculate_average_scores()
        print(f"\n평균 점수:")
        print(f"  BM25: {avg_scores['avg_bm25']:.4f}")
        print(f"  Vector: {avg_scores['avg_vector']:.4f}")
        print(f"  전체 비율: {avg_scores['overall_ratio']:.2f}")
        
        print("\n효과성 평가:")
        for eff in self.summarize_effectiveness():
            print(f"  - {eff}")
        
        print("\n" + "="*60)


def main():
    """메인 실행 함수"""
    tester = DetailedDiagnosticTester()
    
    # 초기화
    tester.initialize_rag()
    tester.initialize_llm()
    
    # 테스트 실행 (5문제)
    tester.run_test(num_questions=5)
    
    logger.info("\n상세 진단 테스트 완료!")


if __name__ == "__main__":
    main()