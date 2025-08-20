#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BM25 vs Vector 독립 성능 비교 실험
각 검색 방법을 단독으로 사용하여 20문제 전체에 대한 성능 평가
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


class ComparisonTester:
    """BM25 vs Vector 비교 테스터"""
    
    def __init__(self):
        self.rag_system = None
        self.model = None
        self.tokenizer = None
        self.bm25_results = []
        self.vector_results = []
        
    def initialize_rag(self):
        """RAG 시스템 초기화"""
        logger.info("RAG 시스템 초기화 중...")
        
        from scripts.load_rag_v2 import RAGSystemV2
        
        self.rag_system = RAGSystemV2()
        self.rag_system.load_all()
        
        logger.info("RAG 초기화 완료")
    
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
    
    def create_bm25_only_retriever(self):
        """BM25 전용 retriever 생성"""
        retriever = self.rag_system.create_hybrid_retriever()
        retriever.bm25_weight = 1.0
        retriever.vector_weight = 0.0
        logger.info("BM25 전용 retriever 생성 (100% BM25)")
        return retriever
    
    def create_vector_only_retriever(self):
        """Vector 전용 retriever 생성"""
        retriever = self.rag_system.create_hybrid_retriever()
        retriever.bm25_weight = 0.0
        retriever.vector_weight = 1.0
        logger.info("Vector 전용 retriever 생성 (100% Vector)")
        return retriever
    
    def is_multiple_choice(self, text: str) -> bool:
        """객관식 문제 판별"""
        lines = text.strip().split('\n')
        options = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and len(line) > 2:
                if line[1] in ['.', ')', ' ', ':']:
                    options.append(line)
        return len(options) >= 2
    
    def create_prompt(self, question: str, contexts: List[str]) -> str:
        """프롬프트 생성"""
        is_mc = self.is_multiple_choice(question)
        
        docs_text = ""
        for i, ctx in enumerate(contexts[:5], 1):
            # 문서 내용 정리
            if ctx:
                preview = ctx[:400].replace('\n\n', '\n')
                docs_text += f"\n[문서 {i}]\n{preview}\n"
        
        if is_mc:
            prompt = f"""당신은 금융보안 전문가입니다.

질문: {question}

참고 문서:
{docs_text}

위 문서를 참고하여 정답 번호만 답하세요.
답: """
        else:
            prompt = f"""당신은 금융보안 전문가입니다.

질문: {question}

참고 문서:
{docs_text}

위 문서를 참고하여 간략하게 답하세요.
답: """
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """LLM으로 답변 생성"""
        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2500)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.3,
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 답변 추출
            if "답: " in response:
                response = response.split("답: ")[-1].strip()
            elif "답:" in response:
                response = response.split("답:")[-1].strip()
            
            return response[:500]  # 최대 500자
            
        except Exception as e:
            logger.error(f"생성 오류: {e}")
            return "답변 생성 실패"
    
    def extract_answer(self, response: str, question: str) -> str:
        """답변에서 최종 답 추출"""
        is_mc = self.is_multiple_choice(question)
        
        if is_mc:
            # 숫자 추출
            import re
            # 처음 나오는 1-5 사이 숫자 찾기
            numbers = re.findall(r'[1-5]', response)
            if numbers:
                return numbers[0]
            return "1"
        else:
            # 주관식은 첫 문장 또는 전체 응답
            sentences = response.split('.')
            if sentences:
                return sentences[0].strip() + '.'
            return response.strip()
    
    def test_single_method(self, retriever, method_name: str, questions: List[Dict]) -> List[Dict]:
        """단일 검색 방법으로 테스트"""
        results = []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{method_name} 방식 테스트 시작")
        logger.info(f"{'='*60}")
        
        for idx, q in enumerate(questions, 1):
            logger.info(f"\n[{idx}/{len(questions)}] {q['ID']} 처리 중...")
            start_time = time.time()
            
            # 문서 검색
            search_results = retriever.search(q['Question'], k=5)
            
            # 문서 내용과 메타데이터 추출
            contexts = []
            doc_titles = []
            
            for result in search_results:
                content = getattr(result, 'content', '')
                contexts.append(content)
                
                # 문서 제목 추출 (첫 줄 또는 ##로 시작하는 부분)
                if content:
                    lines = content.split('\n')
                    title = "제목 없음"
                    for line in lines:
                        if line.strip():
                            if line.startswith("##"):
                                title = line[2:].strip()[:50]
                            else:
                                title = line.strip()[:50]
                            break
                    doc_titles.append(title)
                else:
                    doc_titles.append("제목 없음")
            
            # 프롬프트 생성 및 답변
            prompt = self.create_prompt(q['Question'], contexts)
            raw_answer = self.generate_answer(prompt)
            final_answer = self.extract_answer(raw_answer, q['Question'])
            
            elapsed = time.time() - start_time
            
            # 결과 저장
            result = {
                'question_id': q['ID'],
                'question_type': 'multiple_choice' if self.is_multiple_choice(q['Question']) else 'descriptive',
                'question_full': q['Question'],
                'answer': final_answer,
                'raw_response': raw_answer[:200],
                'referenced_documents': doc_titles[:5],
                'method': method_name,
                'processing_time': f"{elapsed:.1f}초"
            }
            
            results.append(result)
            
            logger.info(f"  답변: {final_answer[:50]}...")
            logger.info(f"  참고 문서 수: {len(doc_titles)}")
            logger.info(f"  처리 시간: {elapsed:.1f}초")
        
        return results
    
    def load_test_questions(self) -> List[Dict]:
        """20문제 로드"""
        # test.csv에서 처음 20문제 로드
        try:
            df = pd.read_csv('test.csv')
            questions = []
            
            # 객관식 10개, 주관식 10개 선별
            mc_count = 0
            desc_count = 0
            
            for idx, row in df.iterrows():
                question = row['Question']
                is_mc = self.is_multiple_choice(question)
                
                if is_mc and mc_count < 10:
                    questions.append({
                        'ID': row['ID'],
                        'Question': question
                    })
                    mc_count += 1
                elif not is_mc and desc_count < 10:
                    questions.append({
                        'ID': row['ID'],
                        'Question': question
                    })
                    desc_count += 1
                
                if mc_count >= 10 and desc_count >= 10:
                    break
            
            logger.info(f"로드된 문제: 객관식 {mc_count}개, 주관식 {desc_count}개")
            return questions
            
        except Exception as e:
            logger.error(f"문제 로드 실패: {e}")
            # 기본 테스트 문제 사용
            return self.get_default_questions()
    
    def get_default_questions(self) -> List[Dict]:
        """기본 테스트 문제 (이전 실험에서 사용한 문제들)"""
        return [
            {"ID": "TEST_000", "Question": "금융산업의 이해와 관련하여 금융투자업의 구분에 해당하지 않는 것은?\n1 소비자금융업\n2 투자자문업\n3 투자매매업\n4 투자중개업\n5 보험중개업"},
            {"ID": "TEST_001", "Question": "위험 관리 계획 수립 시 고려해야 할 요소로 적절하지 않은 것은?\n1 수행인력\n2 위험 수용\n3 위험 대응 전략 선정\n4 대상\n5 기간"},
            {"ID": "TEST_002", "Question": "관리체계 수립 및 운영'의 '정책 수립' 단계에서 가장 중요한 요소는 무엇인가?\n1 정보보호 및 개인정보보호 정책의 제·개정\n2 경영진의 참여\n3 최고책임자의 지정\n4 자원 할당\n5 내부 감사 절차의 수립"},
            {"ID": "TEST_003", "Question": "다음 중 개인정보 보호법상 개인정보처리자가 아닌 것은?\n1 법인\n2 단체\n3 개인\n4 공공기관\n5 개인정보를 처리하지 않는 자"},
            {"ID": "TEST_004", "Question": "트로이 목마(Trojan) 기반 원격제어 악성코드(RAT)의 특징과 주요 탐지 지표를 설명하세요."},
            {"ID": "TEST_005", "Question": "APT 공격의 단계별 특징을 설명하세요."},
            {"ID": "TEST_006", "Question": "랜섬웨어 대응 방안을 기술하세요."},
            {"ID": "TEST_007", "Question": "전자금융거래법에 따라 이용자가 금융 분쟁조정을 신청할 수 있는 기관을 기술하세요."},
            {"ID": "TEST_008", "Question": "개인정보 영향평가의 주요 단계를 설명하세요."},
            {"ID": "TEST_009", "Question": "클라우드 보안 위협과 대응 방안을 설명하세요."},
            {"ID": "TEST_010", "Question": "금융회사의 정보보호 최고책임자(CISO)의 자격 요건은?\n1 정보보호 관련 석사 이상\n2 5년 이상 경력\n3 10년 이상 경력\n4 정보보호 자격증 보유\n5 법적 요건 없음"},
            {"ID": "TEST_011", "Question": "개인정보의 안전성 확보조치 기준에서 요구하는 암호화 대상이 아닌 것은?\n1 비밀번호\n2 바이오정보\n3 주민등록번호\n4 신용카드번호\n5 공개된 정보"},
            {"ID": "TEST_012", "Question": "정보보호 관리체계(ISMS) 인증 유효기간은?\n1 1년\n2 2년\n3 3년\n4 4년\n5 5년"},
            {"ID": "TEST_013", "Question": "제로 트러스트(Zero Trust) 보안 모델의 핵심 원칙을 설명하세요."},
            {"ID": "TEST_014", "Question": "소프트웨어 정의 경계(SDP)의 구성 요소와 동작 방식을 설명하세요."},
            {"ID": "TEST_015", "Question": "금융권 마이데이터 서비스의 보안 요구사항을 기술하세요."},
            {"ID": "TEST_016", "Question": "블록체인 기반 금융 서비스의 보안 고려사항을 설명하세요."},
            {"ID": "TEST_017", "Question": "AI/ML 기반 이상거래 탐지 시스템의 구현 방안을 제시하세요."},
            {"ID": "TEST_018", "Question": "양자 컴퓨팅이 현재 암호체계에 미치는 영향과 대응 방안을 설명하세요."},
            {"ID": "TEST_019", "Question": "금융 API 보안을 위한 OAuth 2.0과 OpenID Connect의 활용 방안을 설명하세요."}
        ]
    
    def run_comparison(self):
        """비교 실험 실행"""
        # 테스트 문제 로드
        questions = self.load_test_questions()
        
        total_start = time.time()
        
        # BM25 전용 테스트
        logger.info("\n" + "#"*60)
        logger.info("# BM25 전용 모드 테스트 시작")
        logger.info("#"*60)
        bm25_retriever = self.create_bm25_only_retriever()
        self.bm25_results = self.test_single_method(bm25_retriever, "BM25-Only", questions)
        
        # 메모리 정리
        import gc
        gc.collect()
        time.sleep(2)
        
        # Vector 전용 테스트
        logger.info("\n" + "#"*60)
        logger.info("# Vector 전용 모드 테스트 시작")
        logger.info("#"*60)
        vector_retriever = self.create_vector_only_retriever()
        self.vector_results = self.test_single_method(vector_retriever, "Vector-Only", questions)
        
        total_elapsed = time.time() - total_start
        logger.info(f"\n전체 실험 시간: {total_elapsed/60:.1f}분")
        
        # 결과 저장
        self.save_results()
    
    def save_results(self):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 통합 결과
        combined_results = {
            'test_info': {
                'timestamp': timestamp,
                'total_questions': len(self.bm25_results),
                'test_type': 'BM25 vs Vector Independent Comparison'
            },
            'bm25_results': self.bm25_results,
            'vector_results': self.vector_results,
            'comparison_summary': self.create_comparison_summary()
        }
        
        output_file = f"comparison_results_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n결과 저장 완료: {output_file}")
        
        # 요약 출력
        self.print_summary()
    
    def create_comparison_summary(self) -> Dict:
        """비교 요약 생성"""
        # 답변 비교
        comparison = []
        for bm25_r, vector_r in zip(self.bm25_results, self.vector_results):
            comparison.append({
                'question_id': bm25_r['question_id'],
                'question_type': bm25_r['question_type'],
                'bm25_answer': bm25_r['answer'][:100],
                'vector_answer': vector_r['answer'][:100],
                'same_answer': bm25_r['answer'] == vector_r['answer']
            })
        
        # 통계
        same_count = sum(1 for c in comparison if c['same_answer'])
        mc_questions = [c for c in comparison if c['question_type'] == 'multiple_choice']
        desc_questions = [c for c in comparison if c['question_type'] == 'descriptive']
        
        summary = {
            'total_questions': len(comparison),
            'same_answers': same_count,
            'different_answers': len(comparison) - same_count,
            'agreement_rate': f"{same_count/len(comparison)*100:.1f}%",
            'mc_agreement': sum(1 for c in mc_questions if c['same_answer']),
            'desc_agreement': sum(1 for c in desc_questions if c['same_answer']),
            'detailed_comparison': comparison
        }
        return summary
    
    def print_summary(self):
        """요약 출력"""
        print("\n" + "="*60)
        print("BM25 vs Vector 비교 실험 완료")
        print("="*60)
        
        summary = self.create_comparison_summary()
        
        print(f"\n총 문제 수: {summary['total_questions']}")
        print(f"동일 답변: {summary['same_answers']}개")
        print(f"다른 답변: {summary['different_answers']}개")
        print(f"일치율: {summary['agreement_rate']}")
        print(f"\n객관식 일치: {summary['mc_agreement']}개")
        print(f"주관식 일치: {summary['desc_agreement']}개")
        
        print("\n" + "="*60)
        print("각 문제의 상세 답변은 JSON 파일을 확인하세요.")
        print("="*60)


def main():
    """메인 실행 함수"""
    tester = ComparisonTester()
    
    # 초기화
    tester.initialize_rag()
    tester.initialize_llm()
    
    # 비교 실험 실행
    tester.run_comparison()
    
    logger.info("\n비교 실험 완료!")


if __name__ == "__main__":
    main()