#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BM25 vs Vector 독립 성능 비교 실험 - 개선된 프롬프트 버전
generate_final_submission_bm25_070.py의 프롬프트와 설정 적용
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
import re

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedComparisonTester:
    """개선된 프롬프트를 사용한 BM25 vs Vector 비교 테스터"""
    
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
        """LLM 초기화 - Qwen2.5-7B-Instruct (16-bit)"""
        logger.info("LLM 초기화 중 (16-bit)...")
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        
        # 16-bit 모델 로드
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
    
    def get_max_choice_number(self, question: str) -> int:
        """객관식 문제의 최대 보기 번호 추출"""
        lines = question.strip().split('\n')
        max_num = 4  # 기본값
        
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                # 첫 번째 숫자 추출
                match = re.match(r'^(\d+)', line)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)
        
        return max_num
    
    def retrieve_context(self, retriever, query: str, k: int = 5) -> List[str]:
        """RAG 검색 수행"""
        try:
            results = retriever.search(query, k=k)
            contexts = []
            for result in results:
                content = getattr(result, 'content', '')
                if content:
                    contexts.append(content)
            return contexts
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def create_improved_prompt(self, question: str, contexts: List[str]) -> str:
        """개선된 프롬프트 생성 - generate_final_submission_bm25_070.py와 동일"""
        
        is_mc = self.is_multiple_choice(question)
        
        if is_mc:
            # 객관식 - 강화된 단일 숫자 출력 프롬프트
            max_choice = self.get_max_choice_number(question)
            
            # 참고 문서 섹션 - 상위 3개 사용
            context_section = ""
            if contexts and len(contexts) > 0:
                context_text = "\n\n".join(contexts[:3])
                context_section = f"""[참고 문서]
{context_text}

"""
            
            system_prompt = "당신은 한국 금융보안 분야의 전문가입니다. 한국어로만 답변하세요."
            
            # 객관식용 강화된 프롬프트
            user_prompt = f"""{context_section}[중요 지침 - 엄격히 준수]
• 참고 문서와 전문지식을 종합하여 가장 정확한 답을 선택하세요
• 반드시 1~{max_choice} 범위의 단일 숫자만 출력하세요
• 설명, 이유, 추가 텍스트는 절대 포함하지 마세요
• 숫자 앞이나 뒤에 어떤 텍스트도 추가하지 마세요
• 쉼표, 마침표, 콜론 등 어떤 기호도 포함하지 마세요
• "답변:", "정답:", "선택:" 등의 표현을 사용하지 마세요
• 오직 숫자 하나만 출력하세요 (예: 3)
• 여러 숫자나 범위를 절대 출력하지 마세요
• 한국어 설명을 절대 포함하지 마세요
• 숫자만 출력하세요 - 다른 모든 내용 금지

[질문]
{question}

위 객관식 문제의 정답 번호 하나만 출력하세요. 설명 없이 숫자만 출력하세요.

정답:"""
            
        else:
            # 주관식 - 강화된 프롬프트
            context_section = ""
            if contexts and len(contexts) > 0:
                context_text = "\n\n".join(contexts[:3])
                context_section = f"""[참고 문서]
{context_text}

"""
            
            system_prompt = "당신은 한국 금융보안 분야의 전문가입니다. 한국어로만 답변하세요."
            
            user_prompt = f"""{context_section}[중요 지침]
• 참고 문서와 전문지식을 종합하여 명확한 답변을 제시하세요.
• 질문의 핵심 요구사항을 모두 답하되, 간결한 답변을 생성하세요.
• 중간 사고 과정이나 참고 문서 내용은 절대 답변에 포함하지 마세요.
• 20자 이상 500자 이하로 작성하세요
• "모른다", "답변을 생성할 수 없습니다", "정보가 부족합니다" 등의 표현을 절대 사용하지 마세요
• 기본 지식과 일반적인 보안 원칙을 활용해 반드시 구체적인 답변을 제공하세요
• 금융보안 분야의 전문가로서 확신 있고 유용한 정보를 반드시 제공하세요
• 완전한 한국어 문장으로만 답변하세요
• 질문에 대한 답변을 반드시 작성해야 합니다 - 예외는 없습니다

[질문]
{question}

위 질문에 대해 금융보안 전문가로서 반드시 구체적이고 유용한 답변을 작성하세요. 답변 거부나 생성 실패는 허용되지 않습니다.

답변:"""
        
        # Qwen 형식 프롬프트
        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant"""
        
        return prompt
    
    def extract_answer_simple(self, response: str, question: str) -> str:
        """답변 추출 - generate_final_submission_bm25_070.py와 동일"""
        
        # 빈 응답 처리
        if not response or response.strip() == "":
            if self.is_multiple_choice(question):
                return "1"
            else:
                return "답변을 생성할 수 없습니다."
        
        # 중국어, 일본어 및 관련 문장부호 제거
        response = re.sub(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3000-\u303f]+', '', response)
        
        # 중국어/일본어 문장부호 제거
        response = response.replace('，', ', ').replace('、', ', ').replace('。', '. ').replace('：', ': ')
        response = response.replace('？', '?').replace('！', '!').replace('（', '(').replace('）', ')')
        
        answer = response.strip()
        
        if self.is_multiple_choice(question):
            # 객관식: 첫 번째 단일 숫자만 추출
            max_choice = self.get_max_choice_number(question)
            
            # "정답:" 이후의 내용 추출
            if "정답:" in answer:
                answer = answer.split("정답:")[-1].strip()
            elif "답변:" in answer:
                answer = answer.split("답변:")[-1].strip()
            
            # 모든 숫자 찾기
            numbers = re.findall(r'\d+', answer)
            
            if numbers:
                # 첫 번째 숫자를 int로 변환
                first_num = int(numbers[0])
                
                # 보기 범위 내의 숫자인지 확인
                if 1 <= first_num <= max_choice:
                    return str(first_num)
                
                # 범위 밖이면 다른 숫자 찾기
                for num_str in numbers[1:]:
                    num = int(num_str)
                    if 1 <= num <= max_choice:
                        return str(num)
            
            # 숫자를 찾지 못하면 기본값
            return "1"
        
        else:
            # 주관식
            # 영어 단어 필터링 (일부 허용되는 단어 제외)
            allowed_english = ['API', 'CPU', 'GPU', 'URL', 'DNS', 'SQL', 'VPN', 'SSL', 'TLS', 'HTTP', 'HTTPS',
                              'AI', 'ML', 'DL', 'IoT', 'PKI', 'OTP', 'MFA', '2FA', 'FIDO', 'CISO', 'ISMS',
                              'ISO', 'IDS', 'IPS', 'DDoS', 'XSS', 'CSRF', 'SQLi', 'APT', 'RAT', 'C&C']
            words = response.split()
            filtered_words = []
            for word in words:
                # 영어 단어인지 확인 (알파벳만 포함)
                if re.match(r'^[A-Za-z]+$', word):
                    # 허용된 단어인지 확인
                    if word.upper() in allowed_english:
                        filtered_words.append(word)
                    # 아니면 제거
                else:
                    filtered_words.append(word)
            response = ' '.join(filtered_words)
            
            answer = response.strip()
            
            # "답변:" 이후의 내용 추출
            if "답변:" in answer:
                answer = answer.split("답변:")[-1].strip()
            
            # 프롬프트 유출 제거
            if answer.startswith("assistant"):
                answer = answer[9:].strip()
            elif answer.startswith("user"):
                answer = answer[4:].strip()
            
            # 최대 길이 제한 - 500자
            if len(answer) > 500:
                answer = answer[:500]
            
            return answer
    
    def generate_answer(self, prompt: str, question: str) -> str:
        """LLM으로 답변 생성 - 개선된 생성 파라미터"""
        try:
            import torch
            
            # 토큰화
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 객관식/주관식에 따른 생성 파라미터 조정
            if self.is_multiple_choice(question):
                # 객관식: 매우 짧게, deterministic하게
                generation_config = {
                    "max_new_tokens": 5,  # 극도로 짧게 - 숫자만 생성
                    "temperature": 0.01,   # 거의 deterministic
                    "top_p": 0.95,
                    "do_sample": False,    # Greedy decoding
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "repetition_penalty": 1.2  # 반복 방지
                }
            else:
                # 주관식: 안정적인 생성
                generation_config = {
                    "max_new_tokens": 512,
                    "temperature": 0.05,  # 매우 낮춰서 안정적인 생성
                    "top_p": 0.95,
                    "do_sample": False,  # Greedy decoding
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "repetition_penalty": 1.1,
                    "no_repeat_ngram_size": 3
                }
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
            # 디코딩
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 프롬프트 제거하고 답변만 추출
            if "<|im_start|>assistant" in full_response:
                parts = full_response.split("<|im_start|>assistant")
                if len(parts) > 1:
                    response = parts[-1].strip()
                else:
                    response = full_response.strip()
            elif "정답:" in full_response:
                parts = full_response.split("정답:")
                if len(parts) > 1:
                    response = parts[-1].strip()
                else:
                    response = full_response.strip()
            elif "답변:" in full_response:
                parts = full_response.split("답변:")
                if len(parts) > 1:
                    response = parts[-1].strip()
                else:
                    response = full_response.strip()
            else:
                input_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                if full_response.startswith(input_text):
                    response = full_response[len(input_text):].strip()
                else:
                    response = full_response.strip()
            
            # 답변 추출
            answer = self.extract_answer_simple(response, question)
            
            logger.debug(f"Raw response: {response[:100]}...")
            logger.debug(f"Extracted answer: {answer}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            
            if self.is_multiple_choice(question):
                return "1"
            else:
                return "답변 생성 중 오류가 발생했습니다."
    
    def test_single_method(self, retriever, method_name: str, questions: List[Dict]) -> List[Dict]:
        """단일 검색 방법으로 테스트"""
        results = []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{method_name} 방식 테스트 시작")
        logger.info(f"{'='*60}")
        
        for idx, q in enumerate(questions, 1):
            logger.info(f"\n[{idx}/{len(questions)}] {q['ID']} 처리 중...")
            start_time = time.time()
            
            # 문서 검색 (상위 5개)
            contexts = self.retrieve_context(retriever, q['Question'], k=5)
            
            # 문서 제목 추출
            doc_titles = []
            for ctx in contexts:
                if ctx:
                    lines = ctx.split('\n')
                    title = "제목 없음"
                    for line in lines:
                        if line.strip():
                            if line.startswith("##"):
                                title = line[2:].strip()[:50]
                            elif line.startswith("#"):
                                title = line[1:].strip()[:50]
                            else:
                                title = line.strip()[:50]
                            break
                    doc_titles.append(title)
            
            # 프롬프트 생성 (개선된 버전)
            prompt = self.create_improved_prompt(q['Question'], contexts)
            
            # 답변 생성
            answer = self.generate_answer(prompt, q['Question'])
            
            elapsed = time.time() - start_time
            
            # 결과 저장
            result = {
                'question_id': q['ID'],
                'question_type': 'multiple_choice' if self.is_multiple_choice(q['Question']) else 'descriptive',
                'question_full': q['Question'],
                'answer': answer,
                'referenced_documents': doc_titles[:5],
                'method': method_name,
                'processing_time': f"{elapsed:.1f}초"
            }
            
            results.append(result)
            
            logger.info(f"  답변: {answer[:50]}...")
            logger.info(f"  참고 문서 수: {len(doc_titles)}")
            logger.info(f"  처리 시간: {elapsed:.1f}초")
        
        return results
    
    def load_test_questions(self) -> List[Dict]:
        """20문제 로드"""
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
            return []
    
    def run_comparison(self):
        """비교 실험 실행"""
        # 테스트 문제 로드
        questions = self.load_test_questions()
        
        if not questions:
            logger.error("테스트 문제를 로드할 수 없습니다.")
            return
        
        total_start = time.time()
        
        # BM25 전용 테스트
        logger.info("\n" + "#"*60)
        logger.info("# BM25 전용 모드 테스트 시작 (개선된 프롬프트)")
        logger.info("#"*60)
        bm25_retriever = self.create_bm25_only_retriever()
        self.bm25_results = self.test_single_method(bm25_retriever, "BM25-Only", questions)
        
        # 메모리 정리
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)
        
        # Vector 전용 테스트
        logger.info("\n" + "#"*60)
        logger.info("# Vector 전용 모드 테스트 시작 (개선된 프롬프트)")
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
                'test_type': 'BM25 vs Vector Comparison with Improved Prompts',
                'prompt_version': 'generate_final_submission_bm25_070.py'
            },
            'bm25_results': self.bm25_results,
            'vector_results': self.vector_results,
            'comparison_summary': self.create_comparison_summary()
        }
        
        output_file = f"improved_comparison_results_{timestamp}.json"
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
                'bm25_answer': bm25_r['answer'],
                'vector_answer': vector_r['answer'],
                'same_answer': bm25_r['answer'] == vector_r['answer']
            })
        
        # 통계
        same_count = sum(1 for c in comparison if c['same_answer'])
        mc_questions = [c for c in comparison if c['question_type'] == 'multiple_choice']
        desc_questions = [c for c in comparison if c['question_type'] == 'descriptive']
        
        mc_same = sum(1 for c in mc_questions if c['same_answer'])
        desc_same = sum(1 for c in desc_questions if c['same_answer'])
        
        summary = {
            'total_questions': len(comparison),
            'same_answers': same_count,
            'different_answers': len(comparison) - same_count,
            'agreement_rate': f"{same_count/len(comparison)*100:.1f}%" if comparison else "0%",
            'mc_total': len(mc_questions),
            'mc_agreement': mc_same,
            'mc_agreement_rate': f"{mc_same/len(mc_questions)*100:.1f}%" if mc_questions else "0%",
            'desc_total': len(desc_questions),
            'desc_agreement': desc_same,
            'desc_agreement_rate': f"{desc_same/len(desc_questions)*100:.1f}%" if desc_questions else "0%",
            'detailed_comparison': comparison
        }
        return summary
    
    def print_summary(self):
        """요약 출력"""
        print("\n" + "="*60)
        print("BM25 vs Vector 비교 실험 완료 (개선된 프롬프트)")
        print("="*60)
        
        summary = self.create_comparison_summary()
        
        print(f"\n총 문제 수: {summary['total_questions']}")
        print(f"동일 답변: {summary['same_answers']}개")
        print(f"다른 답변: {summary['different_answers']}개")
        print(f"전체 일치율: {summary['agreement_rate']}")
        
        print(f"\n객관식 ({summary['mc_total']}문제):")
        print(f"  일치: {summary['mc_agreement']}개")
        print(f"  일치율: {summary['mc_agreement_rate']}")
        
        print(f"\n주관식 ({summary['desc_total']}문제):")
        print(f"  일치: {summary['desc_agreement']}개")
        print(f"  일치율: {summary['desc_agreement_rate']}")
        
        print("\n" + "="*60)
        print("상세 결과는 JSON 파일을 확인하세요.")
        print("="*60)


def main():
    """메인 실행 함수"""
    tester = ImprovedComparisonTester()
    
    # 초기화
    tester.initialize_rag()
    tester.initialize_llm()
    
    # 비교 실험 실행
    tester.run_comparison()
    
    logger.info("\n개선된 비교 실험 완료!")


if __name__ == "__main__":
    main()