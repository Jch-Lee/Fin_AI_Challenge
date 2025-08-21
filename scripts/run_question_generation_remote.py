#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
원격 서버에서 실제 Qwen 모델을 사용한 질문 생성 스크립트
"""

import os
import sys
import json
import time
import torch
from pathlib import Path
from typing import List, Dict, Optional
import logging
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import yaml

# vLLM 임포트
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Using transformers instead.")

# Transformers 임포트 (fallback)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 모듈 임포트
from utils.diversity_sampler import DiversitySampler
from utils.question_validator import QuestionValidator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('remote_question_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QwenQuestionGenerator:
    """Qwen 모델을 사용한 실제 질문 생성기"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",
                 config_path: str = "configs/question_generation_config.yaml",
                 use_vllm: bool = True):
        """
        Args:
            model_name: Qwen 모델 이름
            config_path: 설정 파일 경로
            use_vllm: vLLM 사용 여부
        """
        self.model_name = model_name
        self.config = self._load_config(config_path)
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        
        # 모델 초기화
        self.model = None
        self.tokenizer = None
        self.sampling_params = None
        
        self._initialize_model()
        
        # 청크 데이터 로드
        self.chunks = self._load_chunks()
        
        # 샘플러와 검증기 초기화
        self.sampler = DiversitySampler(self.chunks)
        self.validator = QuestionValidator()
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    def _initialize_model(self):
        """모델 초기화"""
        logger.info(f"Initializing model: {self.model_name}")
        
        if self.use_vllm:
            # vLLM 사용
            logger.info("Using vLLM for inference...")
            self.model = LLM(
                model=self.model_name,
                tensor_parallel_size=torch.cuda.device_count(),
                gpu_memory_utilization=0.90,
                trust_remote_code=True,
                dtype="half",  # AWQ 모델은 half precision
                max_model_len=4096
            )
            
            self.sampling_params = SamplingParams(
                temperature=self.config['model']['temperature'],
                top_p=self.config['model']['top_p'],
                max_tokens=self.config['model']['max_tokens'],
                stop=["<|endoftext|>", "<|im_end|>"]
            )
        else:
            # Transformers 사용 (fallback)
            logger.info("Using transformers for inference...")
            
            # 4-bit 양자화 설정
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_chunks(self) -> List[Dict]:
        """청크 데이터 로드"""
        chunks_file = Path("data/rag/chunks_2300.json")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks
    
    def generate_question(self, chunk_content: str, question_type: str) -> str:
        """
        실제 모델을 사용한 질문 생성
        
        Args:
            chunk_content: 청크 내용
            question_type: 질문 유형
            
        Returns:
            생성된 질문
        """
        # 프롬프트 생성
        prompt = self._create_prompt(chunk_content, question_type)
        
        if self.use_vllm:
            # vLLM 추론
            outputs = self.model.generate([prompt], self.sampling_params)
            generated_text = outputs[0].outputs[0].text
        else:
            # Transformers 추론
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    temperature=self.config['model']['temperature'],
                    top_p=self.config['model']['top_p'],
                    max_new_tokens=self.config['model']['max_tokens'],
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        
        # 질문 추출 및 정제
        question = self._extract_question(generated_text)
        
        return question
    
    def _create_prompt(self, chunk_content: str, question_type: str) -> str:
        """프롬프트 생성"""
        type_guides = {
            'definition': "핵심 개념이나 용어의 정의를 묻는",
            'process': "절차나 프로세스의 단계를 묻는",
            'regulation': "규정, 법률, 기준을 묻는",
            'example': "구체적인 사례나 예시를 묻는",
            'comparison': "둘 이상의 개념의 차이점이나 비교를 묻는",
            'application': "실무 적용이나 활용 방법을 묻는"
        }
        
        # Qwen 모델용 시스템 프롬프트
        system_prompt = """당신은 대한민국 교육평가원에서 근무하는 프롬프트 엔지니어링 연구관입니다.
주어진 자료를 바탕으로 주관식 시험문항을 생성합니다.

절대 규칙:
- 문서에 직접 근거한 문항만 출제
- 주관식: 단답 또는 간결한 서술로 답 가능하게
- 한국어로 작성, 질문은 한 문장
- 문제만 출력 (해설, 근거, 난이도 제외)"""

        user_prompt = f"""원문:
{chunk_content[:1500]}

위 원문을 바탕으로 {type_guides.get(question_type, '일반적인')} 주관식 1문항을 생성하세요.
질문만 한 문장으로 작성하고, 물음표로 끝내세요."""

        if self.use_vllm:
            # vLLM용 포맷
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Transformers용 포맷
            prompt = f"{system_prompt}\n\n{user_prompt}\n\n질문:"
        
        return prompt
    
    def _extract_question(self, generated_text: str) -> str:
        """생성된 텍스트에서 질문 추출"""
        # 여러 줄이 생성된 경우 첫 번째 줄만 추출
        lines = generated_text.strip().split('\n')
        question = lines[0].strip()
        
        # 물음표가 없으면 추가
        if question and not question.endswith('?'):
            question += '?'
        
        # 번호나 라벨 제거 (예: "1. ", "질문: " 등)
        import re
        question = re.sub(r'^[\d]+\.\s*', '', question)
        question = re.sub(r'^질문[:：]\s*', '', question)
        
        return question
    
    def run_generation(self, n_questions: int = 30) -> pd.DataFrame:
        """질문 생성 실행"""
        logger.info("=" * 60)
        logger.info(f"Starting Question Generation with {self.model_name}")
        logger.info("=" * 60)
        
        # 1. 청크 샘플링
        logger.info("Step 1: Sampling diverse chunks...")
        sampled_chunks = self.sampler.sample_diverse_chunks(
            n_samples=min(1000, len(self.chunks)),
            min_per_document=self.config['sampling']['min_per_document']
        )
        
        # 2. 질문 생성
        logger.info(f"Step 2: Generating {n_questions} questions...")
        generated_questions = []
        type_counts = defaultdict(int)
        type_limits = self.config['diversity']['question_types']
        question_types = list(type_limits.keys())
        
        with tqdm(total=n_questions, desc="Generating") as pbar:
            for i, chunk in enumerate(sampled_chunks):
                if len(generated_questions) >= n_questions:
                    break
                
                # 질문 유형 선택
                available_types = [t for t in question_types 
                                 if type_counts[t] < type_limits[t]]
                
                if not available_types:
                    continue
                
                question_type = available_types[i % len(available_types)]
                
                try:
                    # 질문 생성
                    question = self.generate_question(
                        chunk['content'], 
                        question_type
                    )
                    
                    if question and len(question) > 10:
                        generated_questions.append({
                            'question': question,
                            'source_chunk_id': chunk['id'],
                            'source_file': chunk.get('source', 'unknown'),
                            'question_type': question_type
                        })
                        type_counts[question_type] += 1
                        pbar.update(1)
                        
                except Exception as e:
                    logger.error(f"Error generating question: {e}")
                    continue
        
        # 3. 검증
        logger.info("Step 3: Validating questions...")
        validated_questions = self.validator.validate_questions(generated_questions)
        
        # 4. 결과 저장
        logger.info("Step 4: Saving results...")
        df = self._save_results(validated_questions)
        
        # 5. 메트릭 출력
        metrics = self.validator.calculate_diversity_metrics(validated_questions)
        report = self.validator.generate_validation_report(validated_questions)
        logger.info("\n" + report)
        
        return df
    
    def _save_results(self, questions: List[Dict]) -> pd.DataFrame:
        """결과 저장"""
        output_dir = Path(self.config['output']['path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # DataFrame 생성
        df = pd.DataFrame(questions)
        df['id'] = range(1, len(df) + 1)
        
        # CSV 저장
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"generated_questions_{timestamp}.csv"
        df[['id', 'question', 'source_file', 'question_type']].to_csv(
            output_file, 
            index=False, 
            encoding=self.config['output']['encoding']
        )
        logger.info(f"Saved {len(df)} questions to {output_file}")
        
        # 상세 정보 저장
        detail_file = output_dir / f"question_details_{timestamp}.json"
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
        
        return df


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate questions using Qwen model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ", help="Model name")
    parser.add_argument("--config", default="configs/question_generation_config.yaml", help="Config file")
    parser.add_argument("--n-questions", type=int, default=30, help="Number of questions")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for inference")
    
    args = parser.parse_args()
    
    # GPU 확인
    if not torch.cuda.is_available():
        logger.error("GPU not available! This script requires GPU.")
        sys.exit(1)
    
    logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 생성기 초기화 및 실행
    generator = QwenQuestionGenerator(
        model_name=args.model,
        config_path=args.config,
        use_vllm=args.use_vllm
    )
    
    # 질문 생성
    df = generator.run_generation(n_questions=args.n_questions)
    
    print("\n" + "=" * 60)
    print(f"Successfully generated {len(df)} questions!")
    print("=" * 60)


if __name__ == "__main__":
    main()