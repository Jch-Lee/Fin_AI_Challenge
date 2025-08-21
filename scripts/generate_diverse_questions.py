#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
다양성을 최대화한 RAG 기반 질문 생성 스크립트
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 모듈 임포트
from packages.rag.rag_pipeline import RAGPipeline, load_rag_v2_pipeline
from utils.diversity_sampler import DiversitySampler
from utils.question_validator import QuestionValidator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('question_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DiverseQuestionGenerator:
    """다양성을 최대화한 질문 생성기"""
    
    def __init__(self, 
                 rag_dir: str = "data/rag",
                 config_path: str = "configs/question_generation_config.yaml",
                 model_name: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"):
        """
        Args:
            rag_dir: RAG 데이터 디렉토리
            config_path: 설정 파일 경로
            model_name: 사용할 LLM 모델
        """
        self.rag_dir = Path(rag_dir)
        self.config_path = Path(config_path)
        self.model_name = model_name
        
        # 설정 로드
        self.config = self._load_config()
        
        # RAG 시스템 초기화
        logger.info("Loading RAG system...")
        self.rag_pipeline = self._initialize_rag()
        
        # 청크 데이터 로드
        self.chunks = self._load_chunks()
        logger.info(f"Loaded {len(self.chunks)} chunks")
        
        # 샘플러와 검증기 초기화
        self.sampler = DiversitySampler(self.chunks)
        self.validator = QuestionValidator()
        
    def _load_config(self) -> Dict:
        """설정 파일 로드"""
        if self.config_path.exists():
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 기본 설정
            return {
                'model': {
                    'temperature': 0.8,
                    'top_p': 0.95,
                    'max_tokens': 500
                },
                'sampling': {
                    'total_chunks': 1000,
                    'min_per_document': 10,
                    'similarity_threshold': 0.95
                },
                'diversity': {
                    'question_types': {
                        'definition': 6,
                        'process': 6,
                        'regulation': 6,
                        'example': 4,
                        'comparison': 4,
                        'application': 4
                    }
                },
                'output': {
                    'path': 'data/synthetic_questions/',
                    'format': 'csv'
                }
            }
    
    def _initialize_rag(self) -> Optional[RAGPipeline]:
        """RAG 시스템 초기화 (옵션)"""
        try:
            # RAG v2 형식으로 로드 시도
            pipeline = load_rag_v2_pipeline(
                version="2300",
                rag_dir=str(self.rag_dir),
                config_path="configs/rag_config.yaml"
            )
            return pipeline
        except Exception as e:
            logger.warning(f"Failed to load full RAG pipeline: {e}")
            logger.info("Continuing without RAG pipeline - using direct chunk access")
            return None
    
    def _load_chunks(self) -> List[Dict]:
        """청크 데이터 로드"""
        chunks_file = self.rag_dir / "chunks_2300.json"
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        return chunks
    
    def generate_question_prompt(self, chunk_content: str, question_type: str) -> str:
        """질문 생성을 위한 프롬프트 생성"""
        
        # 질문 유형별 가이드
        type_guides = {
            'definition': "핵심 개념이나 용어의 정의를 묻는",
            'process': "절차나 프로세스의 단계를 묻는",
            'regulation': "규정, 법률, 기준을 묻는",
            'example': "구체적인 사례나 예시를 묻는",
            'comparison': "둘 이상의 개념의 차이점이나 비교를 묻는",
            'application': "실무 적용이나 활용 방법을 묻는"
        }
        
        prompt = f"""<지침>
당신은 대한민국 교육평가원에서 근무하는 프롬프트 엔지니어링 연구관입니다. 
주어진 자료를 바탕으로 {type_guides.get(question_type, '일반적인')} 주관식 시험문항을 생성합니다.

<절대 규칙>
- 문서에 직접 근거한 문항만 출제(외부 지식·추측 금지)
- 주관식: 단답 또는 간결한 서술로 답 가능하게
- 표현/형식: 한국어, 질문은 한 문장
- 출력 제한: 문제만 출력(해설/근거/난이도/태그/시간/채점기준 금지)
- 플레이스홀더 금지: 실제 내용으로 모든 부분 채우기

<원문>
{chunk_content}

<문제 유형>
{type_guides.get(question_type, '일반적인')} 질문

<프롬프트>
위 원문을 바탕으로 **주관식 1문항**을 생성하라. 문제만 포함하고, 다음 형식을 따르라:

[질문 내용을 여기에 작성]"""
        
        return prompt
    
    def generate_questions_with_model(self, sampled_chunks: List[Dict]) -> List[Dict]:
        """모델을 사용한 질문 생성 (시뮬레이션)"""
        logger.info(f"Generating questions for {len(sampled_chunks)} chunks...")
        
        generated_questions = []
        question_types = list(self.config['diversity']['question_types'].keys())
        type_counts = defaultdict(int)
        type_limits = self.config['diversity']['question_types']
        
        # 진행 상황 표시
        with tqdm(total=len(sampled_chunks), desc="Generating questions") as pbar:
            for chunk in sampled_chunks:
                # 질문 유형 선택 (제한에 따라)
                available_types = [t for t in question_types 
                                 if type_counts[t] < type_limits[t]]
                
                if not available_types:
                    pbar.update(1)
                    continue
                
                question_type = random.choice(available_types)
                type_counts[question_type] += 1
                
                # 프롬프트 생성
                prompt = self.generate_question_prompt(
                    chunk['content'], 
                    question_type
                )
                
                # TODO: 실제 모델 호출 구현
                # 현재는 시뮬레이션으로 대체
                generated_question = self._simulate_model_generation(
                    chunk, question_type
                )
                
                generated_questions.append({
                    'question': generated_question,
                    'source_chunk_id': chunk['id'],
                    'source_file': chunk.get('source', 'unknown'),
                    'question_type': question_type,
                    'chunk_content': chunk['content'][:200] + '...',  # 미리보기용
                    'prompt': prompt[:500] + '...'  # 디버깅용
                })
                
                pbar.update(1)
                
                # 목표 달성 확인
                if len(generated_questions) >= 30:
                    break
        
        return generated_questions[:30]  # 30개로 제한
    
    def _simulate_model_generation(self, chunk: Dict, question_type: str) -> str:
        """모델 생성 시뮬레이션 (실제 구현 전 테스트용)"""
        # 실제로는 Qwen 모델 호출
        # 여기서는 테스트를 위한 예시 질문 생성
        
        templates = {
            'definition': [
                f"{chunk['content'][:50]}...에서 언급된 주요 개념은 무엇인가?",
                f"문서에서 설명하는 핵심 용어의 정의는 무엇인가?"
            ],
            'process': [
                f"문서에서 제시된 절차의 주요 단계는 무엇인가?",
                f"설명된 프로세스의 순서는 어떻게 되는가?"
            ],
            'regulation': [
                f"문서에서 언급된 주요 규정은 무엇인가?",
                f"적용되는 법률 조항은 무엇인가?"
            ],
            'example': [
                f"문서에서 제시된 구체적인 사례는 무엇인가?",
                f"어떤 예시가 포함되어 있는가?"
            ],
            'comparison': [
                f"문서에서 비교된 두 개념의 차이점은 무엇인가?",
                f"제시된 항목들 간의 주요 차이는 무엇인가?"
            ],
            'application': [
                f"이 내용을 실무에 어떻게 적용할 수 있는가?",
                f"문서의 내용이 실제로 활용되는 상황은 무엇인가?"
            ]
        }
        
        # 청크 내용에서 키워드 추출하여 질문에 포함
        content_preview = chunk['content'][:100]
        keywords = [word for word in content_preview.split() if len(word) > 2][:3]
        
        base_question = random.choice(templates.get(question_type, templates['definition']))
        
        # 키워드를 포함한 구체적인 질문 생성
        if keywords:
            keyword = random.choice(keywords)
            base_question = base_question.replace("문서", f"'{keyword}'가 언급된 부분")
        
        return base_question
    
    def run_experiment(self, sample_size: int = 30) -> pd.DataFrame:
        """실험 실행"""
        logger.info("=" * 60)
        logger.info("Starting Diverse Question Generation Experiment")
        logger.info("=" * 60)
        
        # 1. 다양성 기반 청크 샘플링
        logger.info("\n[Step 1/4] Sampling diverse chunks...")
        sampled_chunks = self.sampler.sample_diverse_chunks(
            n_samples=min(1000, len(self.chunks)),
            min_per_document=self.config['sampling']['min_per_document']
        )
        logger.info(f"Sampled {len(sampled_chunks)} chunks")
        
        # 2. 질문 생성
        logger.info("\n[Step 2/4] Generating questions...")
        generated_questions = self.generate_questions_with_model(
            sampled_chunks[:100]  # 처음 100개만 사용 (30개 질문 생성용)
        )
        
        # 3. 품질 검증 및 중복 제거
        logger.info("\n[Step 3/4] Validating questions...")
        validated_questions = self.validator.validate_questions(generated_questions)
        
        # 4. 결과 저장
        logger.info("\n[Step 4/4] Saving results...")
        df = self.save_results(validated_questions)
        
        # 다양성 메트릭 출력
        metrics = self.validator.calculate_diversity_metrics(validated_questions)
        logger.info("\n" + "=" * 60)
        logger.info("Diversity Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
        
        return df
    
    def save_results(self, questions: List[Dict]) -> pd.DataFrame:
        """결과 저장"""
        output_dir = Path(self.config['output']['path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # DataFrame 생성
        df = pd.DataFrame(questions)
        df['id'] = range(1, len(df) + 1)
        
        # 저장할 컬럼 선택
        columns_to_save = ['id', 'question', 'source_file', 'question_type', 'source_chunk_id']
        df_save = df[columns_to_save]
        
        # CSV 저장
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"synthetic_questions_{timestamp}.csv"
        df_save.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"Saved {len(df)} questions to {output_file}")
        
        # 상세 정보도 별도 저장 (디버깅용)
        detail_file = output_dir / f"question_details_{timestamp}.json"
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved detailed information to {detail_file}")
        
        return df_save


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate diverse synthetic questions using RAG")
    parser.add_argument("--rag-dir", default="data/rag", help="RAG data directory")
    parser.add_argument("--config", default="configs/question_generation_config.yaml", help="Config file")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of questions to generate")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ", help="Model name")
    
    args = parser.parse_args()
    
    # 생성기 초기화
    generator = DiverseQuestionGenerator(
        rag_dir=args.rag_dir,
        config_path=args.config,
        model_name=args.model
    )
    
    # 실험 실행
    df = generator.run_experiment(sample_size=args.sample_size)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("Generated Questions Summary:")
    print(f"Total questions: {len(df)}")
    print(f"\nQuestion type distribution:")
    print(df['question_type'].value_counts())
    print("=" * 60)


if __name__ == "__main__":
    main()