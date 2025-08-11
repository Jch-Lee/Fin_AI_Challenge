"""
Synthetic Data Generator Component
Architecture.md의 ISyntheticDataGenerator 인터페이스 구현
Teacher Model을 사용한 Q&A 쌍 생성 (Distill-M 2 방식)
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SyntheticQAPair:
    """합성 Q&A 쌍 데이터 클래스"""
    question_id: str
    question: str
    answer: str
    context: str
    metadata: Dict[str, Any]
    question_type: str  # "multiple_choice" or "open_ended"
    confidence: float
    source_chunk_id: str
    generated_at: str


class ISyntheticDataGenerator(ABC):
    """합성 데이터 생성 컴포넌트 인터페이스"""
    
    @abstractmethod
    def generate_qa_pairs(self, context: str, num_pairs: int = 3) -> List[SyntheticQAPair]:
        """컨텍스트로부터 Q&A 쌍 생성"""
        pass
    
    @abstractmethod
    def validate_qa_pair(self, qa_pair: SyntheticQAPair) -> bool:
        """생성된 Q&A 쌍의 품질 검증"""
        pass
    
    @abstractmethod
    def save_dataset(self, qa_pairs: List[SyntheticQAPair], output_path: str) -> bool:
        """데이터셋 저장"""
        pass


class SyntheticDataGenerator(ISyntheticDataGenerator):
    """
    Teacher Model 기반 합성 데이터 생성기
    요구사항정의서 Story 1.4: 고성능 교사 모델로 Q&A 쌍 생성
    Pipeline.md Epic 1.5: SyntheticQAPair 데이터셋 생성
    """
    
    def __init__(self, 
                 teacher_model_name: str = "Meta-Llama-3.1-70B-Instruct",
                 output_dir: str = "data/finetune",
                 min_answer_length: int = 20,
                 max_answer_length: int = 512):
        """
        Args:
            teacher_model_name: 교사 모델명 (라이선스 검증 필수)
            output_dir: 생성된 데이터 저장 경로
            min_answer_length: 최소 답변 길이
            max_answer_length: 최대 답변 길이
        """
        self.teacher_model_name = teacher_model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_answer_length = min_answer_length
        self.max_answer_length = max_answer_length
        
        # 금융/보안 도메인 특화 질문 템플릿
        self.question_templates = {
            "definition": [
                "{term}의 정의와 주요 특징을 설명하시오.",
                "{term}이란 무엇이며, 금융 분야에서 어떻게 활용되는가?",
                "{term}의 개념과 중요성을 서술하시오."
            ],
            "comparison": [
                "{concept1}과 {concept2}의 차이점을 비교 분석하시오.",
                "{concept1}와 {concept2} 중 어느 것이 더 적합한지 설명하시오."
            ],
            "application": [
                "{technology}를 금융 서비스에 적용할 때 고려사항은?",
                "{method}의 실제 구현 방법과 주의점을 설명하시오."
            ],
            "security": [
                "{system}의 보안 취약점과 대응 방안은?",
                "{threat}에 대한 방어 전략을 제시하시오."
            ],
            "multiple_choice": [
                "다음 중 {topic}에 대한 설명으로 옳은 것은?",
                "{subject}의 특징으로 적절하지 않은 것은?"
            ]
        }
        
        # 품질 검증 기준
        self.quality_criteria = {
            "min_question_length": 10,
            "max_question_length": 500,
            "min_answer_length": min_answer_length,
            "max_answer_length": max_answer_length,
            "required_korean_ratio": 0.3,  # 최소 30% 한글 포함
            "prohibited_patterns": ["죄송", "모르", "알 수 없", "정보 부족"]
        }
        
        self.generated_count = 0
        self.validation_stats = {
            "total_generated": 0,
            "passed_validation": 0,
            "failed_validation": 0
        }
        
        logger.info(f"SyntheticDataGenerator initialized with teacher model: {teacher_model_name}")
    
    def generate_qa_pairs(self, context: str, num_pairs: int = 3) -> List[SyntheticQAPair]:
        """
        컨텍스트로부터 Q&A 쌍 생성
        
        Args:
            context: 소스 텍스트 (청크)
            num_pairs: 생성할 Q&A 쌍 개수
            
        Returns:
            List[SyntheticQAPair]: 생성된 Q&A 쌍 리스트
        """
        qa_pairs = []
        
        # 컨텍스트 분석
        context_keywords = self._extract_keywords(context)
        
        for i in range(num_pairs):
            # 질문 유형 선택 (70% 주관식, 30% 객관식)
            question_type = "open_ended" if np.random.random() > 0.3 else "multiple_choice"
            
            # 질문 생성
            question = self._generate_question(context, context_keywords, question_type)
            
            # 답변 생성 (Teacher Model 시뮬레이션)
            # 실제 구현에서는 vLLM을 사용하여 Teacher Model 호출
            answer = self._generate_answer(context, question, question_type)
            
            # Q&A 쌍 생성
            qa_pair = SyntheticQAPair(
                question_id=f"synth_{self.generated_count:06d}",
                question=question,
                answer=answer,
                context=context,
                metadata={
                    "keywords": context_keywords,
                    "template_type": self._get_template_type(question),
                    "context_length": len(context),
                    "teacher_model": self.teacher_model_name
                },
                question_type=question_type,
                confidence=0.85,  # Teacher model confidence
                source_chunk_id=f"chunk_{i}",
                generated_at=datetime.now().isoformat()
            )
            
            # 품질 검증
            if self.validate_qa_pair(qa_pair):
                qa_pairs.append(qa_pair)
                self.generated_count += 1
                self.validation_stats["passed_validation"] += 1
            else:
                self.validation_stats["failed_validation"] += 1
                logger.debug(f"QA pair failed validation: {qa_pair.question_id}")
            
            self.validation_stats["total_generated"] += 1
        
        logger.info(f"Generated {len(qa_pairs)} valid Q&A pairs from context")
        return qa_pairs
    
    def validate_qa_pair(self, qa_pair: SyntheticQAPair) -> bool:
        """
        생성된 Q&A 쌍의 품질 검증
        요구사항정의서 Story 1.4: 부적절한 데이터 필터링
        
        Args:
            qa_pair: 검증할 Q&A 쌍
            
        Returns:
            bool: 품질 기준 통과 여부
        """
        # 1. 길이 검증
        if len(qa_pair.question) < self.quality_criteria["min_question_length"]:
            return False
        if len(qa_pair.question) > self.quality_criteria["max_question_length"]:
            return False
        if len(qa_pair.answer) < self.quality_criteria["min_answer_length"]:
            return False
        if len(qa_pair.answer) > self.quality_criteria["max_answer_length"]:
            return False
        
        # 2. 한글 포함 비율 검증
        korean_ratio = self._calculate_korean_ratio(qa_pair.answer)
        if korean_ratio < self.quality_criteria["required_korean_ratio"]:
            logger.debug(f"Korean ratio too low: {korean_ratio:.2f}")
            return False
        
        # 3. 금지 패턴 검증
        for pattern in self.quality_criteria["prohibited_patterns"]:
            if pattern in qa_pair.answer.lower():
                logger.debug(f"Prohibited pattern found: {pattern}")
                return False
        
        # 4. 답변-컨텍스트 관련성 검증
        if not self._check_relevance(qa_pair.context, qa_pair.answer):
            return False
        
        # 5. 객관식 답변 형식 검증
        if qa_pair.question_type == "multiple_choice":
            if not qa_pair.answer.isdigit() or int(qa_pair.answer) < 1 or int(qa_pair.answer) > 10:
                return False
        
        return True
    
    def save_dataset(self, qa_pairs: List[SyntheticQAPair], output_path: str = None) -> bool:
        """
        데이터셋 저장
        요구사항정의서 Story 1.4: /data/finetune 폴더에 구조화된 형식으로 저장
        
        Args:
            qa_pairs: 저장할 Q&A 쌍 리스트
            output_path: 저장 경로 (기본값: data/finetune/synthetic_qa.jsonl)
            
        Returns:
            bool: 저장 성공 여부
        """
        if not qa_pairs:
            logger.warning("No QA pairs to save")
            return False
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"synthetic_qa_{timestamp}.jsonl"
        else:
            output_path = Path(output_path)
        
        try:
            # JSONL 형식으로 저장 (한 줄에 하나의 JSON 객체)
            with open(output_path, 'w', encoding='utf-8') as f:
                for qa_pair in qa_pairs:
                    # dataclass를 dict로 변환
                    qa_dict = asdict(qa_pair)
                    # 한 줄로 저장
                    f.write(json.dumps(qa_dict, ensure_ascii=False) + '\n')
            
            # 메타데이터 파일 생성
            metadata = {
                "total_pairs": len(qa_pairs),
                "generation_timestamp": datetime.now().isoformat(),
                "teacher_model": self.teacher_model_name,
                "validation_stats": self.validation_stats,
                "question_type_distribution": self._get_type_distribution(qa_pairs),
                "average_answer_length": np.mean([len(qa.answer) for qa in qa_pairs])
            }
            
            metadata_path = output_path.with_suffix('.meta.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(qa_pairs)} QA pairs to {output_path}")
            logger.info(f"Metadata saved to {metadata_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            return False
    
    def _extract_keywords(self, context: str) -> List[str]:
        """컨텍스트에서 키워드 추출"""
        # 간단한 키워드 추출 (실제로는 KoNLPy 사용)
        keywords = []
        important_terms = ["AI", "보안", "금융", "시스템", "데이터", "모델", "리스크", "규제"]
        for term in important_terms:
            if term in context:
                keywords.append(term)
        return keywords[:5]  # 상위 5개만
    
    def _generate_question(self, context: str, keywords: List[str], question_type: str) -> str:
        """질문 생성 (템플릿 기반)"""
        # 실제로는 Teacher Model 사용
        if question_type == "multiple_choice":
            template = np.random.choice(self.question_templates["multiple_choice"])
            topic = keywords[0] if keywords else "금융 AI"
            question = template.format(topic=topic, subject=topic)
            
            # 선택지 추가
            choices = [
                "\n1) 올바른 설명입니다",
                "\n2) 부분적으로 맞습니다",
                "\n3) 틀린 설명입니다",
                "\n4) 추가 정보가 필요합니다"
            ]
            question += "".join(choices)
        else:
            category = np.random.choice(["definition", "application", "security"])
            template = np.random.choice(self.question_templates[category])
            
            # 키워드 기반 질문 생성
            if "{term}" in template:
                term = keywords[0] if keywords else "금융 AI 시스템"
                question = template.format(term=term)
            elif "{technology}" in template:
                question = template.format(technology="머신러닝")
            elif "{system}" in template:
                question = template.format(system="금융 거래 시스템")
            elif "{threat}" in template:
                question = template.format(threat="사이버 공격")
            elif "{method}" in template:
                question = template.format(method="이상 탐지")
            else:
                question = template
        
        return question
    
    def _generate_answer(self, context: str, question: str, question_type: str) -> str:
        """답변 생성 (Teacher Model 시뮬레이션)"""
        if question_type == "multiple_choice":
            # 객관식은 번호만 반환
            return str(np.random.randint(1, 5))
        else:
            # 주관식은 컨텍스트 기반 답변 생성
            # 실제로는 Teacher Model (Llama-3.1-70B) 사용
            answer = f"이 질문에 대한 답변은 다음과 같습니다. {context[:200]}... "
            answer += "따라서 금융 AI 시스템에서는 보안과 성능을 모두 고려해야 합니다."
            return answer
    
    def _get_template_type(self, question: str) -> str:
        """질문의 템플릿 유형 판별"""
        if "정의" in question or "무엇" in question:
            return "definition"
        elif "차이" in question or "비교" in question:
            return "comparison"
        elif "적용" in question or "구현" in question:
            return "application"
        elif "보안" in question or "취약" in question:
            return "security"
        else:
            return "general"
    
    def _calculate_korean_ratio(self, text: str) -> float:
        """텍스트의 한글 비율 계산"""
        korean_chars = sum(1 for c in text if '가' <= c <= '힣')
        total_chars = len(text)
        return korean_chars / total_chars if total_chars > 0 else 0
    
    def _check_relevance(self, context: str, answer: str) -> bool:
        """답변과 컨텍스트의 관련성 검증"""
        # 간단한 관련성 체크 (실제로는 더 정교한 방법 필요)
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        
        # 최소 10% 단어 겹침
        overlap = len(context_words & answer_words)
        return overlap > len(answer_words) * 0.1
    
    def _get_type_distribution(self, qa_pairs: List[SyntheticQAPair]) -> Dict[str, int]:
        """질문 유형별 분포"""
        distribution = {"multiple_choice": 0, "open_ended": 0}
        for qa in qa_pairs:
            distribution[qa.question_type] += 1
        return distribution
    
    def generate_from_chunks(self, chunks: List[Dict[str, Any]], 
                            qa_per_chunk: int = 2) -> List[SyntheticQAPair]:
        """
        여러 청크로부터 대량 Q&A 생성
        
        Args:
            chunks: 문서 청크 리스트
            qa_per_chunk: 청크당 생성할 Q&A 개수
            
        Returns:
            List[SyntheticQAPair]: 전체 생성된 Q&A 쌍
        """
        all_qa_pairs = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            chunk_content = chunk.get('content', chunk.get('text', ''))
            if not chunk_content:
                continue
            
            qa_pairs = self.generate_qa_pairs(chunk_content, qa_per_chunk)
            
            # 청크 ID 업데이트
            for qa in qa_pairs:
                qa.source_chunk_id = chunk.get('chunk_id', f'chunk_{i}')
            
            all_qa_pairs.extend(qa_pairs)
        
        logger.info(f"Total QA pairs generated: {len(all_qa_pairs)}")
        logger.info(f"Validation stats: {self.validation_stats}")
        
        return all_qa_pairs


def main():
    """테스트 및 데모"""
    # 생성기 초기화
    generator = SyntheticDataGenerator(
        teacher_model_name="Meta-Llama-3.1-70B-Instruct",
        output_dir="data/finetune"
    )
    
    # 테스트 컨텍스트
    test_context = """
    금융 AI 시스템의 보안은 매우 중요합니다. 특히 개인정보 보호와 관련된 
    규제 준수는 필수적입니다. AI 모델의 학습 데이터에는 민감한 금융 정보가 
    포함될 수 있으므로, 적절한 암호화와 접근 제어가 필요합니다. 
    또한 적대적 공격에 대한 방어 메커니즘을 구축해야 합니다.
    """
    
    # Q&A 쌍 생성
    qa_pairs = generator.generate_qa_pairs(test_context, num_pairs=3)
    
    # 결과 출력
    for qa in qa_pairs:
        print(f"\n{'='*50}")
        print(f"Question ID: {qa.question_id}")
        print(f"Type: {qa.question_type}")
        print(f"Question: {qa.question}")
        print(f"Answer: {qa.answer[:100]}...")
        print(f"Confidence: {qa.confidence:.2%}")
    
    # 데이터셋 저장
    if qa_pairs:
        success = generator.save_dataset(qa_pairs)
        print(f"\nDataset saved: {success}")
        print(f"Validation stats: {generator.validation_stats}")


if __name__ == "__main__":
    main()