"""
Question Classifier Component
Architecture.md의 IQuestionClassifier 인터페이스 구현
객관식/주관식 질문 분류 및 선택지 추출
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedQuestion:
    """분류된 질문 데이터 클래스"""
    question_id: str
    original_text: str
    question_type: str  # "multiple_choice", "open_ended"
    parsed_question: str
    choices: Optional[Dict[str, str]] = None
    confidence: float = 0.0


class IQuestionClassifier(ABC):
    """질문 분류 컴포넌트 인터페이스"""
    
    @abstractmethod
    def classify(self, question_text: str, question_id: str = "") -> ClassifiedQuestion:
        """질문을 객관식/주관식으로 분류하고 파싱"""
        pass
    
    @abstractmethod
    def extract_choices(self, question_text: str) -> Optional[Dict[str, str]]:
        """객관식 선택지 추출"""
        pass


class QuestionClassifier(IQuestionClassifier):
    """
    질문 분류기 구현
    Pipeline.md 2.2.1 요구사항: ≥95% 분류 정확도
    """
    
    def __init__(self):
        # 객관식 패턴 정의
        self.choice_patterns = [
            r'^\s*(\d+)\s*[.)]\s*(.+)$',  # 1) 선택지 or 1. 선택지
            r'^\s*([①②③④⑤⑥⑦⑧⑨⑩])\s*(.+)$',  # ① 선택지
            r'^\s*([가나다라마바사아자차카타파하])\s*[.)]\s*(.+)$',  # 가) 선택지
            r'^\s*([ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ])\s*[.)]\s*(.+)$',  # ㄱ) 선택지
            r'^\s*([A-Za-z])\s*[.)]\s*(.+)$',  # A) 선택지
        ]
        
        # 질문 유형 키워드
        self.mc_keywords = [
            '다음 중', '아래 중', '선택', '고르', '맞는 것', '틀린 것',
            '해당하는 것', '옳은 것', '올바른 것', '적절한 것'
        ]
        
        self.oe_keywords = [
            '설명하', '서술하', '논하', '분석하', '평가하', '제시하',
            '작성하', '기술하', '정의하', '비교하', '요약하'
        ]
        
        logger.info("QuestionClassifier initialized")
    
    def classify(self, question_text: str, question_id: str = "") -> ClassifiedQuestion:
        """
        질문을 객관식/주관식으로 분류
        
        Args:
            question_text: 질문 텍스트
            question_id: 질문 ID
            
        Returns:
            ClassifiedQuestion: 분류된 질문 객체
        """
        if not question_text:
            return ClassifiedQuestion(
                question_id=question_id,
                original_text="",
                question_type="open_ended",
                parsed_question="",
                confidence=0.0
            )
        
        # 선택지 추출 시도
        choices = self.extract_choices(question_text)
        
        # 분류 로직
        confidence = 0.0
        question_type = "open_ended"
        parsed_question = question_text
        
        if choices:
            # 선택지가 2개 이상이면 객관식
            if len(choices) >= 2:
                question_type = "multiple_choice"
                confidence = 0.95
                # 선택지 앞의 텍스트를 질문으로 추출
                parsed_question = self._extract_question_part(question_text, choices)
            else:
                confidence = 0.3
        else:
            # 키워드 기반 분류
            mc_score = self._calculate_keyword_score(question_text, self.mc_keywords)
            oe_score = self._calculate_keyword_score(question_text, self.oe_keywords)
            
            if mc_score > oe_score and mc_score > 0:
                question_type = "multiple_choice"
                confidence = min(0.7, mc_score)
            elif oe_score > 0:
                question_type = "open_ended"
                confidence = min(0.8, oe_score)
            else:
                # 기본값: 주관식
                question_type = "open_ended"
                confidence = 0.6
        
        result = ClassifiedQuestion(
            question_id=question_id,
            original_text=question_text,
            question_type=question_type,
            parsed_question=parsed_question,
            choices=choices,
            confidence=confidence
        )
        
        logger.debug(f"Classified question {question_id}: type={question_type}, confidence={confidence:.2f}")
        
        return result
    
    def extract_choices(self, question_text: str) -> Optional[Dict[str, str]]:
        """
        객관식 선택지 추출
        
        Args:
            question_text: 질문 텍스트
            
        Returns:
            Dict[str, str]: 선택지 번호와 내용 매핑 (예: {"1": "선택지1", "2": "선택지2"})
        """
        lines = question_text.split('\n')
        choices = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in self.choice_patterns:
                match = re.match(pattern, line)
                if match:
                    choice_num = match.group(1)
                    choice_text = match.group(2).strip()
                    
                    # 번호 정규화
                    if choice_num.isdigit():
                        normalized_num = choice_num
                    elif choice_num in '①②③④⑤⑥⑦⑧⑨⑩':
                        normalized_num = str('①②③④⑤⑥⑦⑧⑨⑩'.index(choice_num) + 1)
                    elif choice_num in '가나다라마바사아자차카타파하':
                        normalized_num = str('가나다라마바사아자차카타파하'.index(choice_num) + 1)
                    elif choice_num in 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ':
                        normalized_num = str('ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ'.index(choice_num) + 1)
                    elif choice_num.upper() in 'ABCDEFGHIJ':
                        normalized_num = str('ABCDEFGHIJ'.index(choice_num.upper()) + 1)
                    else:
                        normalized_num = choice_num
                    
                    choices[normalized_num] = choice_text
                    break
        
        return choices if choices else None
    
    def _extract_question_part(self, full_text: str, choices: Dict[str, str]) -> str:
        """선택지를 제외한 질문 부분만 추출"""
        lines = full_text.split('\n')
        question_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            # 선택지가 아닌 라인만 추가
            is_choice = False
            for pattern in self.choice_patterns:
                if re.match(pattern, line_stripped):
                    is_choice = True
                    break
            
            if not is_choice and line_stripped:
                question_lines.append(line_stripped)
        
        return ' '.join(question_lines)
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """키워드 기반 점수 계산"""
        score = 0.0
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword in text:
                score += 0.3
        
        return min(1.0, score)


def main():
    """테스트 함수"""
    classifier = QuestionClassifier()
    
    # 테스트 케이스
    test_questions = [
        """금융 AI 시스템의 보안 위협 중 가장 심각한 것은?
        1) 데이터 유출
        2) 모델 변조
        3) 서비스 거부 공격
        4) 프라이버시 침해""",
        
        """금융 서비스에서 AI를 활용할 때 고려해야 할 윤리적 측면을 설명하시오.""",
        
        """다음 중 머신러닝 모델의 보안 취약점이 아닌 것은?
        ① 적대적 예제
        ② 모델 추출
        ③ 데이터 중독
        ④ 하드웨어 가속""",
    ]
    
    for i, question in enumerate(test_questions):
        result = classifier.classify(question, f"Q{i+1}")
        print(f"\n{'='*50}")
        print(f"Question ID: {result.question_id}")
        print(f"Type: {result.question_type}")
        print(f"Confidence: {result.confidence:.2%}")
        if result.choices:
            print(f"Choices: {result.choices}")
        print(f"Parsed Question: {result.parsed_question[:100]}...")


if __name__ == "__main__":
    main()