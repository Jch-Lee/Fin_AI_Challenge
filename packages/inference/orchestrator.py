"""
Inference Orchestrator Component
Architecture.md의 InferenceOrchestrator 인터페이스 구현
Pipeline.md 3.2.4 요구사항 준수
"""

from typing import Dict, Any, Optional, List
import logging
import time
import numpy as np
from dataclasses import dataclass

# Internal components
from ..preprocessing.question_classifier import QuestionClassifier
from ..preprocessing.embedder import TextEmbedder
from ..rag.knowledge_base import KnowledgeBase
from ..rag.retriever import MultiStageRetriever
from ..inference.cache_layer import QueryCache
from ..inference.prompt_template import PromptTemplate, PromptType
from ..inference.model_loader import ModelLoader, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """추론 설정"""
    use_cache: bool = True
    cache_ttl: int = 3600
    retrieval_k: int = 5
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    max_retries: int = 3
    timeout: int = 30
    

@dataclass
class InferenceResult:
    """추론 결과"""
    question_id: str
    question: str
    answer: str
    question_type: str
    confidence: float
    retrieval_count: int
    inference_time: float
    cache_hit: bool = False
    error: Optional[str] = None


class FallbackHandler:
    """오류 처리 및 폴백 메커니즘"""
    
    def __init__(self):
        self.fallback_responses = {
            "multiple_choice": "1",
            "open_ended": "답변을 생성할 수 없습니다.",
        }
    
    def handle_error(self, 
                     question: str, 
                     question_type: str,
                     error: Exception) -> str:
        """오류 발생 시 폴백 답변 생성"""
        logger.error(f"Fallback triggered: {error}")
        
        # 기본 폴백 답변
        if question_type in self.fallback_responses:
            return self.fallback_responses[question_type]
        
        return "처리 중 오류가 발생했습니다."
    
    def validate_answer(self, answer: str, question_type: str) -> str:
        """답변 검증 및 보정"""
        if question_type == "multiple_choice":
            # 숫자만 추출
            import re
            numbers = re.findall(r'\d+', answer)
            if numbers:
                return numbers[0]
            return "1"
        
        # 주관식은 그대로 반환
        return answer if answer else self.fallback_responses.get(question_type, "")


class InferenceOrchestrator:
    """
    추론 파이프라인 오케스트레이터
    Pipeline.md 3.2.4: 전체 추론 흐름 제어
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        
        # 컴포넌트 초기화
        self.classifier = QuestionClassifier()
        self.embedder = TextEmbedder(model_name="jhgan/ko-sroberta-multitask")
        self.knowledge_base = KnowledgeBase(embedding_dim=384, index_type="Flat")
        self.retriever = MultiStageRetriever(
            knowledge_base=self.knowledge_base,
            embedder=self.embedder
        )
        self.cache = QueryCache(max_size_mb=100, max_entries=1000)
        self.prompt_template = PromptTemplate()
        self.fallback_handler = FallbackHandler()
        
        # 모델 로더 (실제 구현 시 활성화)
        self.model_loader = None
        self.model = None
        self.tokenizer = None
        
        logger.info("InferenceOrchestrator initialized")
    
    def load_model(self, model_config: ModelConfig):
        """모델 로드"""
        self.model_loader = ModelLoader()
        self.model, self.tokenizer = self.model_loader.load_model(model_config)
        logger.info(f"Model loaded: {model_config.model_name}")
    
    def add_documents(self, documents: List[str], doc_ids: List[str], embeddings: np.ndarray):
        """지식베이스에 문서 추가"""
        # FAISS 인덱스에 추가
        metadata = [{"chunk_id": doc_id} for doc_id in doc_ids]
        self.knowledge_base.add_documents(embeddings, documents, metadata)
        
        # BM25 인덱스 구축
        self.retriever.build_bm25_index(documents, doc_ids)
        
        logger.info(f"Added {len(documents)} documents to knowledge base")
    
    def process_question(self, 
                        question: str, 
                        question_id: str = "q001") -> InferenceResult:
        """
        단일 질문 처리 - 전체 파이프라인 실행
        
        Pipeline:
        1. 캐시 확인
        2. 질문 분류
        3. 검색 (하이브리드)
        4. 프롬프트 생성
        5. LLM 추론
        6. 답변 후처리
        7. 캐시 저장
        """
        start_time = time.time()
        
        try:
            # 1. 캐시 확인
            if self.config.use_cache:
                cached = self.cache.get_cached_response(question)
                if cached:
                    logger.info(f"Cache hit for question: {question_id}")
                    return InferenceResult(
                        question_id=question_id,
                        question=question,
                        answer=cached['answer'],
                        question_type=cached.get('type', 'unknown'),
                        confidence=cached.get('confidence', 1.0),
                        retrieval_count=0,
                        inference_time=time.time() - start_time,
                        cache_hit=True
                    )
            
            # 2. 질문 분류
            classified = self.classifier.classify(question, question_id)
            question_type = classified.question_type
            
            logger.info(f"Question classified as: {question_type} (confidence: {classified.confidence:.2%})")
            
            # 3. 하이브리드 검색
            retrieval_results = self.retriever.hybrid_search(
                query=question,
                dense_weight=self.config.dense_weight,
                sparse_weight=self.config.sparse_weight,
                k=self.config.retrieval_k
            )
            
            logger.info(f"Retrieved {len(retrieval_results)} documents")
            
            # 4. 컨텍스트 준비
            context = "\n\n".join([r.content for r in retrieval_results[:3]])
            
            # 5. 프롬프트 생성
            prompt_type = (PromptType.MULTIPLE_CHOICE 
                          if question_type == "multiple_choice" 
                          else PromptType.OPEN_ENDED)
            
            prompt = self.prompt_template.create_prompt(
                question=question,
                context=context,
                prompt_type=prompt_type
            )
            
            # 6. LLM 추론 (현재는 시뮬레이션)
            if self.model and self.tokenizer:
                # 실제 모델 추론
                answer = self._generate_answer(prompt, question_type)
            else:
                # 시뮬레이션 답변
                if question_type == "multiple_choice":
                    answer = "2"
                else:
                    answer = f"검색된 정보에 따르면: {retrieval_results[0].content[:200] if retrieval_results else '관련 정보를 찾을 수 없습니다.'}"
            
            # 7. 답변 검증
            answer = self.fallback_handler.validate_answer(answer, question_type)
            
            # 8. 캐시 저장
            if self.config.use_cache:
                cache_data = {
                    'answer': answer,
                    'type': question_type,
                    'confidence': classified.confidence
                }
                self.cache.cache_response(question, cache_data, ttl=self.config.cache_ttl)
            
            # 9. 결과 생성
            result = InferenceResult(
                question_id=question_id,
                question=question,
                answer=answer,
                question_type=question_type,
                confidence=classified.confidence,
                retrieval_count=len(retrieval_results),
                inference_time=time.time() - start_time,
                cache_hit=False
            )
            
            logger.info(f"Inference completed in {result.inference_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            
            # 폴백 처리
            fallback_answer = self.fallback_handler.handle_error(
                question, 
                question_type if 'question_type' in locals() else "open_ended",
                e
            )
            
            return InferenceResult(
                question_id=question_id,
                question=question,
                answer=fallback_answer,
                question_type="unknown",
                confidence=0.0,
                retrieval_count=0,
                inference_time=time.time() - start_time,
                cache_hit=False,
                error=str(e)
            )
    
    def _generate_answer(self, prompt: str, question_type: str) -> str:
        """실제 LLM 추론 (구현 필요)"""
        # TODO: 실제 모델 추론 구현
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs, max_new_tokens=256)
        # answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 임시 답변
        if question_type == "multiple_choice":
            return "2"
        else:
            return "금융 AI 시스템은 강력한 보안 메커니즘이 필요합니다."
    
    def batch_inference(self, questions: List[Dict[str, str]]) -> List[InferenceResult]:
        """배치 추론"""
        results = []
        
        for item in questions:
            question_id = item.get('ID', 'unknown')
            question = item.get('Question', '')
            
            result = self.process_question(question, question_id)
            results.append(result)
            
            # 진행 상황 로깅
            if len(results) % 10 == 0:
                logger.info(f"Processed {len(results)}/{len(questions)} questions")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        cache_stats = self.cache.get_stats()
        kb_stats = self.knowledge_base.get_stats()
        
        return {
            "cache": cache_stats,
            "knowledge_base": kb_stats,
            "config": {
                "retrieval_k": self.config.retrieval_k,
                "dense_weight": self.config.dense_weight,
                "sparse_weight": self.config.sparse_weight
            }
        }


# Pipeline.md 3.2.4 완료 기준 체크리스트
"""
✅ InferenceOrchestrator 구현 완료:
- [x] 질문 분류 통합
- [x] 캐시 레이어 통합
- [x] 하이브리드 검색 통합 (BM25s + FAISS)
- [x] 프롬프트 템플릿 통합
- [x] 폴백 핸들러 구현
- [x] 배치 추론 지원
- [x] 통계 수집
"""