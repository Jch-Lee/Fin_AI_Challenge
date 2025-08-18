#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
최종 제출파일 생성 - 상세 로그 버전
8-bit 양자화 + 개선된 프롬프트 + 하이브리드 RAG
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
import time
from datetime import datetime
from scripts.simple_improved_prompt import SimpleImprovedPrompt
from scripts.improved_answer_extraction import ImprovedAnswerExtractor

# 상세 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('final_submission_verbose.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from packages.rag.rag_pipeline import RAGPipeline
from packages.rag.embeddings.kure_embedder import KUREEmbedder
from packages.llm.qwen_quantized import QuantizedQwenLLM

def validate_answer_length(answer: str, question_id: str, is_mc: bool) -> str:
    """답변 길이 검증 및 수정"""
    if is_mc:
        if len(answer) > 2:
            logger.warning(f"{question_id}: 객관식 답변이 너무 김 ({len(answer)}자) -> '1'로 수정")
            return "1"
        return answer
    else:
        if len(answer) > 500:
            logger.warning(f"{question_id}: 주관식 답변이 너무 김 ({len(answer)}자) -> 500자로 자름")
            return answer[:500]
        if len(answer) < 10:
            logger.warning(f"{question_id}: 주관식 답변이 너무 짧음 ({len(answer)}자)")
        return answer

def generate_final_submission():
    """최종 제출파일 생성 - 상세 로그 버전"""
    
    logger.info("="*80)
    logger.info("🚀 최종 제출파일 생성 시작 (상세 로그 버전)")
    logger.info("="*80)
    logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("설정: 8-bit 양자화 + 하이브리드 RAG + 개선된 프롬프트")
    
    start_time = time.time()
    
    # 1. 테스트 데이터 로드
    logger.info("📂 테스트 데이터 로드 중...")
    test_df = pd.read_csv("data/competition/test.csv")
    total_questions = len(test_df)
    logger.info(f"   총 {total_questions}개 질문 로드 완료")
    
    # 2. 컴포넌트 초기화
    logger.info("🔧 시스템 컴포넌트 초기화...")
    
    # Embedder
    logger.info("   1. KURE Embedder 초기화...")
    embedder = KUREEmbedder(
        model_name="nlpai-lab/KURE-v1",
        batch_size=32,
        show_progress=False
    )
    logger.info("      ✅ Embedder 로드 완료")
    
    # RAG Pipeline
    logger.info("   2. 하이브리드 RAG Pipeline 초기화...")
    rag_pipeline = RAGPipeline(
        embedder=embedder,
        retriever_type="hybrid",
        knowledge_base_path="data/rag/knowledge_base_fixed",
        enable_reranking=False,
        initial_retrieve_k=10,
        final_k=5
    )
    logger.info("      ✅ 하이브리드 RAG Pipeline 초기화 완료")
    
    # LLM (8-bit)
    logger.info("   3. Qwen2.5-7B-Instruct (8-bit) 초기화...")
    llm_start = time.time()
    llm = QuantizedQwenLLM(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        quantization_type="8bit",
        cache_dir="./models"
    )
    llm_init_time = time.time() - llm_start
    logger.info(f"      ✅ 8-bit LLM 로드 완료 ({llm_init_time:.1f}초)")
    
    memory_info = llm.get_memory_footprint()
    logger.info(f"      GPU 메모리: {memory_info}")
    
    # 프롬프트 및 추출기
    prompt_generator = SimpleImprovedPrompt()
    extractor = ImprovedAnswerExtractor()
    
    # 3. 전체 질문 처리
    logger.info(f"🤖 전체 {total_questions}개 질문 처리 시작...")
    logger.info("="*80)
    
    results = []
    failed_count = 0
    mc_count = 0
    oe_count = 0
    total_inference_time = 0
    
    for idx, row in test_df.iterrows():
        question_id = row['ID']
        question = row['Question']
        
        current_num = idx + 1
        elapsed = time.time() - start_time
        
        # 매 질문마다 상세 로그
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 질문 {current_num}/{total_questions} ({current_num/total_questions*100:.1f}%)")
        logger.info(f"🆔 ID: {question_id}")
        logger.info(f"❓ 질문: {question[:80]}...")
        logger.info(f"⏱️ 경과시간: {elapsed/60:.1f}분")
        
        if current_num > 1:
            avg_time = elapsed / (current_num - 1)
            remaining_questions = total_questions - current_num
            eta = remaining_questions * avg_time
            eta_time = datetime.fromtimestamp(time.time() + eta)
            logger.info(f"🕐 예상완료: {eta_time.strftime('%H:%M:%S')} (약 {eta/60:.0f}분 후)")
        
        try:
            # RAG 검색
            logger.info("🔍 하이브리드 RAG 검색 중...")
            search_start = time.time()
            
            contexts_list = rag_pipeline.retrieve(
                query=question,
                top_k=5,
                use_reranking=False
            )
            
            search_time = time.time() - search_start
            
            if contexts_list:
                contexts = [doc.get('content', '') for doc in contexts_list]
                logger.info(f"   ✅ 검색 완료: {len(contexts_list)}개 문서 ({search_time:.2f}초)")
                logger.info(f"   📊 첫 번째 문서 점수: {contexts_list[0].get('score', 0):.4f}")
            else:
                contexts = []
                logger.info(f"   ⚠️ 검색 결과 없음 ({search_time:.2f}초)")
            
            # 객관식 여부 판단
            is_mc = any(f"{i}." in question or f"{i} " in question for i in range(1, 10))
            question_type = "객관식" if is_mc else "주관식"
            logger.info(f"📝 질문 유형: {question_type}")
            
            if is_mc:
                mc_count += 1
            else:
                oe_count += 1
            
            # 개선된 프롬프트 생성
            logger.info("📝 개선된 프롬프트 생성 중...")
            prompt = prompt_generator.create_improved_prompt(
                question=question,
                contexts=contexts,
                is_mc=is_mc
            )
            logger.info(f"   ✅ 프롬프트 생성 완료 (길이: {len(prompt)}자)")
            
            # LLM 생성
            logger.info("🤖 LLM 답변 생성 중...")
            inference_start = time.time()
            
            response = llm.generate_optimized(
                prompt=prompt,
                max_new_tokens=128 if is_mc else 256,
                temperature=0.1,
                use_cache=False
            )
            
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            logger.info(f"   ✅ 답변 생성 완료 ({inference_time:.2f}초)")
            logger.info(f"   📄 생성된 답변: {response[:100]}...")
            
            # 개선된 답변 추출
            if is_mc:
                max_choice = extractor.extract_choice_range(question)
                answer = extractor.extract_answer_from_response(
                    response=response,
                    max_choice=max_choice,
                    question=question
                )
                logger.info(f"   🎯 선택지 범위: 1-{max_choice}")
                logger.info(f"   ✅ 추출된 답변: '{answer}'")
            else:
                answer = extractor.extract_open_ended_answer(response)
                logger.info(f"   ✅ 추출된 답변: {answer[:100]}... (총 {len(answer)}자)")
            
            # 답변 길이 검증
            validated_answer = validate_answer_length(answer, question_id, is_mc)
            if validated_answer != answer:
                logger.info(f"   🔧 답변 수정됨: '{answer}' → '{validated_answer}'")
                answer = validated_answer
            
            total_time = search_time + inference_time
            logger.info(f"   ⏱️ 총 처리시간: {total_time:.2f}초 (검색: {search_time:.2f}초, 생성: {inference_time:.2f}초)")
            logger.info(f"   🎉 최종 답변: '{answer}'")
            
        except Exception as e:
            logger.error(f"   ❌ 질문 {question_id} 처리 실패: {e}")
            answer = "1" if is_mc else "처리 실패"
            failed_count += 1
            logger.info(f"   🔧 기본값 사용: '{answer}'")
        
        results.append({
            "ID": question_id,
            "Answer": answer
        })
        
        # 통계 요약 (매 25개마다)
        if current_num % 25 == 0:
            logger.info(f"\n📊 중간 통계 (질문 {current_num}/{total_questions}):")
            logger.info(f"   객관식: {mc_count}개, 주관식: {oe_count}개, 실패: {failed_count}개")
            avg_inference = total_inference_time / current_num
            logger.info(f"   평균 추론시간: {avg_inference:.2f}초/질문")
            
            # 메모리 정리
            import torch
            torch.cuda.empty_cache()
            logger.info("   🧹 GPU 메모리 정리 완료")
    
    total_time = time.time() - start_time
    avg_inference_time = total_inference_time / total_questions
    
    # 4. 결과 저장
    logger.info(f"\n💾 결과 저장 중...")
    
    submission_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"submission_8bit_verbose_{timestamp}.csv"
    
    submission_df.to_csv(submission_path, index=False, encoding='utf-8-sig')
    logger.info(f"   ✅ 제출파일 저장: {submission_path}")
    
    backup_path = f"test_results/submission_8bit_verbose_backup_{timestamp}.csv"
    submission_df.to_csv(backup_path, index=False, encoding='utf-8-sig')
    logger.info(f"   ✅ 백업파일 저장: {backup_path}")
    
    # 5. 최종 통계
    logger.info(f"\n" + "="*80)
    logger.info("🎉 최종 제출파일 생성 완료!")
    logger.info("="*80)
    
    logger.info(f"\n📊 처리 통계:")
    logger.info(f"   총 질문 수: {total_questions}개")
    logger.info(f"   객관식: {mc_count}개 ({mc_count/total_questions*100:.1f}%)")
    logger.info(f"   주관식: {oe_count}개 ({oe_count/total_questions*100:.1f}%)")
    logger.info(f"   처리 실패: {failed_count}개 ({failed_count/total_questions*100:.1f}%)")
    
    logger.info(f"\n⏱️ 시간 통계:")
    logger.info(f"   총 처리 시간: {total_time/60:.1f}분")
    logger.info(f"   LLM 초기화: {llm_init_time:.1f}초")
    logger.info(f"   평균 추론 시간: {avg_inference_time:.2f}초/질문")
    logger.info(f"   총 추론 시간: {total_inference_time/60:.1f}분")
    
    logger.info(f"\n🖥️ 시스템 정보:")
    final_memory = llm.get_memory_footprint()
    logger.info(f"   최종 GPU 메모리: {final_memory}")
    logger.info(f"   양자화 방식: 8-bit")
    logger.info(f"   RAG 방식: 하이브리드 (BM25 + Vector)")
    
    logger.info(f"\n📁 생성된 파일:")
    logger.info(f"   메인 제출파일: {submission_path}")
    logger.info(f"   백업 파일: {backup_path}")
    
    # 6. 파일 검증
    logger.info(f"\n🔍 제출파일 검증...")
    try:
        verify_df = pd.read_csv(submission_path)
        
        assert len(verify_df) == total_questions, f"행 수 불일치: {len(verify_df)} != {total_questions}"
        assert list(verify_df.columns) == ['ID', 'Answer'], f"컬럼명 오류: {list(verify_df.columns)}"
        assert verify_df['ID'].isnull().sum() == 0, "ID에 null 값 존재"
        assert verify_df['Answer'].isnull().sum() == 0, "Answer에 null 값 존재"
        
        long_answers = verify_df[verify_df['Answer'].str.len() > 500]
        if len(long_answers) > 0:
            logger.warning(f"   ⚠️ 긴 답변 {len(long_answers)}개 발견 (500자 초과)")
        
        logger.info(f"   ✅ 제출파일 검증 완료")
        logger.info(f"   - 총 행 수: {len(verify_df)}")
        logger.info(f"   - ID 범위: {verify_df['ID'].min()} ~ {verify_df['ID'].max()}")
        logger.info(f"   - 답변 평균 길이: {verify_df['Answer'].str.len().mean():.1f}자")
        
        # 샘플 답변 출력
        logger.info(f"\n🔍 샘플 답변 (처음 10개):")
        for i in range(min(10, len(verify_df))):
            row = verify_df.iloc[i]
            logger.info(f"   {row['ID']}: {row['Answer'][:50]}...")
        
    except Exception as e:
        logger.error(f"   ❌ 제출파일 검증 실패: {e}")
        return False
    
    logger.info(f"\n🏁 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return True


if __name__ == "__main__":
    success = generate_final_submission()
    if success:
        logger.info("\n🎯 최종 제출파일 생성 성공!")
    else:
        logger.error("\n💥 최종 제출파일 생성 실패!")
        sys.exit(1)