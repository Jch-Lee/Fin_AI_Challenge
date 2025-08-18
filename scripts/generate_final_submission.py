#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
최종 제출파일 생성 - 8-bit 양자화 + 개선된 프롬프트 + 하이브리드 RAG
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

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from packages.rag.rag_pipeline import RAGPipeline
from packages.rag.embeddings.kure_embedder import KUREEmbedder
from packages.llm.qwen_quantized import QuantizedQwenLLM

def validate_answer_length(answer: str, question_id: str, is_mc: bool) -> str:
    """
    답변 길이 검증 및 수정
    
    Args:
        answer: 원본 답변
        question_id: 질문 ID
        is_mc: 객관식 여부
        
    Returns:
        검증된 답변
    """
    if is_mc:
        # 객관식: 1-2자리 숫자여야 함
        if len(answer) > 2:
            logger.warning(f"{question_id}: 객관식 답변이 너무 김 ({len(answer)}자) -> '1'로 수정")
            return "1"
        return answer
    else:
        # 주관식: 최대 500자로 제한
        if len(answer) > 500:
            logger.warning(f"{question_id}: 주관식 답변이 너무 김 ({len(answer)}자) -> 500자로 자름")
            return answer[:500]
        # 너무 짧은 답변도 확인
        if len(answer) < 10:
            logger.warning(f"{question_id}: 주관식 답변이 너무 짧음 ({len(answer)}자)")
        return answer

def generate_final_submission():
    """최종 제출파일 생성"""
    
    print("="*80)
    print("🚀 최종 제출파일 생성 시작")
    print("="*80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("설정: 8-bit 양자화 + 하이브리드 RAG + 개선된 프롬프트")
    
    start_time = time.time()
    
    # 1. 테스트 데이터 로드
    print("\n📂 테스트 데이터 로드 중...")
    test_df = pd.read_csv("data/competition/test.csv")
    total_questions = len(test_df)
    print(f"   총 {total_questions}개 질문 로드 완료")
    
    # 2. 컴포넌트 초기화
    print("\n🔧 시스템 컴포넌트 초기화...")
    
    # Embedder
    print("   1. KURE Embedder 초기화...")
    embedder = KUREEmbedder(
        model_name="nlpai-lab/KURE-v1",
        batch_size=32,
        show_progress=False
    )
    print("      ✅ Embedder 로드 완료")
    
    # RAG Pipeline
    print("   2. 하이브리드 RAG Pipeline 초기화...")
    rag_pipeline = RAGPipeline(
        embedder=embedder,
        retriever_type="hybrid",
        knowledge_base_path="data/rag/knowledge_base_fixed",
        enable_reranking=False,
        initial_retrieve_k=10,
        final_k=5
    )
    print("      ✅ 하이브리드 RAG Pipeline 초기화 완료")
    
    # LLM (8-bit)
    print("   3. Qwen2.5-7B-Instruct (8-bit) 초기화...")
    llm_start = time.time()
    llm = QuantizedQwenLLM(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        quantization_type="8bit",
        cache_dir="./models"
    )
    llm_init_time = time.time() - llm_start
    print(f"      ✅ 8-bit LLM 로드 완료 ({llm_init_time:.1f}초)")
    
    memory_info = llm.get_memory_footprint()
    print(f"      GPU 메모리: {memory_info}")
    
    # 프롬프트 및 추출기
    prompt_generator = SimpleImprovedPrompt()
    extractor = ImprovedAnswerExtractor()
    
    # 3. 전체 질문 처리
    print(f"\n🤖 전체 {total_questions}개 질문 처리 시작...")
    print("="*80)
    
    results = []
    failed_count = 0
    mc_count = 0
    oe_count = 0
    total_inference_time = 0
    
    # 진행률 표시를 위한 체크포인트
    checkpoints = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 515]
    
    for idx, row in test_df.iterrows():
        question_id = row['ID']
        question = row['Question']
        
        # 진행률 표시
        current_num = idx + 1
        if current_num in checkpoints or current_num % 25 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / current_num if current_num > 0 else 0
            remaining = (total_questions - current_num) * avg_time
            print(f"\n📊 진행률: {current_num}/{total_questions} ({current_num/total_questions*100:.1f}%)")
            print(f"   경과 시간: {elapsed/60:.1f}분, 예상 남은 시간: {remaining/60:.1f}분")
            print(f"   실패: {failed_count}개, 객관식: {mc_count}개, 주관식: {oe_count}개")
        
        try:
            # RAG 검색
            contexts_list = rag_pipeline.retrieve(
                query=question,
                top_k=5,
                use_reranking=False
            )
            
            if contexts_list:
                contexts = [doc.get('content', '') for doc in contexts_list]
            else:
                contexts = []
            
            # 객관식 여부 판단
            is_mc = any(f"{i}." in question or f"{i} " in question for i in range(1, 10))
            if is_mc:
                mc_count += 1
            else:
                oe_count += 1
            
            # 개선된 프롬프트 생성
            prompt = prompt_generator.create_improved_prompt(
                question=question,
                contexts=contexts,
                is_mc=is_mc
            )
            
            # LLM 생성
            inference_start = time.time()
            response = llm.generate_optimized(
                prompt=prompt,
                max_new_tokens=128 if is_mc else 256,
                temperature=0.1,
                use_cache=False
            )
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # 개선된 답변 추출
            if is_mc:
                max_choice = extractor.extract_choice_range(question)
                answer = extractor.extract_answer_from_response(
                    response=response,
                    max_choice=max_choice,
                    question=question
                )
            else:
                answer = extractor.extract_open_ended_answer(response)
            
            # 답변 길이 검증
            answer = validate_answer_length(answer, question_id, is_mc)
            
        except Exception as e:
            logger.error(f"질문 {question_id} 처리 실패: {e}")
            answer = "1" if is_mc else "처리 실패"
            failed_count += 1
        
        results.append({
            "ID": question_id,
            "Answer": answer
        })
        
        # 메모리 정리 (매 100개마다)
        if current_num % 100 == 0:
            import torch
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    avg_inference_time = total_inference_time / total_questions
    
    # 4. 결과 저장
    print(f"\n💾 결과 저장 중...")
    
    # submission.csv 생성
    submission_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"submission_8bit_{timestamp}.csv"
    
    submission_df.to_csv(submission_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 제출파일 저장: {submission_path}")
    
    # 백업 저장
    backup_path = f"test_results/submission_8bit_backup_{timestamp}.csv"
    submission_df.to_csv(backup_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 백업파일 저장: {backup_path}")
    
    # 5. 최종 통계
    print(f"\n" + "="*80)
    print("🎉 최종 제출파일 생성 완료!")
    print("="*80)
    
    print(f"\n📊 처리 통계:")
    print(f"   총 질문 수: {total_questions}개")
    print(f"   객관식: {mc_count}개 ({mc_count/total_questions*100:.1f}%)")
    print(f"   주관식: {oe_count}개 ({oe_count/total_questions*100:.1f}%)")
    print(f"   처리 실패: {failed_count}개 ({failed_count/total_questions*100:.1f}%)")
    
    print(f"\n⏱️ 시간 통계:")
    print(f"   총 처리 시간: {total_time/60:.1f}분")
    print(f"   LLM 초기화: {llm_init_time:.1f}초")
    print(f"   평균 추론 시간: {avg_inference_time:.2f}초/질문")
    print(f"   총 추론 시간: {total_inference_time/60:.1f}분")
    
    print(f"\n🖥️ 시스템 정보:")
    final_memory = llm.get_memory_footprint()
    print(f"   최종 GPU 메모리: {final_memory}")
    print(f"   양자화 방식: 8-bit")
    print(f"   RAG 방식: 하이브리드 (BM25 + Vector)")
    
    print(f"\n📁 생성된 파일:")
    print(f"   메인 제출파일: {submission_path}")
    print(f"   백업 파일: {backup_path}")
    
    # 6. 파일 검증
    print(f"\n🔍 제출파일 검증...")
    try:
        # 파일 읽기 테스트
        verify_df = pd.read_csv(submission_path)
        
        # 기본 검증
        assert len(verify_df) == total_questions, f"행 수 불일치: {len(verify_df)} != {total_questions}"
        assert list(verify_df.columns) == ['ID', 'Answer'], f"컬럼명 오류: {list(verify_df.columns)}"
        assert verify_df['ID'].isnull().sum() == 0, "ID에 null 값 존재"
        assert verify_df['Answer'].isnull().sum() == 0, "Answer에 null 값 존재"
        
        # 답변 길이 검증
        long_answers = verify_df[verify_df['Answer'].str.len() > 500]
        if len(long_answers) > 0:
            print(f"   ⚠️ 긴 답변 {len(long_answers)}개 발견 (500자 초과)")
        
        print(f"   ✅ 제출파일 검증 완료")
        print(f"   - 총 행 수: {len(verify_df)}")
        print(f"   - ID 범위: {verify_df['ID'].min()} ~ {verify_df['ID'].max()}")
        print(f"   - 답변 평균 길이: {verify_df['Answer'].str.len().mean():.1f}자")
        
    except Exception as e:
        print(f"   ❌ 제출파일 검증 실패: {e}")
        return False
    
    print(f"\n🏁 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return True


if __name__ == "__main__":
    success = generate_final_submission()
    if success:
        print("\n🎯 최종 제출파일 생성 성공!")
    else:
        print("\n💥 최종 제출파일 생성 실패!")
        sys.exit(1)