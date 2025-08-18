#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 파이프라인 진단 및 품질 테스트
각 단계별로 상세 진단 수행
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import torch

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.rag.rag_pipeline import RAGPipeline
from packages.rag.embeddings.kure_embedder import KUREEmbedder
from packages.llm.qwen_quantized import QuantizedQwenLLM
from packages.llm.prompt_templates import FinancePromptTemplate

def print_diagnostic(stage: str, status: str, message: str):
    """진단 메시지 출력"""
    status_symbol = "✓" if status == "OK" else "✗" if status == "ERROR" else "⚠"
    print(f"\n[{stage}] {status_symbol} {status}: {message}")

def diagnose_rag_retrieval(rag_pipeline: RAGPipeline, question: str) -> Tuple[List[Dict], Dict]:
    """
    RAG 검색 단계 진단
    """
    diagnosis = {
        "stage": "RAG_RETRIEVAL",
        "status": "OK",
        "issues": [],
        "metrics": {}
    }
    
    print("\n" + "="*60)
    print("RAG 검색 진단")
    print("="*60)
    
    try:
        # 1. Embedder 테스트
        print("\n1. Embedder 테스트...")
        query_embedding = rag_pipeline.embedder.embed(question, is_query=True)
        print(f"   - 임베딩 차원: {len(query_embedding)}")
        print(f"   - 임베딩 norm: {np.linalg.norm(query_embedding):.4f}")
        diagnosis["metrics"]["embedding_dim"] = len(query_embedding)
        
        # 2. Knowledge Base 상태 확인
        print("\n2. Knowledge Base 확인...")
        try:
            # get_statistics 메서드가 없을 수 있으므로 직접 확인
            if hasattr(rag_pipeline.knowledge_base, 'num_documents'):
                num_docs = rag_pipeline.knowledge_base.num_documents
            elif hasattr(rag_pipeline.knowledge_base, 'documents'):
                num_docs = len(rag_pipeline.knowledge_base.documents)
            else:
                # FAISS 인덱스로 확인
                num_docs = rag_pipeline.knowledge_base.faiss_index.ntotal if hasattr(rag_pipeline.knowledge_base, 'faiss_index') else 0
            
            print(f"   - 총 문서 수: {num_docs}")
            
            if num_docs == 0:
                diagnosis["status"] = "ERROR"
                diagnosis["issues"].append("Knowledge Base가 비어있음")
                print_diagnostic("Knowledge Base", "ERROR", "문서가 없습니다!")
                return [], diagnosis
        except Exception as e:
            print(f"   - Knowledge Base 확인 실패: {e}")
            diagnosis["issues"].append(f"KB 확인 실패: {str(e)}")
        
        # 3. 검색 수행 (Reranking 없이 - 30개 초기 검색)
        print("\n3. 기본 검색 테스트 (30개 초기 검색)...")
        start_time = time.time()
        results_no_rerank = rag_pipeline.retrieve(
            query=question,
            top_k=30,  # 초기 30개 검색
            use_reranking=False
        )
        retrieval_time = time.time() - start_time
        
        print(f"   - 검색 시간: {retrieval_time:.2f}초")
        print(f"   - 검색된 문서: {len(results_no_rerank)}개")
        
        if len(results_no_rerank) == 0:
            diagnosis["status"] = "ERROR"
            diagnosis["issues"].append("검색 결과가 없음")
            print_diagnostic("Vector Search", "ERROR", "문서를 검색할 수 없습니다!")
        else:
            # 검색 결과 품질 확인
            scores = [doc.get('score', 0) for doc in results_no_rerank]
            print(f"   - 점수 범위: {min(scores):.4f} ~ {max(scores):.4f}")
            print(f"   - 평균 점수: {np.mean(scores):.4f}")
            
            # 상위 3개 문서 내용 확인
            print("\n   상위 3개 문서:")
            for i, doc in enumerate(results_no_rerank[:3], 1):
                content = doc.get('content', '')
                print(f"   [{i}] Score: {doc.get('score', 0):.4f}")
                print(f"       Content: {content[:100]}...")
                
                # Legacy document 확인
                if content.startswith("Legacy document"):
                    diagnosis["issues"].append(f"문서 {i}: Legacy placeholder 발견")
        
        # 4. Reranking 테스트 (30→5 파이프라인)
        print("\n4. Reranking 테스트 (30→5 파이프라인)...")
        if rag_pipeline.enable_reranking and rag_pipeline.reranker:
            print("   - 30개 문서를 5개로 리랭킹")
            start_time = time.time()
            results_with_rerank = rag_pipeline.retrieve(
                query=question,
                top_k=5,  # 최종 5개
                use_reranking=True
            )
            rerank_time = time.time() - start_time
            
            print(f"   - Reranking 시간: {rerank_time:.2f}초")
            print(f"   - 최종 문서: {len(results_with_rerank)}개 (30→5 리랭킹)")
            
            if len(results_with_rerank) > 0:
                # Reranking 효과 확인
                rerank_scores = [doc.get('score', 0) for doc in results_with_rerank]
                print(f"   - Reranked 점수 범위: {min(rerank_scores):.4f} ~ {max(rerank_scores):.4f}")
                
                # 순서 변경 확인
                print("\n   Reranking 후 상위 3개:")
                for i, doc in enumerate(results_with_rerank[:3], 1):
                    print(f"   [{i}] Score: {doc.get('score', 0):.4f}, Rerank: {doc.get('rerank_score', 0):.4f}")
                    print(f"       Content: {doc.get('content', '')[:100]}...")
                
                print(f"\n   ✓ 30→5 리랭킹 파이프라인 정상 작동")
            
            diagnosis["metrics"]["reranking_enabled"] = True
            return results_with_rerank, diagnosis
        else:
            print("   - Reranking 비활성화 상태")
            diagnosis["metrics"]["reranking_enabled"] = False
            return results_no_rerank[:5], diagnosis
            
    except Exception as e:
        diagnosis["status"] = "ERROR"
        diagnosis["issues"].append(f"Exception: {str(e)}")
        print_diagnostic("RAG Retrieval", "ERROR", str(e))
        return [], diagnosis

def diagnose_llm_generation(llm: QuantizedQwenLLM, question: str, contexts: List[str]) -> Tuple[str, Dict]:
    """
    LLM 생성 단계 진단
    """
    diagnosis = {
        "stage": "LLM_GENERATION",
        "status": "OK",
        "issues": [],
        "metrics": {}
    }
    
    print("\n" + "="*60)
    print("LLM 생성 진단")
    print("="*60)
    
    try:
        # 1. 컨텍스트 품질 확인
        print("\n1. 컨텍스트 확인...")
        if not contexts:
            print("   - 경고: 컨텍스트가 없음 (RAG 실패)")
            diagnosis["issues"].append("컨텍스트 없음")
            contexts = ["관련 문서를 찾을 수 없습니다."]
        else:
            total_length = sum(len(c) for c in contexts)
            print(f"   - 컨텍스트 수: {len(contexts)}개")
            print(f"   - 총 길이: {total_length}자")
            diagnosis["metrics"]["context_count"] = len(contexts)
            diagnosis["metrics"]["context_length"] = total_length
        
        # 2. 질문 유형 판단
        print("\n2. 질문 유형 판단...")
        is_mc = any(f"{i}." in question or f"{i} " in question for i in range(1, 10))
        question_type = "객관식" if is_mc else "주관식"
        print(f"   - 질문 유형: {question_type}")
        diagnosis["metrics"]["question_type"] = question_type
        
        # 3. 프롬프트 생성
        print("\n3. 프롬프트 생성...")
        if is_mc:
            prompt = f"""당신은 금융보안 전문가입니다.

참고 문서:
{chr(10).join(contexts[:3]) if contexts else "관련 문서 없음"}

다음 객관식 질문에 대해 정답 번호만 출력하세요.

질문: {question}

답변:"""
        else:
            prompt = f"""당신은 금융보안 전문가입니다.

참고 문서:
{chr(10).join(contexts[:3]) if contexts else "관련 문서 없음"}

다음 질문에 대해 정확하고 간략한 답변을 한국어로 작성하세요.

질문: {question}

답변:"""
        
        print(f"   - 프롬프트 길이: {len(prompt)}자")
        
        # 4. LLM 생성
        print("\n4. LLM 답변 생성...")
        start_time = time.time()
        response = llm.generate_optimized(
            prompt=prompt,
            max_new_tokens=64 if is_mc else 256,
            temperature=0.3,
            use_cache=False
        )
        generation_time = time.time() - start_time
        
        print(f"   - 생성 시간: {generation_time:.2f}초")
        print(f"   - 답변 길이: {len(response)}자")
        print(f"   - 답변: {response[:200]}...")
        
        diagnosis["metrics"]["generation_time"] = generation_time
        diagnosis["metrics"]["answer_length"] = len(response)
        
        # 5. 답변 품질 검증
        print("\n5. 답변 품질 검증...")
        if is_mc:
            # 객관식 답변 검증
            if any(str(i) in response for i in range(1, 10)):
                print("   - 객관식 답변 형식: OK")
            else:
                print("   - 경고: 객관식 답변에 번호가 없음")
                diagnosis["issues"].append("객관식 답변 형식 오류")
        else:
            # 주관식 답변 검증
            if len(response) < 10:
                print("   - 경고: 답변이 너무 짧음")
                diagnosis["issues"].append("답변 길이 부족")
            elif "죄송" in response or "알 수 없" in response:
                print("   - 경고: 불확실한 답변")
                diagnosis["issues"].append("불확실한 답변")
            else:
                print("   - 주관식 답변: OK")
        
        return response, diagnosis
        
    except Exception as e:
        diagnosis["status"] = "ERROR"
        diagnosis["issues"].append(f"Exception: {str(e)}")
        print_diagnostic("LLM Generation", "ERROR", str(e))
        return "생성 실패", diagnosis

def main():
    """메인 진단 함수"""
    print("="*60)
    print("RAG 파이프라인 종합 진단")
    print("="*60)
    
    # GPU 확인
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n경고: GPU 사용 불가")
    
    # 1. 테스트 데이터 로드
    print("\n[데이터 로드]")
    test_csv_path = "data/competition/test.csv"
    
    if not Path(test_csv_path).exists():
        print(f"ERROR: {test_csv_path} 파일이 없습니다")
        return
    
    df = pd.read_csv(test_csv_path)
    print(f"총 질문 수: {len(df)}개")
    
    # 다양한 샘플 선택 (처음, 중간, 끝)
    sample_indices = [0, 100, 200, 300, 400]
    sample_df = df.iloc[sample_indices]
    print(f"테스트 샘플: {sample_df['ID'].tolist()}")
    
    # 2. 컴포넌트 초기화
    print("\n[컴포넌트 초기화]")
    all_diagnoses = []
    
    try:
        # Embedder
        print("- KURE Embedder 로딩...")
        embedder = KUREEmbedder(
            model_name="nlpai-lab/KURE-v1",
            batch_size=32,
            show_progress=False
        )
        
        # RAG Pipeline
        print("- RAG Pipeline 로딩 (Vector Retriever + Reranking)...")
        rag_pipeline = RAGPipeline(
            embedder=embedder,
            retriever_type="vector",  # Vector retriever 사용 (HybridRetriever 대신)
            knowledge_base_path="data/rag/knowledge_base_fixed",
            enable_reranking=True,
            initial_retrieve_k=30,
            final_k=5
        )
        
        # LLM
        print("- Qwen2.5-7B-Instruct 로딩...")
        llm = QuantizedQwenLLM(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            quantization_type="4bit",
            cache_dir="./models"
        )
        print(f"- GPU 메모리 사용량: {llm.get_memory_footprint()}")
        
    except Exception as e:
        print(f"\nERROR: 초기화 실패 - {e}")
        return
    
    # 3. 샘플별 진단
    print("\n" + "="*60)
    print("샘플별 상세 진단 시작")
    print("="*60)
    
    for idx, row in sample_df.iterrows():
        question_id = row['ID']
        question = row['Question']
        
        print(f"\n\n{'='*60}")
        print(f"샘플: {question_id}")
        print(f"{'='*60}")
        print(f"질문: {question[:100]}...")
        
        sample_diagnosis = {
            "id": question_id,
            "stages": []
        }
        
        # RAG 검색 진단
        contexts_list, rag_diagnosis = diagnose_rag_retrieval(rag_pipeline, question)
        sample_diagnosis["stages"].append(rag_diagnosis)
        
        if rag_diagnosis["status"] == "ERROR" and "Knowledge Base가 비어있음" in rag_diagnosis["issues"]:
            print("\n[중단] Knowledge Base 문제로 진단 중단")
            all_diagnoses.append(sample_diagnosis)
            break
        
        # 컨텍스트 준비
        contexts = [doc.get('content', '') for doc in contexts_list]
        
        # LLM 생성 진단
        answer, llm_diagnosis = diagnose_llm_generation(llm, question, contexts)
        sample_diagnosis["stages"].append(llm_diagnosis)
        
        # 종합 평가
        print("\n[종합 평가]")
        total_issues = len(rag_diagnosis["issues"]) + len(llm_diagnosis["issues"])
        if total_issues == 0:
            print("✓ 모든 단계 정상 작동")
        else:
            print(f"⚠ 발견된 문제: {total_issues}개")
            for issue in rag_diagnosis["issues"] + llm_diagnosis["issues"]:
                print(f"  - {issue}")
        
        all_diagnoses.append(sample_diagnosis)
        
        # 심각한 문제 발견 시 중단
        if rag_diagnosis["status"] == "ERROR" or llm_diagnosis["status"] == "ERROR":
            print("\n[중단] 심각한 오류 발견으로 진단 중단")
            break
    
    # 4. 최종 보고서
    print("\n\n" + "="*60)
    print("진단 결과 요약")
    print("="*60)
    
    # 문제점 집계
    all_issues = []
    for diag in all_diagnoses:
        for stage in diag["stages"]:
            all_issues.extend(stage["issues"])
    
    if not all_issues:
        print("\n✓ 파이프라인 정상 작동 확인")
    else:
        print(f"\n⚠ 총 {len(all_issues)}개 문제 발견:")
        issue_counts = {}
        for issue in all_issues:
            key = issue.split(":")[0]
            issue_counts[key] = issue_counts.get(key, 0) + 1
        
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {issue_type}: {count}회")
    
    # 성능 메트릭
    print("\n[성능 메트릭]")
    generation_times = []
    for diag in all_diagnoses:
        for stage in diag["stages"]:
            if stage["stage"] == "LLM_GENERATION":
                if "generation_time" in stage["metrics"]:
                    generation_times.append(stage["metrics"]["generation_time"])
    
    if generation_times:
        print(f"  - 평균 생성 시간: {np.mean(generation_times):.2f}초")
        print(f"  - 예상 전체 시간: {np.mean(generation_times) * len(df) / 60:.1f}분")
    
    # 결과 저장
    output_path = "test_results/pipeline_diagnosis.json"
    Path("test_results").mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(all_diagnoses),
            "diagnoses": all_diagnoses,
            "summary": {
                "total_issues": len(all_issues),
                "issue_types": issue_counts if all_issues else {}
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n진단 결과 저장: {output_path}")

if __name__ == "__main__":
    main()