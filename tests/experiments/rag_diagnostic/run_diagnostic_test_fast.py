#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG System Fast Diagnostic Test (5 questions only)
Simplified version for quick testing
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

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quick_diagnostic_test():
    """Run quick diagnostic test with 5 questions"""
    
    logger.info("="*60)
    logger.info("RAG Fast Diagnostic Test (5 questions)")
    logger.info("="*60)
    
    # Initialize RAG
    logger.info("Initializing RAG system...")
    from scripts.load_rag_v2 import RAGSystemV2
    
    rag = RAGSystemV2()
    rag.load_all()
    retriever = rag.create_hybrid_retriever()
    
    logger.info(f"RAG initialized: BM25={retriever.bm25_weight:.1%}, Vector={retriever.vector_weight:.1%}")
    
    # Test questions (3 MC, 2 Descriptive)
    test_questions = [
        {"ID": "TEST_000", "Type": "multiple_choice", 
         "Question": "금융산업의 이해와 관련하여 금융투자업의 구분에 해당하지 않는 것은?\n1 소비자금융업\n2 투자자문업\n3 투자매매업\n4 투자중개업\n5 보험중개업"},
        {"ID": "TEST_001", "Type": "multiple_choice",
         "Question": "위험 관리 계획 수립 시 고려해야 할 요소로 적절하지 않은 것은?\n1 수행인력\n2 위험 수용\n3 위험 대응 전략 선정\n4 대상\n5 기간"},
        {"ID": "TEST_002", "Type": "multiple_choice",
         "Question": "관리체계 수립 및 운영'의 '정책 수립' 단계에서 가장 중요한 요소는 무엇인가?\n1 정보보호 및 개인정보보호 정책의 제·개정\n2 경영진의 참여\n3 최고책임자의 지정\n4 자원 할당\n5 내부 감사 절차의 수립"},
        {"ID": "TEST_004", "Type": "descriptive",
         "Question": "트로이 목마(Trojan) 기반 원격제어 악성코드(RAT)의 특징과 주요 탐지 지표를 설명하세요."},
        {"ID": "TEST_007", "Type": "descriptive",
         "Question": "전자금융거래법에 따라 이용자가 금융 분쟁조정을 신청할 수 있는 기관을 기술하세요."}
    ]
    
    results = []
    
    for idx, q in enumerate(test_questions):
        logger.info(f"\n[{idx+1}/5] Analyzing {q['ID']} ({q['Type']})...")
        
        try:
            # Retrieve documents
            search_results = retriever.search(q['Question'], k=5)
            
            # Analyze scores
            bm25_scores = []
            vector_scores = []
            retrieval_info = []
            
            for result in search_results:
                bm25_score = getattr(result, 'bm25_score', 0)
                vector_score = getattr(result, 'vector_score', 0)
                hybrid_score = getattr(result, 'hybrid_score', 0)
                
                bm25_scores.append(bm25_score)
                vector_scores.append(vector_score)
                
                retrieval_info.append({
                    'content_preview': getattr(result, 'content', '')[:200] + '...',
                    'bm25_score': float(bm25_score),
                    'vector_score': float(vector_score),
                    'hybrid_score': float(hybrid_score),
                    'retrieval_methods': getattr(result, 'retrieval_methods', [])
                })
            
            # Calculate dominance
            avg_bm25 = np.mean(bm25_scores) if bm25_scores else 0
            avg_vector = np.mean(vector_scores) if vector_scores else 0
            ratio = avg_bm25 / (avg_vector + 1e-10)
            
            score_analysis = {
                'avg_bm25_score': float(avg_bm25),
                'avg_vector_score': float(avg_vector),
                'bm25_to_vector_ratio': float(ratio),
                'bm25_dominant': bool(ratio > 1.5),
                'vector_dominant': bool(ratio < 0.67),
                'balanced': bool(0.67 <= ratio <= 1.5)
            }
            
            # Store result
            result = {
                'question_id': q['ID'],
                'question_type': q['Type'],
                'question': q['Question'],
                'retrieval_results': {
                    'top_5_documents': retrieval_info,
                    'score_analysis': score_analysis
                }
            }
            
            results.append(result)
            
            # Log dominance
            if score_analysis['bm25_dominant']:
                logger.info(f"  → BM25 dominant (ratio: {ratio:.2f})")
            elif score_analysis['vector_dominant']:
                logger.info(f"  → Vector dominant (ratio: {ratio:.2f})")
            else:
                logger.info(f"  → Balanced (ratio: {ratio:.2f})")
                
        except Exception as e:
            logger.error(f"Error processing {q['ID']}: {e}")
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"fast_diagnostic_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    
    # Generate summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    mc_results = [r for r in results if r['question_type'] == 'multiple_choice']
    desc_results = [r for r in results if r['question_type'] == 'descriptive']
    
    bm25_dominant = sum(1 for r in results if r['retrieval_results']['score_analysis']['bm25_dominant'])
    vector_dominant = sum(1 for r in results if r['retrieval_results']['score_analysis']['vector_dominant'])
    balanced = sum(1 for r in results if r['retrieval_results']['score_analysis']['balanced'])
    
    logger.info(f"Total questions: {len(results)}")
    logger.info(f"- Multiple choice: {len(mc_results)}")
    logger.info(f"- Descriptive: {len(desc_results)}")
    logger.info("")
    logger.info(f"Score dominance:")
    logger.info(f"- BM25 dominant: {bm25_dominant} ({bm25_dominant/len(results)*100:.0f}%)")
    logger.info(f"- Vector dominant: {vector_dominant} ({vector_dominant/len(results)*100:.0f}%)")
    logger.info(f"- Balanced: {balanced} ({balanced/len(results)*100:.0f}%)")
    
    # MC vs Desc patterns
    mc_bm25 = sum(1 for r in mc_results if r['retrieval_results']['score_analysis']['bm25_dominant'])
    desc_vector = sum(1 for r in desc_results if r['retrieval_results']['score_analysis']['vector_dominant'])
    
    if mc_results:
        logger.info(f"\nMultiple choice: {mc_bm25}/{len(mc_results)} are BM25 dominant")
    if desc_results:
        logger.info(f"Descriptive: {desc_vector}/{len(desc_results)} are Vector dominant")
    
    logger.info("\n" + "="*60)
    logger.info("Fast diagnostic test completed!")
    logger.info("="*60)


if __name__ == "__main__":
    quick_diagnostic_test()