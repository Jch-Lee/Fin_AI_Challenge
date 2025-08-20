#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test to verify the diagnostic system works
Tests with only 2 questions (1 MC, 1 Descriptive) for rapid validation
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def quick_test():
    """Run quick diagnostic test with 2 questions"""
    print("=" * 60)
    print("RAG Diagnostic Quick Test")
    print("=" * 60)
    
    # Test questions
    test_questions = [
        {
            'ID': 'TEST_000',
            'Question': """금융산업의 이해와 관련하여 금융투자업의 구분에 해당하지 않는 것은?
1 소비자금융업
2 투자자문업
3 투자매매업
4 투자중개업
5 보험중개업""",
            'Type': 'multiple_choice'
        },
        {
            'ID': 'TEST_004',
            'Question': "트로이 목마(Trojan) 기반 원격제어 악성코드(RAT)의 특징과 주요 탐지 지표를 설명하세요.",
            'Type': 'descriptive'
        }
    ]
    
    print(f"\nTest Questions:")
    for q in test_questions:
        print(f"- {q['ID']}: {q['Type']}")
    
    # Initialize RAG
    print("\n1. Initializing RAG system...")
    try:
        from scripts.load_rag_v2 import RAGSystemV2
        rag = RAGSystemV2()
        rag.load_all()
        retriever = rag.create_hybrid_retriever()
        print("[OK] RAG initialized successfully")
        print(f"   BM25 weight: {retriever.bm25_weight:.1%}")
        print(f"   Vector weight: {retriever.vector_weight:.1%}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize RAG: {e}")
        return
    
    # Test retrieval for each question
    print("\n2. Testing retrieval...")
    for q in test_questions:
        print(f"\n   {q['ID']} ({q['Type']}):")
        try:
            results = retriever.search(q['Question'], k=3)
            print(f"   [OK] Retrieved {len(results)} documents")
            
            if results:
                top_result = results[0]
                print(f"      Top result scores:")
                print(f"      - BM25: {getattr(top_result, 'bm25_score', 0):.4f}")
                print(f"      - Vector: {getattr(top_result, 'vector_score', 0):.4f}")
                print(f"      - Hybrid: {getattr(top_result, 'hybrid_score', 0):.4f}")
                
                # Check score dominance
                bm25 = getattr(top_result, 'bm25_score', 0)
                vector = getattr(top_result, 'vector_score', 0)
                if bm25 > vector * 1.5:
                    print(f"      -> BM25 dominant")
                elif vector > bm25 * 1.5:
                    print(f"      -> Vector dominant")
                else:
                    print(f"      -> Balanced")
        except Exception as e:
            print(f"   [ERROR] Retrieval failed: {e}")
    
    # Test prompts
    print("\n3. Testing prompt generation...")
    try:
        from diagnostic_prompts import create_diagnostic_prompt_mc, create_simple_prompt
        
        # Test MC prompt
        mc_prompt = create_diagnostic_prompt_mc(
            test_questions[0]['Question'],
            ["Sample context 1", "Sample context 2"],
            [{'bm25_score': 0.8, 'vector_score': 0.6, 'hybrid_score': 0.7}] * 2
        )
        print(f"   [OK] MC diagnostic prompt: {len(mc_prompt)} chars")
        
        # Test simple prompt
        simple_prompt = create_simple_prompt(
            test_questions[0]['Question'],
            ["Sample context"]
        )
        print(f"   [OK] Simple prompt: {len(simple_prompt)} chars")
        
    except Exception as e:
        print(f"   [ERROR] Prompt generation failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Quick Test Complete!")
    print("If all checks passed, the diagnostic system is ready.")
    print("\nNext steps:")
    print("1. Run full diagnostic: python run_diagnostic_test.py")
    print("2. Analyze results: python analyze_results.py")
    print("=" * 60)


if __name__ == "__main__":
    quick_test()