#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KoNLPy 형태소 분석기 비교 테스트
Vision 추출 텍스트를 사용한 여러 한국어 형태소 분석기 비교
"""

import sys
import io
import json
import time
from typing import List, Dict, Any

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 테스트할 텍스트 샘플들 (kiwi 테스트와 동일)
TEST_SAMPLES = [
    # Page 1 텍스트
    "금융분야 AI 보안 가이드라인",
    
    # Page 10 텍스트
    "AI 서비스란 AI 기능을 활용하여 작동하는 프로그램 또는 시스템을 말한다.",
    "'AI 알고리즘'이란 데이터의 규칙·패턴을 해석하거나 지식을 추론할 수 있게 만든 방법 또는 절차를 말한다.",
    
    # Page 20 텍스트 (복잡한 문장)
    "데이터를 통한 유의미한 결과를 도출하기 위해서는 데이터의 종류·형태에 따라 적절한 처리 과정이 필요하다.",
    
    # 기술 용어가 포함된 문장
    "(클리닝) 데이터로부터 불필요한 요소를 제거한다.",
    "결측치 대체, 노이즈 데이터의 평활(smoothing), 이상치 관리, 불일치 값 처리 등"
]

def test_mecab():
    """Mecab 형태소 분석기 테스트"""
    try:
        from konlpy.tag import Mecab
        mecab = Mecab()
        
        results = []
        print("\n" + "="*80)
        print("Mecab 형태소 분석기 테스트")
        print("="*80)
        
        for i, text in enumerate(TEST_SAMPLES, 1):
            print(f"\n[샘플 {i}] {text[:50]}..." if len(text) > 50 else f"\n[샘플 {i}] {text}")
            
            start_time = time.time()
            
            # 형태소 분석
            morphs = mecab.morphs(text)
            pos = mecab.pos(text)
            nouns = mecab.nouns(text)
            
            elapsed = time.time() - start_time
            
            print(f"형태소: {morphs[:10]}..." if len(morphs) > 10 else f"형태소: {morphs}")
            print(f"품사 태깅: {pos[:5]}..." if len(pos) > 5 else f"품사 태깅: {pos}")
            print(f"명사: {nouns}")
            print(f"처리 시간: {elapsed:.4f}초")
            
            results.append({
                "text": text,
                "morphs": morphs,
                "pos": pos,
                "nouns": nouns,
                "time": elapsed
            })
        
        return "mecab", results
    except Exception as e:
        print(f"Mecab 테스트 실패: {e}")
        return "mecab", None

def test_okt():
    """Okt (구 Twitter) 형태소 분석기 테스트"""
    try:
        from konlpy.tag import Okt
        okt = Okt()
        
        results = []
        print("\n" + "="*80)
        print("Okt (Open Korean Text) 형태소 분석기 테스트")
        print("="*80)
        
        for i, text in enumerate(TEST_SAMPLES, 1):
            print(f"\n[샘플 {i}] {text[:50]}..." if len(text) > 50 else f"\n[샘플 {i}] {text}")
            
            start_time = time.time()
            
            # 형태소 분석
            morphs = okt.morphs(text)
            pos = okt.pos(text)
            nouns = okt.nouns(text)
            phrases = okt.phrases(text)  # Okt 특유 기능
            
            elapsed = time.time() - start_time
            
            print(f"형태소: {morphs[:10]}..." if len(morphs) > 10 else f"형태소: {morphs}")
            print(f"품사 태깅: {pos[:5]}..." if len(pos) > 5 else f"품사 태깅: {pos}")
            print(f"명사: {nouns}")
            print(f"구문: {phrases}")
            print(f"처리 시간: {elapsed:.4f}초")
            
            results.append({
                "text": text,
                "morphs": morphs,
                "pos": pos,
                "nouns": nouns,
                "phrases": phrases,
                "time": elapsed
            })
        
        return "okt", results
    except Exception as e:
        print(f"Okt 테스트 실패: {e}")
        return "okt", None

def test_komoran():
    """Komoran 형태소 분석기 테스트"""
    try:
        from konlpy.tag import Komoran
        komoran = Komoran()
        
        results = []
        print("\n" + "="*80)
        print("Komoran 형태소 분석기 테스트")
        print("="*80)
        
        for i, text in enumerate(TEST_SAMPLES, 1):
            print(f"\n[샘플 {i}] {text[:50]}..." if len(text) > 50 else f"\n[샘플 {i}] {text}")
            
            start_time = time.time()
            
            # 형태소 분석
            morphs = komoran.morphs(text)
            pos = komoran.pos(text)
            nouns = komoran.nouns(text)
            
            elapsed = time.time() - start_time
            
            print(f"형태소: {morphs[:10]}..." if len(morphs) > 10 else f"형태소: {morphs}")
            print(f"품사 태깅: {pos[:5]}..." if len(pos) > 5 else f"품사 태깅: {pos}")
            print(f"명사: {nouns}")
            print(f"처리 시간: {elapsed:.4f}초")
            
            results.append({
                "text": text,
                "morphs": morphs,
                "pos": pos,
                "nouns": nouns,
                "time": elapsed
            })
        
        return "komoran", results
    except Exception as e:
        print(f"Komoran 테스트 실패: {e}")
        return "komoran", None

def test_hannanum():
    """Hannanum 형태소 분석기 테스트"""
    try:
        from konlpy.tag import Hannanum
        hannanum = Hannanum()
        
        results = []
        print("\n" + "="*80)
        print("Hannanum 형태소 분석기 테스트")
        print("="*80)
        
        for i, text in enumerate(TEST_SAMPLES, 1):
            print(f"\n[샘플 {i}] {text[:50]}..." if len(text) > 50 else f"\n[샘플 {i}] {text}")
            
            start_time = time.time()
            
            # 형태소 분석
            morphs = hannanum.morphs(text)
            pos = hannanum.pos(text)
            nouns = hannanum.nouns(text)
            
            elapsed = time.time() - start_time
            
            print(f"형태소: {morphs[:10]}..." if len(morphs) > 10 else f"형태소: {morphs}")
            print(f"품사 태깅: {pos[:5]}..." if len(pos) > 5 else f"품사 태깅: {pos}")
            print(f"명사: {nouns}")
            print(f"처리 시간: {elapsed:.4f}초")
            
            results.append({
                "text": text,
                "morphs": morphs,
                "pos": pos,
                "nouns": nouns,
                "time": elapsed
            })
        
        return "hannanum", results
    except Exception as e:
        print(f"Hannanum 테스트 실패: {e}")
        return "hannanum", None

def test_kkma():
    """Kkma 형태소 분석기 테스트"""
    try:
        from konlpy.tag import Kkma
        kkma = Kkma()
        
        results = []
        print("\n" + "="*80)
        print("Kkma 형태소 분석기 테스트")
        print("="*80)
        
        # Kkma는 느리므로 샘플 수를 줄임
        for i, text in enumerate(TEST_SAMPLES[:3], 1):  # 처음 3개만
            print(f"\n[샘플 {i}] {text[:50]}..." if len(text) > 50 else f"\n[샘플 {i}] {text}")
            
            start_time = time.time()
            
            # 형태소 분석
            morphs = kkma.morphs(text)
            pos = kkma.pos(text)
            nouns = kkma.nouns(text)
            sentences = kkma.sentences(text)  # Kkma 특유 기능
            
            elapsed = time.time() - start_time
            
            print(f"형태소: {morphs[:10]}..." if len(morphs) > 10 else f"형태소: {morphs}")
            print(f"품사 태깅: {pos[:5]}..." if len(pos) > 5 else f"품사 태깅: {pos}")
            print(f"명사: {nouns}")
            print(f"문장 분리: {sentences}")
            print(f"처리 시간: {elapsed:.4f}초")
            
            results.append({
                "text": text,
                "morphs": morphs,
                "pos": pos,
                "nouns": nouns,
                "sentences": sentences,
                "time": elapsed
            })
        
        return "kkma", results
    except Exception as e:
        print(f"Kkma 테스트 실패: {e}")
        return "kkma", None

def compare_kiwi():
    """Kiwi 형태소 분석기 재테스트 (비교용)"""
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        
        results = []
        print("\n" + "="*80)
        print("Kiwi 형태소 분석기 테스트 (비교용)")
        print("="*80)
        
        for i, text in enumerate(TEST_SAMPLES, 1):
            print(f"\n[샘플 {i}] {text[:50]}..." if len(text) > 50 else f"\n[샘플 {i}] {text}")
            
            start_time = time.time()
            
            # 형태소 분석
            tokens = kiwi.tokenize(text)
            morphs = [token.form for token in tokens]
            pos = [(token.form, token.tag) for token in tokens]
            nouns = [token.form for token in tokens if token.tag.startswith('N')]
            
            elapsed = time.time() - start_time
            
            print(f"형태소: {morphs[:10]}..." if len(morphs) > 10 else f"형태소: {morphs}")
            print(f"품사 태깅: {pos[:5]}..." if len(pos) > 5 else f"품사 태깅: {pos}")
            print(f"명사: {nouns}")
            print(f"처리 시간: {elapsed:.4f}초")
            
            results.append({
                "text": text,
                "morphs": morphs,
                "pos": pos,
                "nouns": nouns,
                "time": elapsed
            })
        
        return "kiwi", results
    except Exception as e:
        print(f"Kiwi 테스트 실패: {e}")
        return "kiwi", None

def save_comparison_results(all_results):
    """비교 결과를 JSON 파일로 저장"""
    comparison = {
        "test_date": "2025-01-15",
        "test_samples": len(TEST_SAMPLES),
        "analyzers": {},
        "performance_comparison": {},
        "noun_extraction_comparison": {}
    }
    
    for analyzer_name, results in all_results.items():
        if results:
            comparison["analyzers"][analyzer_name] = {
                "status": "success",
                "sample_count": len(results),
                "avg_time": sum(r["time"] for r in results) / len(results),
                "results": results
            }
            
            # 성능 비교
            comparison["performance_comparison"][analyzer_name] = {
                "avg_time_ms": (sum(r["time"] for r in results) / len(results)) * 1000,
                "total_time": sum(r["time"] for r in results)
            }
            
            # 명사 추출 비교 (첫 번째 샘플)
            if results:
                comparison["noun_extraction_comparison"][analyzer_name] = results[0]["nouns"]
        else:
            comparison["analyzers"][analyzer_name] = {
                "status": "failed",
                "sample_count": 0
            }
    
    with open("konlpy_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print("\n결과가 konlpy_comparison_results.json에 저장되었습니다.")

def main():
    print("="*80)
    print("KoNLPy 형태소 분석기 비교 테스트")
    print("Vision 추출 텍스트 사용")
    print("="*80)
    
    # 먼저 KoNLPy 설치 확인
    try:
        import konlpy
        print(f"KoNLPy 버전: {konlpy.__version__}")
    except ImportError:
        print("KoNLPy가 설치되지 않았습니다. 설치 중...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "konlpy"])
        import konlpy
        print(f"KoNLPy 설치 완료. 버전: {konlpy.__version__}")
    
    # JVM 경로 설정 시도 (필요한 경우)
    import os
    if not os.environ.get('JAVA_HOME'):
        print("주의: JAVA_HOME이 설정되지 않았습니다. 일부 분석기가 작동하지 않을 수 있습니다.")
    
    # 각 분석기 테스트
    all_results = {}
    
    # Kiwi (비교용)
    analyzer, results = compare_kiwi()
    if results:
        all_results[analyzer] = results
    
    # Okt
    analyzer, results = test_okt()
    if results:
        all_results[analyzer] = results
    
    # Komoran
    analyzer, results = test_komoran()
    if results:
        all_results[analyzer] = results
    
    # Hannanum
    analyzer, results = test_hannanum()
    if results:
        all_results[analyzer] = results
    
    # Kkma (느림)
    analyzer, results = test_kkma()
    if results:
        all_results[analyzer] = results
    
    # Mecab (설치 필요)
    analyzer, results = test_mecab()
    if results:
        all_results[analyzer] = results
    
    # 결과 저장
    save_comparison_results(all_results)
    
    # 요약 출력
    print("\n" + "="*80)
    print("테스트 요약")
    print("="*80)
    
    for analyzer_name in all_results:
        if all_results[analyzer_name]:
            avg_time = sum(r["time"] for r in all_results[analyzer_name]) / len(all_results[analyzer_name])
            print(f"{analyzer_name}: 성공 (평균 처리 시간: {avg_time*1000:.2f}ms)")
        else:
            print(f"{analyzer_name}: 실패")

if __name__ == "__main__":
    main()