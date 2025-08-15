#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kiwi 형태소 분석 테스트 스크립트
Vision 추출 텍스트를 사용한 한국어 형태소 분석 검증
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from kiwipiepy import Kiwi
import json
from typing import List, Dict

def test_kiwi_analysis():
    """kiwi 라이브러리의 형태소 분석 기능 테스트"""
    
    # kiwi 초기화
    kiwi = Kiwi()
    
    # 테스트할 텍스트 샘플들
    test_samples = [
        # Page 1 텍스트
        "금융분야 AI 보안 가이드라인",
        
        # Page 10 텍스트
        "AI 서비스란 AI 기능을 활용하여 작동하는 프로그램 또는 시스템을 말한다.",
        "'AI 알고리즘'이란 데이터의 규칙·패턴을 해석하거나 지식을 추론할 수 있게 만든 방법 또는 절차를 말한다.",
        "'적대적 공격'이란 적대적 예제를 활용하여 AI 모델이 잘못 판단하도록 유도하는 공격을 말한다.",
        
        # Page 20 텍스트 (복잡한 문장)
        "데이터를 통한 유의미한 결과를 도출하기 위해서는 데이터의 종류·형태에 따라 적절한 처리 과정이 필요하다.",
        "적절한 처리 과정을 거치지 않은 데이터를 학습에 활용할 경우, AI 모델이 왜곡된 기능을 할 수 있다.",
        
        # 기술 용어가 포함된 문장
        "(클리닝) 데이터로부터 불필요한 요소를 제거한다.",
        "결측치 대체, 노이즈 데이터의 평활(smoothing), 이상치 관리, 불일치 값 처리 등"
    ]
    
    print("=" * 80)
    print("KIWI 형태소 분석 테스트 - Vision 추출 텍스트 사용")
    print("=" * 80)
    
    for i, text in enumerate(test_samples, 1):
        print(f"\n[샘플 {i}] {text[:50]}..." if len(text) > 50 else f"\n[샘플 {i}] {text}")
        print("-" * 40)
        
        # 형태소 분석 수행
        result = kiwi.tokenize(text)
        
        # 토큰 리스트
        tokens = [(token.form, token.tag) for token in result]
        print(f"토큰 수: {len(tokens)}")
        print(f"형태소: {tokens}")
        
        # 명사만 추출
        nouns = [token.form for token in result if token.tag.startswith('N')]
        print(f"추출된 명사: {nouns}")
        
        # 동사만 추출
        verbs = [token.form for token in result if token.tag.startswith('V')]
        print(f"추출된 동사: {verbs}")
        
        # 문장 분석
        sentences = kiwi.split_into_sents(text)
        print(f"문장 수: {len(sentences)}")
        for j, sent in enumerate(sentences, 1):
            print(f"  문장 {j}: {sent.text}")

def test_advanced_features():
    """kiwi의 고급 기능 테스트"""
    
    kiwi = Kiwi()
    
    print("\n" + "=" * 80)
    print("KIWI 고급 기능 테스트")
    print("=" * 80)
    
    # 복잡한 기술 문서 텍스트
    complex_text = """
    학습 데이터 처리 시 다음과 같은 기법을 활용하여 학습에 불필요한 요소를 제거하고 데이터를 단순화한다.
    (클리닝) 데이터로부터 불필요한 요소를 제거한다. 결측치 대체, 노이즈 데이터의 평활(smoothing), 이상치 관리, 불일치 값 처리 등.
    (통합) 중복이거나 유사 값을 하나의 값으로 통합한다.
    (변환) 정규화, 집합화, 요약, 계층 생성 등의 기법을 사용하여 학습에 적합한 값으로 변환한다.
    """
    
    # 1. 상세 형태소 분석
    print("\n[상세 형태소 분석]")
    result = kiwi.tokenize(complex_text, normalize_coda=True)
    
    # 품사별 분류
    pos_dict = {}
    for token in result:
        tag_category = token.tag[0]  # 첫 글자로 대분류
        if tag_category not in pos_dict:
            pos_dict[tag_category] = []
        pos_dict[tag_category].append(token.form)
    
    print("품사별 분류:")
    for pos, words in pos_dict.items():
        print(f"  {pos}: {list(set(words))[:10]}")  # 중복 제거 후 최대 10개만 표시
    
    # 2. 문장 경계 검출
    print("\n[문장 경계 검출]")
    sentences = kiwi.split_into_sents(complex_text)
    for i, sent in enumerate(sentences, 1):
        print(f"  문장 {i} (시작:{sent.start}, 끝:{sent.end}): {sent.text[:50]}...")
    
    # 3. 명사구 추출 시뮬레이션 (연속된 명사들)
    print("\n[명사구 추출]")
    tokens = kiwi.tokenize(complex_text)
    noun_phrases = []
    current_phrase = []
    
    for token in tokens:
        if token.tag.startswith('N'):
            current_phrase.append(token.form)
        else:
            if len(current_phrase) > 1:
                noun_phrases.append(' '.join(current_phrase))
            current_phrase = []
    
    if len(current_phrase) > 1:
        noun_phrases.append(' '.join(current_phrase))
    
    print(f"추출된 명사구: {noun_phrases}")

def test_special_characters():
    """특수문자 및 영문 혼재 텍스트 처리 테스트"""
    
    kiwi = Kiwi()
    
    print("\n" + "=" * 80)
    print("특수문자 및 영문 혼재 처리 테스트")
    print("=" * 80)
    
    special_texts = [
        "관계형 DB(RDB), CSV 등",
        "크롤링(Crawling), 스크래핑(Scraping), 오픈API",
        "HTML, XML, JSON, 웹문서, 웹로그, 센서 데이터 등",
        "AI(Artificial Intelligence) 모델",
        "평활(smoothing)** 관측치를 평균값으로 대체"
    ]
    
    for text in special_texts:
        print(f"\n원문: {text}")
        result = kiwi.tokenize(text)
        tokens = [(token.form, token.tag) for token in result]
        print(f"분석 결과: {tokens}")

if __name__ == "__main__":
    # 메인 테스트 실행
    test_kiwi_analysis()
    test_advanced_features()
    test_special_characters()
    
    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("kiwi 라이브러리가 Vision 추출 텍스트를 정상적으로 분석합니다.")
    print("=" * 80)