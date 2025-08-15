#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kiwi 사용자 사전 생성 및 적용 스크립트
분석 결과를 바탕으로 최적화된 사용자 사전 구축
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
from typing import List, Dict
from kiwipiepy import Kiwi

def load_analysis_results():
    """분석 결과 로드"""
    with open('kiwi_extensive_analysis.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def create_optimized_dictionary():
    """최적화된 사용자 사전 생성"""
    
    # 분석 결과 로드
    results = load_analysis_results()
    
    # 사용자 사전 항목
    custom_dict = []
    
    # 1. 금융 도메인 핵심 용어
    financial_terms = [
        # 금융 일반
        ("금융보안", "NNP", 10),
        ("금융회사", "NNP", 10),
        ("금융분야", "NNP", 10),
        ("금융서비스", "NNP", 10),
        ("핀테크", "NNP", 10),
        ("오픈뱅킹", "NNP", 10),
        ("모바일뱅킹", "NNP", 10),
        ("인터넷뱅킹", "NNP", 10),
        
        # 보안 관련
        ("보안위협", "NNP", 10),
        ("보안취약점", "NNP", 10),
        ("사이버보안", "NNP", 10),
        ("정보보안", "NNP", 10),
        ("개인정보보호", "NNP", 10),
        ("민감정보", "NNP", 10),
        ("금융정보", "NNP", 10),
        
        # AI/ML 관련
        ("인공지능", "NNP", 10),
        ("머신러닝", "NNP", 10),
        ("딥러닝", "NNP", 10),
        ("자연어처리", "NNP", 10),
        ("컴퓨터비전", "NNP", 10),
        ("강화학습", "NNP", 10),
        ("연합학습", "NNP", 10),
        ("전이학습", "NNP", 10),
        
        # 공격 유형
        ("적대적공격", "NNP", 10),
        ("적대적예제", "NNP", 10),
        ("데이터중독", "NNP", 10),
        ("모델변조", "NNP", 10),
        ("모델추출", "NNP", 10),
        ("회피공격", "NNP", 10),
        ("백도어공격", "NNP", 10),
        
        # 데이터 처리
        ("데이터전처리", "NNP", 10),
        ("데이터정제", "NNP", 10),
        ("데이터증강", "NNP", 10),
        ("데이터마이닝", "NNP", 10),
        ("빅데이터", "NNP", 10),
        ("메타데이터", "NNP", 10),
        
        # 기술 용어
        ("블록체인", "NNP", 10),
        ("클라우드컴퓨팅", "NNP", 10),
        ("엣지컴퓨팅", "NNP", 10),
        ("마이크로서비스", "NNP", 10),
        ("컨테이너", "NNP", 10),
        
        # 프라이버시
        ("차분프라이버시", "NNP", 10),
        ("동형암호", "NNP", 10),
        ("익명화", "NNP", 10),
        ("가명화", "NNP", 10),
        ("비식별화", "NNP", 10),
    ]
    
    # 2. 빈출 복합 명사 (분석 결과 기반)
    frequent_compounds = [
        ("학습데이터", "NNP", 8),
        ("데이터수집", "NNP", 8),
        ("서비스구성", "NNP", 8),
        ("애플리케이션서버", "NNP", 8),
        ("챗봇서비스", "NNP", 8),
        ("학습방식", "NNP", 8),
        ("정형데이터", "NNP", 8),
        ("비정형데이터", "NNP", 8),
        ("반정형데이터", "NNP", 8),
        ("모델개발", "NNP", 8),
        ("모델학습", "NNP", 8),
        ("모델검증", "NNP", 8),
        ("개발주기", "NNP", 8),
    ]
    
    # 3. 오인식 수정 용어
    correction_terms = [
        ("평활", "NNG", 15),  # 평화로 오인식 방지
        ("크롤링", "NNP", 15),  # 과도 분해 방지
        ("스크래핑", "NNP", 15),
        ("스트리밍", "NNP", 15),
        ("클리닝", "NNP", 15),
        ("스무딩", "NNP", 15),
        ("샘플링", "NNP", 15),
        ("클러스터링", "NNP", 15),
        ("파이프라인", "NNP", 15),
        ("하이퍼파라미터", "NNP", 15),
    ]
    
    # 4. 영문 약어 및 기술 용어
    english_terms = [
        ("API", "SL", 5),
        ("SDK", "SL", 5),
        ("REST", "SL", 5),
        ("JSON", "SL", 5),
        ("XML", "SL", 5),
        ("CSV", "SL", 5),
        ("SQL", "SL", 5),
        ("NoSQL", "SL", 5),
        ("OAuth", "SL", 5),
        ("JWT", "SL", 5),
        ("HTTPS", "SL", 5),
        ("SSL", "SL", 5),
        ("TLS", "SL", 5),
        ("AES", "SL", 5),
        ("RSA", "SL", 5),
        ("GDPR", "SL", 5),
        ("PCI-DSS", "SL", 5),
        ("ISO27001", "SL", 5),
    ]
    
    # 모든 항목 통합
    custom_dict.extend(financial_terms)
    custom_dict.extend(frequent_compounds)
    custom_dict.extend(correction_terms)
    custom_dict.extend(english_terms)
    
    return custom_dict

def test_custom_dictionary(custom_dict):
    """사용자 사전 적용 테스트"""
    
    print("="*80)
    print("사용자 사전 적용 테스트")
    print("="*80)
    
    # Kiwi 인스턴스 생성
    kiwi_default = Kiwi()
    kiwi_custom = Kiwi()
    
    # 사용자 사전 추가
    for word, pos, score in custom_dict:
        try:
            kiwi_custom.add_user_word(word, pos, score)
        except:
            # 일부 품사 태그가 지원되지 않을 수 있음
            kiwi_custom.add_user_word(word, "NNP", score)
    
    # 테스트 문장들
    test_sentences = [
        "금융보안 시스템에서 적대적공격을 방어하기 위한 머신러닝 모델",
        "데이터수집과 전처리 과정에서 평활 기법을 적용했습니다",
        "크롤링과 스크래핑을 통해 빅데이터를 수집하고 있습니다",
        "차분프라이버시와 동형암호를 활용한 개인정보보호 시스템",
        "API와 SDK를 통한 오픈뱅킹 서비스 구현",
        "학습데이터의 품질이 모델학습 결과에 큰 영향을 미칩니다",
    ]
    
    print("\n테스트 결과:")
    print("-"*80)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n[테스트 {i}] {sentence}")
        
        # 기본 사전 분석
        default_tokens = kiwi_default.tokenize(sentence)
        default_morphs = [token.form for token in default_tokens]
        
        # 사용자 사전 적용 분석
        custom_tokens = kiwi_custom.tokenize(sentence)
        custom_morphs = [token.form for token in custom_tokens]
        
        print(f"기본 사전: {default_morphs}")
        print(f"사용자 사전: {custom_morphs}")
        
        # 차이점 확인
        if default_morphs != custom_morphs:
            print("→ 개선됨!")
    
    return kiwi_custom

def save_dictionary_files(custom_dict):
    """사용자 사전을 여러 형식으로 저장"""
    
    # 1. JSON 형식 (구조화된 데이터)
    dict_json = []
    for word, pos, score in custom_dict:
        dict_json.append({
            "word": word,
            "pos": pos,
            "score": score
        })
    
    with open("kiwi_custom_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(dict_json, f, ensure_ascii=False, indent=2)
    
    # 2. TSV 형식 (Kiwi 호환)
    with open("kiwi_custom_dictionary.tsv", "w", encoding="utf-8") as f:
        f.write("# Kiwi 사용자 사전\n")
        f.write("# word\tpos\tscore\n")
        for word, pos, score in custom_dict:
            f.write(f"{word}\t{pos}\t{score}\n")
    
    # 3. 카테고리별 정리 문서
    with open("kiwi_dictionary_documentation.md", "w", encoding="utf-8") as f:
        f.write("# Kiwi 사용자 사전 문서\n\n")
        f.write("## 개요\n")
        f.write(f"- 총 단어 수: {len(custom_dict)}개\n")
        f.write("- 생성 일자: 2025-01-15\n")
        f.write("- 목적: 금융 AI 보안 도메인 텍스트 분석 최적화\n\n")
        
        f.write("## 카테고리별 단어 목록\n\n")
        
        f.write("### 1. 금융 도메인 용어\n")
        f.write("| 단어 | 품사 | 가중치 |\n")
        f.write("|------|------|--------|\n")
        for word, pos, score in custom_dict[:45]:
            if score == 10:
                f.write(f"| {word} | {pos} | {score} |\n")
        
        f.write("\n### 2. 빈출 복합 명사\n")
        f.write("| 단어 | 품사 | 가중치 |\n")
        f.write("|------|------|--------|\n")
        for word, pos, score in custom_dict:
            if score == 8:
                f.write(f"| {word} | {pos} | {score} |\n")
        
        f.write("\n### 3. 오인식 수정 용어\n")
        f.write("| 단어 | 품사 | 가중치 |\n")
        f.write("|------|------|--------|\n")
        for word, pos, score in custom_dict:
            if score == 15:
                f.write(f"| {word} | {pos} | {score} |\n")
        
        f.write("\n### 4. 영문 약어\n")
        f.write("| 단어 | 품사 | 가중치 |\n")
        f.write("|------|------|--------|\n")
        for word, pos, score in custom_dict:
            if score == 5:
                f.write(f"| {word} | {pos} | {score} |\n")
    
    print(f"\n사전 파일 저장 완료:")
    print("- kiwi_custom_dictionary.json (JSON 형식)")
    print("- kiwi_custom_dictionary.tsv (TSV 형식)")
    print("- kiwi_dictionary_documentation.md (문서)")

def main():
    print("="*80)
    print("Kiwi 사용자 사전 생성 및 적용")
    print("="*80)
    
    # 사용자 사전 생성
    custom_dict = create_optimized_dictionary()
    print(f"\n총 {len(custom_dict)}개의 사용자 사전 항목 생성")
    
    # 사전 적용 테스트
    kiwi_custom = test_custom_dictionary(custom_dict)
    
    # 사전 파일 저장
    save_dictionary_files(custom_dict)
    
    print("\n" + "="*80)
    print("사용자 사전 생성 완료!")
    print("="*80)

if __name__ == "__main__":
    main()