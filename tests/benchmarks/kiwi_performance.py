#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kiwi 확장 테스트 - 여러 페이지의 Vision 추출 텍스트 분석
사용자 사전 구축을 위한 미인식/오인식 단어 파악
"""

import sys
import io
import json
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from kiwipiepy import Kiwi

def load_vision_texts(base_path: str, pages: List[int]) -> Dict[int, str]:
    """Vision 추출 텍스트 파일들을 로드"""
    texts = {}
    for page_num in pages:
        page_dir = f"page_{page_num:03d}"
        file_path = Path(base_path) / page_dir / "vl_model_v2.txt"
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                texts[page_num] = f.read()
                print(f"페이지 {page_num} 로드 완료")
        else:
            print(f"페이지 {page_num} 파일 없음: {file_path}")
    
    return texts

def extract_technical_terms(text: str) -> Set[str]:
    """텍스트에서 기술 용어 후보 추출"""
    technical_patterns = [
        r'\b[A-Z]{2,}\b',  # 대문자 약어 (AI, ML, API 등)
        r'\b[A-Za-z]+[0-9]+\b',  # 영문+숫자 조합
        r'\([A-Za-z]+\)',  # 괄호 안 영문 용어
        r'[가-힣]+\([A-Za-z]+\)',  # 한글(영문) 패턴
        r'[가-힣]+·[가-힣]+',  # 가운뎃점으로 연결된 한글
    ]
    
    terms = set()
    for pattern in technical_patterns:
        matches = re.findall(pattern, text)
        terms.update(matches)
    
    return terms

def analyze_with_kiwi(kiwi: Kiwi, text: str) -> Dict:
    """Kiwi로 텍스트 분석"""
    result = {
        "tokens": [],
        "morphs": [],
        "pos": [],
        "nouns": [],
        "verbs": [],
        "unknown_tokens": [],
        "english_tokens": [],
        "compound_nouns": []
    }
    
    # 형태소 분석
    tokens = kiwi.tokenize(text)
    
    for token in tokens:
        result["tokens"].append((token.form, token.tag))
        result["morphs"].append(token.form)
        result["pos"].append(token.tag)
        
        # 품사별 분류
        if token.tag.startswith('N'):
            result["nouns"].append(token.form)
        elif token.tag.startswith('V'):
            result["verbs"].append(token.form)
        elif token.tag == 'SL':  # 외국어
            result["english_tokens"].append(token.form)
        elif token.tag in ['UN', 'UNK']:  # 미확인 토큰
            result["unknown_tokens"].append(token.form)
    
    # 연속된 명사 찾기 (복합 명사 후보)
    for i in range(len(tokens) - 1):
        if tokens[i].tag.startswith('N') and tokens[i+1].tag.startswith('N'):
            compound = f"{tokens[i].form} {tokens[i+1].form}"
            result["compound_nouns"].append(compound)
    
    return result

def identify_problematic_words(analysis_results: Dict) -> Dict:
    """문제가 있는 단어들 식별"""
    problems = {
        "misrecognized": [],  # 오인식된 단어
        "over_segmented": [],  # 과도하게 분해된 단어
        "unknown": [],  # 미인식 단어
        "technical_terms": [],  # 기술 용어
        "domain_specific": []  # 도메인 특화 용어
    }
    
    # 검증된 오인식 패턴
    known_issues = {
        "평활": "평화",  # 평활 → 평화로 잘못 인식
        "크롤링": ["크로", "ᆯ", "링"],  # 과도 분해
    }
    
    # 금융/보안 도메인 용어
    domain_terms = [
        "금융", "보안", "인공지능", "머신러닝", "딥러닝",
        "블록체인", "암호화", "인증", "보안위협", "취약점",
        "데이터마이닝", "빅데이터", "클라우드", "API", "SDK",
        "적대적공격", "적대적예제", "모델변조", "데이터중독",
        "프라이버시", "개인정보", "민감정보", "익명화", "가명화"
    ]
    
    return problems

def generate_user_dictionary(all_results: Dict, problematic_words: Dict) -> List[Dict]:
    """사용자 사전 항목 생성"""
    user_dict = []
    
    # 1. 오인식 단어 추가
    for word, wrong in problematic_words.get("misrecognized", []):
        user_dict.append({
            "word": word,
            "pos": "NNG",  # 일반 명사
            "priority": 1,
            "reason": f"오인식 수정: {wrong} → {word}"
        })
    
    # 2. 과도 분해 단어 추가
    for word in problematic_words.get("over_segmented", []):
        user_dict.append({
            "word": word,
            "pos": "NNG",
            "priority": 2,
            "reason": "과도 분해 방지"
        })
    
    # 3. 기술 용어 추가
    for term in problematic_words.get("technical_terms", []):
        user_dict.append({
            "word": term,
            "pos": "NNG",
            "priority": 3,
            "reason": "기술 용어"
        })
    
    # 4. 도메인 특화 용어 추가
    for term in problematic_words.get("domain_specific", []):
        user_dict.append({
            "word": term,
            "pos": "NNG",
            "priority": 4,
            "reason": "금융/보안 도메인 용어"
        })
    
    return user_dict

def main():
    print("="*80)
    print("Kiwi 확장 테스트 - Vision 추출 텍스트 전체 분석")
    print("="*80)
    
    # Kiwi 초기화
    kiwi = Kiwi()
    
    # 테스트할 페이지 범위 (1-30 페이지)
    test_pages = list(range(1, 31))
    base_path = "data/vision_extraction_benchmark/full_extraction_20250814_052410"
    
    # Vision 텍스트 로드
    print("\n텍스트 파일 로드 중...")
    vision_texts = load_vision_texts(base_path, test_pages)
    
    # 전체 분석 결과 저장
    all_results = {}
    all_nouns = defaultdict(int)
    all_unknown = set()
    all_english = set()
    all_compounds = defaultdict(int)
    problematic_patterns = defaultdict(list)
    
    print(f"\n총 {len(vision_texts)}개 페이지 분석 시작...")
    print("="*80)
    
    for page_num, text in vision_texts.items():
        print(f"\n[페이지 {page_num}] 분석 중...")
        
        # 기술 용어 추출
        technical_terms = extract_technical_terms(text)
        
        # Kiwi 분석
        analysis = analyze_with_kiwi(kiwi, text)
        
        # 통계 수집
        for noun in analysis["nouns"]:
            all_nouns[noun] += 1
        
        all_unknown.update(analysis["unknown_tokens"])
        all_english.update(analysis["english_tokens"])
        
        for compound in analysis["compound_nouns"]:
            all_compounds[compound] += 1
        
        # 문제 패턴 찾기
        if "평화" in analysis["morphs"] and "smoothing" in text:
            problematic_patterns["평활→평화"].append(page_num)
        
        if "크로" in analysis["morphs"] and "링" in analysis["morphs"]:
            problematic_patterns["크롤링 과도분해"].append(page_num)
        
        # 결과 저장
        all_results[page_num] = {
            "text_length": len(text),
            "token_count": len(analysis["tokens"]),
            "noun_count": len(analysis["nouns"]),
            "verb_count": len(analysis["verbs"]),
            "english_count": len(analysis["english_tokens"]),
            "unknown_count": len(analysis["unknown_tokens"]),
            "compound_noun_count": len(analysis["compound_nouns"]),
            "technical_terms": list(technical_terms),
            "sample_nouns": analysis["nouns"][:20],  # 샘플로 20개만
            "sample_compounds": analysis["compound_nouns"][:10]
        }
        
        print(f"  - 토큰 수: {len(analysis['tokens'])}")
        print(f"  - 명사: {len(analysis['nouns'])}개")
        print(f"  - 영문: {len(analysis['english_tokens'])}개")
        print(f"  - 복합명사 후보: {len(analysis['compound_nouns'])}개")
    
    # 빈도 분석
    print("\n" + "="*80)
    print("빈도 분석 결과")
    print("="*80)
    
    # 상위 빈출 명사
    top_nouns = sorted(all_nouns.items(), key=lambda x: x[1], reverse=True)[:30]
    print("\n[상위 30개 빈출 명사]")
    for noun, count in top_nouns:
        print(f"  {noun}: {count}회")
    
    # 상위 복합 명사
    top_compounds = sorted(all_compounds.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\n[상위 20개 복합 명사 후보]")
    for compound, count in top_compounds:
        print(f"  {compound}: {count}회")
    
    # 문제 패턴
    print("\n[발견된 문제 패턴]")
    for pattern, pages in problematic_patterns.items():
        print(f"  {pattern}: {len(pages)}개 페이지에서 발견 (페이지: {pages[:5]}...)")
    
    # 사용자 사전 후보 생성
    print("\n" + "="*80)
    print("사용자 사전 후보 생성")
    print("="*80)
    
    # 사용자 사전에 추가할 단어들
    user_dict_candidates = []
    
    # 1. 알려진 오인식 단어
    user_dict_candidates.extend([
        {"word": "평활", "pos": "NNG", "priority": 1, "reason": "평화로 오인식"},
        {"word": "크롤링", "pos": "NNG", "priority": 1, "reason": "과도 분해"},
        {"word": "스크래핑", "pos": "NNG", "priority": 1, "reason": "기술 용어"},
    ])
    
    # 2. 자주 나타나는 복합 명사 (5회 이상)
    for compound, count in top_compounds:
        if count >= 5:
            user_dict_candidates.append({
                "word": compound.replace(" ", ""),
                "pos": "NNG",
                "priority": 2,
                "frequency": count,
                "reason": "빈출 복합명사"
            })
    
    # 3. 금융/보안 도메인 특화 용어
    domain_terms = [
        "적대적공격", "적대적예제", "데이터중독", "모델변조",
        "프라이버시", "개인정보보호", "민감정보", "익명화", "가명화",
        "데이터마이닝", "빅데이터", "클라우드컴퓨팅", "블록체인",
        "머신러닝", "딥러닝", "인공지능", "자연어처리",
        "금융보안", "사이버보안", "보안위협", "취약점분석"
    ]
    
    for term in domain_terms:
        user_dict_candidates.append({
            "word": term,
            "pos": "NNG",
            "priority": 3,
            "reason": "도메인 특화 용어"
        })
    
    # 결과 저장
    final_results = {
        "analysis_date": "2025-01-15",
        "pages_analyzed": len(vision_texts),
        "statistics": {
            "total_unique_nouns": len(all_nouns),
            "total_unique_compounds": len(all_compounds),
            "total_unknown_tokens": len(all_unknown),
            "total_english_tokens": len(all_english),
            "problematic_patterns": dict(problematic_patterns)
        },
        "top_nouns": dict(top_nouns),
        "top_compounds": dict(top_compounds),
        "page_results": all_results,
        "user_dictionary_candidates": user_dict_candidates
    }
    
    # JSON 파일로 저장
    output_file = "kiwi_extensive_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n분석 결과가 {output_file}에 저장되었습니다.")
    
    # 사용자 사전 파일 별도 저장
    user_dict_file = "kiwi_user_dictionary.json"
    with open(user_dict_file, 'w', encoding='utf-8') as f:
        json.dump(user_dict_candidates, f, ensure_ascii=False, indent=2)
    
    print(f"사용자 사전 후보가 {user_dict_file}에 저장되었습니다.")
    
    # 간단한 사용자 사전 텍스트 파일도 생성
    with open("kiwi_user_dictionary.txt", 'w', encoding='utf-8') as f:
        f.write("# Kiwi 사용자 사전\n")
        f.write("# 형식: 단어\t품사\t우선순위\n\n")
        for entry in user_dict_candidates:
            f.write(f"{entry['word']}\t{entry['pos']}\t{entry['priority']}\t# {entry['reason']}\n")
    
    print("사용자 사전 텍스트 파일이 kiwi_user_dictionary.txt에 저장되었습니다.")
    
    print("\n" + "="*80)
    print("테스트 완료!")
    print(f"총 {len(user_dict_candidates)}개의 사용자 사전 항목이 생성되었습니다.")
    print("="*80)

if __name__ == "__main__":
    main()