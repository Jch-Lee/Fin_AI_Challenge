#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kiwi vs KURE 토크나이저 비교 실험
- Kiwi: 형태소 분석 기반 토크나이징
- KURE: SentenceTransformer 내장 토크나이저
"""

import sys
import io
import time
from typing import List, Dict, Any
import re

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Kiwi 임포트
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False
    print("Kiwi not available")

# KURE 임포트
try:
    from sentence_transformers import SentenceTransformer
    import torch
    KURE_AVAILABLE = True
except ImportError:
    KURE_AVAILABLE = False
    print("SentenceTransformer not available")

def test_kiwi_tokenizer(texts: List[str]) -> Dict[str, Any]:
    """Kiwi 토크나이저 테스트"""
    if not KIWI_AVAILABLE:
        return {"error": "Kiwi not available"}
    
    kiwi = Kiwi()
    results = {
        "name": "Kiwi",
        "method": "형태소 분석",
        "tokens": [],
        "stats": [],
        "processing_time": 0
    }
    
    start_time = time.time()
    
    for text in texts:
        # 띄어쓰기 교정
        corrected_text = kiwi.space(text)
        
        # 형태소 분석
        morphemes = kiwi.tokenize(text)
        
        # 의미 있는 품사만 추출 (BM25용)
        meaningful_tokens = []
        for token in morphemes:
            if token.tag.startswith(('N', 'V', 'VA', 'SL')):  # 명사, 동사, 형용사, 외국어
                if len(token.form) >= 2 or token.tag.startswith('SL'):
                    meaningful_tokens.append(token.form)
        
        # 모든 형태소 (분석용)
        all_tokens = [f"{token.form}/{token.tag}" for token in morphemes]
        
        results["tokens"].append({
            "original": text,
            "corrected": corrected_text,
            "meaningful_tokens": meaningful_tokens,
            "all_morphemes": all_tokens,
            "token_count": len(meaningful_tokens),
            "morpheme_count": len(all_tokens)
        })
        
        results["stats"].append({
            "char_count": len(text),
            "meaningful_tokens": len(meaningful_tokens),
            "total_morphemes": len(all_tokens),
            "avg_token_length": sum(len(t) for t in meaningful_tokens) / max(len(meaningful_tokens), 1)
        })
    
    results["processing_time"] = time.time() - start_time
    return results

def test_kure_tokenizer(texts: List[str]) -> Dict[str, Any]:
    """KURE 토크나이저 테스트"""
    if not KURE_AVAILABLE:
        return {"error": "KURE not available"}
    
    # KURE 모델 로드 (토크나이저만 사용)
    try:
        model = SentenceTransformer("nlpai-lab/KURE-v1")
    except:
        try:
            model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        except:
            model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    tokenizer = model.tokenizer
    
    results = {
        "name": "KURE",
        "method": "SentenceTransformer 내장 토크나이저",
        "model_name": model.model_name if hasattr(model, 'model_name') else "Unknown",
        "tokens": [],
        "stats": [],
        "processing_time": 0
    }
    
    start_time = time.time()
    
    for text in texts:
        # 토크나이징
        encoded = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(encoded)
        
        # 서브워드 토큰을 단어로 복원 시도
        decoded_tokens = []
        current_word = ""
        
        for token in tokens:
            if token.startswith("##") or token.startswith("▁"):
                # 서브워드 연결
                if token.startswith("##"):
                    current_word += token[2:]
                else:
                    if current_word:
                        decoded_tokens.append(current_word)
                    current_word = token[1:]
            else:
                if current_word:
                    decoded_tokens.append(current_word)
                current_word = token
        
        if current_word:
            decoded_tokens.append(current_word)
        
        # 의미 있는 토큰만 필터링 (한글, 영문, 숫자)
        meaningful_tokens = []
        for token in decoded_tokens:
            # 특수문자 제거 후 길이 확인
            clean_token = re.sub(r'[^\w가-힣]', '', token)
            if len(clean_token) >= 1:  # KURE는 서브워드라 더 짧을 수 있음
                meaningful_tokens.append(token)
        
        results["tokens"].append({
            "original": text,
            "raw_tokens": tokens[:20],  # 처음 20개만 표시
            "decoded_tokens": decoded_tokens,
            "meaningful_tokens": meaningful_tokens,
            "token_count": len(meaningful_tokens),
            "raw_token_count": len(tokens)
        })
        
        results["stats"].append({
            "char_count": len(text),
            "meaningful_tokens": len(meaningful_tokens),
            "raw_tokens": len(tokens),
            "avg_token_length": sum(len(t) for t in meaningful_tokens) / max(len(meaningful_tokens), 1)
        })
    
    results["processing_time"] = time.time() - start_time
    return results

def compare_tokenizers():
    """토크나이저 비교 실험"""
    
    # 테스트 텍스트 (다양한 패턴)
    test_texts = [
        "AI 기술이 금융 서비스에 혁신을 가져오고 있다",
        "인터넷뱅킹에서 2단계 인증이 필수가 되었습니다",
        "가상자산 거래소의 보안 강화가 시급합니다",
        "머신러닝과 딥러닝을 활용한 사기 탐지 시스템",
        "CBDC(Central Bank Digital Currency) 도입 논의",
        "API 보안과 OAuth 2.0 인증 프로토콜",
        "핀테크 스타트업의 규제샌드박스 참여",
        "블록체인 기술을 이용한 스마트 컨트랙트 개발"
    ]
    
    print("="*80)
    print("Kiwi vs KURE 토크나이저 비교 실험")
    print("="*80)
    
    # Kiwi 테스트
    print("\n[1] Kiwi 토크나이저 테스트")
    print("-" * 50)
    kiwi_results = test_kiwi_tokenizer(test_texts)
    
    if "error" not in kiwi_results:
        print(f"처리 시간: {kiwi_results['processing_time']:.4f}초")
        print(f"방법: {kiwi_results['method']}")
        
        for i, result in enumerate(kiwi_results['tokens'][:3]):  # 처음 3개만 출력
            print(f"\n예시 {i+1}: {result['original']}")
            print(f"  교정: {result['corrected']}")
            print(f"  의미 토큰: {result['meaningful_tokens']}")
            print(f"  토큰 수: {result['token_count']}")
    else:
        print(f"오류: {kiwi_results['error']}")
    
    # KURE 테스트
    print("\n\n[2] KURE 토크나이저 테스트")
    print("-" * 50)
    kure_results = test_kure_tokenizer(test_texts)
    
    if "error" not in kure_results:
        print(f"처리 시간: {kure_results['processing_time']:.4f}초")
        print(f"방법: {kure_results['method']}")
        print(f"모델: {kure_results.get('model_name', 'Unknown')}")
        
        for i, result in enumerate(kure_results['tokens'][:3]):  # 처음 3개만 출력
            print(f"\n예시 {i+1}: {result['original']}")
            print(f"  디코딩 토큰: {result['decoded_tokens']}")
            print(f"  의미 토큰: {result['meaningful_tokens']}")
            print(f"  토큰 수: {result['token_count']}")
    else:
        print(f"오류: {kure_results['error']}")
    
    # 비교 분석
    if "error" not in kiwi_results and "error" not in kure_results:
        print("\n\n[3] 비교 분석")
        print("="*60)
        
        # 속도 비교
        kiwi_time = kiwi_results['processing_time']
        kure_time = kure_results['processing_time'] 
        speed_ratio = kure_time / kiwi_time if kiwi_time > 0 else float('inf')
        
        print(f"처리 속도:")
        print(f"  Kiwi: {kiwi_time:.4f}초")
        print(f"  KURE: {kure_time:.4f}초")
        print(f"  비율: KURE가 Kiwi보다 {speed_ratio:.2f}배 {'빠름' if speed_ratio < 1 else '느림'}")
        
        # 토큰 수 비교
        kiwi_avg_tokens = sum(s['meaningful_tokens'] for s in kiwi_results['stats']) / len(kiwi_results['stats'])
        kure_avg_tokens = sum(s['meaningful_tokens'] for s in kure_results['stats']) / len(kure_results['stats'])
        
        print(f"\n평균 토큰 수:")
        print(f"  Kiwi: {kiwi_avg_tokens:.1f}개")
        print(f"  KURE: {kure_avg_tokens:.1f}개")
        print(f"  비율: {kure_avg_tokens/kiwi_avg_tokens:.2f}배")
        
        # 토큰 길이 비교
        kiwi_avg_length = sum(s['avg_token_length'] for s in kiwi_results['stats']) / len(kiwi_results['stats'])
        kure_avg_length = sum(s['avg_token_length'] for s in kure_results['stats']) / len(kure_results['stats'])
        
        print(f"\n평균 토큰 길이:")
        print(f"  Kiwi: {kiwi_avg_length:.1f}자")
        print(f"  KURE: {kure_avg_length:.1f}자")
        
        # 상세 비교 테이블
        print(f"\n상세 비교 (처음 5개 텍스트):")
        print(f"{'텍스트':<10} {'Kiwi토큰':<8} {'KURE토큰':<8} {'Kiwi길이':<8} {'KURE길이':<8}")
        print("-" * 50)
        
        for i in range(min(5, len(test_texts))):
            kiwi_stat = kiwi_results['stats'][i]
            kure_stat = kure_results['stats'][i]
            
            print(f"텍스트{i+1:<3} {kiwi_stat['meaningful_tokens']:<8} {kure_stat['meaningful_tokens']:<8} "
                  f"{kiwi_stat['avg_token_length']:<8.1f} {kure_stat['avg_token_length']:<8.1f}")
        
        # 장단점 분석
        print(f"\n장단점 분석:")
        print(f"[Kiwi]")
        print(f"  ✅ 형태소 분석으로 의미 단위 정확")
        print(f"  ✅ 한국어 특화 품사 태깅")
        print(f"  ✅ 띄어쓰기 교정 기능")
        print(f"  ❌ 상대적으로 느린 처리 속도")
        
        print(f"\n[KURE]")
        print(f"  ✅ 빠른 처리 속도")
        print(f"  ✅ 사전 훈련된 임베딩과 일관성")
        print(f"  ✅ 서브워드 단위로 OOV 처리")
        print(f"  ❌ 형태소 경계와 불일치 가능")
    
    # 사용 권장사항
    print(f"\n\n[4] 사용 권장사항")
    print("="*60)
    print(f"📊 BM25 키워드 검색: Kiwi 토크나이저")
    print(f"   - 형태소 분석으로 정확한 의미 단위 추출")
    print(f"   - 한국어 문법 구조 고려")
    print(f"   - 검색 정확도 우선")
    
    print(f"\n🎯 벡터 임베딩 생성: KURE 토크나이저")
    print(f"   - 사전 훈련된 모델과 동일한 토크나이징")
    print(f"   - 빠른 처리 속도")
    print(f"   - 임베딩 품질 우선")
    
    print(f"\n🔄 현재 하이브리드 구조:")
    print(f"   - BM25: Kiwi로 토큰화 → 키워드 매칭")
    print(f"   - Vector: KURE로 임베딩 → 유사도 계산")
    print(f"   - 최적의 조합으로 판단됨 ✅")

def detailed_token_analysis():
    """토큰 분석 상세 버전"""
    
    analysis_text = "AI 기술을 활용한 핀테크 서비스가 기존 금융업계에 혁신을 가져오고 있습니다."
    
    print(f"\n\n[상세 분석] 텍스트: {analysis_text}")
    print("="*80)
    
    if KIWI_AVAILABLE:
        kiwi = Kiwi()
        
        # Kiwi 상세 분석
        print(f"\n[Kiwi 상세 분석]")
        corrected = kiwi.space(analysis_text)
        print(f"띄어쓰기 교정: {corrected}")
        
        morphemes = kiwi.tokenize(analysis_text)
        print(f"\n형태소 분석 결과:")
        for token in morphemes:
            print(f"  {token.form:<10} {token.tag:<8}")
        
        meaningful = [token.form for token in morphemes 
                     if token.tag.startswith(('N', 'V', 'VA', 'SL')) and len(token.form) >= 2]
        print(f"\nBM25용 토큰: {meaningful}")
    
    if KURE_AVAILABLE:
        try:
            model = SentenceTransformer("nlpai-lab/KURE-v1")
            tokenizer = model.tokenizer
            
            print(f"\n[KURE 상세 분석]")
            encoded = tokenizer.encode(analysis_text, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(encoded)
            
            print(f"서브워드 토큰:")
            for i, token in enumerate(tokens):
                print(f"  {i:2d}: {token}")
            
            # 텍스트 복원
            decoded = tokenizer.decode(encoded)
            print(f"\n복원된 텍스트: {decoded}")
            
        except Exception as e:
            print(f"KURE 분석 실패: {e}")

def main():
    """메인 실행 함수"""
    compare_tokenizers()
    detailed_token_analysis()
    
    print(f"\n" + "="*80)
    print(f"비교 실험 완료!")
    print(f"="*80)

if __name__ == "__main__":
    main()