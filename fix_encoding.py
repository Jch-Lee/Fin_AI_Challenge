#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
인코딩 문제 해결 및 데이터 확인
"""
import pandas as pd
import chardet

# 파일 인코딩 감지
with open('data/competition/test.csv', 'rb') as f:
    raw_data = f.read()
    encoding = chardet.detect(raw_data)
    print(f"Detected encoding: {encoding}")

# 다양한 인코딩 시도
encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1']

for enc in encodings:
    try:
        df = pd.read_csv('data/competition/test.csv', encoding=enc)
        print(f"\n=== Success with {enc} ===")
        print(f"Total questions: {len(df)}")
        
        # 첫 번째 질문 확인
        question = df.iloc[0]['Question']
        print(f"First question preview: {question[:100]}")
        
        # 한글이 제대로 보이는지 확인
        if '금융' in question or '보안' in question:
            print("✅ Korean text detected correctly!")
            
            # 이 인코딩으로 정상 로드됨
            print(f"\n=== Sample Questions with {enc} ===")
            for i in range(3):
                print(f"ID: {df.iloc[i]['ID']}")
                print(f"Q: {df.iloc[i]['Question'][:150]}...")
                print()
            break
        else:
            print("❌ Korean text still broken")
            
    except Exception as e:
        print(f"Failed with {enc}: {e}")