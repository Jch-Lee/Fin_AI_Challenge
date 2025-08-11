#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test 데이터 확인 스크립트
"""
import pandas as pd

# UTF-8 with BOM으로 읽기
try:
    df = pd.read_csv('data/competition/test.csv', encoding='utf-8-sig')
    print(f"Total questions: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\n=== Sample questions ===")
    for i in range(3):
        print(f"ID: {df.iloc[i]['ID']}")
        question = df.iloc[i]['Question']
        print(f"Question: {question[:150]}...")
        print()
        
except Exception as e:
    print(f"Error: {e}")
    # 다른 인코딩 시도
    try:
        df = pd.read_csv('data/competition/test.csv', encoding='cp949')
        print("Loaded with cp949 encoding")
        print(f"Total questions: {len(df)}")
    except Exception as e2:
        print(f"cp949 also failed: {e2}")

# submission 파일도 확인
try:
    sub = pd.read_csv('data/competition/sample_submission.csv', encoding='utf-8-sig')
    print(f"\n=== Submission format ===")
    print(f"Total entries: {len(sub)}")
    print(f"Columns: {sub.columns.tolist()}")
    print(sub.head())
except Exception as e:
    print(f"Submission file error: {e}")