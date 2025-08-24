#!/usr/bin/env python3
"""
test.csv에서 주관식 문제 10개를 추출하여 remote_test_subjective.csv 생성
원격 서버 실험용
"""

import pandas as pd
import os
from pathlib import Path

def is_multiple_choice(question: str) -> bool:
    """객관식 문제 여부 판단"""
    lines = question.split('\n')
    choices = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and len(line) > 1:
            # 숫자로 시작하고 뒤에 내용이 있는 라인
            if '.' in line[:3] or ')' in line[:3] or ' ' in line[:3]:
                choices.append(line)
    return len(choices) >= 2

def main():
    # 프로젝트 루트 경로
    project_root = Path(__file__).parent.parent
    
    # test.csv 로드
    test_file = project_root / "test.csv"
    if not test_file.exists():
        print(f"ERROR: test.csv 파일이 없습니다: {test_file}")
        return
    
    df = pd.read_csv(test_file)
    print(f"전체 문제 수: {len(df)}개")
    
    # 주관식 문제만 필터링
    subjective_questions = []
    for idx, row in df.iterrows():
        if not is_multiple_choice(row['Question']):
            subjective_questions.append(row)
    
    print(f"주관식 문제 수: {len(subjective_questions)}개")
    
    # 상위 10개만 선택
    selected_questions = subjective_questions[:10]
    
    if len(selected_questions) < 10:
        print(f"WARNING: 주관식 문제가 10개 미만입니다: {len(selected_questions)}개")
    
    # DataFrame으로 변환
    test_df = pd.DataFrame(selected_questions)
    
    # 결과 파일 저장
    output_file = project_root / "remote_test_subjective.csv"
    test_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"SUCCESS: 주관식 10문제 저장 완료: {output_file}")
    print(f"선택된 문제 ID: {', '.join(map(str, test_df['ID'].tolist()))}")
    
    # 샘플 출력
    print("\n--- 첫 3개 문제 샘플 ---")
    for i in range(min(3, len(test_df))):
        question = test_df.iloc[i]['Question']
        question_preview = question[:100] + "..." if len(question) > 100 else question
        print(f"ID {test_df.iloc[i]['ID']}: {question_preview}")

if __name__ == "__main__":
    main()