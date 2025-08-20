#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract 20 test questions (10 multiple choice, 10 descriptive) from test.csv
"""

import pandas as pd
import re
from pathlib import Path

def is_multiple_choice(text: str) -> bool:
    """Check if question is multiple choice"""
    lines = text.strip().split('\n')
    options = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and len(line) > 2:
            if line[1] in ['.', ')', ' ', ':']:
                options.append(line)
    return len(options) >= 2

def extract_test_questions():
    """Extract 20 test questions from test.csv"""
    
    # Load test data
    test_df = pd.read_csv('data/competition/test.csv')
    
    # Separate multiple choice and descriptive questions
    mc_questions = []
    desc_questions = []
    
    for idx, row in test_df.iterrows():
        question_id = row['ID']
        question_text = row['Question']
        
        if is_multiple_choice(question_text):
            mc_questions.append({
                'ID': question_id,
                'Question': question_text,
                'Type': 'multiple_choice'
            })
        else:
            desc_questions.append({
                'ID': question_id,
                'Question': question_text,
                'Type': 'descriptive'
            })
    
    # Select 10 of each type
    selected_mc = mc_questions[:10]
    selected_desc = desc_questions[:10]
    
    # Combine and sort by ID
    all_questions = selected_mc + selected_desc
    all_questions.sort(key=lambda x: x['ID'])
    
    # Save to CSV
    output_df = pd.DataFrame(all_questions)
    output_path = Path('tests/experiments/rag_diagnostic/test_questions_20.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Extracted {len(selected_mc)} multiple choice questions")
    print(f"Extracted {len(selected_desc)} descriptive questions")
    print(f"Total: {len(all_questions)} questions saved to {output_path}")
    
    # Print sample
    print("\n=== Sample Questions ===")
    print("Multiple Choice IDs:", [q['ID'] for q in selected_mc[:5]])
    print("Descriptive IDs:", [q['ID'] for q in selected_desc[:5]])

if __name__ == "__main__":
    extract_test_questions()