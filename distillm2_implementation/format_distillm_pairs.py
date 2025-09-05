import json
import pandas as pd
from pathlib import Path

def format_for_distillm2(input_file, output_dir):
    """DistiLLM-2 학습을 위한 데이터 포맷팅"""
    
    # CSV 파일 읽기
    df = pd.read_csv(input_file)
    
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # OpenAI 메시지 형식으로 변환
    formatted_data = []
    for _, row in df.iterrows():
        formatted_pair = {
            'messages': [
                {'role': 'user', 'content': row['prompt']},
                {'role': 'assistant', 'content': ''}
            ],
            'chosen': [
                {'role': 'user', 'content': row['prompt']},
                {'role': 'assistant', 'content': row['chosen']}
            ],
            'rejected': [
                {'role': 'user', 'content': row['prompt']},
                {'role': 'assistant', 'content': row['rejected']}
            ]
        }
        formatted_data.append(formatted_pair)
    
    # Train/Test 분할 (90/10)
    split_idx = int(len(formatted_data) * 0.9)
    train_data = formatted_data[:split_idx]
    test_data = formatted_data[split_idx:]
    
    # JSONL 파일로 저장
    with open(output_dir / 'train.jsonl', 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(output_dir / 'test.jsonl', 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Train: {len(train_data)} samples saved")
    print(f"Test: {len(test_data)} samples saved")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python format_distillm_pairs.py <input_csv> <output_dir>")
        sys.exit(1)
    
    format_for_distillm2(sys.argv[1], sys.argv[2])
