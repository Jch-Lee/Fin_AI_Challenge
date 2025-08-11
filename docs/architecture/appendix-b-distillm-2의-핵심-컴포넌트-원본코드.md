# Appendix B: Distillm-2의 핵심 컴포넌트 원본코드

## **1. 데이터 재형식화 (Data Reformatting) reformat.py:1-54**

이 스크립트는 teacher와 student 모델의 개별 응답을 DPO 훈련에 적합한 chosen/rejected 쌍으로 변환합니다.

```python
import os
import json
import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tqdm import tqdm
import argparse

def main(args):
    teacher_data = load_dataset('json', data_files=args.teacher_file, split='train')
    student_data = load_dataset('json', data_files=args.student_file, split='train')

    # make sure the pair
    samples = []
    dict_teacher = {x['prompt']: str(x) for x in teacher_data}
    dict_student = {x['prompt']: str(x) for x in student_data}

    for p in teacher_data['prompt']:
        try:
            chosen, rejected = eval(dict_teacher[p]), eval(dict_student[p])
            chosen = [
                {"content": p, "role": "user"},
                {"content": chosen['generated_text'], "role": "assistant"}
            ]
            rejected = [
                {"content": p, "role": "user"},
                {"content": rejected['generated_text'], "role": "assistant"}
            ]
            samples.append({"prompt": p, "chosen": chosen, "rejected": rejected})

        except:
            continue

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(f'{args.output_dir}/train.json', 'w') as json_file:
        json.dump(samples, json_file)

    dataset = DatasetDict({
        'train': load_dataset('json', data_files=f'{args.output_dir}/train.json', split='train'),
        'test': load_dataset('json', data_files=f'{args.output_dir}/train.json', split='train').select(range(500)),
    })
    dataset.save_to_disk(args.output_dir)
    print (f"Binarized datasets save to {os.path.join(args.output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_file", type=str, required=True)
    parser.add_argument("--student_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
```

## **2. vLLM 응답 생성 (Response Generation) generate_vllm.py:1-102**

vLLM을 사용한 고효율 응답 생성 스크립트로, teacher와 student 모델 모두에서 사용됩니다.

```python
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

import argparse
import json
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Decode with vLLM')
parser.add_argument('--data_dir', type=str, default="ultrachat",
                    help='Directory containing the data')
parser.add_argument('--iter', type=int, default='1', help='training iteration')
parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', 
                    help='Path to the SLM model')
parser.add_argument('--teacher-model', type=str, default=None, 
                    help='Path to the LLM model.')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--eps', type=float, default=0.04,
                    help='epsilon for typical acceptance sampler')
parser.add_argument('--max_tokens', type=int, default=1024,
                    help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--output_dir', type=str, default="datasets/phi3_ultrafeedback",
                    help='output_dir')
parser.add_argument('--split', type=str, default='train_prefs')
parser.add_argument('--frac_idx', type=int, default=0)
parser.add_argument('--frac_size', type=int, default=0)
parser.add_argument('--lora_path', type=str, default=None)

args = parser.parse_args()

data_dir = args.data_dir
