#!/bin/bash

# DistiLLM-2 학습 실행 스크립트

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline

# 학습 실행
accelerate launch \
    --config_file accelerate_configs/zero3_offload.yaml \
    --num_processes 4 \
    distillm-2/train.py \
    training_configs/qwen_7b_distillm2_formatted.yaml

echo "학습 완료!"
