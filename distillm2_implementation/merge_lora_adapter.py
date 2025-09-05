from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import argparse

def merge_lora_with_base(base_model_path, adapter_path, output_path):
    """LoRA 어댑터를 베이스 모델과 병합"""
    
    print(f"베이스 모델 로드: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"LoRA 어댑터 로드: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("모델 병합 중...")
    model = model.merge_and_unload()
    
    print(f"병합된 모델 저장: {output_path}")
    model.save_pretrained(output_path)
    
    # 토크나이저도 복사
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("병합 완료!")
    
    # 메모리 정리
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    args = parser.parse_args()
    
    merge_lora_with_base(args.base_model, args.adapter, args.output)
