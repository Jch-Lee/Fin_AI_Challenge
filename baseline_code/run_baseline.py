import re
import os
import pandas as pd
from tqdm import tqdm

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def is_multiple_choice(question_text):
    """
    객관식 여부를 판단: 2개 이상의 숫자 선택지가 줄 단위로 존재할 경우 객관식으로 간주
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2

def extract_question_and_choices(full_text):
    """
    전체 질문 문자열에서 질문 본문과 선택지 리스트를 분리
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []

    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    
    question = " ".join(q_lines)
    return question, options

def make_prompt_auto(text):
    """프롬프트 생성기"""
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"
                )
    else:
        prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
                f"질문: {text}\n\n"
                "답변:"
                )   
    return prompt

def extract_answer_only(generated_text: str, original_question: str) -> str:
    """
    - "답변:" 이후 텍스트만 추출
    - 객관식 문제면: 정답 숫자만 추출 (실패 시 전체 텍스트 또는 기본값 반환)
    - 주관식 문제면: 전체 텍스트 그대로 반환
    - 공백 또는 빈 응답 방지: 최소 "미응답" 반환
    """
    # "답변:" 기준으로 텍스트 분리
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()
    
    # 공백 또는 빈 문자열일 경우 기본값 지정
    if not text:
        return "미응답"

    # 객관식 여부 판단
    is_mc = is_multiple_choice(original_question)

    if is_mc:
        # 숫자만 추출
        match = re.match(r"\D*([1-9][0-9]?)", text)
        if match:
            return match.group(1)
        else:
            # 숫자 추출 실패 시 "0" 반환
            return "0"
    else:
        return text

def main():
    print("=== 베이스라인 금융보안 AI 모델 실행 ===")
    
    # GPU 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")
    if device == "cuda":
        print(f"GPU 개수: {torch.cuda.device_count()}")
        print(f"현재 GPU: {torch.cuda.get_device_name()}")
    
    # 데이터 로드
    print("테스트 데이터 로딩 중...")
    test = pd.read_csv('../data/competition/test.csv', encoding='utf-8')
    print(f"테스트 데이터 크기: {len(test)} 샘플")
    
    # 모델 로드 (GPU 최적화)
    model_name = "beomi/gemma-ko-7b"
    print(f"모델 로딩 중: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 토크나이저 안전 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # BitsAndBytesConfig로 4bit 양자화 설정
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Inference pipeline (accelerate와 호환)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16
    )
    
    print("모델 로딩 완료. 추론 시작...")
    
    # 추론 실행
    preds = []
    
    for q in tqdm(test['Question'], desc="추론 진행중"):
        try:
            prompt = make_prompt_auto(q)
            output = pipe(
                prompt, 
                max_new_tokens=64,  # 토큰 수 감소로 안정성 향상
                temperature=0.3,
                top_p=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            pred_answer = extract_answer_only(output[0]["generated_text"], original_question=q)
            preds.append(pred_answer)
        except Exception as e:
            print(f"에러 발생 (샘플 {len(preds)+1}): {e}")
            # 에러 발생 시 기본값 사용
            if is_multiple_choice(q):
                preds.append("1")  # 객관식 기본값
            else:
                preds.append("답변 생성 실패")  # 주관식 기본값
    
    # 제출 파일 생성
    print("제출 파일 생성 중...")
    sample_submission = pd.read_csv('../data/competition/sample_submission.csv', encoding='utf-8')
    
    # 예측 결과가 sample_submission보다 많을 경우, 처음 부분만 사용
    if len(preds) > len(sample_submission):
        preds = preds[:len(sample_submission)]
    
    sample_submission['Answer'] = preds
    sample_submission.to_csv('./baseline_submission.csv', index=False, encoding='utf-8-sig')
    
    print(f"베이스라인 제출 파일 생성 완료: baseline_submission.csv")
    print(f"총 {len(sample_submission)} 개 샘플 예측 완료")

if __name__ == "__main__":
    main() 