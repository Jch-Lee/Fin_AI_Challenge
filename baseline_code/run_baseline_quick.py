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
    print("=== 베이스라인 금융보안 AI 모델 빠른 테스트 ===")
    
    # 데이터 로드
    print("테스트 데이터 로딩 중...")
    test = pd.read_csv('./test.csv', encoding='utf-8')
    print(f"전체 테스트 데이터 크기: {len(test)} 샘플")
    
    # 첫 10개 샘플만 사용
    test_sample = test.head(10)
    print(f"테스트 샘플 크기: {len(test_sample)} 샘플")
    
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")
    
    # 모델 로드
    model_name = "beomi/gemma-ko-7b"
    print(f"모델 로딩 중: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if device == "cuda":
            # GPU 사용 시 4bit 양자화
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16
            )
        else:
            # CPU 사용 시 일반 로딩
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            
        # Inference pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1
        )
        
        print("모델 로딩 완료. 추론 시작...")
        
        # 추론 실행
        preds = []
        
        for idx, q in enumerate(tqdm(test_sample['Question'], desc="추론 진행중")):
            try:
                prompt = make_prompt_auto(q)
                output = pipe(prompt, max_new_tokens=64, temperature=0.2, top_p=0.9)
                pred_answer = extract_answer_only(output[0]["generated_text"], original_question=q)
                preds.append(pred_answer)
                print(f"샘플 {idx+1}: {pred_answer}")
            except Exception as e:
                print(f"샘플 {idx+1} 추론 중 오류: {e}")
                preds.append("0")
        
        # 제출 파일 생성 (빠른 테스트용)
        print("테스트 결과 파일 생성 중...")
        test_result = pd.DataFrame({
            'ID': test_sample['ID'].iloc[:len(preds)],
            'Answer': preds
        })
        test_result.to_csv('./quick_test_result.csv', index=False, encoding='utf-8-sig')
        
        print(f"빠른 테스트 완료: quick_test_result.csv")
        print(f"총 {len(preds)} 개 샘플 테스트 완료")
        
    except Exception as e:
        print(f"모델 로딩 또는 추론 중 오류: {e}")
        print("CPU로 더 가벼운 모델 또는 다른 방법을 시도해보세요.")

if __name__ == "__main__":
    main() 