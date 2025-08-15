#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 프롬프트를 적용한 VL 모델 추출 실험
기존 결과 디렉토리에 vl_model_v2.txt 파일만 추가
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import torch
import pymupdf
from PIL import Image
import io
import traceback

def create_advanced_semantic_prompt():
    """Version 1: 상세 규칙 기반 프롬프트"""
    prompt = """
이 문서 페이지를 분석하여 의미 있는 정보만 마크다운으로 추출하세요.

## 추출 우선순위
1급 (반드시 추출):
- 모든 제목과 소제목
- 본문 단락의 완전한 문장
- 정의, 설명, 결론
- 핵심 수치와 그 맥락

2급 (요약하여 추출):
- 그래프/차트의 핵심 메시지
- 표의 주요 데이터
- 이미지의 설명

3급 (제외):
- 축 눈금값, 격자선 숫자
- 페이지 번호, 머리글/바닥글
- 반복되는 레이블

## 세부 처리 규칙

### 📈 그래프/차트 처리
출력 형식:
### [차트] {차트 제목}
- **측정 항목**: {Y축 레이블}
- **기간/범위**: {X축 범위}
- **핵심 발견**: {주요 트렌드나 특이점}
- **주요 수치**: {최대/최소/변화율 등 의미있는 값만}

### 📊 표(Table) 처리
- 헤더는 항상 포함
- 데이터는 다음 중 하나 선택:
  a) 5행 이하: 전체 포함
  b) 6-10행: 상위 3개 + "... 외 N개"
  c) 10행 초과: 요약 통계만

### 📝 텍스트 처리
- 완전한 문장: 그대로 유지
- 나열된 단어/숫자: 문맥 있는 것만 유지

### 🖼️ 이미지/다이어그램 처리
![다이어그램] {설명}
- 구성요소: {주요 요소 나열}
- 관계/흐름: {요소 간 관계 설명}
- 핵심 메시지: {다이어그램이 전달하는 핵심}

## 금지 사항
절대 포함하지 마세요:
- 단독 숫자 나열 (0, 5, 10, 15, 20...)
- 날짜만 나열 (1월, 2월, 3월...)
- "그림 1", "표 2" 같은 참조 번호
- 범례의 색상 설명 (빨간색, 파란색...)
- 격자선, 축 눈금값

## 출력 검증
추출 후 자가 검증:
- 각 추출 항목이 독립적으로 의미가 있는가?
- 컨텍스트 없이도 이해 가능한가?
- RAG 검색 시 유용한 정보인가?
"""
    return prompt

def run_improved_extraction():
    """개선된 프롬프트로 추출 실행"""
    
    # 기존 결과 디렉토리 사용
    existing_output_dir = Path('outputs/full_extraction_20250814_052410')
    pdf_path = Path('data/금융분야 AI 보안 가이드라인.pdf')
    
    if not existing_output_dir.exists():
        print(f"❌ Error: Directory not found: {existing_output_dir}")
        return
    
    print('🚀 Improved VL Extraction Experiment (Version 2 Prompt)')
    print('=' * 60)
    print(f'📄 PDF: {pdf_path}')
    print(f'📁 Adding results to: {existing_output_dir}')
    
    # PDF 정보 확인
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    
    print(f'📖 Total pages: {total_pages}')
    print('=' * 60)
    
    # 모델 로드
    print('\n📦 Loading Qwen2.5-VL model...')
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            'Qwen/Qwen2.5-VL-7B-Instruct',
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        processor = AutoProcessor.from_pretrained(
            'Qwen/Qwen2.5-VL-7B-Instruct',
            trust_remote_code=True
        )
        
        print('✅ Model loaded successfully!')
        
        if torch.cuda.is_available():
            memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f'📊 GPU memory used: {memory:.1f} GB')
    
    except Exception as e:
        print(f'❌ Failed to load model: {e}')
        return
    
    # 개선된 프롬프트 사용
    prompt = create_advanced_semantic_prompt()
    
    # 결과 저장용
    results = []
    total_v2_chars = 0
    successful_pages = 0
    failed_pages = []
    
    start_time = time.time()
    
    # 페이지별 처리
    for page_num in range(total_pages):
        print(f'\n📖 Processing page {page_num + 1}/{total_pages}...')
        
        page_dir = existing_output_dir / f'page_{page_num+1:03d}'
        
        try:
            # 이미지 로드 (기존 이미지 사용)
            image_path = page_dir / 'original.png'
            if not image_path.exists():
                print(f'  ⚠️ Original image not found, converting from PDF...')
                doc = pymupdf.open(pdf_path)
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img_data = pix.tobytes('png')
                doc.close()
                image = Image.open(io.BytesIO(img_data))
                image.save(image_path)
            else:
                image = Image.open(image_path)
            
            # VL 모델 추출 (Version 2 프롬프트)
            print('  - Extracting with improved VL prompt...')
            vl_start = time.time()
            
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError:
                process_vision_info = None
            
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': prompt}
                ]
            }]
            
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            if process_vision_info:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs = [image]
                video_inputs = []
            
            inputs = processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors='pt'
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # 안전한 디코딩
            if isinstance(inputs, dict):
                input_ids = inputs.get('input_ids', None)
            else:
                input_ids = inputs.input_ids
            
            if input_ids is not None and outputs is not None:
                generated_ids = []
                for inp, out in zip(input_ids, outputs):
                    inp_len = len(inp) if hasattr(inp, '__len__') else inp.shape[0]
                    out_len = len(out) if hasattr(out, '__len__') else out.shape[0]
                    if out_len > inp_len:
                        generated_ids.append(out[inp_len:])
                    else:
                        generated_ids.append(out)
                
                if generated_ids:
                    vl_text_v2 = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                else:
                    vl_text_v2 = '[디코딩 실패]'
            else:
                vl_text_v2 = '[생성 실패]'
            
            vl_time = time.time() - vl_start
            
            # Version 2 결과 저장
            (page_dir / 'vl_model_v2.txt').write_text(vl_text_v2, encoding='utf-8')
            vl_v2_chars = len(vl_text_v2)
            total_v2_chars += vl_v2_chars
            
            # 기존 결과와 비교를 위해 읽기
            pymupdf_chars = 0
            vl_v1_chars = 0
            
            pymupdf_file = page_dir / 'pymupdf.txt'
            if pymupdf_file.exists():
                pymupdf_text = pymupdf_file.read_text(encoding='utf-8')
                pymupdf_chars = len(pymupdf_text)
            
            vl_v1_file = page_dir / 'vl_model.txt'
            if vl_v1_file.exists():
                vl_v1_text = vl_v1_file.read_text(encoding='utf-8')
                vl_v1_chars = len(vl_v1_text)
            
            # 결과 기록
            reduction_from_v1 = ((vl_v2_chars - vl_v1_chars) / vl_v1_chars * 100) if vl_v1_chars > 0 else 0
            improvement_from_pymupdf = ((vl_v2_chars - pymupdf_chars) / pymupdf_chars * 100) if pymupdf_chars > 0 else 0
            
            result = {
                'page': page_num + 1,
                'pymupdf_chars': pymupdf_chars,
                'vl_v1_chars': vl_v1_chars,
                'vl_v2_chars': vl_v2_chars,
                'v2_vs_v1_change': reduction_from_v1,
                'v2_vs_pymupdf_change': improvement_from_pymupdf,
                'vl_time': vl_time,
                'success': True
            }
            
            results.append(result)
            successful_pages += 1
            
            print(f'  ✅ PyMuPDF: {pymupdf_chars:,} chars')
            print(f'  ✅ VL V1 (original): {vl_v1_chars:,} chars')
            print(f'  ✅ VL V2 (improved): {vl_v2_chars:,} chars')
            print(f'  📊 V2 vs V1: {reduction_from_v1:+.1f}%')
            print(f'  📊 V2 vs PyMuPDF: {improvement_from_pymupdf:+.1f}%')
            print(f'  ⏱️ Processing time: {vl_time:.2f}s')
            
        except Exception as e:
            print(f'  ❌ Failed: {e}')
            failed_pages.append(page_num + 1)
            results.append({
                'page': page_num + 1,
                'error': str(e),
                'success': False
            })
        
        # GPU 메모리 정리 (5페이지마다)
        if (page_num + 1) % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print('  🧹 GPU memory cleared')
    
    total_time = time.time() - start_time
    
    # 전체 통계 계산
    total_pymupdf = sum(r['pymupdf_chars'] for r in results if r.get('success'))
    total_v1 = sum(r['vl_v1_chars'] for r in results if r.get('success'))
    
    # 최종 요약
    print('\n' + '=' * 60)
    print('📊 FINAL COMPARISON RESULTS')
    print('=' * 60)
    print(f'✅ Successful pages: {successful_pages}/{total_pages}')
    if failed_pages:
        print(f'❌ Failed pages: {failed_pages}')
    print(f'\n📝 Total character counts:')
    print(f'  - PyMuPDF: {total_pymupdf:,} chars')
    print(f'  - VL V1 (original prompt): {total_v1:,} chars')
    print(f'  - VL V2 (improved prompt): {total_v2_chars:,} chars')
    print(f'\n📈 Improvements:')
    if total_v1 > 0:
        print(f'  - V2 vs V1: {((total_v2_chars - total_v1) / total_v1 * 100):+.1f}%')
    if total_pymupdf > 0:
        print(f'  - V2 vs PyMuPDF: {((total_v2_chars - total_pymupdf) / total_pymupdf * 100):+.1f}%')
    print(f'\n⏱️ Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)')
    print(f'⏱️ Average time per page: {total_time/total_pages:.2f}s')
    
    # 비교 요약 저장
    comparison_summary = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_pages': total_pages,
        'successful_pages': successful_pages,
        'failed_pages': failed_pages,
        'total_chars': {
            'pymupdf': total_pymupdf,
            'vl_v1_original': total_v1,
            'vl_v2_improved': total_v2_chars
        },
        'improvements': {
            'v2_vs_v1_percent': ((total_v2_chars - total_v1) / total_v1 * 100) if total_v1 > 0 else 0,
            'v2_vs_pymupdf_percent': ((total_v2_chars - total_pymupdf) / total_pymupdf * 100) if total_pymupdf > 0 else 0,
            'v2_vs_v1_chars': total_v2_chars - total_v1,
            'v2_vs_pymupdf_chars': total_v2_chars - total_pymupdf
        },
        'total_time': total_time,
        'average_time_per_page': total_time / total_pages,
        'page_results': results
    }
    
    summary_file = existing_output_dir / 'comparison_summary_v2.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_summary, f, indent=2, ensure_ascii=False)
    
    print(f'\n📁 Results added to: {existing_output_dir}')
    print(f'📄 Comparison summary: {summary_file}')
    
    return existing_output_dir

if __name__ == '__main__':
    run_improved_extraction()