#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전체 PDF 텍스트 추출 실험
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

# GPU 메모리 정리 함수
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_full_extraction():
    """전체 PDF 추출 실행"""
    
    pdf_path = Path('data/금융분야 AI 보안 가이드라인.pdf')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('outputs') / f'full_extraction_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('🚀 Full PDF Text Extraction Experiment')
    print('=' * 60)
    print(f'📄 PDF: {pdf_path}')
    print(f'📁 Output: {output_dir}')
    
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
    
    # 프롬프트 설정
    prompt = """
이 문서 페이지의 모든 텍스트를 추출해주세요.

규칙:
1. 페이지에 있는 모든 텍스트를 빠짐없이 원문 그대로 추출
2. 표나 차트가 있다면 그 안의 데이터와 텍스트도 모두 추출
3. 이미지에 포함된 텍스트도 추출
4. 텍스트의 순서와 구조를 최대한 유지

주의: 색상, 폰트, 레이아웃 등 시각적 스타일 설명은 제외하고 실제 텍스트 내용만 추출하세요.
"""
    
    # 결과 저장용
    results = []
    total_pymupdf_chars = 0
    total_vl_chars = 0
    successful_pages = 0
    failed_pages = []
    
    start_time = time.time()
    
    # 페이지별 처리
    for page_num in range(total_pages):
        print(f'\n📖 Processing page {page_num + 1}/{total_pages}...')
        
        page_dir = output_dir / f'page_{page_num+1:03d}'
        page_dir.mkdir(exist_ok=True)
        
        try:
            # PyMuPDF 추출
            print('  - Extracting with PyMuPDF...')
            doc = pymupdf.open(pdf_path)
            page = doc[page_num]
            pymupdf_text = page.get_text()
            doc.close()
            
            (page_dir / 'pymupdf.txt').write_text(pymupdf_text, encoding='utf-8')
            pymupdf_chars = len(pymupdf_text)
            total_pymupdf_chars += pymupdf_chars
            
            # 이미지 변환
            print('  - Converting to image...')
            doc = pymupdf.open(pdf_path)
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes('png')
            doc.close()
            
            image = Image.open(io.BytesIO(img_data))
            image.save(page_dir / 'original.png')
            
            # VL 모델 추출
            print('  - Extracting with VL model...')
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
                    vl_text = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                else:
                    vl_text = '[디코딩 실패]'
            else:
                vl_text = '[생성 실패]'
            
            vl_time = time.time() - vl_start
            
            (page_dir / 'vl_model.txt').write_text(vl_text, encoding='utf-8')
            vl_chars = len(vl_text)
            total_vl_chars += vl_chars
            
            # 결과 기록
            improvement = vl_chars - pymupdf_chars
            improvement_rate = (improvement / pymupdf_chars * 100) if pymupdf_chars > 0 else 0
            
            result = {
                'page': page_num + 1,
                'pymupdf_chars': pymupdf_chars,
                'vl_chars': vl_chars,
                'improvement': improvement,
                'improvement_rate': improvement_rate,
                'vl_time': vl_time,
                'success': True
            }
            
            results.append(result)
            successful_pages += 1
            
            print(f'  ✅ PyMuPDF: {pymupdf_chars:,} chars')
            print(f'  ✅ VL Model: {vl_chars:,} chars')
            print(f'  ✅ Improvement: {improvement:+,} chars ({improvement_rate:+.1f}%)')
            print(f'  ✅ VL Time: {vl_time:.2f}s')
            
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
            clear_gpu_memory()
            print('  🧹 GPU memory cleared')
    
    total_time = time.time() - start_time
    
    # 최종 요약
    print('\n' + '=' * 60)
    print('📊 FINAL RESULTS')
    print('=' * 60)
    print(f'✅ Successful pages: {successful_pages}/{total_pages}')
    if failed_pages:
        print(f'❌ Failed pages: {failed_pages}')
    print(f'📝 Total PyMuPDF chars: {total_pymupdf_chars:,}')
    print(f'📝 Total VL Model chars: {total_vl_chars:,}')
    print(f'📈 Total improvement: {total_vl_chars - total_pymupdf_chars:+,} chars')
    if total_pymupdf_chars > 0:
        overall_improvement = ((total_vl_chars - total_pymupdf_chars) / total_pymupdf_chars * 100)
        print(f'📈 Overall improvement rate: {overall_improvement:+.1f}%')
    print(f'⏱️ Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)')
    print(f'⏱️ Average time per page: {total_time/total_pages:.2f}s')
    
    # 요약 저장
    summary = {
        'timestamp': timestamp,
        'pdf_path': str(pdf_path),
        'total_pages': total_pages,
        'successful_pages': successful_pages,
        'failed_pages': failed_pages,
        'total_pymupdf_chars': total_pymupdf_chars,
        'total_vl_chars': total_vl_chars,
        'total_improvement': total_vl_chars - total_pymupdf_chars,
        'overall_improvement_rate': ((total_vl_chars - total_pymupdf_chars) / total_pymupdf_chars * 100) if total_pymupdf_chars > 0 else 0,
        'total_time': total_time,
        'average_time_per_page': total_time / total_pages,
        'results': results
    }
    
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f'\n📁 Results saved to: {output_dir}')
    print(f'📄 Summary: {summary_file}')
    
    return output_dir

if __name__ == '__main__':
    run_full_extraction()