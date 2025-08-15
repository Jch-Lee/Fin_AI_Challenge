#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VL 모델 텍스트 추출 비교 실험 (원격 서버용)
독립 실행 가능한 단일 스크립트

이 스크립트는:
1. GPU 환경을 자동으로 감지하고 최적화
2. PyMuPDF vs Qwen2.5-VL 텍스트 추출 비교
3. 결과를 페이지별로 저장
4. HTML 비교 리포트 생성
"""

import os
import sys
import json
import time
import io
from pathlib import Path
from datetime import datetime
import torch
import pymupdf
from PIL import Image
import traceback

def check_gpu_environment():
    """GPU 환경 확인 및 정보 출력"""
    print("=" * 60)
    print("🔍 GPU Environment Check")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n🖥️ GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
            reserved_memory = torch.cuda.memory_reserved(i) / 1024**3
            
            print(f"  - Total Memory: {total_memory:.1f} GB")
            print(f"  - Memory Allocated: {allocated_memory:.1f} GB")
            print(f"  - Memory Reserved: {reserved_memory:.1f} GB")
            print(f"  - Available Memory: {total_memory - allocated_memory:.1f} GB")
            
            return True, total_memory
    else:
        print("❌ CUDA is not available")
        print("⚠️ Will use CPU (this will be very slow)")
        return False, 0
    
    print("=" * 60)

def select_best_quantization(gpu_memory_gb):
    """GPU 메모리에 따른 최적 양자화 선택"""
    if gpu_memory_gb >= 40:
        return None, "FP16"  # 전체 정밀도
    elif gpu_memory_gb >= 20:
        return "8bit", "8-bit"
    elif gpu_memory_gb >= 8:
        return "4bit", "4-bit"
    else:
        return "4bit", "4-bit (CPU fallback)"

class VLExtractionComparator:
    """독립형 VL vs PyMuPDF 비교기"""
    
    def __init__(self, pdf_path, output_dir):
        self.pdf_path = Path(pdf_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"vl_comparison_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = None
        self.model = None
        self.processor = None
        self.quantization_info = None
        
        # 로그 파일 설정
        self.log_file = self.output_dir / "experiment.log"
        self.log("Experiment initialized", timestamp=True)
        
    def log(self, message, timestamp=False):
        """로그 메시지 출력 및 파일 저장"""
        if timestamp:
            full_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        else:
            full_message = message
            
        print(full_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(full_message + '\n')
        
    def setup_model(self):
        """GPU 환경에 맞춰 모델 설정"""
        self.log("\n📦 Setting up VL model...", timestamp=True)
        
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        except ImportError as e:
            # Fallback to Qwen2VL if Qwen2_5_VL not available
            try:
                from transformers import Qwen2VLForConditionalGeneration as Qwen2_5_VLForConditionalGeneration
            except ImportError:
                self.log(f"❌ Failed to import transformers: {e}")
                raise
        
        # GPU 환경 확인
        has_gpu, gpu_memory = check_gpu_environment()
        
        if has_gpu:
            quantization_type, quantization_name = select_best_quantization(gpu_memory)
            
            self.log(f"\n📊 Model Configuration:")
            self.log(f"  - GPU Memory: {gpu_memory:.1f} GB")
            self.log(f"  - Quantization: {quantization_name}")
            
            # 양자화 임시 비활성화 (호환성 문제 해결)
            quantization_config = None
            quantization_name = "FP16 (No quantization)"
                
            torch_dtype = torch.float16
            self.quantization_info = quantization_name
        else:
            self.log("\n⚠️ Using CPU - this will be very slow")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32
            )
            torch_dtype = torch.float32
            self.quantization_info = "4-bit (CPU)"
        
        self.log("\n📥 Loading Qwen2.5-VL-7B model...")
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )
            
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", 
                trust_remote_code=True
            )
            
            self.device = "cuda" if has_gpu else "cpu"
            self.log("✅ Model loaded successfully!")
            
            if has_gpu:
                current_memory = torch.cuda.memory_allocated(0) / 1024**3
                self.log(f"📊 Current GPU memory usage: {current_memory:.1f} GB")
            
        except Exception as e:
            self.log(f"❌ Failed to load model: {e}")
            self.log(f"Traceback: {traceback.format_exc()}")
            raise
        
    def extract_pymupdf(self, page_num):
        """PyMuPDF 텍스트 추출"""
        try:
            doc = pymupdf.open(self.pdf_path)
            page = doc[page_num]
            text = page.get_text()
            doc.close()
            return text
        except Exception as e:
            self.log(f"❌ PyMuPDF extraction failed for page {page_num + 1}: {e}")
            return f"[PyMuPDF 추출 실패: {str(e)}]"
    
    def pdf_to_image(self, page_num, dpi=150):
        """PDF 페이지를 이미지로 변환"""
        try:
            doc = pymupdf.open(self.pdf_path)
            page = doc[page_num]
            pix = page.get_pixmap(dpi=dpi)
            img_data = pix.tobytes("png")
            doc.close()
            return Image.open(io.BytesIO(img_data))
        except Exception as e:
            self.log(f"❌ PDF to image conversion failed for page {page_num + 1}: {e}")
            raise
    
    def extract_vl(self, image):
        """VL 모델로 텍스트 추출"""
        prompt = """
이 문서 페이지의 내용을 다음 규칙에 따라 추출해주세요:

1. 텍스트 추출:
   - 페이지의 모든 텍스트를 원문 그대로 정확히 추출
   - 제목, 본문, 각주 등 모든 텍스트 포함
   - 순서와 구조 유지

2. 표/차트/그래프 처리:
   - 데이터와 수치만 추출 (색상, 위치 설명 제외)
   - 표: 헤더와 데이터를 구조적으로 표현
   - 차트: 축 레이블, 데이터 값, 범례 정보만 추출
   - 그래프: 추세, 수치, 주요 포인트만 설명

3. 이미지/다이어그램 처리:
   - 이미지 내 텍스트와 핵심 정보만 추출
   - 프로세스 흐름이나 관계도는 논리적 연결만 설명
   - 아이콘이나 로고는 이름만 언급

중요: 색상, 폰트, 레이아웃, 디자인 요소 등 시각적 스타일 설명은 제외하고
실제 정보와 데이터만 추출하세요. 배경색, 텍스트 위치, 장식적 요소는 언급하지 마세요.
"""
        
        try:
            # qwen_vl_utils를 사용한 메시지 처리
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError:
                # qwen_vl_utils가 없을 경우 기본 처리
                process_vision_info = None
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # 프롬프트 템플릿 적용
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # process_vision_info가 있으면 사용, 없으면 대체 처리
            if process_vision_info:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                # 대체 처리: 직접 이미지 추출
                image_inputs = [image]
                video_inputs = []
            
            # 입력 데이터 검증
            if not text:
                self.log("⚠️ Empty text input")
                return "[텍스트 입력이 비어있습니다]"
            
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt"
            )
            
            # GPU로 이동
            if self.device == "cuda":
                inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # 생성 설정
            generation_config = {
                "max_new_tokens": 2048,  # 더 많은 토큰 생성
                "temperature": 0.7,  # 조금 더 창의적인 출력
                "do_sample": True,
                "top_p": 0.9,
                "pad_token_id": self.processor.tokenizer.eos_token_id
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
            # 방어적 코드: 출력 데이터 검증
            if outputs is None or len(outputs) == 0:
                self.log("⚠️ Model generated empty output")
                return "[모델이 빈 출력을 생성했습니다]"
            
            # inputs가 딕셔너리인 경우 input_ids 추출
            if isinstance(inputs, dict):
                input_ids_tensor = inputs.get('input_ids', None)
            else:
                input_ids_tensor = inputs.input_ids
            
            if input_ids_tensor is None:
                self.log("⚠️ No input_ids found")
                return "[input_ids를 찾을 수 없습니다]"
            
            # 입력 부분 제거 후 디코딩
            generated_ids = []
            for input_ids, output_ids in zip(input_ids_tensor, outputs):
                # 안전한 슬라이싱을 위한 길이 확인
                input_len = len(input_ids) if hasattr(input_ids, '__len__') else input_ids.shape[0]
                output_len = len(output_ids) if hasattr(output_ids, '__len__') else output_ids.shape[0]
                
                if output_len > input_len:
                    generated_ids.append(output_ids[input_len:])
                else:
                    # 출력이 입력보다 짧은 경우 전체 출력 사용
                    generated_ids.append(output_ids)
            
            # 생성된 ID가 없는 경우 처리
            if not generated_ids:
                self.log("⚠️ No valid generated IDs")
                return "[유효한 생성 결과가 없습니다]"
            
            # 디코딩 시 방어적 처리
            decoded_outputs = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # 디코딩 결과 검증
            if decoded_outputs and len(decoded_outputs) > 0:
                response = decoded_outputs[0]
            else:
                self.log("⚠️ Decoding produced empty result")
                response = "[디코딩 결과가 비어있습니다]"
            
            return response.strip() if response else "[빈 응답]"
            
        except Exception as e:
            self.log(f"❌ VL model extraction failed: {e}")
            return f"[VL 모델 추출 실패: {str(e)}]"
    
    def run_comparison(self, max_pages=10):
        """비교 실험 실행"""
        self.log(f"\n🚀 Starting VL extraction comparison experiment", timestamp=True)
        self.log(f"📄 PDF: {self.pdf_path}")
        self.log(f"📁 Output: {self.output_dir}")
        
        # PDF 확인
        if not self.pdf_path.exists():
            self.log(f"❌ PDF file not found: {self.pdf_path}")
            return None
        
        # 모델 설정
        self.setup_model()
        
        # PDF 페이지 수 확인
        try:
            doc = pymupdf.open(self.pdf_path)
            total_pdf_pages = len(doc)
            doc.close()
            
            total_pages = min(max_pages, total_pdf_pages)
            
            self.log(f"\n📖 Processing {total_pages} pages (out of {total_pdf_pages} total)")
            self.log("=" * 60)
            
        except Exception as e:
            self.log(f"❌ Failed to open PDF: {e}")
            return None
        
        results = []
        total_start_time = time.time()
        
        for page_num in range(total_pages):
            self.log(f"\n📖 Processing page {page_num + 1}/{total_pages}...")
            page_start_time = time.time()
            
            # 페이지 디렉토리 생성
            page_dir = self.output_dir / f"page_{page_num+1:03d}"
            page_dir.mkdir(exist_ok=True)
            
            try:
                # PyMuPDF 추출
                self.log("  - Extracting with PyMuPDF...")
                pymupdf_text = self.extract_pymupdf(page_num)
                (page_dir / "pymupdf.txt").write_text(pymupdf_text, encoding="utf-8")
                
                # 이미지 변환 및 저장
                self.log("  - Converting to image...")
                image = self.pdf_to_image(page_num)
                image.save(page_dir / "original.png")
                
                # VL 추출
                self.log("  - Extracting with VL model...")
                vl_start_time = time.time()
                vl_text = self.extract_vl(image)
                vl_process_time = time.time() - vl_start_time
                (page_dir / "vl_model.txt").write_text(vl_text, encoding="utf-8")
                
                # 전체 처리 시간
                total_page_time = time.time() - page_start_time
                
                # 결과 기록
                page_result = {
                    "page": page_num + 1,
                    "pymupdf_chars": len(pymupdf_text),
                    "vl_chars": len(vl_text),
                    "improvement": len(vl_text) - len(pymupdf_text),
                    "improvement_rate": ((len(vl_text) - len(pymupdf_text)) / len(pymupdf_text) * 100) if len(pymupdf_text) > 0 else 0,
                    "vl_process_time": vl_process_time,
                    "total_page_time": total_page_time,
                    "success": True
                }
                
                self.log(f"  ✅ Results:")
                self.log(f"     - PyMuPDF: {page_result['pymupdf_chars']:,} chars")
                self.log(f"     - VL Model: {page_result['vl_chars']:,} chars")
                self.log(f"     - Improvement: +{page_result['improvement']:,} chars ({page_result['improvement_rate']:.1f}%)")
                self.log(f"     - VL Time: {vl_process_time:.2f}s")
                self.log(f"     - Total Time: {total_page_time:.2f}s")
                
            except Exception as e:
                self.log(f"  ❌ Page processing failed: {e}")
                page_result = {
                    "page": page_num + 1,
                    "pymupdf_chars": 0,
                    "vl_chars": 0,
                    "improvement": 0,
                    "improvement_rate": 0,
                    "vl_process_time": 0,
                    "total_page_time": 0,
                    "success": False,
                    "error": str(e)
                }
            
            results.append(page_result)
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_experiment_time = time.time() - total_start_time
        
        # 성공한 페이지만 필터링하여 요약 계산
        successful_results = [r for r in results if r["success"]]
        
        if successful_results:
            # 요약 통계 생성
            summary = {
                "timestamp": self.timestamp,
                "pdf_path": str(self.pdf_path),
                "pdf_name": self.pdf_path.name,
                "total_pages_processed": total_pages,
                "successful_pages": len(successful_results),
                "failed_pages": len(results) - len(successful_results),
                "device": self.device,
                "quantization": self.quantization_info,
                "gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "page_results": results,
                "statistics": {
                    "total_improvement": sum(r["improvement"] for r in successful_results),
                    "average_improvement_rate": sum(r["improvement_rate"] for r in successful_results) / len(successful_results),
                    "total_vl_time": sum(r["vl_process_time"] for r in successful_results),
                    "total_experiment_time": total_experiment_time,
                    "average_vl_time_per_page": sum(r["vl_process_time"] for r in successful_results) / len(successful_results),
                    "total_pymupdf_chars": sum(r["pymupdf_chars"] for r in successful_results),
                    "total_vl_chars": sum(r["vl_chars"] for r in successful_results)
                }
            }
        else:
            self.log("❌ No pages were successfully processed!")
            summary = {
                "timestamp": self.timestamp,
                "pdf_path": str(self.pdf_path),
                "error": "No pages were successfully processed",
                "page_results": results
            }
        
        # 요약 저장
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # HTML 리포트 생성 (성공한 페이지가 있을 때만)
        if successful_results:
            self.generate_html_report(successful_results, summary)
        
        # 최종 결과 출력
        self.log("\n" + "=" * 60)
        self.log(f"✅ Experiment completed!", timestamp=True)
        self.log(f"📁 Results saved to: {self.output_dir}")
        
        if successful_results:
            stats = summary["statistics"]
            self.log(f"📊 Statistics:")
            self.log(f"   - Successful pages: {len(successful_results)}/{total_pages}")
            self.log(f"   - Average improvement: {stats['average_improvement_rate']:.1f}%")
            self.log(f"   - Total additional chars: {stats['total_improvement']:,}")
            self.log(f"   - Average VL time/page: {stats['average_vl_time_per_page']:.2f}s")
            self.log(f"   - Total experiment time: {stats['total_experiment_time']:.1f}s")
        else:
            self.log("❌ Experiment failed - no successful page processing")
        
        return summary
    
    def generate_html_report(self, results, summary):
        """HTML 비교 리포트 생성"""
        self.log("📄 Generating HTML comparison report...")
        
        stats = summary["statistics"]
        
        # HTML 헤더
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>VL 모델 텍스트 추출 비교 - {summary['pdf_name']}</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .summary {{ 
            background: white; 
            padding: 20px; 
            margin-bottom: 30px; 
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .summary h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .page-section {{ 
            margin-bottom: 50px; 
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .page-header {{
            background: #667eea;
            color: white;
            padding: 15px 20px;
            font-size: 18px;
            font-weight: bold;
        }}
        .page-content {{
            padding: 20px;
        }}
        .comparison {{ 
            display: flex; 
            gap: 20px; 
            margin-top: 20px;
        }}
        .column {{ 
            flex: 1; 
        }}
        .column h3 {{
            color: #333;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #ddd;
        }}
        .text-box {{ 
            border: 1px solid #ddd; 
            padding: 15px; 
            height: 400px; 
            overflow-y: auto;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.4;
            border-radius: 5px;
        }}
        .pymupdf {{ 
            background: #fff5f5; 
            border-left: 4px solid #ff6b6b;
        }}
        .vl {{ 
            background: #f5fff5; 
            border-left: 4px solid #51cf66;
        }}
        .stats {{ 
            margin-top: 15px; 
            font-size: 14px;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
        }}
        .improvement {{ 
            color: #51cf66; 
            font-weight: bold; 
        }}
        .original-image {{ 
            max-width: 100%; 
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .nav-buttons {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        }}
        .nav-button {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 15px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }}
        .nav-button:hover {{
            background: #5a67d8;
        }}
        @media (max-width: 768px) {{
            .comparison {{
                flex-direction: column;
            }}
            .nav-buttons {{
                position: relative;
                text-align: center;
                margin-bottom: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 VL 모델 텍스트 추출 비교 결과</h1>
        <p>문서: {summary['pdf_name']} | 실험 일시: {summary['timestamp']}</p>
        <p>GPU: {summary['gpu_info']} ({summary['quantization']})</p>
    </div>
    
    <div class="nav-buttons">
        <a href="#summary" class="nav-button">📊 요약</a>
        <a href="#page1" class="nav-button">📄 페이지</a>
    </div>
    
    <div id="summary" class="summary">
        <h2>📊 실험 요약</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{len(results)}</div>
                <div class="stat-label">처리된 페이지</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['average_improvement_rate']:.1f}%</div>
                <div class="stat-label">평균 개선율</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['total_improvement']:,}</div>
                <div class="stat-label">추가 문자 수</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['average_vl_time_per_page']:.1f}s</div>
                <div class="stat-label">평균 VL 처리시간</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['total_pymupdf_chars']:,}</div>
                <div class="stat-label">PyMuPDF 총 문자</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['total_vl_chars']:,}</div>
                <div class="stat-label">VL 모델 총 문자</div>
            </div>
        </div>
    </div>
"""
        
        # 페이지별 결과
        for i, result in enumerate(results):
            page_num = result['page']
            page_dir = self.output_dir / f"page_{page_num:03d}"
            
            # 텍스트 파일 읽기
            try:
                pymupdf_text = (page_dir / "pymupdf.txt").read_text(encoding="utf-8")[:5000]  # 처음 5000자만
                vl_text = (page_dir / "vl_model.txt").read_text(encoding="utf-8")[:5000]  # 처음 5000자만
            except:
                pymupdf_text = "[텍스트 읽기 실패]"
                vl_text = "[텍스트 읽기 실패]"
            
            # HTML 이스케이프 처리
            pymupdf_text = pymupdf_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            vl_text = vl_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            html += f"""
    <div id="page{page_num}" class="page-section">
        <div class="page-header">
            📄 페이지 {page_num} 
            <span style="float: right; font-size: 14px;">
                처리시간: {result['total_page_time']:.2f}s
            </span>
        </div>
        <div class="page-content">
            <img src="page_{page_num:03d}/original.png" alt="Page {page_num}" class="original-image">
            <div class="comparison">
                <div class="column">
                    <h3>🔤 PyMuPDF 추출</h3>
                    <div class="text-box pymupdf">{pymupdf_text}</div>
                    <div class="stats">
                        📊 문자 수: {result['pymupdf_chars']:,}
                    </div>
                </div>
                <div class="column">
                    <h3>🤖 VL 모델 추출</h3>
                    <div class="text-box vl">{vl_text}</div>
                    <div class="stats">
                        📊 문자 수: {result['vl_chars']:,}
                        <span class="improvement">(+{result['improvement']:,}, {result['improvement_rate']:.1f}%)</span><br>
                        ⏱️ VL 처리시간: {result['vl_process_time']:.2f}s
                    </div>
                </div>
            </div>
        </div>
    </div>
"""
        
        html += """
    <script>
        // 페이지 네비게이션을 위한 스크립트
        document.addEventListener('DOMContentLoaded', function() {
            const navButtons = document.querySelector('.nav-buttons');
            const pages = document.querySelectorAll('.page-section');
            
            pages.forEach((page, index) => {
                const pageNum = index + 1;
                const button = document.createElement('a');
                button.href = `#page${pageNum}`;
                button.className = 'nav-button';
                button.textContent = pageNum;
                button.style.fontSize = '12px';
                button.style.padding = '5px 10px';
                navButtons.appendChild(button);
            });
        });
    </script>
</body>
</html>"""
        
        # HTML 파일 저장
        html_file = self.output_dir / "comparison_report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html)
        
        self.log(f"📄 HTML report saved: {html_file}")

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VL 모델과 PyMuPDF 텍스트 추출 비교 실험",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python vl_extraction_comparison.py --pdf data/금융분야\ AI\ 보안\ 가이드라인.pdf --max-pages 10
  python vl_extraction_comparison.py --pdf document.pdf --output results --max-pages 5
        """
    )
    
    parser.add_argument(
        "--pdf", 
        default="data/금융분야 AI 보안 가이드라인.pdf",
        help="분석할 PDF 파일 경로"
    )
    parser.add_argument(
        "--output", 
        default="outputs",
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--max-pages", 
        type=int, 
        default=10,
        help="처리할 최대 페이지 수"
    )
    
    args = parser.parse_args()
    
    print("🚀 VL 모델 텍스트 추출 비교 실험")
    print("=" * 60)
    print(f"📄 PDF: {args.pdf}")
    print(f"📁 Output: {args.output}")
    print(f"📖 Max pages: {args.max_pages}")
    print("=" * 60)
    
    try:
        comparator = VLExtractionComparator(args.pdf, args.output)
        summary = comparator.run_comparison(args.max_pages)
        
        if summary and "statistics" in summary:
            print("\n🎉 실험 완료! HTML 리포트를 확인하세요.")
            print(f"📁 결과 디렉토리: {comparator.output_dir}")
            html_report = comparator.output_dir / "comparison_report.html"
            if html_report.exists():
                print(f"🌐 HTML 리포트: {html_report}")
        else:
            print("\n❌ 실험 실패")
            
    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 실험이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 실험 실행 중 오류 발생: {e}")
        print(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    main()