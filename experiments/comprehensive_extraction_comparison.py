#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
포괄적인 VL vs 기존 텍스트 추출 방법 비교 실험
실제 Qwen2.5-VL 모델과 PyMuPDF 기반 기존 방법을 직접 비교
"""

import os
import sys
import time
import json
import torch
import psutil
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

# 기존 구현 import
from packages.preprocessing.pdf_processor_advanced import AdvancedPDFProcessor

# VL 모델 관련 import
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    import pymupdf
    import io
    VL_AVAILABLE = True
    print("VL 모델 라이브러리 사용 가능")
except ImportError as e:
    VL_AVAILABLE = False
    print(f"VL 모델 라이브러리 없음: {e}")

# ============================================================================
# 메트릭 데이터 클래스
# ============================================================================

@dataclass
class ExtractionMetrics:
    """텍스트 추출 메트릭"""
    method_name: str
    processing_time: float
    peak_memory_mb: float
    char_count: int
    token_count: int
    unique_tokens: int
    chunk_count: int
    visual_elements_detected: int
    error_count: int
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ComparisonResults:
    """비교 실험 결과"""
    experiment_id: str
    timestamp: str
    document_path: str
    traditional_metrics: ExtractionMetrics
    vl_metrics: Optional[ExtractionMetrics]
    content_comparison: Dict
    performance_analysis: Dict
    
    def to_dict(self):
        return asdict(self)

# ============================================================================
# 메모리 최적화 VL 프로세서
# ============================================================================

class MemoryOptimizedVLProcessor:
    """메모리 제약 환경을 위한 VL 프로세서"""
    
    def __init__(self, memory_budget_gb: float = 20.0):
        self.memory_budget = memory_budget_gb * 1024  # MB 단위
        self.model = None
        self.processor = None
        self.model_loaded = False
        
    def _get_available_memory(self) -> float:
        """사용 가능한 GPU 메모리 반환 (MB)"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            allocated_memory = torch.cuda.memory_allocated(0) / 1024 / 1024
            return total_memory - allocated_memory
        return 0
    
    def _estimate_model_memory(self, quantization_config: Dict) -> float:
        """모델 메모리 사용량 추정 (MB)"""
        base_memory = 14000  # 14GB for FP16
        
        if quantization_config.get("load_in_4bit"):
            return base_memory * 0.25  # ~3.5GB
        elif quantization_config.get("load_in_8bit"):
            return base_memory * 0.5   # ~7GB
        else:
            return base_memory         # ~14GB
    
    def load_model(self) -> bool:
        """적응형 모델 로딩"""
        if not VL_AVAILABLE:
            print("VL 모델 라이브러리가 설치되지 않음")
            return False
            
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        available_memory = self._get_available_memory()
        
        # 양자화 설정 우선순위
        quantization_configs = [
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.float16,
                "name": "4-bit"
            },
            {
                "load_in_8bit": True,
                "name": "8-bit"
            },
            {
                "torch_dtype": torch.float16,
                "name": "FP16"
            }
        ]
        
        for config in quantization_configs:
            estimated_memory = self._estimate_model_memory(config)
            
            print(f"시도: {config['name']} 양자화 (예상 메모리: {estimated_memory:.0f}MB)")
            
            if estimated_memory < available_memory:
                try:
                    # BitsAndBytesConfig 설정
                    if config.get("load_in_4bit") or config.get("load_in_8bit"):
                        from transformers import BitsAndBytesConfig
                        bnb_config = BitsAndBytesConfig(**{k: v for k, v in config.items() if k != "name"})
                        
                        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                            model_name,
                            quantization_config=bnb_config,
                            device_map="auto",
                            trust_remote_code=True
                        )
                    else:
                        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                            model_name,
                            torch_dtype=config.get("torch_dtype", torch.float16),
                            device_map="auto",
                            trust_remote_code=True
                        )
                    
                    self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                    self.model_loaded = True
                    
                    actual_memory = torch.cuda.memory_allocated(0) / 1024 / 1024
                    print(f"✅ 모델 로딩 성공: {config['name']} (실제 메모리: {actual_memory:.0f}MB)")
                    return True
                    
                except Exception as e:
                    print(f"❌ {config['name']} 로딩 실패: {e}")
                    # 메모리 정리
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            else:
                print(f"⚠️ 메모리 부족으로 {config['name']} 스킵")
        
        print("❌ 모든 양자화 설정 실패")
        return False
    
    def process_pdf_pages(self, pdf_path: str, max_pages: int = 10) -> List[str]:
        """PDF 페이지를 VL 모델로 처리"""
        if not self.model_loaded:
            print("모델이 로드되지 않음")
            return []
        
        results = []
        doc = pymupdf.open(pdf_path)
        
        try:
            for page_num in range(min(max_pages, len(doc))):
                print(f"페이지 {page_num + 1}/{min(max_pages, len(doc))} 처리 중...")
                
                page = doc[page_num]
                
                # 페이지를 이미지로 변환
                pix = page.get_pixmap(dpi=150)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # VL 모델 프롬프트
                prompt = """이 문서 페이지의 모든 텍스트와 정보를 추출하세요.
                
다음에 주의하세요:
- 모든 텍스트를 정확히 읽어주세요
- 차트나 그래프가 있다면 데이터 포인트와 수치를 추출하세요
- 테이블이 있다면 모든 셀 내용을 읽어주세요
- 다이어그램이 있다면 텍스트와 연결 관계를 설명하세요

형태나 구조 설명보다는 실제 내용과 데이터에 집중해주세요."""
                
                # 메시지 구성
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }]
                
                try:
                    # 모델 추론
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    inputs = self.processor(
                        text=[text],
                        images=[image],
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    # GPU로 이동
                    inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    
                    # 생성
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.eos_token_id
                        )
                    
                    # 디코딩
                    generated_ids = [
                        output_ids[len(input_ids):] 
                        for input_ids, output_ids in zip(inputs['input_ids'], outputs)
                    ]
                    
                    response = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    
                    results.append(response.strip())
                    
                    # 메모리 정리
                    del inputs, outputs, generated_ids
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"페이지 {page_num + 1} 처리 오류: {e}")
                    results.append(f"[페이지 {page_num + 1} 처리 실패: {str(e)}]")
        
        finally:
            doc.close()
        
        return results
    
    def unload_model(self):
        """모델 언로드 및 메모리 정리"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self.model_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("모델 언로드 완료")

# ============================================================================
# 기존 방법 추출기
# ============================================================================

class TraditionalExtractor:
    """기존 PyMuPDF 기반 텍스트 추출"""
    
    def __init__(self):
        self.pdf_processor = AdvancedPDFProcessor()
    
    def extract_from_pdf(self, pdf_path: str) -> str:
        """기존 방법으로 PDF에서 텍스트 추출"""
        try:
            # 기존 AdvancedPDFProcessor 사용
            result = self.pdf_processor.extract_pdf(pdf_path)
            
            # 마크다운과 페이지 텍스트를 결합
            combined_text = []
            
            if result.markdown:
                combined_text.append("=== MARKDOWN 추출 결과 ===")
                combined_text.append(result.markdown)
            
            if result.page_texts:
                combined_text.append("=== 페이지별 텍스트 ===")
                for i, page_text in enumerate(result.page_texts):
                    if page_text.strip():
                        combined_text.append(f"[페이지 {i+1}]")
                        combined_text.append(page_text)
            
            return "\n\n".join(combined_text)
        
        except Exception as e:
            print(f"기존 방법 추출 오류: {e}")
            return f"[추출 실패: {str(e)}]"

# ============================================================================
# 실험 실행기
# ============================================================================

class ExtractionComparisonExperiment:
    """텍스트 추출 방법 비교 실험"""
    
    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.traditional_extractor = TraditionalExtractor()
        self.vl_processor = MemoryOptimizedVLProcessor()
        
    def _measure_performance(self, func, *args, **kwargs) -> Tuple[any, float, float]:
        """함수 실행 시간과 메모리 사용량 측정"""
        # 초기 메모리 상태
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated(0) / 1024 / 1024
        else:
            start_gpu_memory = 0
        
        # 함수 실행
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # 최종 메모리 상태
        end_memory = process.memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_gpu_memory = torch.cuda.memory_allocated(0) / 1024 / 1024
        else:
            end_gpu_memory = 0
        
        execution_time = end_time - start_time
        peak_memory = max(end_memory - start_memory, end_gpu_memory - start_gpu_memory)
        
        return result, execution_time, peak_memory
    
    def _analyze_content(self, text: str) -> Dict:
        """텍스트 콘텐츠 분석"""
        words = text.split()
        
        return {
            "char_count": len(text),
            "word_count": len(words),
            "unique_words": len(set(words)),
            "lines": len(text.split('\n')),
            "visual_keywords": sum(1 for word in words if word.lower() in 
                                 ['그림', '표', '차트', '그래프', '도표', 'figure', 'table', 'chart']),
            "financial_keywords": sum(1 for word in words if word.lower() in
                                    ['금융', '보안', '위험', '규제', '컴플라이언스', '리스크'])
        }
    
    def run_traditional_extraction(self, pdf_path: str) -> ExtractionMetrics:
        """기존 방법으로 텍스트 추출 실행"""
        print("\n=== 기존 방법 텍스트 추출 ===")
        
        result, exec_time, peak_mem = self._measure_performance(
            self.traditional_extractor.extract_from_pdf, pdf_path
        )
        
        # 결과 저장
        output_file = self.output_dir / "traditional_extraction_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        
        # 콘텐츠 분석
        content_analysis = self._analyze_content(result)
        
        metrics = ExtractionMetrics(
            method_name="Traditional_PyMuPDF",
            processing_time=exec_time,
            peak_memory_mb=peak_mem,
            char_count=content_analysis["char_count"],
            token_count=content_analysis["word_count"],
            unique_tokens=content_analysis["unique_words"],
            chunk_count=content_analysis["lines"],
            visual_elements_detected=content_analysis["visual_keywords"],
            error_count=1 if "추출 실패" in result else 0
        )
        
        print(f"처리 시간: {exec_time:.2f}초")
        print(f"메모리 사용: {peak_mem:.0f}MB")
        print(f"추출 문자 수: {metrics.char_count:,}")
        print(f"결과 저장: {output_file}")
        
        return metrics
    
    def run_vl_extraction(self, pdf_path: str, max_pages: int = 10) -> Optional[ExtractionMetrics]:
        """VL 방법으로 텍스트 추출 실행"""
        print("\n=== VL 모델 텍스트 추출 ===")
        
        # 모델 로딩
        print("VL 모델 로딩 중...")
        if not self.vl_processor.load_model():
            print("VL 모델 로딩 실패")
            return None
        
        try:
            # VL 추출 실행
            result, exec_time, peak_mem = self._measure_performance(
                self.vl_processor.process_pdf_pages, pdf_path, max_pages
            )
            
            # 결과를 하나의 텍스트로 결합
            combined_text = "\n\n".join([f"[페이지 {i+1}]\n{content}" for i, content in enumerate(result)])
            
            # 결과 저장
            output_file = self.output_dir / "vl_extraction_output.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            
            # 콘텐츠 분석
            content_analysis = self._analyze_content(combined_text)
            
            metrics = ExtractionMetrics(
                method_name="VL_Qwen2.5-VL-7B",
                processing_time=exec_time,
                peak_memory_mb=peak_mem,
                char_count=content_analysis["char_count"],
                token_count=content_analysis["word_count"],
                unique_tokens=content_analysis["unique_words"],
                chunk_count=len(result),
                visual_elements_detected=content_analysis["visual_keywords"],
                error_count=sum(1 for r in result if "처리 실패" in r)
            )
            
            print(f"처리 시간: {exec_time:.2f}초")
            print(f"메모리 사용: {peak_mem:.0f}MB")
            print(f"추출 문자 수: {metrics.char_count:,}")
            print(f"처리 페이지: {len(result)}개")
            print(f"결과 저장: {output_file}")
            
            return metrics
        
        finally:
            # 모델 언로드
            self.vl_processor.unload_model()
    
    def compare_methods(self, pdf_path: str, max_pages: int = 10) -> ComparisonResults:
        """두 방법 비교 실행"""
        print(f"\n📄 문서 분석: {pdf_path}")
        print("=" * 60)
        
        experiment_id = hashlib.md5(f"{pdf_path}_{datetime.now()}".encode()).hexdigest()[:8]
        
        # 1. 기존 방법 실행
        traditional_metrics = self.run_traditional_extraction(pdf_path)
        
        # 2. VL 방법 실행
        vl_metrics = self.run_vl_extraction(pdf_path, max_pages)
        
        # 3. 비교 분석
        content_comparison = self._compare_content(traditional_metrics, vl_metrics)
        performance_analysis = self._analyze_performance(traditional_metrics, vl_metrics)
        
        # 4. 결과 구성
        results = ComparisonResults(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            document_path=pdf_path,
            traditional_metrics=traditional_metrics,
            vl_metrics=vl_metrics,
            content_comparison=content_comparison,
            performance_analysis=performance_analysis
        )
        
        # 5. 결과 저장
        self._save_results(results)
        
        return results
    
    def _compare_content(self, traditional: ExtractionMetrics, vl: Optional[ExtractionMetrics]) -> Dict:
        """콘텐츠 비교 분석"""
        if vl is None:
            return {
                "comparison_possible": False,
                "reason": "VL extraction failed"
            }
        
        comparison = {
            "comparison_possible": True,
            "content_increase": {
                "char_count": ((vl.char_count - traditional.char_count) / traditional.char_count * 100) if traditional.char_count > 0 else 0,
                "token_count": ((vl.token_count - traditional.token_count) / traditional.token_count * 100) if traditional.token_count > 0 else 0,
                "unique_tokens": ((vl.unique_tokens - traditional.unique_tokens) / traditional.unique_tokens * 100) if traditional.unique_tokens > 0 else 0
            },
            "visual_elements": {
                "traditional": traditional.visual_elements_detected,
                "vl": vl.visual_elements_detected,
                "improvement": vl.visual_elements_detected - traditional.visual_elements_detected
            },
            "error_comparison": {
                "traditional_errors": traditional.error_count,
                "vl_errors": vl.error_count
            }
        }
        
        return comparison
    
    def _analyze_performance(self, traditional: ExtractionMetrics, vl: Optional[ExtractionMetrics]) -> Dict:
        """성능 비교 분석"""
        if vl is None:
            return {
                "analysis_possible": False,
                "reason": "VL extraction failed"
            }
        
        analysis = {
            "analysis_possible": True,
            "processing_time": {
                "traditional": traditional.processing_time,
                "vl": vl.processing_time,
                "ratio": vl.processing_time / traditional.processing_time if traditional.processing_time > 0 else float('inf')
            },
            "memory_usage": {
                "traditional": traditional.peak_memory_mb,
                "vl": vl.peak_memory_mb,
                "additional_memory": vl.peak_memory_mb - traditional.peak_memory_mb
            },
            "efficiency": {
                "traditional_chars_per_sec": traditional.char_count / traditional.processing_time if traditional.processing_time > 0 else 0,
                "vl_chars_per_sec": vl.char_count / vl.processing_time if vl.processing_time > 0 else 0
            }
        }
        
        return analysis
    
    def _save_results(self, results: ComparisonResults):
        """실험 결과 저장"""
        # JSON 결과 저장
        json_file = self.output_dir / f"comparison_results_{results.experiment_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 요약 리포트 생성
        self._generate_summary_report(results)
        
        print(f"\n📊 실험 결과 저장 완료:")
        print(f"  - JSON 결과: {json_file}")
        print(f"  - 요약 리포트: {self.output_dir}/comparison_summary.md")
    
    def _generate_summary_report(self, results: ComparisonResults):
        """요약 리포트 생성"""
        report = []
        report.append("# 텍스트 추출 방법 비교 실험 결과\n")
        report.append(f"**실험 ID**: {results.experiment_id}")
        report.append(f"**실험 일시**: {results.timestamp}")
        report.append(f"**문서**: {Path(results.document_path).name}\n")
        
        # 기본 메트릭 비교
        report.append("## 📊 기본 메트릭 비교\n")
        report.append("| 메트릭 | 기존 방법 | VL 방법 | 개선율 |")
        report.append("|--------|-----------|---------|--------|")
        
        trad = results.traditional_metrics
        vl = results.vl_metrics
        
        if vl:
            comp = results.content_comparison
            report.append(f"| 추출 문자 수 | {trad.char_count:,} | {vl.char_count:,} | {comp['content_increase']['char_count']:+.1f}% |")
            report.append(f"| 토큰 수 | {trad.token_count:,} | {vl.token_count:,} | {comp['content_increase']['token_count']:+.1f}% |")
            report.append(f"| 고유 토큰 | {trad.unique_tokens:,} | {vl.unique_tokens:,} | {comp['content_increase']['unique_tokens']:+.1f}% |")
            report.append(f"| 처리 시간 | {trad.processing_time:.2f}초 | {vl.processing_time:.2f}초 | {((vl.processing_time/trad.processing_time-1)*100):+.1f}% |")
            report.append(f"| 메모리 사용 | {trad.peak_memory_mb:.0f}MB | {vl.peak_memory_mb:.0f}MB | +{vl.peak_memory_mb-trad.peak_memory_mb:.0f}MB |")
        else:
            report.append(f"| 추출 문자 수 | {trad.char_count:,} | N/A | N/A |")
            report.append(f"| 토큰 수 | {trad.token_count:,} | N/A | N/A |")
            report.append(f"| 처리 시간 | {trad.processing_time:.2f}초 | N/A | N/A |")
        
        # 성능 분석
        if vl and results.performance_analysis.get("analysis_possible"):
            perf = results.performance_analysis
            report.append("\n## ⚡ 성능 분석\n")
            report.append(f"- **처리 속도 비율**: VL 방법이 기존 방법 대비 {perf['processing_time']['ratio']:.1f}배")
            report.append(f"- **처리 효율성**: 기존 {perf['efficiency']['traditional_chars_per_sec']:.0f} vs VL {perf['efficiency']['vl_chars_per_sec']:.0f} 문자/초")
            report.append(f"- **추가 메모리**: {perf['memory_usage']['additional_memory']:.0f}MB")
        
        # 시각적 요소 분석
        if vl and results.content_comparison.get("comparison_possible"):
            visual = results.content_comparison["visual_elements"]
            report.append("\n## 👁️ 시각적 요소 감지\n")
            report.append(f"- **기존 방법**: {visual['traditional']}개")
            report.append(f"- **VL 방법**: {visual['vl']}개")
            report.append(f"- **개선**: {visual['improvement']:+d}개")
        
        # 결론 및 권장사항
        report.append("\n## 💡 결론 및 권장사항\n")
        
        if vl:
            if results.content_comparison["content_increase"]["char_count"] > 10:
                report.append("✅ **VL 통합 권장**: 콘텐츠 추출량이 유의미하게 증가")
            elif results.content_comparison["visual_elements"]["improvement"] > 5:
                report.append("✅ **VL 통합 권장**: 시각적 요소 처리 능력 향상")
            else:
                report.append("⚠️ **신중한 검토 필요**: 성능 향상이 제한적")
                
            if vl.peak_memory_mb > 20000:  # 20GB
                report.append("⚠️ **메모리 최적화 필요**: 대회 환경 메모리 제한 고려")
                
            if vl.processing_time > trad.processing_time * 3:
                report.append("⚠️ **처리 시간 최적화 필요**: 대회 시간 제한 고려")
        else:
            report.append("❌ **VL 통합 불가**: 모델 로딩 또는 처리 실패")
            report.append("📝 **권장사항**: 메모리 제약 완화 또는 모델 경량화 검토")
        
        # 리포트 저장
        report_file = self.output_dir / "comparison_summary.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("=" * 60)
    print("🔍 포괄적인 텍스트 추출 방법 비교 실험")
    print("=" * 60)
    
    # 테스트 문서 찾기
    test_documents = [
        "data/raw/금융분야 AI 보안 가이드라인.pdf",
        "docs/금융분야 AI 보안 가이드라인.pdf",
        "data/documents/금융분야 AI 보안 가이드라인.pdf",
    ]
    
    pdf_path = None
    for doc_path in test_documents:
        if Path(doc_path).exists():
            pdf_path = doc_path
            break
    
    if not pdf_path:
        print("❌ 테스트 문서를 찾을 수 없습니다.")
        print("다음 경로 중 하나에 PDF 문서를 배치하세요:")
        for doc_path in test_documents:
            print(f"  - {doc_path}")
        return
    
    # 실험 실행
    experiment = ExtractionComparisonExperiment()
    
    try:
        results = experiment.compare_methods(pdf_path, max_pages=10)
        
        print("\n" + "=" * 60)
        print("✅ 실험 완료!")
        print("=" * 60)
        
        if results.vl_metrics:
            print(f"📈 콘텐츠 증가: {results.content_comparison['content_increase']['char_count']:+.1f}%")
            print(f"⚡ 처리 시간 비율: {results.performance_analysis['processing_time']['ratio']:.1f}x")
            print(f"💾 추가 메모리: {results.performance_analysis['memory_usage']['additional_memory']:.0f}MB")
        else:
            print("⚠️ VL 방법 실행 실패 - 기존 방법 결과만 저장됨")
        
        print(f"\n📁 결과 디렉토리: {experiment.output_dir}")
        
    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 실험 중단됨")
    except Exception as e:
        print(f"\n❌ 실험 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()