#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real Qwen2.5-VL Model PDF Extraction Comparison Test
Compares actual VL model extraction with PyMuPDF extraction
"""

import os
import sys
import json
import time
import gc
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import pymupdf
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Memory monitoring
import psutil

# Model imports
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)

# Import prompts from the example file
sys.path.append(str(Path(__file__).parent))
from vl_prompt_example import (
    GENERAL_EXTRACTION_PROMPT,
    CHART_EXTRACTION_PROMPT,
    TABLE_EXTRACTION_PROMPT,
    DIAGRAM_EXTRACTION_PROMPT,
    FINANCIAL_DOC_PROMPT
)

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Test configuration"""
    # Model settings
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    USE_4BIT = True  # Enable 4-bit quantization for memory efficiency
    MAX_NEW_TOKENS = 1024
    TEMPERATURE = 0.1  # Low temperature for accuracy
    
    # Test settings
    TEST_PDF_PATH = "../data/pdfs/financial_sample.pdf"  # Update with actual path
    MAX_PAGES = 5  # Limit pages for initial testing
    DPI = 150  # Image resolution for PDF conversion
    
    # Output settings
    RESULTS_DIR = "./results"
    REPORT_FILE = "real_vl_comparison_report.json"
    
    # Memory management
    CLEAR_CACHE_EVERY_N_PAGES = 2
    MAX_IMAGE_SIZE = (1920, 1920)  # Resize large images

# ============================================================================
# Model Manager
# ============================================================================

class VLModelManager:
    """Manages Qwen2.5-VL model loading and inference"""
    
    def __init__(self, use_4bit: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.use_4bit = use_4bit
        
    def load_model(self) -> bool:
        """Load the VL model with appropriate configuration"""
        try:
            print(f"Loading Qwen2.5-VL model on {self.device}...")
            
            # Configure quantization if enabled
            if self.use_4bit and self.device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    Config.MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                print("Model loaded with 4-bit quantization")
            else:
                # Load without quantization (requires more memory)
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    Config.MODEL_NAME,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                print(f"Model loaded on {self.device}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                Config.MODEL_NAME,
                trust_remote_code=True
            )
            
            # Log memory usage
            self._log_memory_usage("After model loading")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return False
    
    def extract_from_image(self, image: Image.Image, prompt: str) -> str:
        """Extract text from image using VL model"""
        try:
            # Resize image if too large
            if image.size[0] > Config.MAX_IMAGE_SIZE[0] or image.size[1] > Config.MAX_IMAGE_SIZE[1]:
                image.thumbnail(Config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # Prepare messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text,
                images=image,
                return_tensors="pt"
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=Config.MAX_NEW_TOKENS,
                    temperature=Config.TEMPERATURE,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            result = self.processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return result.strip()
            
        except Exception as e:
            print(f"Error during extraction: {e}")
            traceback.print_exc()
            return f"[Extraction Error: {str(e)}]"
    
    def _log_memory_usage(self, stage: str):
        """Log current memory usage"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"{stage} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        process = psutil.Process()
        ram_usage = process.memory_info().rss / 1024**3
        print(f"{stage} - RAM Usage: {ram_usage:.2f}GB")
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

# ============================================================================
# PDF Processor
# ============================================================================

class PDFProcessor:
    """Handles PDF processing and extraction"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = None
        
    def open(self):
        """Open PDF document"""
        self.doc = pymupdf.open(self.pdf_path)
        return len(self.doc)
    
    def close(self):
        """Close PDF document"""
        if self.doc:
            self.doc.close()
    
    def extract_with_pymupdf(self, page_num: int) -> Dict:
        """Extract text using PyMuPDF"""
        page = self.doc[page_num]
        
        # Extract text
        text = page.get_text()
        
        # Extract tables if present
        tables = []
        try:
            # PyMuPDF table extraction (if available in version)
            tabs = page.find_tables()
            if tabs:
                for tab in tabs:
                    tables.append(tab.extract())
        except:
            pass
        
        # Get page metadata
        metadata = {
            "page_num": page_num + 1,
            "width": page.rect.width,
            "height": page.rect.height,
            "rotation": page.rotation
        }
        
        return {
            "text": text,
            "tables": tables,
            "metadata": metadata
        }
    
    def page_to_image(self, page_num: int, dpi: int = 150) -> Image.Image:
        """Convert PDF page to PIL Image"""
        page = self.doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img_data = pix.tobytes("png")
        return Image.open(BytesIO(img_data))
    
    def detect_content_type(self, page_num: int) -> str:
        """Detect the type of content on the page"""
        page = self.doc[page_num]
        text = page.get_text().lower()
        
        # Simple heuristic for content type detection
        if any(word in text for word in ["table", "표", "테이블"]):
            return "table"
        elif any(word in text for word in ["chart", "graph", "차트", "그래프"]):
            return "chart"
        elif any(word in text for word in ["diagram", "flow", "다이어그램", "플로우"]):
            return "diagram"
        elif any(word in text for word in ["금융", "financial", "계좌", "거래"]):
            return "financial"
        else:
            return "general"

# ============================================================================
# Comparison Engine
# ============================================================================

class ExtractionComparator:
    """Compares VL and PyMuPDF extraction results"""
    
    def __init__(self):
        self.results = []
        
    def compare_extractions(self, vl_text: str, pymupdf_text: str) -> Dict:
        """Compare two extraction results"""
        from difflib import SequenceMatcher
        
        # Calculate similarity
        similarity = SequenceMatcher(None, vl_text, pymupdf_text).ratio()
        
        # Count statistics
        vl_chars = len(vl_text)
        pymupdf_chars = len(pymupdf_text)
        vl_lines = vl_text.count('\n') + 1
        pymupdf_lines = pymupdf_text.count('\n') + 1
        
        # Find unique content
        vl_words = set(vl_text.split())
        pymupdf_words = set(pymupdf_text.split())
        unique_to_vl = vl_words - pymupdf_words
        unique_to_pymupdf = pymupdf_words - vl_words
        
        return {
            "similarity": similarity,
            "vl_stats": {
                "chars": vl_chars,
                "lines": vl_lines,
                "words": len(vl_words)
            },
            "pymupdf_stats": {
                "chars": pymupdf_chars,
                "lines": pymupdf_lines,
                "words": len(pymupdf_words)
            },
            "unique_to_vl": len(unique_to_vl),
            "unique_to_pymupdf": len(unique_to_pymupdf)
        }
    
    def add_result(self, page_num: int, content_type: str, vl_result: str,
                   pymupdf_result: Dict, comparison: Dict, timing: Dict):
        """Add comparison result"""
        self.results.append({
            "page": page_num + 1,
            "content_type": content_type,
            "vl_extraction": vl_result[:500] + "..." if len(vl_result) > 500 else vl_result,
            "pymupdf_extraction": pymupdf_result["text"][:500] + "..." if len(pymupdf_result["text"]) > 500 else pymupdf_result["text"],
            "comparison": comparison,
            "timing": timing,
            "metadata": pymupdf_result["metadata"]
        })
    
    def generate_report(self) -> Dict:
        """Generate comprehensive comparison report"""
        if not self.results:
            return {"error": "No results to report"}
        
        # Calculate aggregates
        avg_similarity = np.mean([r["comparison"]["similarity"] for r in self.results])
        total_vl_time = sum(r["timing"]["vl_time"] for r in self.results)
        total_pymupdf_time = sum(r["timing"]["pymupdf_time"] for r in self.results)
        
        # Content type distribution
        content_types = {}
        for r in self.results:
            ct = r["content_type"]
            if ct not in content_types:
                content_types[ct] = 0
            content_types[ct] += 1
        
        return {
            "summary": {
                "total_pages": len(self.results),
                "average_similarity": avg_similarity,
                "total_vl_time": total_vl_time,
                "total_pymupdf_time": total_pymupdf_time,
                "speed_ratio": total_vl_time / total_pymupdf_time if total_pymupdf_time > 0 else 0,
                "content_types": content_types
            },
            "detailed_results": self.results,
            "metadata": {
                "test_date": datetime.now().isoformat(),
                "model": Config.MODEL_NAME,
                "quantization": "4-bit" if Config.USE_4BIT else "fp16",
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        }

# ============================================================================
# Main Test Function
# ============================================================================

def run_comparison_test():
    """Run the complete comparison test"""
    print("=" * 80)
    print("Real Qwen2.5-VL Model PDF Extraction Comparison Test")
    print("=" * 80)
    
    # Initialize components
    vl_manager = VLModelManager(use_4bit=Config.USE_4BIT)
    comparator = ExtractionComparator()
    
    # Load VL model
    print("\n1. Loading VL Model...")
    if not vl_manager.load_model():
        print("Failed to load VL model. Exiting.")
        return
    
    # Check if test PDF exists
    if not os.path.exists(Config.TEST_PDF_PATH):
        print(f"\nTest PDF not found: {Config.TEST_PDF_PATH}")
        print("Creating a simple test with available PDFs...")
        # Try to find any PDF in the data directory
        pdf_files = list(Path("../data").rglob("*.pdf"))
        if pdf_files:
            Config.TEST_PDF_PATH = str(pdf_files[0])
            print(f"Using: {Config.TEST_PDF_PATH}")
        else:
            print("No PDF files found. Please provide a test PDF.")
            return
    
    # Process PDF
    print(f"\n2. Processing PDF: {Config.TEST_PDF_PATH}")
    pdf_processor = PDFProcessor(Config.TEST_PDF_PATH)
    
    try:
        num_pages = pdf_processor.open()
        pages_to_process = min(num_pages, Config.MAX_PAGES)
        print(f"Total pages: {num_pages}, Processing: {pages_to_process}")
        
        # Select appropriate prompts
        prompt_map = {
            "general": GENERAL_EXTRACTION_PROMPT,
            "table": TABLE_EXTRACTION_PROMPT,
            "chart": CHART_EXTRACTION_PROMPT,
            "diagram": DIAGRAM_EXTRACTION_PROMPT,
            "financial": FINANCIAL_DOC_PROMPT
        }
        
        # Process each page
        print("\n3. Extracting and Comparing...")
        for page_num in tqdm(range(pages_to_process), desc="Processing pages"):
            print(f"\n--- Page {page_num + 1} ---")
            
            # Detect content type
            content_type = pdf_processor.detect_content_type(page_num)
            print(f"Content type: {content_type}")
            
            # PyMuPDF extraction
            start_time = time.time()
            pymupdf_result = pdf_processor.extract_with_pymupdf(page_num)
            pymupdf_time = time.time() - start_time
            
            # Convert page to image for VL model
            page_image = pdf_processor.page_to_image(page_num, Config.DPI)
            
            # VL model extraction
            prompt = prompt_map.get(content_type, GENERAL_EXTRACTION_PROMPT)
            start_time = time.time()
            vl_result = vl_manager.extract_from_image(page_image, prompt)
            vl_time = time.time() - start_time
            
            # Compare results
            comparison = comparator.compare_extractions(vl_result, pymupdf_result["text"])
            
            # Store results
            comparator.add_result(
                page_num=page_num,
                content_type=content_type,
                vl_result=vl_result,
                pymupdf_result=pymupdf_result,
                comparison=comparison,
                timing={"vl_time": vl_time, "pymupdf_time": pymupdf_time}
            )
            
            # Print comparison summary
            print(f"Similarity: {comparison['similarity']:.2%}")
            print(f"VL extraction time: {vl_time:.2f}s")
            print(f"PyMuPDF extraction time: {pymupdf_time:.4f}s")
            print(f"VL unique words: {comparison['unique_to_vl']}")
            
            # Clear cache periodically
            if (page_num + 1) % Config.CLEAR_CACHE_EVERY_N_PAGES == 0:
                vl_manager.clear_cache()
                print("Cache cleared")
        
        # Generate report
        print("\n4. Generating Report...")
        report = comparator.generate_report()
        
        # Save report
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        report_path = os.path.join(Config.RESULTS_DIR, Config.REPORT_FILE)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Pages processed: {report['summary']['total_pages']}")
        print(f"Average similarity: {report['summary']['average_similarity']:.2%}")
        print(f"Total VL time: {report['summary']['total_vl_time']:.2f}s")
        print(f"Total PyMuPDF time: {report['summary']['total_pymupdf_time']:.2f}s")
        print(f"Speed ratio (VL/PyMuPDF): {report['summary']['speed_ratio']:.1f}x")
        print(f"Content types: {report['summary']['content_types']}")
        print(f"\nReport saved to: {report_path}")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        traceback.print_exc()
    
    finally:
        pdf_processor.close()
        vl_manager.clear_cache()
        
        # Final memory report
        print("\n5. Final Memory Report")
        vl_manager._log_memory_usage("Test completion")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Check CUDA availability
    print("System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    
    # Run test
    run_comparison_test()