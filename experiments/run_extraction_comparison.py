#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real extraction comparison experiment between traditional PyMuPDF and Qwen2.5-VL model.
This script performs actual extraction using both methods and saves results for comparison.
"""

import os
import sys
import json
import time
import logging
import torch
import gc
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import our existing PDF processor
from packages.preprocessing.pdf_processor_advanced import AdvancedPDFProcessor

# Import transformers for VL model
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from PIL import Image
import fitz  # PyMuPDF for image extraction

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionMetrics:
    """Metrics for extraction comparison"""
    method: str
    extraction_time: float
    token_count: int
    char_count: int
    line_count: int
    table_count: int
    memory_usage_mb: float
    success: bool
    error_message: str = ""
    
    
@dataclass 
class ComparisonResults:
    """Complete comparison results"""
    pdf_path: str
    pdf_pages: int
    pdf_size_mb: float
    traditional_metrics: ExtractionMetrics
    vl_metrics: ExtractionMetrics
    content_similarity: float
    timestamp: str


class VLModelExtractor:
    """Vision-Language model based PDF extractor using Qwen2.5-VL"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-2B-Instruct", use_4bit: bool = True):
        """Initialize VL model with quantization"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing VL model: {model_name} on {self.device}")
        
        try:
            # Configure 4-bit quantization
            if use_4bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            else:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
            
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            logger.info("VL model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load VL model: {e}")
            raise
            
    def extract_from_pdf(self, pdf_path: str, max_pages: int = 10) -> Tuple[str, List[Dict]]:
        """
        Extract text from PDF using VL model
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to process (for memory management)
            
        Returns:
            Tuple of (extracted_text, page_results)
        """
        logger.info(f"Extracting from PDF using VL model: {pdf_path}")
        
        # Open PDF with PyMuPDF for image extraction
        doc = fitz.open(pdf_path)
        total_pages = min(len(doc), max_pages)
        
        all_text = []
        page_results = []
        
        for page_num in range(total_pages):
            logger.info(f"Processing page {page_num + 1}/{total_pages}")
            
            try:
                # Convert page to image
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img_data = pix.pil_tobytes(format="PNG")
                image = Image.open(io.BytesIO(img_data))
                
                # Prepare prompt for VL model
                prompt = """Please extract and transcribe ALL text content from this document page.
                Include:
                - All headings and titles
                - All body text and paragraphs  
                - Table contents (format as structured text)
                - List items and bullet points
                - Any captions or footnotes
                
                Maintain the original structure and formatting as much as possible.
                Extract the complete text content:"""
                
                # Process with VL model
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # Generate text
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = self.processor(
                    text=text,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                # Decode output
                output_text = self.processor.decode(
                    outputs[0], 
                    skip_special_tokens=True
                )
                
                # Extract only the generated response
                if prompt in output_text:
                    output_text = output_text.split(prompt)[-1].strip()
                
                all_text.append(f"--- Page {page_num + 1} ---\n{output_text}")
                
                page_results.append({
                    "page": page_num + 1,
                    "text": output_text,
                    "success": True
                })
                
                # Clear GPU memory
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {e}")
                page_results.append({
                    "page": page_num + 1,
                    "text": "",
                    "success": False,
                    "error": str(e)
                })
                
        doc.close()
        
        return "\n\n".join(all_text), page_results


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    else:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


def calculate_text_metrics(text: str) -> Dict[str, int]:
    """Calculate basic text metrics"""
    return {
        "char_count": len(text),
        "line_count": text.count('\n'),
        "word_count": len(text.split()),
        "paragraph_count": text.count('\n\n')
    }


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity (character overlap ratio)"""
    if not text1 or not text2:
        return 0.0
    
    # Simple character-based similarity
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
        
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def run_traditional_extraction(pdf_path: str) -> Tuple[str, ExtractionMetrics]:
    """Run traditional PyMuPDF extraction"""
    logger.info("Starting traditional extraction...")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        # Use our existing advanced PDF processor
        processor = AdvancedPDFProcessor(
            use_markdown=True,
            extract_tables=True,
            preserve_layout=True
        )
        
        result = processor.extract_pdf(pdf_path)
        
        extraction_time = time.time() - start_time
        memory_used = get_memory_usage() - start_memory
        
        metrics = ExtractionMetrics(
            method="Traditional (PyMuPDF)",
            extraction_time=extraction_time,
            token_count=len(result.text.split()),
            char_count=len(result.text),
            line_count=result.text.count('\n'),
            table_count=len(result.tables),
            memory_usage_mb=memory_used,
            success=True
        )
        
        return result.text, metrics
        
    except Exception as e:
        logger.error(f"Traditional extraction failed: {e}")
        return "", ExtractionMetrics(
            method="Traditional (PyMuPDF)",
            extraction_time=time.time() - start_time,
            token_count=0,
            char_count=0,
            line_count=0,
            table_count=0,
            memory_usage_mb=0,
            success=False,
            error_message=str(e)
        )


def run_vl_extraction(pdf_path: str, max_pages: int = 5) -> Tuple[str, ExtractionMetrics]:
    """Run VL model extraction"""
    logger.info("Starting VL model extraction...")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        # Initialize VL extractor
        extractor = VLModelExtractor(
            model_name="Qwen/Qwen2.5-VL-2B-Instruct",
            use_4bit=True
        )
        
        # Extract text
        text, page_results = extractor.extract_from_pdf(pdf_path, max_pages=max_pages)
        
        extraction_time = time.time() - start_time
        memory_used = get_memory_usage() - start_memory
        
        # Count successful pages
        successful_pages = sum(1 for p in page_results if p.get("success", False))
        
        metrics = ExtractionMetrics(
            method=f"VL Model (Qwen2.5-VL-2B)",
            extraction_time=extraction_time,
            token_count=len(text.split()),
            char_count=len(text),
            line_count=text.count('\n'),
            table_count=0,  # VL doesn't specifically identify tables
            memory_usage_mb=memory_used,
            success=True,
            error_message=f"Processed {successful_pages}/{len(page_results)} pages"
        )
        
        # Clean up model from memory
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return text, metrics
        
    except Exception as e:
        logger.error(f"VL extraction failed: {e}")
        traceback.print_exc()
        return "", ExtractionMetrics(
            method="VL Model (Qwen2.5-VL-2B)",
            extraction_time=time.time() - start_time,
            token_count=0,
            char_count=0,
            line_count=0,
            table_count=0,
            memory_usage_mb=0,
            success=False,
            error_message=str(e)
        )


def main():
    """Main experiment execution"""
    
    # Setup paths
    pdf_path = PROJECT_ROOT / "data" / "raw" / "금융분야 AI 보안 가이드라인.pdf"
    output_dir = PROJECT_ROOT / "experiments" / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Check if PDF exists
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    logger.info(f"Running extraction comparison on: {pdf_path}")
    logger.info(f"PDF size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Get PDF metadata
    import fitz
    doc = fitz.open(str(pdf_path))
    pdf_pages = len(doc)
    doc.close()
    
    # Run traditional extraction
    traditional_text, traditional_metrics = run_traditional_extraction(str(pdf_path))
    
    # Save traditional output
    with open(output_dir / "traditional_extraction.txt", "w", encoding="utf-8") as f:
        f.write(traditional_text)
    logger.info(f"Traditional extraction saved: {len(traditional_text)} characters")
    
    # Run VL model extraction (limit pages for memory)
    max_pages = min(5, pdf_pages)  # Process first 5 pages for comparison
    vl_text, vl_metrics = run_vl_extraction(str(pdf_path), max_pages=max_pages)
    
    # Save VL output
    with open(output_dir / "vl_model_extraction.txt", "w", encoding="utf-8") as f:
        f.write(vl_text)
    logger.info(f"VL extraction saved: {len(vl_text)} characters")
    
    # Calculate similarity
    similarity = calculate_similarity(traditional_text[:10000], vl_text)
    
    # Create comparison results
    results = ComparisonResults(
        pdf_path=str(pdf_path),
        pdf_pages=pdf_pages,
        pdf_size_mb=pdf_path.stat().st_size / 1024 / 1024,
        traditional_metrics=traditional_metrics,
        vl_metrics=vl_metrics,
        content_similarity=similarity,
        timestamp=datetime.now().isoformat()
    )
    
    # Save comparison metrics
    comparison_dict = {
        "experiment": "PDF Extraction Method Comparison",
        "pdf_info": {
            "path": results.pdf_path,
            "pages": results.pdf_pages,
            "size_mb": results.pdf_size_mb
        },
        "traditional_extraction": asdict(results.traditional_metrics),
        "vl_model_extraction": asdict(results.vl_metrics),
        "comparison": {
            "content_similarity": results.content_similarity,
            "speed_ratio": (
                traditional_metrics.extraction_time / vl_metrics.extraction_time 
                if vl_metrics.extraction_time > 0 else 0
            ),
            "text_length_ratio": (
                traditional_metrics.char_count / vl_metrics.char_count
                if vl_metrics.char_count > 0 else 0
            )
        },
        "timestamp": results.timestamp
    }
    
    with open(output_dir / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(comparison_dict, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION COMPARISON RESULTS")
    print("="*60)
    
    print(f"\nPDF: {pdf_path.name}")
    print(f"Pages: {pdf_pages}, Size: {results.pdf_size_mb:.2f} MB")
    
    print("\n--- Traditional Extraction (PyMuPDF) ---")
    print(f"Time: {traditional_metrics.extraction_time:.2f} seconds")
    print(f"Characters: {traditional_metrics.char_count:,}")
    print(f"Lines: {traditional_metrics.line_count:,}")
    print(f"Tables found: {traditional_metrics.table_count}")
    print(f"Memory used: {traditional_metrics.memory_usage_mb:.2f} MB")
    
    print("\n--- VL Model Extraction (Qwen2.5-VL) ---")
    print(f"Time: {vl_metrics.extraction_time:.2f} seconds")
    print(f"Characters: {vl_metrics.char_count:,}")
    print(f"Lines: {vl_metrics.line_count:,}")
    print(f"Memory used: {vl_metrics.memory_usage_mb:.2f} MB")
    print(f"Note: {vl_metrics.error_message}")
    
    print("\n--- Comparison ---")
    print(f"Content similarity: {similarity:.2%}")
    print(f"Speed difference: Traditional is {comparison_dict['comparison']['speed_ratio']:.2f}x faster")
    print(f"Text completeness: Traditional extracted {comparison_dict['comparison']['text_length_ratio']:.2f}x more text")
    
    print("\n✅ Experiment completed successfully!")
    print(f"Results saved to: {output_dir}")
    

if __name__ == "__main__":
    # Add missing import
    import io
    
    main()