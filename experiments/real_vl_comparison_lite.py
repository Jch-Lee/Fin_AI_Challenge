#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightweight Real VL Model Test - Memory Optimized Version
Tests Qwen2.5-VL with aggressive memory optimization
"""

import os
import sys
import json
import time
import gc
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch
import pymupdf
from PIL import Image
from io import BytesIO

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================================================================
# Lightweight Test Configuration
# ============================================================================

class LiteConfig:
    """Lightweight test configuration"""
    # Model settings - using smallest VL model available
    MODEL_OPTIONS = [
        "Qwen/Qwen2-VL-2B-Instruct",  # Smallest option if available
        "Qwen/Qwen2.5-VL-7B-Instruct",  # Standard model
    ]
    
    # Aggressive optimization settings
    USE_8BIT = True  # Try 8-bit first, fallback to 4-bit
    USE_CPU_OFFLOAD = True  # Offload to CPU if needed
    MAX_NEW_TOKENS = 256  # Reduced token limit
    
    # Test settings - very limited for memory
    MAX_PAGES = 1  # Test only 1 page
    DPI = 100  # Lower resolution
    MAX_IMAGE_SIZE = (1024, 1024)  # Smaller images
    
    # Batch settings
    BATCH_SIZE = 1  # Process one at a time

# ============================================================================
# Minimal VL Model Wrapper
# ============================================================================

class MinimalVLModel:
    """Minimal VL model wrapper with aggressive memory management"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        
    def try_load_model(self) -> bool:
        """Try to load model with various fallback strategies"""
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        for model_name in LiteConfig.MODEL_OPTIONS:
            print(f"\nAttempting to load: {model_name}")
            
            # Try different loading strategies
            strategies = [
                ("8-bit quantization", self._load_8bit),
                ("4-bit quantization", self._load_4bit),
                ("CPU offload", self._load_cpu_offload),
                ("CPU only", self._load_cpu)
            ]
            
            for strategy_name, strategy_func in strategies:
                try:
                    print(f"  Trying {strategy_name}...")
                    if strategy_func(model_name):
                        print(f"  ✓ Successfully loaded with {strategy_name}")
                        self.model_loaded = True
                        return True
                except Exception as e:
                    print(f"  ✗ {strategy_name} failed: {str(e)[:100]}")
                    self._cleanup()
                    continue
        
        print("\nAll loading strategies failed.")
        return False
    
    def _load_8bit(self, model_name: str) -> bool:
        """Try loading with 8-bit quantization"""
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
        
        if self.device != "cuda":
            return False
            
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        return True
    
    def _load_4bit(self, model_name: str) -> bool:
        """Try loading with 4-bit quantization"""
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
        
        if self.device != "cuda":
            return False
            
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        return True
    
    def _load_cpu_offload(self, model_name: str) -> bool:
        """Try loading with CPU offloading"""
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        if self.device != "cuda":
            return False
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            offload_folder="offload",
            offload_state_dict=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        return True
    
    def _load_cpu(self, model_name: str) -> bool:
        """Try loading on CPU only"""
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        # This will be very slow but might work
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.device = "cpu"
        return True
    
    def _cleanup(self):
        """Clean up memory"""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def extract_from_image(self, image: Image.Image, prompt: str) -> str:
        """Extract text from image - minimal version"""
        if not self.model_loaded:
            return "[Model not loaded]"
        
        try:
            # Resize image aggressively
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            # Prepare input
            if hasattr(self.processor, 'image_processor'):
                # VL model path
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                )
            else:
                # Fallback for text-only model
                inputs = self.processor(
                    prompt,
                    return_tensors="pt"
                )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                         for k, v in inputs.items()}
            
            # Generate with minimal tokens
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=LiteConfig.MAX_NEW_TOKENS,
                    temperature=0.1,
                    do_sample=False,  # Greedy for consistency
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # Decode
            result = self.processor.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Clean up intermediate tensors
            del inputs, outputs
            gc.collect()
            
            return result
            
        except Exception as e:
            return f"[Extraction error: {str(e)[:100]}]"

# ============================================================================
# Simple Test Runner
# ============================================================================

def run_minimal_test():
    """Run minimal comparison test"""
    print("=" * 60)
    print("Minimal VL Model Test - Memory Optimized")
    print("=" * 60)
    
    # System info
    print("\nSystem Information:")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {props.total_memory / 1024**3:.2f}GB total")
        print(f"Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated")
    
    # Initialize model
    print("\n1. Loading Model...")
    model = MinimalVLModel()
    
    if not model.try_load_model():
        print("\nCould not load VL model. Trying alternative approach...")
        # Alternative: just test PyMuPDF extraction
        test_pymupdf_only()
        return
    
    # Find a test PDF
    print("\n2. Finding test PDF...")
    pdf_files = list(Path(".").parent.rglob("*.pdf"))[:5]  # Limit search
    
    if not pdf_files:
        print("No PDF files found.")
        return
    
    test_pdf = pdf_files[0]
    print(f"Using: {test_pdf}")
    
    # Test extraction
    print("\n3. Testing Extraction...")
    try:
        doc = pymupdf.open(str(test_pdf))
        
        # Process first page only
        page = doc[0]
        
        # PyMuPDF extraction
        print("\nPyMuPDF extraction:")
        text = page.get_text()
        print(f"  Extracted {len(text)} characters")
        print(f"  First 200 chars: {text[:200]}...")
        
        # Convert to image
        pix = page.get_pixmap(dpi=LiteConfig.DPI)
        img_data = pix.tobytes("png")
        image = Image.open(BytesIO(img_data))
        
        # VL extraction
        print("\nVL Model extraction:")
        prompt = "Extract all text from this image. Output only the text content."
        vl_text = model.extract_from_image(image, prompt)
        print(f"  Extracted {len(vl_text)} characters")
        print(f"  First 200 chars: {vl_text[:200]}...")
        
        # Simple comparison
        print("\n4. Comparison:")
        if len(vl_text) > 50:  # Got meaningful extraction
            # Calculate overlap
            vl_words = set(vl_text.lower().split())
            pdf_words = set(text.lower().split())
            overlap = vl_words & pdf_words
            
            print(f"  VL words: {len(vl_words)}")
            print(f"  PDF words: {len(pdf_words)}")
            print(f"  Common words: {len(overlap)}")
            print(f"  Overlap ratio: {len(overlap) / max(len(vl_words), len(pdf_words), 1):.2%}")
        else:
            print("  VL extraction too short for meaningful comparison")
        
        doc.close()
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        model._cleanup()
        print("\n5. Test complete!")

def test_pymupdf_only():
    """Test only PyMuPDF extraction as fallback"""
    print("\n=== PyMuPDF-Only Test ===")
    
    # Find PDFs
    pdf_files = list(Path(".").parent.rglob("*.pdf"))[:3]
    
    if not pdf_files:
        print("No PDF files found")
        return
    
    for pdf_path in pdf_files:
        print(f"\nTesting: {pdf_path.name}")
        try:
            doc = pymupdf.open(str(pdf_path))
            page = doc[0]
            
            # Extract text
            text = page.get_text()
            print(f"  Text: {len(text)} chars")
            
            # Try table extraction
            tables = page.find_tables()
            print(f"  Tables found: {len(tables)}")
            
            # Get images
            image_list = page.get_images()
            print(f"  Images found: {len(image_list)}")
            
            doc.close()
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nPyMuPDF test complete. VL model comparison requires more memory.")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Set memory fraction for GPU
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
    
    # Run test
    run_minimal_test()