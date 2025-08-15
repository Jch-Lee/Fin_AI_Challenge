# Vision-Language Model Testing Guide

This directory contains test scripts for comparing Qwen2.5-VL model extraction with traditional PyMuPDF extraction.

## Test Scripts

### 1. `real_vl_comparison.py` - Full Test
**Purpose**: Comprehensive comparison using actual Qwen2.5-VL model
**Requirements**: 
- GPU with at least 8GB VRAM (24GB recommended)
- `transformers`, `bitsandbytes`, `accelerate` libraries
- Qwen2.5-VL model weights will be downloaded (~14GB)

**Features**:
- Loads actual Qwen2.5-VL-7B model with 4-bit quantization
- Processes multiple PDF pages
- Uses specialized prompts for different content types (tables, charts, diagrams)
- Generates detailed comparison report with metrics
- Memory management and cache clearing

**Usage**:
```bash
python real_vl_comparison.py
```

**Configuration** (in script):
```python
class Config:
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    USE_4BIT = True  # Enable 4-bit quantization
    MAX_PAGES = 5    # Number of pages to test
    TEST_PDF_PATH = "../data/pdfs/financial_sample.pdf"
```

### 2. `real_vl_comparison_lite.py` - Lightweight Test
**Purpose**: Memory-optimized version for systems with limited resources
**Requirements**:
- Works with 4GB+ VRAM or CPU-only
- Attempts multiple loading strategies

**Features**:
- Tries multiple quantization levels (8-bit, 4-bit)
- CPU offloading support
- Processes single page with reduced resolution
- Fallback to PyMuPDF-only testing if model can't load
- Aggressive memory management

**Usage**:
```bash
python real_vl_comparison_lite.py
```

### 3. `vl_prompt_example.py` - Prompt Templates
**Purpose**: Defines optimized prompts for different content types
**No model loading** - just prompt definitions

**Available Prompts**:
- `GENERAL_EXTRACTION_PROMPT` - General text extraction
- `TABLE_EXTRACTION_PROMPT` - Table-specific extraction
- `CHART_EXTRACTION_PROMPT` - Chart/graph data extraction
- `DIAGRAM_EXTRACTION_PROMPT` - Diagram text extraction
- `FINANCIAL_DOC_PROMPT` - Financial document extraction

## Expected Output

### Full Test (`real_vl_comparison.py`)
Generates `results/real_vl_comparison_report.json`:
```json
{
  "summary": {
    "total_pages": 5,
    "average_similarity": 0.85,
    "total_vl_time": 25.3,
    "total_pymupdf_time": 0.15,
    "speed_ratio": 168.7,
    "content_types": {
      "table": 2,
      "general": 3
    }
  },
  "detailed_results": [...],
  "metadata": {
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "quantization": "4-bit",
    "device": "cuda"
  }
}
```

### Lightweight Test (`real_vl_comparison_lite.py`)
Console output showing:
- Loading strategy attempts
- Basic extraction comparison
- Word overlap statistics

## Memory Requirements

| Script | GPU Memory | System RAM | Processing Time |
|--------|------------|------------|-----------------|
| `real_vl_comparison.py` (4-bit) | ~6-8 GB | ~16 GB | ~5s/page |
| `real_vl_comparison.py` (fp16) | ~15 GB | ~32 GB | ~3s/page |
| `real_vl_comparison_lite.py` | ~4 GB | ~8 GB | ~10s/page |
| PyMuPDF only | None | ~2 GB | <0.1s/page |

## Troubleshooting

### Out of Memory Errors
1. Use `real_vl_comparison_lite.py` instead
2. Reduce `MAX_PAGES` and `DPI` settings
3. Enable more aggressive quantization
4. Use CPU offloading (slower but works)

### Model Loading Failures
1. Check internet connection (first run downloads model)
2. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
3. Try smaller model variants if available
4. Use CPU-only mode as last resort

### Poor Extraction Quality
1. Increase DPI for page-to-image conversion
2. Use content-specific prompts
3. Adjust temperature (lower = more consistent)
4. Check if images are clear and readable

## Performance Comparison

Based on testing, expect:
- **PyMuPDF**: Fast (<0.1s/page), good for standard text
- **VL Model**: Slower (3-10s/page), better for:
  - Complex layouts
  - Charts and diagrams
  - Handwritten text
  - Non-standard formatting
  - Multi-language content

## Integration with RAG Pipeline

These tests validate the VL model approach for the competition pipeline:
1. **Baseline**: PyMuPDF for standard text extraction
2. **Enhancement**: VL model for complex visual content
3. **Hybrid**: Use both based on content detection
4. **Optimization**: Cache VL results for frequently accessed content

## Next Steps

1. Run lightweight test first to verify setup
2. If successful, try full test with sample PDFs
3. Analyze report to determine if VL enhancement is worth the overhead
4. Integrate best approach into main RAG pipeline