# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Financial Security AI Competition project implementing an 8-bit quantized Korean LLM system with hybrid RAG for answering financial security questions. The system achieved 5.75 seconds/question processing speed with 10.7GB VRAM usage on RTX 4090.

### Competition Constraints
- **Single Model**: No ensemble allowed
- **Open Source**: Models published before Aug 1, 2025
- **Offline Environment**: No internet during inference
- **Performance**: 4.5 hours max for 515 questions on RTX 4090 (24GB VRAM)
- **Memory**: Must fit within 24GB VRAM

## Development Commands

### Main Inference Pipeline

```bash
# Quick test (10 samples) - Always start with this
python scripts/generate_submission_standalone.py \
    --test_mode \
    --num_samples 10 \
    --output_file test_result.csv

# Full dataset inference (515 questions)
python scripts/generate_submission_standalone.py \
    --input_file test.csv \
    --output_file submission.csv \
    --data_dir ./data/rag

# Monitor GPU memory during inference
watch -n 1 nvidia-smi
```

### RAG System Building

```bash
# Build hybrid RAG system (BM25 + FAISS)
python scripts/build_hybrid_rag_2300.py \
    --input_dir ./data/documents \
    --output_dir ./data/rag \
    --chunk_size 2300

# Generate synthetic Q&A data with Teacher model
python scripts/generate_bulk_3000.py \
    --num_questions 3000 \
    --output_file data/synthetic_questions/combined_3000_questions.csv
```

### Environment Setup

```bash
# Create Python 3.10 virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify CUDA setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## High-Level Architecture

### Current Implementation (8-bit Quantized Pipeline)

```
Input Question → Question Classifier → Hybrid RAG Retrieval → LLM Generation → Answer Extraction
                        ↓                      ↓                     ↓
                  MC vs Subjective    BM25 + FAISS Top-3    Qwen2.5-7B-Instruct
                                      (6 contexts total)         (8-bit)
```

### Core Components

1. **Standalone Inference Script** (`scripts/generate_submission_standalone.py`)
   - Self-contained with all dependencies included
   - No package imports for competition compatibility
   - 8-bit quantization using BitsAndBytesConfig
   - Deterministic generation (temperature=0.05, do_sample=False)

2. **Hybrid RAG System**
   - **Documents**: 73 PDFs → 8,756 chunks (2,300 chars each)
   - **BM25**: Kiwi tokenizer for Korean morphological analysis
   - **FAISS**: KURE-v1 embeddings (1024-dim)
   - **Retrieval**: Combined Top-3 (independent selection from each method)

3. **Question Processing**
   - **Classification**: `is_multiple_choice()` detects numbered options
   - **Prompting**: Role-based prompts as financial security expert
   - **MC Prompt**: Direct number output instruction
   - **Subjective Prompt**: 5 specific guidelines for comprehensive answers

4. **Model Configuration**
   - **LLM**: Qwen2.5-7B-Instruct with chat template
   - **Quantization**: 8-bit (10.7GB VRAM usage)
   - **Generation**: max_new_tokens=32 (MC) or 256 (subjective)
   - **Batching**: Single sample processing for stability

### Data Flow

```
data/
├── rag/
│   ├── chunks_2300.json         # 8,756 text chunks
│   ├── bm25_index_2300.pkl      # BM25 sparse index
│   ├── faiss_index_2300.index   # FAISS dense index
│   └── metadata_2300.json       # Chunk metadata
├── synthetic_questions/
│   └── combined_3000_questions.csv  # Teacher-generated Q&A
└── documents/                   # Original 73 PDFs
```

### Critical Implementation Details

1. **Chunking Strategy**: 2,300 characters (not 500) for optimal context
2. **BM25 Method**: Uses `get_top_n()` not `get_scores()`
3. **Temperature**: 0.05 for consistency (not 0.1 or 0.3)
4. **Context Usage**: Full document text without length limits
5. **Vision Processing**: Qwen2.5-VL used during RAG building for image/chart extraction

## Performance Characteristics

- **Processing Speed**: 5.75 seconds/question
- **Total Time**: ~49 minutes for 515 questions
- **Memory Usage**: 10.7GB VRAM (8-bit quantization)
- **Context Window**: 6 documents × 2,300 chars
- **Success Rate**: 100% inference stability

## Testing Strategy

```bash
# Component testing
python tests/unit/preprocessing/test_kiwi_tokenizer.py
python tests/integration/test_rag_complete_system.py

# Performance benchmarking
python tests/benchmarks/kiwi_performance.py
python tests/benchmarks/embedding_benchmark.py
```

## Common Issues and Solutions

1. **CUDA Out of Memory**
   - Ensure 8-bit quantization is enabled
   - Close other GPU processes
   - Use `torch.cuda.empty_cache()` between batches

2. **Slow Inference**
   - Check if model is on GPU: `model.device`
   - Verify CUDA is available: `torch.cuda.is_available()`
   - Monitor GPU utilization with `nvidia-smi`

3. **Korean Text Issues**
   - Ensure UTF-8 encoding: `encoding='utf-8-sig'`
   - Use Kiwi tokenizer for proper morphological analysis
   - Verify font support for Korean characters

## Future Development (Teacher-Student Distillation)

Planned but not yet implemented:
- Teacher: Qwen2.5-14B-Instruct
- Student: Qwen2.5-7B-Instruct (optimized)
- Method: Distill-M 2 contrastive distillation
- Data: 3,000 synthetic Q&A pairs (already generated)

## Key Files Reference

- `scripts/generate_submission_standalone.py`: Main inference pipeline
- `scripts/build_hybrid_rag_2300.py`: RAG system builder
- `scripts/generate_bulk_3000.py`: Synthetic data generator
- `data/rag/`: Pre-built indexes and chunks
- `docs/CURRENT_STATUS_2025_08_23.md`: Latest implementation status
- `docs/performance_record.md`: Detailed performance metrics