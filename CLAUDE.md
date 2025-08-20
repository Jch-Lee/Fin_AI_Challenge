# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Financial Security AI Competition** project focused on building a Korean LLM system to answer financial security questions using the **Distill-M 2 (Contrastive Distillation)** approach. The system handles both multiple-choice and open-ended questions about financial security topics.

### Key Constraints
- **Single Model**: Only one LLM allowed (no ensemble)
- **Open Source**: Must use models with non-commercial licenses published before Aug 1, 2025
- **Offline Environment**: Final inference runs without internet connectivity
- **Performance**: 4.5 hours max for full dataset inference on RTX 4090 (24GB VRAM)
- **Architecture**: Student-Teacher distillation using RAG (Retrieval-Augmented Generation)

## Development Commands

### Environment Setup
```bash
# Install all project dependencies (recommended)
pip install -r requirements.txt

# For baseline reference only
pip install -r baseline_code/requirements.txt
```

**Important**: Use the root `requirements.txt` for development. The `baseline_code/requirements.txt` is for reference only.

### Running Baseline Models
```bash
# Quick test (first 10 samples)
python baseline_code/run_baseline_quick.py

# Full baseline inference
python baseline_code/run_baseline.py

# Model used: beomi/gemma-ko-7b with 4-bit quantization
```

### Key Files and Data Flow
```bash
# Input data
./test.csv              # Questions with ID and Question columns  
./sample_submission.csv # Expected output format (ID, Answer)

# Outputs
./baseline_submission.csv    # Full baseline results
./quick_test_result.csv     # Quick test results (10 samples)
```

## Code Architecture

### Core Components

#### 1. Question Processing Pipeline
- **Question Classification**: `is_multiple_choice()` - Detects if question has numbered options (≥2)
- **Content Parsing**: `extract_question_and_choices()` - Separates question text from multiple choice options
- **Prompt Generation**: `make_prompt_auto()` - Creates role-specific prompts for financial security expert

#### 2. Model Architecture Strategy
**Current Baseline**: Single model approach using `beomi/gemma-ko-7b`
**Planned Architecture** (from 아키텍처.md):
- **Student Models**: Mistral-7B-Instruct, Qwen2.5-1.5B-Instruct (검증 완료)
- **Teacher Models**: Llama-3.1-70B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct (검증 완료)
- **Training Method**: Distill-M 2 contrastive distillation
- **Optimization**: 4-bit quantization with BitsAndBytesConfig

#### 3. RAG System Design (Planned)
```
Data Pipeline: External Data → Preprocessing → FAISS Index + Fine-tuning Dataset
Training: Teacher/Student Response Generation → Data Reformatting → DistiLLM-2 Training  
Inference: Question → Multi-Stage Retrieval → Quantized Student Model → Answer
```

#### 4. Answer Extraction Logic
- **Multiple Choice**: Extract numerical answer (1-99) from generated text
- **Open-ended**: Return full descriptive text
- **Error Handling**: Default to "1" for MC, "미응답" for open-ended

### Question Types and Response Strategy

#### Multiple Choice Questions
```python
# Pattern detection: Lines starting with digits followed by space
# Prompt: "정답 선택지 번호만 출력하세요"
# Expected output: Single number (e.g., "3")
```

#### Open-ended Questions  
```python
# No numbered options detected
# Prompt: "정확하고 간략한 설명을 작성하세요"
# Expected output: Descriptive text explanation
```

### Performance Optimization

#### Memory Management
- **4-bit Quantization**: `load_in_4bit=True` with `bnb_4bit_quant_type="nf4"`
- **Device Mapping**: `device_map="auto"` for optimal GPU utilization
- **Token Limits**: `max_new_tokens=64` to control generation length

#### Inference Parameters
```python
temperature=0.3      # Low for consistency
top_p=0.8           # Focused sampling
do_sample=True      # Enable sampling
pad_token_id=tokenizer.eos_token_id
```

## Development Workflow

### 0. File Creation Guidelines
**IMPORTANT**: When creating any files or directories:
- **No Emojis**: Never use emojis in file names, directory names, or file contents
- **Clean Naming**: Use only alphanumeric characters, hyphens, and underscores
- **Professional Format**: Maintain clean, professional file structure without decorative symbols

### 1. Git Workflow Requirements
**MANDATORY**: All development must follow the GitFlow workflow as defined in `docs/git-workflow/`:

#### Branch Strategy
- **main**: Production-ready releases only
- **develop**: Integration branch for features
- **feature branches**: `feat/story-[번호]/[기능명]` (e.g., `feat/story-1.1/init-project`)
- **bugfix branches**: `fix/[이슈번호]/[버그명]`
- **hotfix branches**: `hotfix/[버전]/[수정내용]`

#### Development Process
1. **Start New Feature**: Always branch from `develop`
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feat/story-1.1/feature-name
   ```

2. **Development**: Make small, frequent commits with conventional messages
   ```bash
   git add .
   git commit -m "feat: add new functionality"
   ```

3. **Integration**: Create PR to merge feature → develop
   - Required: Code review before merge
   - Clean up: Delete feature branch after successful merge

4. **Release**: Merge develop → main for stable releases

#### Commit Message Convention
- `feat:` 새로운 기능 추가
- `fix:` 버그 수정
- `docs:` 문서 수정
- `style:` 코드 포맷팅
- `refactor:` 코드 리팩토링
- `test:` 테스트 추가/수정
- `chore:` 빌드 업무, 패키지 관리

### 2. Model Development Pipeline
Based on 상세파이프라인.md, follow this sequence:
1. **Data Preprocessing**: Document parsing, chunking, embedding
2. **Knowledge Base**: FAISS index construction  
3. **Synthetic Data**: Teacher model Q&A generation
4. **Training**: Distill-M 2 contrastive distillation
5. **Optimization**: Model quantization and deployment

### 3. Testing Strategy
```bash
# Always test with quick version first
python baseline_code/run_baseline_quick.py

# Monitor GPU memory usage
nvidia-smi

# Validate output format matches sample_submission.csv
```

### 4. Competition Submission Format
- **Input**: `test.csv` with columns [ID, Question]
- **Output**: CSV with columns [ID, Answer] 
- **Answer Format**: Numbers for MC, Korean text for open-ended
- **File Encoding**: UTF-8 with BOM (`encoding='utf-8-sig'`)

## Technical Stack

### Core Libraries
- **PyTorch**: 2.1.0 (competition requirement)
- **Transformers**: 4.41.2 for LLM loading
- **Accelerate**: 0.30.1 for optimization
- **TRL**: 0.9.6 for DistiLLMTrainer
- **vLLM**: 0.5.4 for response generation

### Planned Additional Components
- **RAG**: FAISS-CPU, sentence-transformers, BM25
- **Korean NLP**: KoNLPy for morphological analysis
- **Document Processing**: PyMuPDF, LangChain
- **Monitoring**: Weights & Biases, tqdm

## Error Handling and Robustness

### Exception Management
```python
# Always wrap inference in try-catch
try:
    # Model inference
    output = pipe(prompt, ...)
    pred_answer = extract_answer_only(output[0]["generated_text"], question)
except Exception as e:
    # Fallback logic
    if is_multiple_choice(question):
        pred_answer = "1"  # Default MC answer
    else:
        pred_answer = "답변 생성 실패"  # Default descriptive answer
```

### Model Loading Safety
- Check GPU availability with `torch.cuda.is_available()`
- Implement CPU fallback for model loading
- Set proper tokenizer pad tokens: `tokenizer.pad_token = tokenizer.eos_token`

## Competition-Specific Notes

### Evaluation Metrics
- **Multiple Choice**: Exact match accuracy
- **Open-ended**: Semantic similarity (likely ROUGE/BERTScore)
- **Combined**: FSKU evaluation metric

### Resource Constraints
- **GPU Memory**: 24GB VRAM limit
- **Inference Time**: 4.5 hours maximum
- **Storage**: 40GB total disk space
- **Environment**: Ubuntu 22.04, Python 3.10, CUDA 11.8

### Critical Success Factors
1. **Model Size**: Balance performance vs. memory constraints
2. **Question Classification**: Accurate MC vs. open-ended detection
3. **Korean Language**: Proper handling of Korean text encoding and morphology
4. **Prompt Engineering**: Effective role-based prompts for financial security domain
5. **Error Recovery**: Robust fallback mechanisms for edge cases

## Future Development Directions

The baseline provides a foundation, but the full system will implement:
- **RAG Integration**: Knowledge base construction and retrieval
- **Teacher-Student Training**: Distill-M 2 contrastive distillation  
- **Multi-stage Optimization**: Progressive model refinement
- **Advanced Inference**: Caching, parallel processing, timeout handling

This architecture balances competitive performance requirements with practical development constraints.

## Key Reference Documents

### Core Development Guidelines
The following documents in the `docs/` folder provide essential guidance for development:

1. **`docs/Architecture.md`**: System architecture and component interface definitions
   - 10 core component specifications with ABC interfaces
   - Data models and type definitions
   - Tech stack and model selection strategy
   
2. **`docs/Pipeline.md`**: Detailed development pipeline with completion criteria
   - Epic 1-3 step-by-step tasks
   - Specific completion criteria for each task
   - Offline environment validation procedures
   
3. **`docs/PROJECT_PLAN.md`**: 3-week development schedule and priority management
   - Weekly goals and daily task planning
   - P0-P3 priority matrix
   - Risk management and success metrics

### Development Workflow
Always refer to these documents when:
- Implementing new components (check Architecture.md interfaces)
- Verifying task completion (check Pipeline.md criteria)
- Planning daily work (check PROJECT_PLAN.md priorities)
- 프로젝트에 대한 문서들은 docs/ 에서 참고하고, 업데이트할 것.