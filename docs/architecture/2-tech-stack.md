# 2. Tech Stack

| Category | Technology | Version | Purpose |
| --- | --- | --- | --- |
| **Language** | Python | `3.10` | 주요 개발 언어 |
| **ML Framework** | PyTorch | `2.1.0` | 딥러닝 프레임워크 |
| **Core ML** | Transformers | `4.41.2` | LLM 로드/사용 |
| **Optimization** | Accelerate | `0.30.1` | 학습/추론 최적화 |
| **Response Gen** | vllm | `0.5.4` | 교사/학생 응답 생성 |
| **Fine-Tuning** | TRL | `0.9.6` | `DistiLLMTrainer` 사용 |
| **Embedding** | sentence-transformers | `2.7.0` | 벡터 임베딩 |
| **Vector DB** | FAISS-CPU | `1.8.0` | 벡터 검색 |
| **Quantization** | auto-gptq / bitsandbytes | `0.7.1`/`0.43.1` | GPTQ/QLoRA 양자화 |
| **Data/Doc Proc** | Pandas, LangChain, PyMuPDF | `2.2.2`,`0.2.1`,`1.24.1` | 데이터/문서 처리 |
| **Search** | bm25s | `0.2.2` | Sparse 검색 |
| **Korean NLP** | konlpy | `0.6.0` | 한국어 형태소 분석 |
| **Monitoring** | tqdm, wandb | `4.66.4`,`0.17.0` | 진행 표시, 실험 추적 |
| **Testing** | pytest | `8.2.0` | 단위/통합 테스트 |

## Model Selection Strategy

- **Student Model Candidates:** `mistralai/Mistral-7B-Instruct-v0.2`, `Solar-10.7B-Instruct` , `Qwen2.5-1.5B-Instruct`
- **Teacher Model Candidates:** `Meta-Llama-3.1-70B-Instruct` , `Qwen2.5-7B-Instruct`

### Model License Verification Process

**대회 규칙 2) 준수를 위한 필수 검증 절차**:

| Model | License | Release Date | Verification Status |
|-------|---------|--------------|-------------------|
| `mistralai/Mistral-7B-Instruct-v0.2` | Apache 2.0 | 2024-03-24 | ✅ 검증 완료 |
| `Solar-10.7B-Instruct` | Apache 2.0 | 2023-12-11 | ⏳ 검증 필요 |
| `Qwen2.5-1.5B-Instruct` | Apache 2.0 | 2024-09-19 | ❌ 2025.08.01 이후 |
| `Meta-Llama-3.1-70B-Instruct` | Llama 3.1 Community | 2024-07-23 | ✅ 검증 완료 |
| `Qwen2.5-7B-Instruct` | Apache 2.0 | 2024-09-19 | ❌ 2025.08.01 이후 |

**라이선스 검증 체크리스트**:
1. **공개 일자**: 2025년 8월 1일 이전 (~2025.07.31) 공식 배포 확인
2. **라이선스 유형**: MIT, Apache 2.0, Llama Community 등 비상업적 이용 허용 여부
3. **가중치 접근성**: Hugging Face Hub 또는 공식 리포지토리에서 다운로드 가능 여부
4. **법적 제약사항**: 대회 사용에 대한 라이선스 제한 조건 검토

**검증된 최종 후보 모델**:
- **Student**: `mistralai/Mistral-7B-Instruct-v0.2` (Apache 2.0, 2024-03-24)
- **Teacher**: `Meta-Llama-3.1-70B-Instruct` (Llama Community, 2024-07-23)

---
