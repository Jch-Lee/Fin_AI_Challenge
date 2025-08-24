# 2. Tech Stack

## 실제 사용 라이브러리 (2025-08-23 업데이트)

| Category | Technology | Version | Purpose | 상태 |
| --- | --- | --- | --- | --- |
| **Language** | Python | `3.10` | 주요 개발 언어 | ✅ 사용중 |
| **ML Framework** | PyTorch | `2.1.0` | 딥러닝 프레임워크 | ✅ 사용중 |
| **Core ML** | Transformers | `4.41.2` | LLM 로드/사용 | ✅ 사용중 |
| **Optimization** | Accelerate | `0.30.1` | 학습/추론 최적화 | ✅ 사용중 |
| **Quantization** | bitsandbytes | `0.43.1` | 8-bit/4-bit 양자화 | ✅ 사용중 |
| **Embedding** | sentence-transformers | `2.7.0` | KURE-v1 임베더 | ✅ 사용중 |
| **Vector DB** | FAISS-CPU | `1.8.0` | 벡터 검색 | ✅ 사용중 |
| **Search** | rank-bm25 | `0.2.2` | BM25 희소 검색 | ✅ 사용중 |
| **Korean NLP** | Kiwipiepy | `0.18.0` | 한국어 형태소 분석 | ✅ 사용중 |
| **Data Processing** | Pandas | `2.2.2` | 데이터프레임 처리 | ✅ 사용중 |
| **Numeric** | NumPy | `1.26.4` | 수치 연산 | ✅ 사용중 |
| **Monitoring** | tqdm | `4.66.4` | 진행 표시 | ✅ 사용중 |
| **Vision-Language** | Qwen2.5-VL-7B-Instruct | `latest` | PDF 이미지 텍스트 추출 | ✅ 사용중 |
| **Image Processing** | Pillow | `10.3.0` | 이미지 변환/처리 | ✅ 사용중 |
| **PDF Processing** | PyMuPDF | `1.24.5` | PDF 텍스트 추출 (폴백) | ✅ 사용중 |
| **Korean Backup** | KoNLPy | `0.6.0` | 한국어 NLP (백업) | ⚠️ 백업용 |
| **Response Gen** | vllm | `0.5.4` | 교사/학생 응답 생성 | ⏳ 계획됨 |
| **Fine-Tuning** | TRL | `0.9.6` | `DistiLLMTrainer` 사용 | ⏳ 계획됨 |
| **Experiment Tracking** | wandb | `0.17.0` | 실험 추적 | ⏳ 계획됨 |
| **Testing** | pytest | `8.2.0` | 단위/통합 테스트 | ⏳ 계획됨 |

## Model Selection Strategy

- **Student Model (Production):** `Qwen2.5-1.5B-Instruct` - 최종 배포용 경량 모델
- **Teacher Model (Training):** `Qwen2.5-7B-Instruct` - Distill-M 2 학습용
- **Vision Model (Production):** `Qwen/Qwen2.5-VL-7B-Instruct` - PDF 텍스트 추출 (41.2% 개선 검증)

### Model License Verification Process

**대회 규칙 2) 준수를 위한 필수 검증 절차**:

| Model | License | Release Date | Verification Status |
|-------|---------|--------------|-------------------|
| `mistralai/Mistral-7B-Instruct-v0.2` | Apache 2.0 | 2024-03-24 | ✅ 검증 완료 |
| `Solar-10.7B-Instruct` | CC-BY-NC-4.0 | 2023-12-11 | ✅ 검증 완료 |
| `Qwen2.5-14B-Instruct` | Apache 2.0 | 2024-09-19 | ✅ 검증 완료 |
| `Qwen2.5-1.5B-Instruct` | Apache 2.0 | 2024-09-19 | ✅ 검증 완료 |
| `Meta-Llama-3.1-70B-Instruct` | Llama 3.1 Community | 2024-07-23 | ✅ 검증 완료 |
| `Qwen2.5-7B-Instruct` | Apache 2.0 | 2024-09-19 | ✅ 검증 완료 |
| `Qwen2.5-VL-7B-Instruct` | Apache 2.0 | 2024-08-29 | ✅ 검증 완료 |

**라이선스 검증 체크리스트**:
1. **공개 일자**: 2025년 8월 1일 이전 (~2025.07.31) 공식 배포 확인
2. **라이선스 유형**: MIT, Apache 2.0, Llama Community 등 비상업적 이용 허용 여부
3. **가중치 접근성**: Hugging Face Hub 또는 공식 리포지토리에서 다운로드 가능 여부
4. **법적 제약사항**: 대회 사용에 대한 라이선스 제한 조건 검토

**검증된 최종 후보 모델**:
- **Student 후보**: 
  - `mistralai/Mistral-7B-Instruct-v0.2` (Apache 2.0, 2024-03-24)
  - `Qwen2.5-1.5B-Instruct` (Apache 2.0, 2024-09-19)
  - `Solar-10.7B-Instruct` (CC-BY-NC-4.0, 2023-12-11)
- **Teacher 후보**: 
  - `Meta-Llama-3.1-70B-Instruct` (Llama Community, 2024-07-23)
  - `Qwen2.5-7B-Instruct` (Apache 2.0, 2024-09-19)
  - `Qwen2.5-14B-Instruct` (Apache 2.0, 2024-09-19)

### Qwen 2.5 모델 상세 사양

**Qwen2.5-1.5B-Instruct**:
- **파라미터**: 1.54B (1.31B non-embedding)
- **아키텍처**: RoPE, SwiGLU, RMSNorm, QKV bias
- **컨텍스트 길이**: 32,768 tokens
- **생성 길이**: 최대 8,192 tokens
- **다국어 지원**: 29+ 언어 (한국어 포함)
- **특징**: 구조화된 데이터 이해, JSON 출력 생성 개선

**Qwen2.5-7B-Instruct**:
- **파라미터**: 7.61B (6.53B non-embedding)
- **아키텍처**: RoPE, SwiGLU, RMSNorm, QKV bias
- **컨텍스트 길이**: 131,072 tokens
- **생성 길이**: 최대 8,192 tokens
- **다국어 지원**: 29+ 언어 (한국어 포함)
- **특징**: 장문 생성, 수학/코딩 능력 강화, 긴 컨텍스트 처리

**Qwen2.5-14B-Instruct**:
- **파라미터**: 14.7B (13.1B non-embedding)
- **아키텍처**: RoPE, SwiGLU, RMSNorm, 48 layers
- **어텐션**: 40 Q heads, 8 KV heads
- **컨텍스트 길이**: 131,072 tokens
- **생성 길이**: 최대 8,192 tokens
- **다국어 지원**: 29+ 언어 (한국어 포함)
- **특징**: YaRN 기법으로 장문 처리, vLLM 최적화 지원

### 추가 적격 모델 (합성 데이터 생성용)

**Qwen3-30B-A3B-Instruct-2507** (Apache 2.0, 2025-07-30) ✅:
- **파라미터**: 30.5B (3.3B active per inference, MoE)
- **전문가**: 128개 중 8개 활성화
- **컨텍스트**: 262,144 tokens (최대 1M tokens 지원)
- **적격성**: 2025년 7월 30일 출시로 대회 기준일(2025년 8월 1일) 이전 출시 ✅
- **용도**: 고품질 합성 QA 데이터 생성에 최적

---
