# RAG 기반 질문 생성 시스템

## 개요
벡터DB(8,270개 청크)를 활용하여 다양한 금융보안 관련 질문을 자동 생성하는 시스템

## 주요 특징
- **다양성 최대화**: 문서별, 위치별, 길이별 균등 샘플링
- **질문 유형 분산**: 6가지 질문 유형 (정의, 프로세스, 규정, 예시, 비교, 적용)
- **품질 검증**: 중복 제거, 유사도 검사, 형식 검증
- **다양성 메트릭**: 종합 다양성 점수 계산

## 로컬 테스트 실행
```bash
# 테스트 (시뮬레이션 모드)
python scripts/generate_diverse_questions.py --sample-size 30
```

## 원격 서버 배포 및 실행

### 1. 서버 정보 준비
서버 주소와 SSH 접속 정보가 필요합니다.

### 2. 배포 (Windows PowerShell)
```powershell
.\scripts\deploy_to_remote.ps1 -ServerAddress <서버주소> -UserName <사용자명>

# 예시
.\scripts\deploy_to_remote.ps1 -ServerAddress 192.168.1.100 -UserName ubuntu
```

### 3. 배포 (Linux/Mac)
```bash
./scripts/deploy_to_remote.sh <서버주소> <사용자명>

# 예시
./scripts/deploy_to_remote.sh 192.168.1.100 ubuntu
```

### 4. 원격 서버에서 실행
```bash
ssh ubuntu@<서버주소>
cd /home/ubuntu/question_generation
./run_generation.sh
```

## 파일 구조
```
scripts/
├── generate_diverse_questions.py     # 로컬 테스트용 (시뮬레이션)
├── run_question_generation_remote.py # 원격 서버용 (실제 모델)
├── utils/
│   ├── diversity_sampler.py         # 다양성 샘플링 모듈
│   └── question_validator.py        # 품질 검증 모듈
├── deploy_to_remote.sh              # Linux/Mac 배포 스크립트
└── deploy_to_remote.ps1             # Windows 배포 스크립트

configs/
└── question_generation_config.yaml  # 설정 파일

data/
├── rag/                             # RAG 데이터
│   ├── chunks_2300.json
│   ├── embeddings_2300.npy
│   └── ...
└── synthetic_questions/             # 생성된 질문 저장
```

## 설정 파일 (configs/question_generation_config.yaml)
- **model**: Qwen 모델 설정 (temperature, top_p 등)
- **sampling**: 청크 샘플링 설정
- **diversity**: 질문 유형별 생성 개수
- **output**: 출력 형식 및 경로

## 다양성 메트릭
- **문서 커버리지**: 사용된 문서 수
- **평균 유사도**: 질문 간 유사도 (낮을수록 좋음)
- **고유 키워드 비율**: 중복되지 않는 키워드 비율
- **질문 유형 엔트로피**: 유형 분포의 균등성
- **종합 다양성 점수**: 0-100점 (높을수록 다양함)

## 필요한 패키지
```
torch>=2.0.0
transformers>=4.41.0
vllm>=0.5.0
accelerate>=0.30.0
bitsandbytes>=0.43.0
sentence-transformers>=2.7.0
faiss-cpu
rank-bm25
kiwipiepy
numpy
pandas
tqdm
pyyaml
```