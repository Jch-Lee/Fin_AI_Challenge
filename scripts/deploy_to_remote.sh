#!/bin/bash
# 원격 서버로 필요한 파일 전송 및 실행 스크립트

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}원격 서버 질문 생성 배포 스크립트${NC}"
echo -e "${GREEN}========================================${NC}"

# 서버 정보 (사용자가 입력해야 함)
if [ -z "$1" ]; then
    echo -e "${RED}Usage: $0 <server_address> [user]${NC}"
    echo -e "${YELLOW}Example: $0 192.168.1.100 ubuntu${NC}"
    exit 1
fi

SERVER=$1
USER=${2:-ubuntu}
REMOTE_DIR="/home/${USER}/question_generation"

echo -e "${YELLOW}서버: ${SERVER}${NC}"
echo -e "${YELLOW}사용자: ${USER}${NC}"
echo -e "${YELLOW}원격 디렉토리: ${REMOTE_DIR}${NC}"

# 필요한 파일 목록
FILES_TO_COPY=(
    "scripts/run_question_generation_remote.py"
    "scripts/utils/diversity_sampler.py"
    "scripts/utils/question_validator.py"
    "configs/question_generation_config.yaml"
    "data/rag/chunks_2300.json"
    "data/rag/embeddings_2300.npy"
    "data/rag/faiss_index_2300.index"
    "data/rag/bm25_index_2300.pkl"
    "data/rag/metadata_2300.json"
)

# 1. 원격 디렉토리 생성
echo -e "${GREEN}1. 원격 디렉토리 생성...${NC}"
ssh ${USER}@${SERVER} "mkdir -p ${REMOTE_DIR}/{scripts/utils,configs,data/rag,data/synthetic_questions}"

# 2. 파일 전송
echo -e "${GREEN}2. 파일 전송 중...${NC}"
for file in "${FILES_TO_COPY[@]}"; do
    echo -e "  전송: $file"
    scp "$file" ${USER}@${SERVER}:${REMOTE_DIR}/$file
done

# 3. requirements.txt 생성 및 전송
echo -e "${GREEN}3. requirements.txt 생성...${NC}"
cat > temp_requirements.txt << EOF
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
EOF

scp temp_requirements.txt ${USER}@${SERVER}:${REMOTE_DIR}/requirements.txt
rm temp_requirements.txt

# 4. 실행 스크립트 생성
echo -e "${GREEN}4. 실행 스크립트 생성...${NC}"
cat > temp_run.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/question_generation

# 가상환경 활성화 (있는 경우)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# 패키지 설치
echo "Installing required packages..."
pip install -r requirements.txt

# 질문 생성 실행
echo "Starting question generation..."
python scripts/run_question_generation_remote.py \
    --model "Qwen/Qwen2.5-32B-Instruct-AWQ" \
    --config configs/question_generation_config.yaml \
    --n-questions 30 \
    --use-vllm

echo "Generation complete!"
EOF

scp temp_run.sh ${USER}@${SERVER}:${REMOTE_DIR}/run_generation.sh
ssh ${USER}@${SERVER} "chmod +x ${REMOTE_DIR}/run_generation.sh"
rm temp_run.sh

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}배포 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}원격 서버에서 실행하려면:${NC}"
echo -e "  ssh ${USER}@${SERVER}"
echo -e "  cd ${REMOTE_DIR}"
echo -e "  ./run_generation.sh"
echo ""
echo -e "${YELLOW}또는 직접 실행:${NC}"
echo -e "  ssh ${USER}@${SERVER} 'cd ${REMOTE_DIR} && ./run_generation.sh'"