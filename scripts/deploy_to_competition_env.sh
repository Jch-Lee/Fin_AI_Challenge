#!/bin/bash
# 대회 환경 배포 스크립트

echo "대회 환경 배포 시작..."

# 1. 필수 디렉토리 생성
ssh -p 27716 root@172.81.127.46 << 'EOF'
cd /root
mkdir -p Fin_AI_Challenge/scripts
mkdir -p Fin_AI_Challenge/data/rag
mkdir -p Fin_AI_Challenge/data/competition
EOF

# 2. 파일 업로드
echo "파일 업로드 중..."

# 스크립트 파일
scp -P 27716 scripts/generate_submission_qwen_updated_db_remote.py root@172.81.127.46:/root/Fin_AI_Challenge/scripts/
scp -P 27716 scripts/load_rag_v2.py root@172.81.127.46:/root/Fin_AI_Challenge/scripts/

# RAG 데이터
scp -P 27716 data/rag/chunks_2300.json root@172.81.127.46:/root/Fin_AI_Challenge/data/rag/
scp -P 27716 data/rag/metadata_2300.json root@172.81.127.46:/root/Fin_AI_Challenge/data/rag/
scp -P 27716 data/rag/faiss_index_2300.index root@172.81.127.46:/root/Fin_AI_Challenge/data/rag/
scp -P 27716 data/rag/bm25_index_2300.pkl root@172.81.127.46:/root/Fin_AI_Challenge/data/rag/
scp -P 27716 data/rag/embeddings_2300.npy root@172.81.127.46:/root/Fin_AI_Challenge/data/rag/

# 테스트 데이터
scp -P 27716 test_subjective_10.csv root@172.81.127.46:/root/Fin_AI_Challenge/test.csv

# Requirements
scp -P 27716 requirements_competition.txt root@172.81.127.46:/root/Fin_AI_Challenge/

echo "파일 업로드 완료!"

# 3. 환경 설정
ssh -p 27716 root@172.81.127.46 << 'EOF'
cd /root/Fin_AI_Challenge

echo "가상환경 생성 중..."
python3.10 -m venv venv_comp
source venv_comp/bin/activate

echo "패키지 설치 중..."
pip install --upgrade pip
pip install -r requirements_competition.txt

echo "환경 설정 완료!"
EOF

echo "배포 완료!"
