#!/usr/bin/env python3
"""
대회 환경용 requirements.txt 생성
대회 환경: Python 3.10, CUDA 11.8, PyTorch 2.1.0
"""

import json
from pathlib import Path

def create_competition_requirements():
    """대회 환경에 맞는 requirements.txt 생성"""
    
    # 메인 파이프라인에서 필요한 패키지들
    # generate_submission_qwen_updated_db_remote.py 분석 결과
    required_packages = [
        # 핵심 라이브러리 (대회 환경 버전)
        "torch==2.1.0",  # 대회 요구사항
        "transformers==4.41.2",  # 현재 사용 중
        
        # RAG 관련
        "sentence-transformers==2.7.0",  # KURE-v1 임베더용
        "faiss-cpu==1.8.0",  # FAISS 인덱스
        "kiwipiepy==0.17.1",  # BM25 토크나이저
        
        # 데이터 처리
        "pandas==2.0.3",
        "numpy==1.24.3",
        
        # 유틸리티
        "tqdm==4.66.1",
        
        # 기타 필수
        "accelerate==0.30.1",  # 모델 로딩 최적화
        "safetensors==0.4.1",  # 모델 파일 포맷
    ]
    
    # requirements_competition.txt 생성
    output_file = Path(__file__).parent.parent / "requirements_competition.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Competition Environment Requirements\n")
        f.write("# Python 3.10, CUDA 11.8, PyTorch 2.1.0\n")
        f.write("# Generated for main pipeline: generate_submission_qwen_updated_db_remote.py\n\n")
        
        for package in required_packages:
            f.write(f"{package}\n")
    
    print(f"대회 환경용 requirements 생성 완료: {output_file}")
    
    # 최소 파일 목록 생성
    minimal_files = {
        "scripts": [
            "generate_submission_qwen_updated_db_remote.py",
            "load_rag_v2.py"
        ],
        "data/rag": [
            "chunks_2300.json",
            "metadata_2300.json",
            "faiss_index_2300.index",
            "bm25_index_2300.pkl",
            "embeddings_2300.npy"
        ],
        "data/competition": [
            "test_subjective_10.csv"  # 테스트용
        ],
        "root": [
            "requirements_competition.txt"
        ]
    }
    
    # 파일 목록 JSON으로 저장
    files_list_path = Path(__file__).parent / "minimal_files_list.json"
    with open(files_list_path, 'w', encoding='utf-8') as f:
        json.dump(minimal_files, f, indent=2, ensure_ascii=False)
    
    print(f"최소 파일 목록 생성 완료: {files_list_path}")
    
    # 배포 스크립트 생성
    deploy_script = Path(__file__).parent / "deploy_to_competition_env.sh"
    
    with open(deploy_script, 'w', encoding='utf-8') as f:
        f.write("""#!/bin/bash
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
""")
    
    print(f"배포 스크립트 생성 완료: {deploy_script}")

if __name__ == "__main__":
    create_competition_requirements()