#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FAISS 인덱스 구축 스크립트
임베딩에서 FAISS 벡터 검색 인덱스 생성
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import logging
import faiss
from typing import Optional

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FAISSIndexBuilder:
    """FAISS 인덱스 구축기"""
    
    def __init__(self, 
                 index_type: str = "IVF1024,Flat",
                 use_gpu: bool = False):
        """
        Args:
            index_type: FAISS 인덱스 타입
            use_gpu: GPU 사용 여부
        """
        self.index_type = index_type
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        if self.use_gpu:
            logger.info(f"GPU available: {faiss.get_num_gpus()} GPUs")
        else:
            logger.info("Using CPU for FAISS")
    
    def load_embeddings(self, embeddings_path: str) -> np.ndarray:
        """
        임베딩 로드
        
        Args:
            embeddings_path: 임베딩 파일 경로
            
        Returns:
            임베딩 배열
        """
        logger.info(f"Loading embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        
        # Float32로 변환 (FAISS 요구사항)
        if embeddings.dtype != np.float32:
            logger.info(f"Converting embeddings from {embeddings.dtype} to float32")
            embeddings = embeddings.astype(np.float32)
        
        logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def create_index(self, dimension: int) -> faiss.Index:
        """
        FAISS 인덱스 생성
        
        Args:
            dimension: 벡터 차원
            
        Returns:
            FAISS 인덱스
        """
        logger.info(f"Creating FAISS index: {self.index_type}")
        
        if self.index_type == "Flat":
            # 단순 L2 거리 인덱스
            index = faiss.IndexFlatL2(dimension)
            
        elif self.index_type == "FlatIP":
            # 내적(코사인 유사도) 인덱스
            index = faiss.IndexFlatIP(dimension)
            
        elif self.index_type.startswith("IVF"):
            # IVF (Inverted File) 인덱스
            parts = self.index_type.split(",")
            nlist = int(parts[0].replace("IVF", ""))
            
            # 양자화기
            quantizer = faiss.IndexFlatL2(dimension)
            
            if len(parts) > 1 and parts[1] == "Flat":
                # IVF + Flat
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            elif len(parts) > 1 and parts[1].startswith("PQ"):
                # IVF + PQ (Product Quantization)
                m = int(parts[1].replace("PQ", ""))
                index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
            else:
                # 기본 IVF
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        elif self.index_type.startswith("HNSW"):
            # HNSW (Hierarchical Navigable Small World)
            M = int(self.index_type.replace("HNSW", "") or "32")
            index = faiss.IndexHNSWFlat(dimension, M)
            
        else:
            # 기본값: Flat L2
            logger.warning(f"Unknown index type: {self.index_type}, using Flat")
            index = faiss.IndexFlatL2(dimension)
        
        # GPU로 이동 (가능한 경우)
        if self.use_gpu:
            logger.info("Moving index to GPU")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        return index
    
    def train_index(self, index: faiss.Index, embeddings: np.ndarray):
        """
        인덱스 학습 (IVF 계열에 필요)
        
        Args:
            index: FAISS 인덱스
            embeddings: 학습용 임베딩
        """
        if not index.is_trained:
            logger.info("Training index...")
            
            # 학습 샘플 수 결정
            n_train = min(len(embeddings), max(1024, len(embeddings) // 10))
            
            # 랜덤 샘플링
            train_indices = np.random.choice(
                len(embeddings), 
                n_train, 
                replace=False
            )
            train_embeddings = embeddings[train_indices]
            
            # 학습
            index.train(train_embeddings)
            logger.info(f"Index trained with {n_train} samples")
        else:
            logger.info("Index does not require training")
    
    def add_embeddings(self, index: faiss.Index, embeddings: np.ndarray):
        """
        인덱스에 임베딩 추가
        
        Args:
            index: FAISS 인덱스
            embeddings: 임베딩 배열
        """
        logger.info(f"Adding {len(embeddings)} embeddings to index")
        
        # 배치 단위로 추가 (메모리 효율)
        batch_size = 10000
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            index.add(batch)
            
            if (i + batch_size) % 50000 == 0:
                logger.info(f"Added {min(i+batch_size, len(embeddings))}/{len(embeddings)} embeddings")
        
        logger.info(f"Total embeddings in index: {index.ntotal}")
    
    def save_index(self, index: faiss.Index, output_path: str):
        """
        인덱스 저장
        
        Args:
            index: FAISS 인덱스
            output_path: 출력 경로
        """
        # GPU 인덱스를 CPU로 이동
        if self.use_gpu:
            logger.info("Moving index from GPU to CPU for saving")
            index = faiss.index_gpu_to_cpu(index)
        
        logger.info(f"Saving index to {output_path}")
        faiss.write_index(index, output_path)
        
        # 파일 크기 확인
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"Index saved: {file_size:.2f} MB")
    
    def build(self, 
              embeddings_path: str,
              output_path: str,
              metadata_path: Optional[str] = None):
        """
        전체 FAISS 인덱스 구축 프로세스
        
        Args:
            embeddings_path: 임베딩 파일 경로
            output_path: 출력 인덱스 경로
            metadata_path: 메타데이터 경로
        """
        # 1. 임베딩 로드
        embeddings = self.load_embeddings(embeddings_path)
        dimension = embeddings.shape[1]
        
        # 2. 인덱스 생성
        index = self.create_index(dimension)
        
        # 3. 인덱스 학습 (필요한 경우)
        self.train_index(index, embeddings)
        
        # 4. 임베딩 추가
        self.add_embeddings(index, embeddings)
        
        # 5. 인덱스 저장
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.save_index(index, output_path)
        
        # 6. 메타데이터 저장
        if metadata_path:
            self._save_metadata(embeddings, output_path, metadata_path)
        
        # 7. 통계 출력
        self._print_statistics(index, embeddings)
    
    def _save_metadata(self, embeddings: np.ndarray, index_path: str, metadata_path: str):
        """메타데이터 저장"""
        metadata = {
            'index_type': self.index_type,
            'num_vectors': len(embeddings),
            'dimension': embeddings.shape[1],
            'index_path': index_path,
            'use_gpu': self.use_gpu
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def _print_statistics(self, index: faiss.Index, embeddings: np.ndarray):
        """통계 출력"""
        print("\n" + "="*60)
        print(" FAISS 인덱스 구축 완료")
        print("="*60)
        print(f"인덱스 타입: {self.index_type}")
        print(f"벡터 수: {index.ntotal}")
        print(f"벡터 차원: {embeddings.shape[1]}")
        print(f"GPU 사용: {self.use_gpu}")
        
        # 인덱스 타입별 정보
        if hasattr(index, 'nlist'):
            print(f"클러스터 수: {index.nlist}")
        if hasattr(index, 'nprobe'):
            print(f"검색 프로브: {index.nprobe}")
        
        print("="*60)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FAISS 인덱스 구축")
    parser.add_argument("--embeddings-path", type=str, required=True,
                       help="임베딩 파일 경로 (.npy)")
    parser.add_argument("--output-path", type=str, required=True,
                       help="출력 인덱스 경로")
    parser.add_argument("--index-type", type=str, default="IVF1024,Flat",
                       help="FAISS 인덱스 타입 (Flat, FlatIP, IVF1024,Flat, HNSW32)")
    parser.add_argument("--use-gpu", action="store_true",
                       help="GPU 사용")
    parser.add_argument("--metadata-path", type=str,
                       help="메타데이터 저장 경로")
    
    args = parser.parse_args()
    
    # 빌더 생성
    builder = FAISSIndexBuilder(
        index_type=args.index_type,
        use_gpu=args.use_gpu
    )
    
    # 인덱스 구축
    builder.build(
        embeddings_path=args.embeddings_path,
        output_path=args.output_path,
        metadata_path=args.metadata_path
    )


if __name__ == "__main__":
    main()