"""
KURE-v1 모델로 임베딩 재생성 스크립트
기존 e5 모델 데이터를 KURE-v1으로 업데이트
"""

import os
import sys
import json
import time
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from packages.preprocessing.embedder import EmbeddingGenerator
from packages.rag.knowledge_base import KnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingMigrator:
    """임베딩 마이그레이션 클래스"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.old_model_name = "dragonkue/multilingual-e5-small-ko"
        self.new_model_name = "nlpai-lab/KURE-v1"
        
        # 새 모델로 임베더 초기화
        self.new_embedder = EmbeddingGenerator(model_name=self.new_model_name)
        
    def load_existing_chunks(self) -> tuple:
        """기존 청크 데이터 로드"""
        logger.info("기존 청크 데이터 로딩 중...")
        
        # 최신 데이터 경로
        latest_dir = self.data_dir / "e5_embeddings" / "latest"
        
        # 청크 데이터 로드
        chunks_file = latest_dir / "chunks.json"
        if not chunks_file.exists():
            raise FileNotFoundError(f"청크 파일을 찾을 수 없습니다: {chunks_file}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # 메타데이터 로드
        metadata_file = latest_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        logger.info(f"로드된 청크 수: {len(chunks)}")
        return chunks, metadata
    
    def generate_new_embeddings(self, chunks: list) -> np.ndarray:
        """새 모델로 임베딩 생성"""
        logger.info(f"KURE-v1 모델로 {len(chunks)} 개 청크의 임베딩 생성 중...")
        
        # 청크 텍스트 추출
        texts = []
        for chunk in chunks:
            if isinstance(chunk, dict) and 'content' in chunk:
                texts.append(chunk['content'])
            elif isinstance(chunk, str):
                texts.append(chunk)
            else:
                logger.warning(f"Unknown chunk format: {type(chunk)}")
                texts.append(str(chunk))
        
        start_time = time.time()
        
        # 배치로 임베딩 생성
        embeddings = self.new_embedder.generate_embeddings_batch(
            texts, 
            normalize=True
        )
        
        generation_time = time.time() - start_time
        
        logger.info(f"임베딩 생성 완료")
        logger.info(f"  - 소요 시간: {generation_time:.2f}초")
        logger.info(f"  - 평균 시간: {(generation_time/len(texts))*1000:.2f}ms per text")
        logger.info(f"  - 임베딩 형태: {embeddings.shape}")
        logger.info(f"  - 임베딩 차원: {embeddings.shape[1]}")
        
        return embeddings
    
    def save_new_embeddings(self, chunks: list, embeddings: np.ndarray, old_metadata: dict):
        """새 임베딩 데이터 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 새 디렉토리 생성
        kure_dir = self.data_dir / "kure_embeddings"
        timestamped_dir = kure_dir / timestamp
        latest_dir = kure_dir / "latest"
        
        for directory in [kure_dir, timestamped_dir, latest_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 새 메타데이터 생성
        new_metadata = {
            "timestamp": timestamp,
            "model": self.new_model_name,
            "dimension": embeddings.shape[1],
            "num_chunks": len(chunks),
            "num_embeddings": embeddings.shape[0],
            "pdf_source": old_metadata.get("pdf_source", "unknown"),
            "chunk_size": old_metadata.get("chunk_size", 1000),
            "chunk_overlap": old_metadata.get("chunk_overlap", 100),
            "migration_from": {
                "old_model": old_metadata.get("model", "unknown"),
                "old_dimension": old_metadata.get("dimension", "unknown"),
                "migration_date": timestamp
            }
        }
        
        # 파일 저장 (timestamped와 latest 둘 다)
        for save_dir in [timestamped_dir, latest_dir]:
            # 청크 데이터 저장
            chunks_file = save_dir / "chunks.json"
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            # 임베딩 저장
            embeddings_file = save_dir / "embeddings.npy"
            np.save(str(embeddings_file), embeddings)
            
            # 메타데이터 저장
            metadata_file = save_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(new_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"새 임베딩 데이터 저장 완료: {kure_dir}")
        return new_metadata
    
    def build_faiss_index(self, embeddings: np.ndarray, chunks: list, save_dir: Path):
        """FAISS 인덱스 생성"""
        logger.info("FAISS 인덱스 생성 중...")
        
        # 지식 베이스 생성
        kb = KnowledgeBase(
            embedding_dim=embeddings.shape[1],
            index_type="Flat"
        )
        
        # 문서 메타데이터 준비
        documents = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                doc_data = {
                    "id": i,
                    "content": chunk.get("content", str(chunk)),
                    "metadata": chunk.get("metadata", {}),
                    "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                    "doc_id": chunk.get("doc_id", f"doc_{i}"),
                    "chunk_index": chunk.get("chunk_index", i)
                }
            else:
                doc_data = {
                    "id": i,
                    "content": str(chunk),
                    "metadata": {},
                    "chunk_id": f"chunk_{i}",
                    "doc_id": f"doc_0",
                    "chunk_index": i
                }
            documents.append(doc_data)
        
        # 인덱스에 추가
        kb.add_documents(embeddings, documents)
        
        # 인덱스 저장
        index_dir = save_dir / "faiss.index"
        kb.save(str(index_dir))
        
        logger.info(f"FAISS 인덱스 저장: {index_dir}")
        return kb
    
    def validate_migration(self, chunks: list, embeddings: np.ndarray) -> dict:
        """마이그레이션 검증"""
        logger.info("마이그레이션 검증 중...")
        
        validation_results = {
            "chunks_count": len(chunks),
            "embeddings_shape": embeddings.shape,
            "embedding_dimension": embeddings.shape[1],
            "model_dimension_match": embeddings.shape[1] == self.new_embedder.embedding_dim,
            "sample_similarity_test": None
        }
        
        # 샘플 유사도 테스트
        if len(chunks) >= 3:
            try:
                # 첫 번째 청크를 쿼리로 사용
                query_embedding = embeddings[0]
                doc_embeddings = embeddings[1:3]  # 다음 2개 청크
                
                similarities = self.new_embedder.compute_similarity(
                    query_embedding, doc_embeddings
                )
                
                validation_results["sample_similarity_test"] = {
                    "success": True,
                    "similarities": similarities.tolist(),
                    "max_similarity": float(np.max(similarities)),
                    "min_similarity": float(np.min(similarities))
                }
            except Exception as e:
                validation_results["sample_similarity_test"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # 검증 결과 출력
        logger.info("검증 결과:")
        logger.info(f"  - 청크 수: {validation_results['chunks_count']}")
        logger.info(f"  - 임베딩 형태: {validation_results['embeddings_shape']}")
        logger.info(f"  - 임베딩 차원: {validation_results['embedding_dimension']}")
        logger.info(f"  - 모델 차원 일치: {validation_results['model_dimension_match']}")
        
        if validation_results["sample_similarity_test"] and validation_results["sample_similarity_test"]["success"]:
            similarities = validation_results["sample_similarity_test"]["similarities"]
            logger.info(f"  - 샘플 유사도 테스트: 성공 (유사도: {similarities})")
        
        return validation_results
    
    def run_migration(self):
        """전체 마이그레이션 실행"""
        logger.info("="*60)
        logger.info("KURE-v1 임베딩 마이그레이션 시작")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # 1. 기존 데이터 로드
            chunks, old_metadata = self.load_existing_chunks()
            
            # 2. 새 임베딩 생성
            new_embeddings = self.generate_new_embeddings(chunks)
            
            # 3. 새 데이터 저장
            new_metadata = self.save_new_embeddings(chunks, new_embeddings, old_metadata)
            
            # 4. FAISS 인덱스 생성
            kure_latest_dir = self.data_dir / "kure_embeddings" / "latest"
            self.build_faiss_index(new_embeddings, chunks, kure_latest_dir)
            
            # 5. 검증
            validation = self.validate_migration(chunks, new_embeddings)
            
            total_time = time.time() - start_time
            
            # 결과 요약
            logger.info("="*60)
            logger.info("마이그레이션 완료!")
            logger.info("="*60)
            logger.info(f"총 소요 시간: {total_time:.2f}초")
            logger.info(f"기존 모델: {old_metadata.get('model', 'unknown')} (차원: {old_metadata.get('dimension', 'unknown')})")
            logger.info(f"새 모델: {self.new_model_name} (차원: {new_metadata['dimension']})")
            logger.info(f"처리된 청크 수: {validation['chunks_count']}")
            logger.info(f"저장 위치: {self.data_dir}/kure_embeddings/")
            
            return {
                "success": True,
                "total_time": total_time,
                "old_metadata": old_metadata,
                "new_metadata": new_metadata,
                "validation": validation
            }
            
        except Exception as e:
            logger.error(f"마이그레이션 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def main():
    """메인 실행 함수"""
    migrator = EmbeddingMigrator()
    result = migrator.run_migration()
    
    if result["success"]:
        print("\n✅ 마이그레이션 성공!")
        print("다음 단계:")
        print("1. 기존 코드에서 'data/e5_embeddings/latest' 경로를 'data/kure_embeddings/latest'로 변경")
        print("2. 임베딩 차원이 384 → 1024로 변경되었으므로 관련 설정 확인")
        print("3. RAG 시스템 테스트 실행")
    else:
        print(f"\n❌ 마이그레이션 실패: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()