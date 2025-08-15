"""
RAG 시스템 구축 스크립트
Architecture.md와 Pipeline.md Epic 1.2-1.4 완전 구현
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.preprocessing.data_preprocessor import DataPreprocessor, ProcessedDocument
from packages.preprocessing.chunker import DocumentChunker, DocumentChunk
from packages.preprocessing import LANGCHAIN_AVAILABLE
if LANGCHAIN_AVAILABLE:
    from packages.preprocessing.hierarchical_chunker import HierarchicalMarkdownChunker
from packages.preprocessing.embedder import TextEmbedder
from packages.rag.knowledge_base import KnowledgeBase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGSystemBuilder:
    """
    RAG 시스템 구축 및 관리 클래스
    Pipeline.md Epic 1 전체 구현
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 model_name: str = "nlpai-lab/KURE-v1"):
        """
        Args:
            data_dir: 데이터 저장 디렉토리
            model_name: 임베딩 모델명
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        
        # 디렉토리 구조 생성
        self._create_directory_structure()
        
        # 컴포넌트 초기화
        self.preprocessor = DataPreprocessor(str(self.data_dir / "processed"))
        
        # 계층적 마크다운 청킹을 우선 사용, 없으면 기본 청킹 사용
        if LANGCHAIN_AVAILABLE:
            logger.info("Using HierarchicalMarkdownChunker for optimal markdown processing")
            self.chunker = HierarchicalMarkdownChunker(
                use_chunk_cleaner=True,
                enable_semantic=False
            )
        else:
            logger.info("Using basic DocumentChunker (install langchain for better chunking)")
            self.chunker = DocumentChunker(
                chunk_size=512,  # Pipeline.md 1.3.1 요구사항
                chunk_overlap=50
            )
        
        self.embedder = TextEmbedder(model_name=model_name)
        self.knowledge_base = None
        
        logger.info(f"RAG System Builder initialized with model: {model_name}")
    
    def _create_directory_structure(self):
        """Pipeline.md 1.1.1 디렉토리 구조 생성"""
        directories = [
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "processed" / "metadata",
            self.data_dir / "finetune",
            self.data_dir / "knowledge_base",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Directory structure created")
    
    def process_documents(self, input_path: str) -> List[ProcessedDocument]:
        """
        Epic 1.2: 문서 처리
        """
        logger.info(f"Processing documents from: {input_path}")
        
        processed_docs = []
        
        if os.path.isfile(input_path):
            # 단일 파일 처리
            if input_path.endswith('.pdf'):
                doc = self.preprocessor.process_pdf(input_path)
                processed_docs.append(doc)
            elif input_path.endswith('.html'):
                doc = self.preprocessor.process_html(input_path)
                processed_docs.append(doc)
            elif input_path.endswith('.txt'):
                doc = self.preprocessor.process_text(input_path)
                processed_docs.append(doc)
        else:
            # 디렉토리 처리
            processed_docs = self.preprocessor.process_directory(input_path)
        
        logger.info(f"Processed {len(processed_docs)} documents")
        return processed_docs
    
    def create_chunks(self, documents: List[ProcessedDocument]) -> List[DocumentChunk]:
        """
        Epic 1.3: 청킹 및 임베딩 준비
        """
        logger.info("Creating document chunks...")
        
        all_chunks = []
        
        for doc in documents:
            # 문서를 청크로 분할
            chunks = self.chunker.chunk_document(
                document=doc.content,
                metadata={
                    "source": doc.source_path,
                    "doc_id": doc.doc_id,
                    "processed_at": doc.processed_at
                }
            )
            
            # 청크에 문서 정보 추가
            for chunk in chunks:
                chunk.metadata.update({
                    "original_doc_id": doc.doc_id,
                    "source_file": doc.source_path
                })
            
            all_chunks.extend(chunks)
            logger.info(f"Document {doc.doc_id}: {len(chunks)} chunks created")
        
        # 청크 저장
        chunks_file = self.data_dir / "processed" / "chunks.jsonl"
        self.chunker.save_chunks(all_chunks, str(chunks_file))
        logger.info(f"Saved {len(all_chunks)} chunks to {chunks_file}")
        
        return all_chunks
    
    def create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """
        Epic 1.3.3: 임베딩 생성
        """
        logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        
        # 청크 텍스트 추출
        texts = [chunk.content for chunk in chunks]
        
        # 배치 임베딩 생성
        embeddings = self.embedder.embed_batch(texts, batch_size=32)
        
        # 임베딩 저장
        embeddings_file = self.data_dir / "processed" / "embeddings.npy"
        np.save(str(embeddings_file), embeddings)
        logger.info(f"Saved embeddings to {embeddings_file}")
        
        return embeddings
    
    def build_knowledge_base(self, 
                            chunks: List[DocumentChunk], 
                            embeddings: np.ndarray) -> KnowledgeBase:
        """
        Epic 1.4: 지식 베이스 구축
        """
        logger.info("Building FAISS knowledge base...")
        
        # 차원 확인
        dimension = embeddings.shape[1]
        logger.info(f"Embedding dimension: {dimension}")
        
        # 지식 베이스 초기화
        self.knowledge_base = KnowledgeBase(
            embedding_dim=dimension,
            index_type="Flat"  # 시작은 Flat으로, 나중에 IVF로 최적화
        )
        
        # 문서 메타데이터 준비
        documents = []
        for i, chunk in enumerate(chunks):
            doc_data = {
                "id": i,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index
            }
            documents.append(doc_data)
        
        # 인덱스에 추가
        self.knowledge_base.add_documents(embeddings, documents)
        
        # 인덱스 저장
        index_path = self.data_dir / "knowledge_base" / "faiss.index"
        self.knowledge_base.save(str(index_path))
        logger.info(f"Knowledge base saved to {index_path}")
        
        return self.knowledge_base
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """검색 수행"""
        if self.knowledge_base is None:
            # 저장된 인덱스 로드
            index_path = self.data_dir / "knowledge_base" / "faiss.index"
            if index_path.exists():
                # KURE-v1의 임베딩 차원은 1024
                self.knowledge_base = KnowledgeBase(embedding_dim=1024)
                self.knowledge_base.load(str(index_path))
            else:
                raise ValueError("Knowledge base not built. Run build_full_pipeline first.")
        
        # 쿼리 임베딩
        query_embedding = self.embedder.embed(query)
        
        # 검색 수행
        results = self.knowledge_base.search(query_embedding, k=k)
        
        return results
    
    def build_full_pipeline(self, input_path: str) -> Dict[str, Any]:
        """
        전체 RAG 파이프라인 실행
        Epic 1.2 ~ 1.4 통합 실행
        """
        logger.info("=" * 50)
        logger.info("Starting full RAG pipeline build...")
        logger.info("=" * 50)
        
        # 1. 문서 처리 (Epic 1.2)
        logger.info("\n[Step 1] Processing documents...")
        documents = self.process_documents(input_path)
        
        # 2. 청킹 (Epic 1.3)
        logger.info("\n[Step 2] Creating chunks...")
        chunks = self.create_chunks(documents)
        
        # 3. 임베딩 생성 (Epic 1.3.3)
        logger.info("\n[Step 3] Generating embeddings...")
        embeddings = self.create_embeddings(chunks)
        
        # 4. 지식베이스 구축 (Epic 1.4)
        logger.info("\n[Step 4] Building knowledge base...")
        knowledge_base = self.build_knowledge_base(chunks, embeddings)
        
        # 5. 통계 정보
        stats = {
            "documents_processed": len(documents),
            "total_chunks": len(chunks),
            "embedding_dimension": embeddings.shape[1],
            "index_size": knowledge_base.index.ntotal if knowledge_base else 0,
            "avg_chunk_size": np.mean([len(c.content) for c in chunks]),
            "total_characters": sum([len(c.content) for c in chunks])
        }
        
        logger.info("\n" + "=" * 50)
        logger.info("RAG Pipeline Build Complete!")
        logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
        logger.info("=" * 50)
        
        return stats
    
    def validate_pipeline(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        파이프라인 검증
        Pipeline.md 완료 기준 확인
        """
        logger.info("\nValidating RAG pipeline...")
        
        validation_results = {
            "search_performance": [],
            "avg_response_time": 0,
            "success_rate": 0
        }
        
        import time
        
        for query in test_queries:
            start_time = time.time()
            try:
                results = self.search(query, k=5)
                response_time = (time.time() - start_time) * 1000  # ms
                
                validation_results["search_performance"].append({
                    "query": query,
                    "response_time_ms": response_time,
                    "results_count": len(results),
                    "success": True
                })
                
                logger.info(f"Query: '{query[:50]}...' - {response_time:.2f}ms - {len(results)} results")
                
            except Exception as e:
                logger.error(f"Query failed: {e}")
                validation_results["search_performance"].append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        # 통계 계산
        successful_searches = [r for r in validation_results["search_performance"] if r["success"]]
        if successful_searches:
            validation_results["avg_response_time"] = np.mean([r["response_time_ms"] for r in successful_searches])
            validation_results["success_rate"] = len(successful_searches) / len(test_queries) * 100
        
        # Pipeline.md 1.4 완료 기준: 응답 시간 < 100ms
        if validation_results["avg_response_time"] < 100:
            logger.info("✅ Performance requirement met: avg response time < 100ms")
        else:
            logger.warning(f"⚠️ Performance requirement not met: avg response time = {validation_results['avg_response_time']:.2f}ms")
        
        return validation_results


def main():
    """메인 실행 함수"""
    # RAG 시스템 빌더 초기화
    builder = RAGSystemBuilder(
        data_dir="data",
        model_name="nlpai-lab/KURE-v1"
    )
    
    # PDF 파일 처리
    pdf_path = "금융분야 AI 보안 가이드라인.pdf"
    
    if os.path.exists(pdf_path):
        # 전체 파이프라인 실행
        stats = builder.build_full_pipeline(pdf_path)
        
        # 테스트 쿼리로 검증
        test_queries = [
            "금융 AI 시스템의 보안 요구사항은?",
            "개인정보 보호를 위한 기술적 조치",
            "AI 모델의 취약점과 대응 방안",
            "금융 서비스의 사이버 보안 위협",
            "머신러닝 모델 공격 방어 전략"
        ]
        
        validation_results = builder.validate_pipeline(test_queries)
        
        print("\n" + "=" * 50)
        print("Validation Results:")
        print(f"Success Rate: {validation_results['success_rate']:.1f}%")
        print(f"Avg Response Time: {validation_results['avg_response_time']:.2f}ms")
        print("=" * 50)
        
    else:
        logger.error(f"PDF file not found: {pdf_path}")


if __name__ == "__main__":
    main()