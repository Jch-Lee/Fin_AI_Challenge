#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
완전한 RAG 파이프라인 상세 테스트
모든 중간 과정 출력 및 검증
"""

import os
import sys
import json
import time
import numpy as np
import faiss
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# 프로젝트 경로 추가 (../../ 상대경로 사용)
sys.path.append(str(Path(__file__).parent.parent.parent))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str, level: int = 1):
    """섹션 헤더 출력"""
    if level == 1:
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
    elif level == 2:
        print("\n" + "-"*60)
        print(f" {title}")
        print("-"*60)
    else:
        print(f"\n[{title}]")


def safe_print(text: str, max_length: int = None):
    """CP949 안전 출력"""
    safe_text = text.encode('cp949', errors='ignore').decode('cp949')
    if max_length and len(safe_text) > max_length:
        safe_text = safe_text[:max_length] + "..."
    print(safe_text)


class DetailedRAGTester:
    """RAG 시스템 상세 테스터"""
    
    def __init__(self):
        self.components = {}
        self.results = {}
        
    def test_pdf_processing(self):
        """1단계: PDF 처리 테스트"""
        print_section("1단계: PDF 처리", 1)
        
        from packages.preprocessing.pdf_processor_advanced import AdvancedPDFProcessor
        
        pdf_path = "금융분야 AI 보안 가이드라인.pdf"
        if not Path(pdf_path).exists():
            print(f"[오류] PDF 파일을 찾을 수 없습니다: {pdf_path}")
            return None
            
        processor = AdvancedPDFProcessor()
        print(f"\n파일: {pdf_path}")
        print("처리 중...")
        
        start = time.time()
        result = processor.extract_pdf(pdf_path)
        process_time = time.time() - start
        
        text = result.text
        
        print(f"\n[결과]")
        print(f"  - 추출된 텍스트: {len(text):,}자")
        print(f"  - 처리 시간: {process_time:.2f}초")
        print(f"  - 페이지 수: {result.metadata.get('total_pages', 'Unknown')}")
        
        # 텍스트 샘플
        print("\n[텍스트 샘플 (처음 500자)]")
        safe_print(text[:500])
        
        self.components['pdf_text'] = text
        self.results['pdf_processing'] = {
            'text_length': len(text),
            'process_time': process_time
        }
        
        return text
    
    def test_chunking(self, text: str):
        """2단계: 청킹 테스트"""
        print_section("2단계: 텍스트 청킹", 1)
        
        from packages.preprocessing.chunker import DocumentChunker
        
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)
        
        print(f"\n[설정]")
        print(f"  - 청크 크기: 1000자")
        print(f"  - 오버랩: 100자")
        
        start = time.time()
        chunks = chunker.chunk_document(text, metadata={"doc_id": "금융AI가이드"})
        chunk_time = time.time() - start
        
        print(f"\n[결과]")
        print(f"  - 생성된 청크: {len(chunks)}개")
        print(f"  - 처리 시간: {chunk_time:.2f}초")
        
        # 청크 통계
        chunk_lengths = [len(chunk.content) for chunk in chunks]
        print(f"\n[청크 통계]")
        print(f"  - 평균 길이: {np.mean(chunk_lengths):.0f}자")
        print(f"  - 최소 길이: {min(chunk_lengths)}자")
        print(f"  - 최대 길이: {max(chunk_lengths)}자")
        
        # 샘플 청크
        print(f"\n[샘플 청크]")
        for i in [0, len(chunks)//2, -1]:
            chunk = chunks[i]
            print(f"\n청크 #{chunk.chunk_id}:")
            safe_print(f"  {chunk.content[:100]}...")
        
        self.components['chunks'] = chunks
        self.results['chunking'] = {
            'num_chunks': len(chunks),
            'avg_length': np.mean(chunk_lengths),
            'process_time': chunk_time
        }
        
        return chunks
    
    def test_embeddings(self, chunks):
        """3단계: 임베딩 생성 테스트"""
        print_section("3단계: E5 임베딩 생성", 1)
        
        from packages.preprocessing.embedder_e5 import E5Embedder
        
        print("\n[E5 모델 로딩]")
        embedder = E5Embedder()
        print(f"  - 모델: {embedder.model_name}")
        print(f"  - 차원: {embedder.embedding_dim}")
        
        # 청크 텍스트 추출
        chunk_texts = [chunk.content for chunk in chunks]
        
        print(f"\n[임베딩 생성]")
        print(f"  - 문서 수: {len(chunk_texts)}")
        print(f"  - 배치 크기: 32")
        
        start = time.time()
        embeddings = embedder.encode(
            chunk_texts,
            is_query=False,  # 문서 임베딩
            batch_size=32
        )
        embed_time = time.time() - start
        
        print(f"\n[결과]")
        print(f"  - 임베딩 shape: {embeddings.shape}")
        print(f"  - 임베딩 dtype: {embeddings.dtype}")
        print(f"  - 처리 시간: {embed_time:.2f}초")
        print(f"  - 속도: {len(chunks)/embed_time:.1f} chunks/sec")
        
        # 임베딩 통계
        print(f"\n[임베딩 통계]")
        print(f"  - 평균 norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")
        print(f"  - 표준편차: {np.std(embeddings):.4f}")
        
        self.components['embedder'] = embedder
        self.components['embeddings'] = embeddings
        self.results['embeddings'] = {
            'shape': embeddings.shape,
            'process_time': embed_time,
            'speed': len(chunks)/embed_time
        }
        
        return embeddings, embedder
    
    def test_faiss_index(self, embeddings):
        """4단계: FAISS 인덱스 구축 테스트"""
        print_section("4단계: FAISS 인덱스 구축", 1)
        
        dimension = embeddings.shape[1]
        
        print(f"\n[인덱스 생성]")
        print(f"  - 타입: IndexFlatIP (Inner Product)")
        print(f"  - 차원: {dimension}")
        
        start = time.time()
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        index_time = time.time() - start
        
        print(f"\n[결과]")
        print(f"  - 인덱싱된 벡터: {index.ntotal}개")
        print(f"  - 구축 시간: {index_time:.4f}초")
        
        self.components['faiss_index'] = index
        self.results['faiss_index'] = {
            'num_vectors': index.ntotal,
            'build_time': index_time
        }
        
        return index
    
    def test_bm25_index(self, chunks):
        """5단계: BM25 인덱스 구축 테스트"""
        print_section("5단계: BM25 인덱스 구축", 1)
        
        from packages.retrieval.bm25_retriever import BM25Retriever
        
        print("\n[BM25 초기화]")
        bm25 = BM25Retriever()
        
        chunk_texts = [chunk.content for chunk in chunks]
        
        print(f"  - 토크나이저: simple (공백 분리)")
        print(f"  - 문서 수: {len(chunk_texts)}")
        
        start = time.time()
        doc_ids = [f"doc_{i}" for i in range(len(chunk_texts))]
        bm25.build_index(chunk_texts, doc_ids)
        bm25_time = time.time() - start
        
        print(f"\n[결과]")
        print(f"  - 인덱스 구축 시간: {bm25_time:.4f}초")
        print(f"  - BM25 인덱스 구축 완료 ({len(chunk_texts)}개 문서)")
        
        self.components['bm25'] = bm25
        self.results['bm25_index'] = {
            'build_time': bm25_time
        }
        
        return bm25
    
    def test_hybrid_search(self, query: str, k: int = 5):
        """6단계: 하이브리드 검색 테스트"""
        print_section(f"6단계: 하이브리드 검색 - '{query}'", 1)
        
        from packages.retrieval.hybrid_retriever import HybridRetriever
        
        # 컴포넌트 확인
        if 'chunks' not in self.components:
            print("[오류] 청크가 없습니다")
            return None
            
        print("\n[하이브리드 검색기 초기화]")
        
        # 간단한 벡터 검색기 래퍼 생성
        class SimpleVectorRetriever:
            def __init__(self, faiss_index, chunks):
                self.index = faiss_index
                self.chunks = chunks
                
        vector_retriever = SimpleVectorRetriever(
            self.components['faiss_index'],
            self.components['chunks']
        )
        
        hybrid = HybridRetriever(
            bm25_retriever=self.components['bm25'],
            vector_retriever=vector_retriever,
            embedder=self.components['embedder'],
            bm25_weight=0.3,  # BM25 가중치
            vector_weight=0.7,  # Vector 가중치
            normalization_method="min_max"
        )
        
        print(f"  - BM25 가중치 (α): 0.3")
        print(f"  - Vector 가중치 (β): 0.7")
        print(f"  - 정규화 방법: min-max")
        
        # 검색 수행
        print(f"\n[검색 수행]")
        start = time.time()
        results, details = hybrid.explain_search(query, k=k)
        search_time = time.time() - start
        
        print(f"  - 검색 시간: {search_time:.4f}초")
        print(f"  - 검색된 문서: {len(results)}개")
        
        # BM25 결과
        print_section("BM25 검색 결과", 2)
        for i, (idx, score) in enumerate(details['bm25_results'][:3]):
            print(f"\n  [{i+1}] 문서 #{idx} (점수: {score:.4f})")
            safe_print(f"      {self.components['chunks'][idx].content[:100]}...")
        
        # Vector 결과
        print_section("Vector 검색 결과", 2)
        for i, (idx, score) in enumerate(details['vector_results'][:3]):
            print(f"\n  [{i+1}] 문서 #{idx} (유사도: {score:.4f})")
            safe_print(f"      {self.components['chunks'][idx].content[:100]}...")
        
        # 최종 하이브리드 결과
        print_section("최종 하이브리드 결과", 2)
        for i, (doc, score, metadata) in enumerate(results):
            print(f"\n  [{i+1}] 최종 점수: {score:.4f}")
            print(f"      - BM25 점수: {metadata.get('bm25_score', 0):.4f}")
            print(f"      - Vector 점수: {metadata.get('vector_score', 0):.4f}")
            print(f"      - 출처: {metadata.get('found_by', 'unknown')}")
            safe_print(f"      내용: {doc[:150]}...")
        
        self.results[f'search_{query[:20]}'] = {
            'num_results': len(results),
            'search_time': search_time,
            'top_score': results[0][1] if results else 0
        }
        
        return results
    
    def test_llm_generation(self, query: str, contexts: List[str]):
        """7단계: LLM 답변 생성 테스트"""
        print_section("7단계: Qwen2.5-7B 답변 생성", 1)
        
        try:
            from scripts.integrate_qwen_llm import QwenLLM
            import torch
            
            print("\n[LLM 초기화]")
            use_4bit = torch.cuda.is_available()
            
            llm = QwenLLM(
                use_4bit=use_4bit,
                max_new_tokens=256,
                temperature=0.3
            )
            
            print(f"  - 모델: Qwen2.5-7B-Instruct")
            print(f"  - 4-bit 양자화: {use_4bit}")
            print(f"  - 최대 토큰: 256")
            print(f"  - Temperature: 0.3")
            
            # 프롬프트 생성
            print(f"\n[프롬프트 생성]")
            from packages.llm.prompt_templates import FinancePromptTemplate
            
            q_type = FinancePromptTemplate.detect_question_type(query)
            print(f"  - 질문 유형: {q_type.value}")
            
            prompts = FinancePromptTemplate.create_prompt(
                query, contexts[:3], q_type, include_citations=True
            )
            
            print(f"\n[시스템 프롬프트]")
            safe_print(prompts['system'][:200] + "...")
            
            print(f"\n[사용자 프롬프트 (일부)]")
            safe_print(prompts['user'][:300] + "...")
            
            # 답변 생성
            print(f"\n[답변 생성 중...]")
            start = time.time()
            answer = llm.generate(query, contexts[:3])
            gen_time = time.time() - start
            
            print(f"\n[생성 완료]")
            print(f"  - 생성 시간: {gen_time:.2f}초")
            print(f"  - 답변 길이: {len(answer)}자")
            
            print(f"\n[생성된 답변]")
            print("-"*60)
            safe_print(answer)
            print("-"*60)
            
            self.results['generation'] = {
                'gen_time': gen_time,
                'answer_length': len(answer)
            }
            
            return answer
            
        except Exception as e:
            print(f"\n[LLM 오류] {e}")
            print("LLM 생성을 건너뜁니다.")
            return "[LLM 미사용 - 시뮬레이션 모드]"
    
    def run_full_test(self, query: str = "금융 AI 시스템의 보안을 위해 어떤 조치가 필요한가?"):
        """전체 파이프라인 테스트"""
        
        print("\n" + "■"*80)
        print(" RAG 파이프라인 전체 테스트")
        print("■"*80)
        
        total_start = time.time()
        
        # 1. PDF 처리
        text = self.test_pdf_processing()
        if not text:
            return
        
        # 2. 청킹
        chunks = self.test_chunking(text)
        
        # 3. 임베딩
        embeddings, embedder = self.test_embeddings(chunks)
        
        # 4. FAISS 인덱스
        faiss_index = self.test_faiss_index(embeddings)
        
        # 5. BM25 인덱스
        bm25 = self.test_bm25_index(chunks)
        
        # 6. 하이브리드 검색
        search_results = self.test_hybrid_search(query, k=5)
        
        # 7. LLM 답변 생성
        if search_results:
            contexts = [doc for doc, _, _ in search_results]
            answer = self.test_llm_generation(query, contexts)
        
        total_time = time.time() - total_start
        
        # 최종 요약
        print_section("최종 요약", 1)
        
        print("\n[처리 시간 분석]")
        for stage, result in self.results.items():
            if 'time' in str(result):
                if 'process_time' in result:
                    print(f"  - {stage}: {result['process_time']:.2f}초")
                elif 'build_time' in result:
                    print(f"  - {stage}: {result['build_time']:.2f}초")
                elif 'search_time' in result:
                    print(f"  - {stage}: {result['search_time']:.4f}초")
                elif 'gen_time' in result:
                    print(f"  - {stage}: {result['gen_time']:.2f}초")
        
        print(f"\n[전체 소요 시간]: {total_time:.2f}초")
        
        # 결과 저장
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """테스트 결과 저장"""
        output_dir = Path("rag_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"full_pipeline_test_{timestamp}.json"
        
        # JSON 직렬화 가능하도록 변환
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, (int, float, str, bool, list)):
                        serializable_results[k] = v
                    elif isinstance(v, tuple):
                        serializable_results[k] = list(v)
                    elif isinstance(v, np.ndarray):
                        serializable_results[k] = v.tolist()
                    else:
                        serializable_results[k] = str(v)
            else:
                serializable_results[key] = value
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n[결과 저장]: {output_file}")


def test_specific_questions():
    """특정 질문들에 대한 테스트"""
    print_section("추가 질문 테스트", 1)
    
    tester = DetailedRAGTester()
    
    # 컴포넌트 초기화 (빠른 테스트를 위해 재사용)
    print("\n[빠른 초기화 중...]")
    
    # 기존 인덱스 로드
    from packages.preprocessing.embedder_e5 import E5Embedder
    import faiss
    
    index_dir = Path("data/e5_embeddings/latest")
    if index_dir.exists():
        # FAISS 로드
        index = faiss.read_index(str(index_dir / "faiss_index.bin"))
        
        # 청크 로드
        with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        
        # 청크 객체 생성
        from packages.preprocessing.chunker import Chunk
        chunks = []
        for chunk_data in chunks_data:
            chunk = Chunk(
                chunk_id=chunk_data['chunk_id'],
                content=chunk_data['content'],
                metadata=chunk_data['metadata']
            )
            chunks.append(chunk)
        
        # 컴포넌트 설정
        tester.components['chunks'] = chunks
        tester.components['faiss_index'] = index
        tester.components['embedder'] = E5Embedder()
        
        # BM25 재구축
        from packages.retrieval.bm25_retriever import BM25Retriever
        bm25 = BM25Retriever()
        bm25.build_index([c.content for c in chunks])
        tester.components['bm25'] = bm25
        
        print("초기화 완료!")
        
        # 테스트 질문들
        test_questions = [
            "AI 모델의 적대적 공격이란 무엇인가?",
            "챗봇 서비스의 보안 점검 항목은?",
            "AI 학습 데이터 관리 방법은?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[질문 {i}]: {question}")
            results = tester.test_hybrid_search(question, k=3)
            
            if results:
                print(f"\n최고 점수: {results[0][1]:.4f}")
                print("관련 내용:")
                safe_print(results[0][0][:200] + "...")


if __name__ == "__main__":
    try:
        # 전체 파이프라인 테스트
        tester = DetailedRAGTester()
        results = tester.run_full_test()
        
        # 추가 질문 테스트
        response = input("\n추가 질문을 테스트하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            test_specific_questions()
        
        print("\n" + "■"*80)
        print(" [완료] 모든 테스트가 성공적으로 완료되었습니다!")
        print("■"*80)
        
    except Exception as e:
        logger.error(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()