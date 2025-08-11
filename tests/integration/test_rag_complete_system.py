#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
완전한 RAG 시스템 통합 검증
모든 컴포넌트의 정확한 작동 확인
"""

import os
import sys
import json
import time
import numpy as np
import faiss
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# 프로젝트 경로 추가 (../../ 상대경로 사용)
sys.path.append(str(Path(__file__).parent.parent.parent))

# UTF-8 출력 설정
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class CompleteRAGValidator:
    """완전한 RAG 시스템 검증기"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        
    def validate_pdf_processing(self):
        """PDF 처리 검증"""
        print("\n" + "="*80)
        print("1. PDF 처리 검증")
        print("="*80)
        
        try:
            from packages.preprocessing.pdf_processor_advanced import AdvancedPDFProcessor
            
            pdf_path = "금융분야 AI 보안 가이드라인.pdf"
            if not Path(pdf_path).exists():
                self.errors.append(f"PDF 파일 없음: {pdf_path}")
                return None
                
            processor = AdvancedPDFProcessor()
            result = processor.extract_pdf(pdf_path)
            
            text = result.text
            self.results['pdf'] = {
                'success': True,
                'text_length': len(text),
                'pages': result.metadata.get('total_pages', 0)
            }
            
            print(f"✅ PDF 처리 성공")
            print(f"   - 텍스트 길이: {len(text):,}자")
            print(f"   - 샘플: {text[:100]}...")
            
            return text
            
        except Exception as e:
            self.errors.append(f"PDF 처리 실패: {e}")
            print(f"❌ PDF 처리 실패: {e}")
            return None
    
    def validate_chunking(self, text):
        """청킹 검증"""
        print("\n" + "="*80)
        print("2. 청킹 검증")
        print("="*80)
        
        try:
            from packages.preprocessing.chunker import DocumentChunker
            
            chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)
            chunks = chunker.chunk_document(text, metadata={"doc_id": "test"})
            
            self.results['chunking'] = {
                'success': True,
                'num_chunks': len(chunks),
                'avg_length': np.mean([len(c.content) for c in chunks])
            }
            
            print(f"✅ 청킹 성공")
            print(f"   - 청크 수: {len(chunks)}개")
            print(f"   - 평균 길이: {self.results['chunking']['avg_length']:.0f}자")
            
            return chunks
            
        except Exception as e:
            self.errors.append(f"청킹 실패: {e}")
            print(f"❌ 청킹 실패: {e}")
            return None
    
    def validate_e5_embeddings(self, chunks):
        """E5 임베딩 검증"""
        print("\n" + "="*80)
        print("3. E5 임베딩 검증")
        print("="*80)
        
        try:
            from packages.preprocessing.embedder_e5 import E5Embedder
            
            embedder = E5Embedder()
            print(f"   모델: {embedder.model_name}")
            print(f"   차원: {embedder.embedding_dim}")
            
            # 문서 임베딩
            chunk_texts = [c.content for c in chunks]
            doc_embeddings = embedder.encode(chunk_texts, is_query=False, batch_size=32)
            
            # 쿼리 임베딩
            test_query = "AI 보안"
            query_embedding = embedder.encode([test_query], is_query=True)
            
            self.results['embeddings'] = {
                'success': True,
                'model': embedder.model_name,
                'dimension': embedder.embedding_dim,
                'doc_shape': doc_embeddings.shape,
                'query_shape': query_embedding.shape
            }
            
            print(f"✅ E5 임베딩 성공")
            print(f"   - 문서 임베딩: {doc_embeddings.shape}")
            print(f"   - 쿼리 임베딩: {query_embedding.shape}")
            
            return embedder, doc_embeddings
            
        except Exception as e:
            self.errors.append(f"E5 임베딩 실패: {e}")
            print(f"❌ E5 임베딩 실패: {e}")
            return None, None
    
    def validate_faiss_index(self, embeddings):
        """FAISS 인덱스 검증"""
        print("\n" + "="*80)
        print("4. FAISS 인덱스 검증")
        print("="*80)
        
        try:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings.astype('float32'))
            
            # 검색 테스트
            test_vector = embeddings[0:1]
            D, I = index.search(test_vector, k=5)
            
            self.results['faiss'] = {
                'success': True,
                'num_vectors': index.ntotal,
                'dimension': dimension,
                'search_test': {'distances': D[0].tolist(), 'indices': I[0].tolist()}
            }
            
            print(f"✅ FAISS 인덱스 성공")
            print(f"   - 인덱싱된 벡터: {index.ntotal}개")
            print(f"   - 검색 테스트: Top-1 거리 = {D[0][0]:.4f}")
            
            return index
            
        except Exception as e:
            self.errors.append(f"FAISS 인덱스 실패: {e}")
            print(f"❌ FAISS 인덱스 실패: {e}")
            return None
    
    def validate_bm25(self, chunks):
        """BM25 검증"""
        print("\n" + "="*80)
        print("5. BM25 검색 검증")
        print("="*80)
        
        try:
            from packages.retrieval.bm25_retriever import BM25Retriever
            
            bm25 = BM25Retriever()
            chunk_texts = [c.content for c in chunks]
            doc_ids = [f"doc_{i}" for i in range(len(chunks))]
            
            bm25.build_index(chunk_texts, doc_ids)
            
            # 검색 테스트
            test_query = "AI 보안"
            results = bm25.search(test_query, k=5)
            
            self.results['bm25'] = {
                'success': True,
                'num_docs': len(chunk_texts),
                'search_results': len(results)
            }
            
            print(f"✅ BM25 인덱스 성공")
            print(f"   - 인덱싱된 문서: {len(chunk_texts)}개")
            print(f"   - 테스트 검색 결과: {len(results)}개")
            
            return bm25
            
        except Exception as e:
            self.errors.append(f"BM25 실패: {e}")
            print(f"❌ BM25 실패: {e}")
            return None
    
    def validate_hybrid_search(self, embedder, faiss_index, bm25, chunks):
        """하이브리드 검색 검증"""
        print("\n" + "="*80)
        print("6. 하이브리드 검색 검증")
        print("="*80)
        
        try:
            # 직접 하이브리드 검색 구현
            test_query = "금융 AI 시스템의 보안 조치"
            
            # BM25 검색
            bm25_results = bm25.search(test_query, k=10)
            print(f"   BM25 검색: {len(bm25_results)}개 결과")
            
            # Vector 검색
            query_embedding = embedder.encode([test_query], is_query=True)
            D, I = faiss_index.search(query_embedding.astype('float32'), k=10)
            print(f"   Vector 검색: {len(I[0])}개 결과")
            
            # 점수 결합 (간단한 방식)
            alpha = 0.3  # BM25 가중치
            beta = 0.7   # Vector 가중치
            
            combined_scores = {}
            
            # BM25 점수 추가
            for i, result in enumerate(bm25_results):
                doc_idx = int(result.doc_id.split('_')[1])
                normalized_score = 1.0 / (i + 1)  # 순위 기반 정규화
                combined_scores[doc_idx] = alpha * normalized_score
            
            # Vector 점수 추가
            for i, (idx, score) in enumerate(zip(I[0], D[0])):
                if idx not in combined_scores:
                    combined_scores[idx] = 0
                normalized_score = score  # 이미 코사인 유사도
                combined_scores[idx] += beta * normalized_score
            
            # 상위 5개 선택
            top_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            self.results['hybrid'] = {
                'success': True,
                'bm25_results': len(bm25_results),
                'vector_results': len(I[0]),
                'combined_results': len(top_results),
                'top_score': top_results[0][1] if top_results else 0
            }
            
            print(f"✅ 하이브리드 검색 성공")
            print(f"   - 최종 결과: {len(top_results)}개")
            print(f"   - 최고 점수: {top_results[0][1]:.4f}")
            
            # 결과 반환
            search_results = []
            for idx, score in top_results:
                if 0 <= idx < len(chunks):
                    search_results.append((chunks[idx].content, score))
            
            return search_results
            
        except Exception as e:
            self.errors.append(f"하이브리드 검색 실패: {e}")
            print(f"❌ 하이브리드 검색 실패: {e}")
            return None
    
    def validate_qwen_llm(self, query, contexts):
        """Qwen LLM 검증"""
        print("\n" + "="*80)
        print("7. Qwen2.5-7B LLM 검증")
        print("="*80)
        
        try:
            from scripts.integrate_qwen_llm import QwenLLM
            from packages.llm.prompt_templates import FinancePromptTemplate
            
            # GPU 확인
            use_gpu = torch.cuda.is_available()
            print(f"   GPU 사용: {use_gpu}")
            
            # LLM 초기화
            llm = QwenLLM(
                use_4bit=use_gpu,
                max_new_tokens=256,
                temperature=0.3
            )
            
            # 프롬프트 생성
            q_type = FinancePromptTemplate.detect_question_type(query)
            print(f"   질문 유형: {q_type.value}")
            
            # 답변 생성
            start = time.time()
            answer = llm.generate(query, contexts[:3])
            gen_time = time.time() - start
            
            self.results['llm'] = {
                'success': True,
                'model': 'Qwen2.5-7B-Instruct',
                'use_gpu': use_gpu,
                'answer_length': len(answer),
                'generation_time': gen_time
            }
            
            print(f"✅ LLM 답변 생성 성공")
            print(f"   - 답변 길이: {len(answer)}자")
            print(f"   - 생성 시간: {gen_time:.2f}초")
            print(f"\n[생성된 답변]")
            print("-"*60)
            print(answer[:500] + "..." if len(answer) > 500 else answer)
            print("-"*60)
            
            return answer
            
        except Exception as e:
            self.errors.append(f"LLM 실패: {e}")
            print(f"❌ LLM 실패: {e}")
            # LLM 실패 시 시뮬레이션
            return f"[LLM 오류: {e}]"
    
    def run_complete_validation(self):
        """전체 시스템 검증 실행"""
        print("\n" + "█"*80)
        print(" RAG 시스템 완전 검증 시작")
        print("█"*80)
        
        start_time = time.time()
        
        # 1. PDF 처리
        text = self.validate_pdf_processing()
        if not text:
            print("\n⚠️ PDF 처리 실패로 검증 중단")
            return False
        
        # 2. 청킹
        chunks = self.validate_chunking(text)
        if not chunks:
            print("\n⚠️ 청킹 실패로 검증 중단")
            return False
        
        # 3. E5 임베딩
        embedder, embeddings = self.validate_e5_embeddings(chunks)
        if embedder is None or embeddings is None:
            print("\n⚠️ E5 임베딩 실패로 검증 중단")
            return False
        
        # 4. FAISS 인덱스
        faiss_index = self.validate_faiss_index(embeddings)
        if not faiss_index:
            print("\n⚠️ FAISS 인덱스 실패로 검증 중단")
            return False
        
        # 5. BM25
        bm25 = self.validate_bm25(chunks)
        if not bm25:
            print("\n⚠️ BM25 실패로 검증 중단")
            return False
        
        # 6. 하이브리드 검색
        test_query = "금융 AI 시스템의 보안을 위해 어떤 조치가 필요한가?"
        search_results = self.validate_hybrid_search(embedder, faiss_index, bm25, chunks)
        if not search_results:
            print("\n⚠️ 하이브리드 검색 실패로 검증 중단")
            return False
        
        # 7. LLM 답변 생성
        contexts = [doc for doc, score in search_results]
        answer = self.validate_qwen_llm(test_query, contexts)
        
        total_time = time.time() - start_time
        
        # 최종 결과
        print("\n" + "█"*80)
        print(" 검증 결과 요약")
        print("█"*80)
        
        success_count = sum(1 for r in self.results.values() if r.get('success', False))
        total_count = len(self.results)
        
        print(f"\n성공: {success_count}/{total_count} 컴포넌트")
        
        for component, result in self.results.items():
            status = "✅" if result.get('success', False) else "❌"
            print(f"  {status} {component}")
        
        if self.errors:
            print(f"\n오류 목록:")
            for error in self.errors:
                print(f"  - {error}")
        
        print(f"\n총 검증 시간: {total_time:.2f}초")
        
        # 결과 저장
        self.save_results()
        
        return success_count == total_count
    
    def save_results(self):
        """검증 결과 저장"""
        output_dir = Path("rag_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"validation_{timestamp}.json"
        
        # JSON 직렬화를 위한 타입 변환
        def convert_to_serializable(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        report = {
            'timestamp': timestamp,
            'results': convert_to_serializable(self.results),
            'errors': self.errors,
            'success': all(r.get('success', False) for r in self.results.values())
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과 저장: {output_file}")


def test_multiple_questions():
    """여러 질문 테스트"""
    print("\n" + "="*80)
    print(" 추가 질문 테스트")
    print("="*80)
    
    # 기존 인덱스 로드
    from packages.preprocessing.embedder_e5 import E5Embedder
    from packages.retrieval.bm25_retriever import BM25Retriever
    import faiss
    
    index_dir = Path("data/e5_embeddings/latest")
    if not index_dir.exists():
        print("인덱스를 찾을 수 없습니다")
        return
    
    # 컴포넌트 로드
    embedder = E5Embedder()
    index = faiss.read_index(str(index_dir / "faiss_index.bin"))
    
    with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    # 테스트 질문들
    test_questions = [
        "AI 모델의 적대적 공격이란?",
        "챗봇 서비스의 보안 점검 항목은?",
        "AI 학습 데이터 관리 방법은?",
        "금융 AI 시스템의 규제 요구사항은?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n질문 {i}: {question}")
        
        # 검색
        query_embedding = embedder.encode([question], is_query=True)
        D, I = index.search(query_embedding.astype('float32'), k=3)
        
        print(f"  최고 유사도: {D[0][0]:.4f}")
        if I[0][0] < len(chunks_data):
            print(f"  관련 내용: {chunks_data[I[0][0]]['content'][:100]}...")


if __name__ == "__main__":
    # 메인 검증 실행
    validator = CompleteRAGValidator()
    success = validator.run_complete_validation()
    
    if success:
        print("\n" + "🎉"*20)
        print(" 모든 RAG 컴포넌트가 정상 작동합니다!")
        print("🎉"*20)
        
        # 추가 테스트
        response = input("\n추가 질문을 테스트하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            test_multiple_questions()
    else:
        print("\n⚠️ 일부 컴포넌트에 문제가 있습니다. 위 오류를 확인하세요.")