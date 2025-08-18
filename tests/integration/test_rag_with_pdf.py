"""
실제 PDF 문서를 사용한 RAG 시스템 테스트
금융분야 AI 보안 가이드라인 PDF를 로드하여 테스트
"""

import sys
import io
import time
import logging
from pathlib import Path
from typing import List, Dict

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# 시스템 경로 추가
sys.path.append(str(Path(__file__).parent))

from packages.preprocessing.pdf_processor_advanced import AdvancedPDFProcessor
from packages.preprocessing.chunker import DocumentChunker
from packages.rag import create_rag_pipeline


def load_pdf_documents(pdf_path: str) -> List[Dict]:
    """PDF 문서를 로드하고 청킹"""
    logger.info(f"\n📄 PDF 로딩: {pdf_path}")
    
    # PDF 프로세서 초기화
    pdf_processor = AdvancedPDFProcessor()
    
    # PDF 처리
    try:
        result = pdf_processor.extract_pdf(pdf_path)
        logger.info(f"✅ PDF에서 {len(result.page_texts)} 페이지 추출")
        logger.info(f"   전체 텍스트 길이: {len(result.text)} 문자")
        
        # 페이지별 문서 생성
        documents = []
        for page_num, page_text in enumerate(result.page_texts, 1):
            doc = {
                'content': page_text,
                'metadata': {
                    'page': page_num,
                    'source': pdf_path,
                    'has_tables': len(result.tables) > 0
                }
            }
            documents.append(doc)
        
        # 처음 3페이지 미리보기
        for i, doc in enumerate(documents[:3]):
            content_preview = doc['content'][:200].replace('\n', ' ') if doc['content'] else ""
            logger.info(f"  페이지 {i+1}: {content_preview}...")
        
        return documents
    except Exception as e:
        logger.error(f"❌ PDF 처리 실패: {e}")
        return []


def chunk_documents(documents: List, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
    """문서를 청킹"""
    logger.info(f"\n✂️ 문서 청킹 (chunk_size={chunk_size}, overlap={overlap})")
    
    # 청커 초기화
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    # 모든 문서를 청킹
    all_chunks = []
    for doc in documents:
        content = doc.get('content', '')
        
        if not content:
            continue
            
        # 청킹
        chunks = chunker.chunk_document(content)
        
        # 메타데이터 추가
        page_num = doc.get('metadata', {}).get('page', 0)
        for i, chunk in enumerate(chunks):
            # DocumentChunk 객체인 경우
            if hasattr(chunk, 'content'):
                chunk_text = chunk.content
            else:
                chunk_text = str(chunk)
            
            chunk_doc = {
                'content': chunk_text,
                'metadata': {
                    'source': 'AI_보안_가이드라인.pdf',
                    'page': page_num,
                    'chunk_id': f"page_{page_num}_chunk_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            }
            all_chunks.append(chunk_doc)
    
    logger.info(f"✅ {len(all_chunks)}개 청크 생성")
    return all_chunks


def test_rag_with_real_pdf():
    """실제 PDF로 RAG 시스템 테스트"""
    
    print("="*60)
    print("실제 PDF 문서를 사용한 RAG 시스템 테스트")
    print("="*60)
    
    # 1. PDF 로드
    pdf_path = "data/raw/금융분야 AI 보안 가이드라인.pdf"
    if not Path(pdf_path).exists():
        logger.error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return False
    
    documents = load_pdf_documents(pdf_path)
    if not documents:
        logger.error("PDF 문서 로드 실패")
        return False
    
    # 2. 문서 청킹
    chunks = chunk_documents(documents, chunk_size=512, overlap=50)
    if not chunks:
        logger.error("문서 청킹 실패")
        return False
    
    # 3. RAG 파이프라인 초기화
    logger.info("\n🚀 RAG 파이프라인 초기화")
    
    try:
        pipeline = create_rag_pipeline(
            embedder_type="kure",
            retriever_type="vector",  # 간단한 벡터 검색 사용
            enable_reranking=False  # 리랭킹 비활성화 (빠른 테스트)
        )
        logger.info("✅ 파이프라인 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 파이프라인 초기화 실패: {e}")
        return False
    
    # 4. 문서 추가
    logger.info("\n📥 지식 베이스에 문서 추가")
    
    try:
        # 청크 텍스트와 메타데이터 분리
        texts = [chunk['content'] for chunk in chunks]
        metadata = [chunk['metadata'] for chunk in chunks]
        
        # 배치로 문서 추가
        start_time = time.time()
        num_added = pipeline.add_documents(
            texts=texts,
            metadata=metadata,
            batch_size=32
        )
        add_time = time.time() - start_time
        
        logger.info(f"✅ {num_added}개 문서 추가 완료 (소요시간: {add_time:.2f}초)")
        logger.info(f"   평균: {add_time/num_added:.3f}초/문서")
        
        # 지식 베이스 저장
        kb_path = "pdf_knowledge_base.pkl"
        pipeline.save_knowledge_base(kb_path)
        logger.info(f"💾 지식 베이스 저장: {kb_path}")
        
    except Exception as e:
        logger.error(f"❌ 문서 추가 실패: {e}")
        return False
    
    # 5. 테스트 질문
    logger.info("\n🔍 검색 테스트")
    
    test_queries = [
        "AI 시스템의 보안 위협은 무엇인가?",
        "금융 AI의 데이터 보호 방법은?",
        "모델 공격에 대한 방어 전략은?",
        "AI 모델의 취약점 평가 방법은?",
        "금융 분야 AI 규제 요구사항은?",
    ]
    
    for query in test_queries:
        logger.info(f"\n질문: {query}")
        logger.info("-" * 40)
        
        try:
            # 검색 수행
            start_time = time.time()
            results = pipeline.retrieve(query, top_k=3)
            search_time = time.time() - start_time
            
            logger.info(f"검색 완료 ({search_time:.3f}초)")
            
            # 결과 출력
            if results:
                for i, doc in enumerate(results, 1):
                    score = doc.get('score', 0)
                    content = doc.get('content', '')[:150]
                    metadata = doc.get('metadata', {})
                    page = metadata.get('page', 'N/A')
                    
                    logger.info(f"\n  [{i}] 점수: {score:.4f} | 페이지: {page}")
                    logger.info(f"      {content}...")
            else:
                logger.info("  검색 결과 없음")
                
            # 컨텍스트 생성
            context = pipeline.generate_context(query, top_k=3, max_length=1000)
            if context:
                logger.info(f"\n생성된 컨텍스트 (첫 300자):")
                logger.info(f"{context[:300]}...")
            
        except Exception as e:
            logger.error(f"❌ 검색 실패: {e}")
    
    # 6. 통계 출력
    logger.info("\n📊 파이프라인 통계")
    stats = pipeline.get_statistics()
    logger.info(f"  - Embedder: {stats['embedder_model']}")
    logger.info(f"  - Embedding 차원: {stats['embedding_dim']}")
    logger.info(f"  - 저장된 문서 수: {stats['num_documents']}")
    logger.info(f"  - 인덱스 크기: {stats['index_size']}")
    
    return True


def test_specific_questions():
    """특정 도메인 질문 테스트"""
    logger.info("\n" + "="*60)
    logger.info("특정 도메인 질문 테스트")
    logger.info("="*60)
    
    # 기존 지식 베이스 로드
    kb_path = "pdf_knowledge_base.pkl"
    if not Path(kb_path).exists():
        logger.warning(f"지식 베이스가 없습니다. 먼저 test_rag_with_real_pdf()를 실행하세요.")
        return False
    
    # 파이프라인 초기화 및 기존 KB 로드
    pipeline = create_rag_pipeline(
        embedder_type="kure",
        retriever_type="vector",
        knowledge_base_path=kb_path,
        enable_reranking=False
    )
    
    # 도메인별 질문
    domain_questions = {
        "보안 위협": [
            "적대적 공격(adversarial attack)이란?",
            "데이터 포이즈닝 공격 방어 방법은?",
            "모델 추출 공격을 방지하는 방법은?"
        ],
        "규제 준수": [
            "AI 시스템의 설명가능성 요구사항은?",
            "개인정보 보호를 위한 AI 설계 원칙은?",
            "금융 AI의 감사 추적 요구사항은?"
        ],
        "모델 보안": [
            "모델 암호화 기법은?",
            "연합학습의 보안 이점은?",
            "차등 프라이버시 적용 방법은?"
        ]
    }
    
    for domain, questions in domain_questions.items():
        logger.info(f"\n[{domain}]")
        for question in questions:
            logger.info(f"\nQ: {question}")
            
            # 검색
            results = pipeline.retrieve(question, top_k=1)
            if results:
                best_match = results[0]
                logger.info(f"A: {best_match.get('content', '')[:200]}...")
                logger.info(f"   (점수: {best_match.get('score', 0):.4f})")
    
    return True


def main():
    """메인 실행 함수"""
    try:
        # 1. 실제 PDF로 RAG 시스템 테스트
        success = test_rag_with_real_pdf()
        
        if success:
            # 2. 특정 도메인 질문 테스트
            test_specific_questions()
            
            print("\n" + "="*60)
            print("✅ 모든 테스트 완료!")
            print("="*60)
        else:
            print("\n❌ 테스트 실패")
        
        return success
        
    except KeyboardInterrupt:
        print("\n\n테스트가 사용자에 의해 중단되었습니다.")
        return False
    except Exception as e:
        print(f"\n\n예기치 않은 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)