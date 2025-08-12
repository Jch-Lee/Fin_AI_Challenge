"""
pipeline_results 형식으로 RAG 실험 결과 저장
오늘 실행한 PDF RAG 테스트의 중간 과정을 체계적으로 저장
"""

import sys
import io
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 시스템 경로 추가
sys.path.append(str(Path(__file__).parent))

from packages.preprocessing.pdf_processor_advanced import AdvancedPDFProcessor
from packages.preprocessing.chunker import DocumentChunker
from packages.rag import create_rag_pipeline


def create_pipeline_results_folder():
    """새로운 pipeline_results 폴더 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"pipeline_results_{timestamp}"
    folder_path = Path(folder_name)
    folder_path.mkdir(exist_ok=True)
    return folder_path


def save_step_1_pdf_extraction(output_dir: Path, pdf_path: str):
    """Step 1: PDF 텍스트 추출"""
    print("\n[Step 1] PDF 텍스트 추출")
    
    processor = AdvancedPDFProcessor()
    result = processor.extract_pdf(pdf_path)
    
    # 전체 텍스트 저장
    text_file = output_dir / "01_extracted_text.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(result.text)
    
    print(f"✅ 저장: {text_file}")
    print(f"   - 텍스트 길이: {len(result.text)} 문자")
    print(f"   - 페이지 수: {len(result.page_texts)}")
    
    return result


def save_step_2_chunking(output_dir: Path, pdf_result):
    """Step 2: 텍스트 청킹"""
    print("\n[Step 2] 텍스트 청킹")
    
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    all_chunks = []
    
    for page_num, page_text in enumerate(pdf_result.page_texts, 1):
        if not page_text:
            continue
        
        chunks = chunker.chunk_document(page_text)
        for i, chunk in enumerate(chunks):
            if hasattr(chunk, 'content'):
                chunk_text = chunk.content
            else:
                chunk_text = str(chunk)
            
            chunk_data = {
                'text': chunk_text,
                'page': page_num,
                'chunk_index': i,
                'metadata': {
                    'source': '금융분야 AI 보안 가이드라인.pdf',
                    'page': page_num,
                    'chunk_id': f"page_{page_num}_chunk_{i}"
                }
            }
            all_chunks.append(chunk_data)
    
    # 청크 저장
    chunks_file = output_dir / "02_chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 저장: {chunks_file}")
    print(f"   - 총 청크 수: {len(all_chunks)}")
    
    return all_chunks


def save_step_3_embeddings(output_dir: Path, chunks):
    """Step 3: 임베딩 생성"""
    print("\n[Step 3] 임베딩 생성")
    
    # RAG 파이프라인으로 임베딩 생성
    pipeline = create_rag_pipeline(
        embedder_type="kure",
        retriever_type="vector",
        enable_reranking=False
    )
    
    # 텍스트만 추출
    texts = [chunk['text'] for chunk in chunks]
    
    # 임베딩 생성
    embeddings = pipeline.embedder.embed_batch(texts, batch_size=32)
    embeddings_array = np.array(embeddings)
    
    # 임베딩 저장
    embeddings_file = output_dir / "03_embeddings.npy"
    np.save(embeddings_file, embeddings_array)
    
    # 메타데이터 저장
    metadata = {
        'model': pipeline.embedder.model_name,
        'total_chunks': len(chunks),
        'embedding_dimension': embeddings_array.shape[1],
        'embedding_shape': embeddings_array.shape
    }
    
    metadata_file = output_dir / "03_embedding_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ 저장: {embeddings_file}")
    print(f"✅ 저장: {metadata_file}")
    print(f"   - 임베딩 shape: {embeddings_array.shape}")
    print(f"   - 모델: {pipeline.embedder.model_name}")
    
    return pipeline, embeddings_array


def save_step_4_index(output_dir: Path, pipeline, embeddings, chunks):
    """Step 4: FAISS 인덱스 구축"""
    print("\n[Step 4] FAISS 인덱스 구축")
    
    # 문서 추가
    texts = [chunk['text'] for chunk in chunks]
    metadata = [chunk['metadata'] for chunk in chunks]
    
    # 지식 베이스에 추가 (이미 임베딩은 생성됨)
    documents = []
    for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
        doc = {
            "id": i,
            "content": text,
            "embedding": embedding,
            "metadata": meta
        }
        documents.append(doc)
    
    pipeline.knowledge_base.add_documents(embeddings, documents)
    
    # FAISS 인덱스 저장
    index_file = output_dir / "04_faiss_index.bin"
    import faiss
    faiss.write_index(pipeline.knowledge_base.index, str(index_file))
    
    print(f"✅ 저장: {index_file}")
    print(f"   - 인덱스 크기: {pipeline.knowledge_base.index.ntotal}")
    
    return pipeline


def save_step_5_search(output_dir: Path, pipeline, test_question):
    """Step 5: 검색 테스트"""
    print("\n[Step 5] 검색 테스트")
    
    # 검색 수행
    results = pipeline.retrieve(test_question, top_k=5)
    
    # 검색 결과 저장
    search_results = {
        'question': test_question,
        'top_k': 5,
        'results': []
    }
    
    for i, doc in enumerate(results):
        search_results['results'].append({
            'rank': i + 1,
            'score': float(doc.get('score', 0)),
            'content': doc.get('content', '')[:200],
            'metadata': doc.get('metadata', {})
        })
    
    search_file = output_dir / "05_search_results.json"
    with open(search_file, 'w', encoding='utf-8') as f:
        json.dump(search_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 저장: {search_file}")
    print(f"   - 검색된 문서: {len(results)}")
    
    return results


def save_step_6_prompt(output_dir: Path, pipeline, question, search_results):
    """Step 6: 프롬프트 생성"""
    print("\n[Step 6] 프롬프트 생성")
    
    # 컨텍스트 생성
    context = pipeline.generate_context(question, top_k=5, max_length=2000)
    
    # 프롬프트 생성
    prompt = f"""당신은 금융 보안 전문가입니다. 주어진 컨텍스트를 바탕으로 질문에 답변해주세요.

=== 관련 컨텍스트 ===
{context}

=== 질문 ===
{question}

=== 답변 ===
"""
    
    prompt_file = output_dir / "06_generated_prompt.txt"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    print(f"✅ 저장: {prompt_file}")
    print(f"   - 프롬프트 길이: {len(prompt)} 문자")
    
    return prompt


def save_step_7_answer(output_dir: Path, question, prompt):
    """Step 7: 답변 생성 (시뮬레이션)"""
    print("\n[Step 7] 답변 생성")
    
    # 실제 LLM이 없으므로 시뮬레이션
    answer = {
        'question': question,
        'prompt_length': len(prompt),
        'generated_answer': "금융 기관의 AI 시스템 보안을 위해서는 다음과 같은 조치들이 필요합니다:\n\n1. 데이터 보안: 개인정보 암호화, 접근 권한 관리\n2. 모델 보안: 적대적 공격 방어, 모델 추출 방지\n3. 시스템 보안: 침입 탐지, 실시간 모니터링\n4. 규제 준수: 금융위원회 가이드라인 준수, 정기 감사",
        'generation_timestamp': datetime.now().isoformat(),
        'model': "시뮬레이션 (실제 LLM 미사용)",
        'note': "실제 LLM 연동 시 이 부분이 실제 답변으로 대체됩니다"
    }
    
    answer_file = output_dir / "07_generated_answer.json"
    with open(answer_file, 'w', encoding='utf-8') as f:
        json.dump(answer, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 저장: {answer_file}")
    
    return answer


def save_pipeline_summary(output_dir: Path, pdf_path: str, chunks, embeddings_shape, question):
    """파이프라인 요약 정보 저장"""
    print("\n[Summary] 파이프라인 요약")
    
    summary = {
        'experiment_timestamp': datetime.now().isoformat(),
        'steps_completed': 7,
        'pdf_file': pdf_path,
        'total_chunks': len(chunks),
        'embedding_model': 'nlpai-lab/KURE-v1',
        'embedding_dimension': embeddings_shape[1],
        'embedding_shape': list(embeddings_shape),
        'test_question': question,
        'retrieval_top_k': 5,
        'pipeline_success': True,
        'output_directory': str(output_dir)
    }
    
    summary_file = output_dir / "pipeline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 저장: {summary_file}")
    
    return summary


def main():
    """메인 실행 함수"""
    print("="*60)
    print("Pipeline Results 형식으로 RAG 실험 결과 저장")
    print("="*60)
    
    # 설정
    pdf_path = "data/raw/금융분야 AI 보안 가이드라인.pdf"
    test_question = "금융 AI 시스템의 보안을 위해 어떤 조치들이 필요한가요?"
    
    # 출력 디렉토리 생성
    output_dir = create_pipeline_results_folder()
    print(f"\n📁 출력 디렉토리: {output_dir}")
    
    try:
        # Step 1: PDF 추출
        pdf_result = save_step_1_pdf_extraction(output_dir, pdf_path)
        
        # Step 2: 청킹
        chunks = save_step_2_chunking(output_dir, pdf_result)
        
        # Step 3: 임베딩
        pipeline, embeddings = save_step_3_embeddings(output_dir, chunks)
        
        # Step 4: 인덱스
        pipeline = save_step_4_index(output_dir, pipeline, embeddings, chunks)
        
        # Step 5: 검색
        search_results = save_step_5_search(output_dir, pipeline, test_question)
        
        # Step 6: 프롬프트
        prompt = save_step_6_prompt(output_dir, pipeline, test_question, search_results)
        
        # Step 7: 답변
        answer = save_step_7_answer(output_dir, test_question, prompt)
        
        # Summary
        summary = save_pipeline_summary(output_dir, pdf_path, chunks, embeddings.shape, test_question)
        
        print("\n" + "="*60)
        print("✅ 모든 파이프라인 결과 저장 완료!")
        print(f"📁 결과 위치: {output_dir}")
        print("="*60)
        
        # 파일 목록 출력
        print("\n생성된 파일:")
        for file in sorted(output_dir.glob("*")):
            size = file.stat().st_size
            print(f"  - {file.name} ({size:,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)