"""
RAG 시스템 중간 생성물 확인 스크립트
"""

import sys
import io
import json
import numpy as np
import pickle
from pathlib import Path

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def view_logs():
    """로그 파일 확인"""
    print("\n=== 로그 파일 ===")
    log_dir = Path("logs")
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            print(f"\n📄 {log_file.name}")
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"   총 {len(lines)}줄")
                # 마지막 5줄 출력
                print("   마지막 5줄:")
                for line in lines[-5:]:
                    print(f"   {line.strip()}")


def view_test_results():
    """테스트 결과 JSON 확인"""
    print("\n=== 테스트 결과 ===")
    log_dir = Path("logs")
    if log_dir.exists():
        for json_file in log_dir.glob("*.json"):
            print(f"\n📊 {json_file.name}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 시스템 정보
                if 'system_info' in data:
                    print("   시스템 정보:")
                    print(f"   - Device: {data['system_info'].get('device', 'N/A')}")
                    print(f"   - GPU: {data['system_info'].get('gpu_name', 'N/A')}")
                
                # 파이프라인 설정
                if 'pipeline_config' in data:
                    print("   파이프라인 설정:")
                    config = data['pipeline_config']
                    print(f"   - Embedder: {config.get('embedder_model', 'N/A')}")
                    print(f"   - Retriever: {config.get('retriever_type', 'N/A')}")
                    print(f"   - Reranking: {config.get('reranking_enabled', False)}")
                
                # 성능 메트릭
                if 'performance_metrics' in data:
                    print("   성능 메트릭:")
                    metrics = data['performance_metrics']
                    print(f"   - 임베딩 평균: {metrics.get('embedding_avg', 0):.3f}초")
                    print(f"   - 검색 평균: {metrics.get('retrieval_avg', 0):.3f}초")
                    print(f"   - 전체 평균: {metrics.get('total_avg', 0):.3f}초")


def view_knowledge_base():
    """지식 베이스 내용 확인"""
    print("\n=== 지식 베이스 ===")
    
    kb_files = [
        "pdf_knowledge_base.pkl",
        "test_knowledge_base.pkl"
    ]
    
    for kb_file in kb_files:
        kb_path = Path(kb_file)
        if kb_path.exists():
            if kb_path.is_dir():
                pkl_file = kb_path / "knowledge_base.pkl"
                if pkl_file.exists():
                    kb_path = pkl_file
                else:
                    continue
            
            print(f"\n💾 {kb_path}")
            try:
                with open(kb_path, 'rb') as f:
                    kb = pickle.load(f)
                    
                    if hasattr(kb, 'documents'):
                        print(f"   문서 수: {len(kb.documents)}")
                        if kb.documents:
                            print("   첫 번째 문서 미리보기:")
                            doc = kb.documents[0]
                            if isinstance(doc, dict):
                                content = doc.get('content', '')[:100]
                            else:
                                content = str(doc)[:100]
                            print(f"   {content}...")
                    
                    if hasattr(kb, 'index'):
                        print(f"   인덱스 크기: {kb.index.ntotal if hasattr(kb.index, 'ntotal') else 'N/A'}")
            except Exception as e:
                print(f"   ❌ 로드 실패: {e}")


def view_embeddings():
    """임베딩 데이터 확인"""
    print("\n=== 임베딩 데이터 ===")
    
    embedding_dirs = [
        Path("data/kure_embeddings/latest"),
        Path("data/e5_embeddings/latest")
    ]
    
    for emb_dir in embedding_dirs:
        if emb_dir.exists():
            print(f"\n📁 {emb_dir}")
            
            # chunks.json
            chunks_file = emb_dir / "chunks.json"
            if chunks_file.exists():
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    print(f"   청크 수: {len(chunks)}")
                    if chunks:
                        # chunks가 딕셔너리 리스트인지 문자열 리스트인지 확인
                        first_chunk = chunks[0]
                        if isinstance(first_chunk, dict):
                            content = first_chunk.get('content', str(first_chunk))[:100]
                        else:
                            content = str(first_chunk)[:100]
                        print(f"   첫 청크: {content}...")
            
            # embeddings.npy
            emb_file = emb_dir / "embeddings.npy"
            if emb_file.exists():
                embeddings = np.load(emb_file)
                print(f"   임베딩 shape: {embeddings.shape}")
                print(f"   임베딩 차원: {embeddings.shape[1] if len(embeddings.shape) > 1 else 'N/A'}")
            
            # metadata.json
            meta_file = emb_dir / "metadata.json"
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    print(f"   메타데이터 키: {list(metadata.keys())}")


def view_pipeline_results():
    """파이프라인 결과 확인"""
    print("\n=== 파이프라인 결과 ===")
    
    pipeline_dir = Path("pipeline_results")
    if pipeline_dir.exists():
        files = sorted(pipeline_dir.glob("*"))
        for file in files:
            print(f"\n📄 {file.name}")
            
            if file.suffix == '.txt':
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"   길이: {len(content)} 문자")
                    print(f"   미리보기: {content[:100]}...")
            
            elif file.suffix == '.json':
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"   항목 수: {len(data)}")
                        if data:
                            print(f"   첫 항목: {str(data[0])[:100]}...")
                    elif isinstance(data, dict):
                        print(f"   키: {list(data.keys())}")
            
            elif file.suffix == '.npy':
                arr = np.load(file)
                print(f"   Shape: {arr.shape}")
                print(f"   dtype: {arr.dtype}")


def main():
    """메인 함수"""
    print("="*60)
    print("RAG 시스템 중간 생성물 확인")
    print("="*60)
    
    # 1. 로그 파일
    view_logs()
    
    # 2. 테스트 결과
    view_test_results()
    
    # 3. 지식 베이스
    view_knowledge_base()
    
    # 4. 임베딩 데이터
    view_embeddings()
    
    # 5. 파이프라인 결과
    view_pipeline_results()
    
    print("\n" + "="*60)
    print("✅ 중간 생성물 확인 완료")
    print("="*60)


if __name__ == "__main__":
    main()