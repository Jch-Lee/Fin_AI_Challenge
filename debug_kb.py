"""
지식베이스 로딩 디버깅
"""

import sys
import os
import json
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from packages.rag.knowledge_base import KnowledgeBase

def debug_knowledge_base():
    print("=== 지식베이스 로딩 디버깅 ===")
    
    # 1. 파일 존재 확인
    kure_index_path = Path("data/kure_embeddings/latest/faiss.index")
    print(f"\n[1] 인덱스 경로 확인: {kure_index_path}")
    print(f"  - 존재 여부: {kure_index_path.exists()}")
    
    if kure_index_path.exists():
        # 하위 파일들 확인
        faiss_file = kure_index_path / "faiss.index"
        pkl_file = kure_index_path / "knowledge_base.pkl"
        
        print(f"  - FAISS 파일: {faiss_file.exists()}")
        print(f"  - PKL 파일: {pkl_file.exists()}")
        
        # 2. PKL 파일 내용 확인
        if pkl_file.exists():
            import pickle
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                print(f"\n[2] PKL 파일 내용:")
                print(f"  - 키: {list(data.keys())}")
                print(f"  - 문서 수: {data.get('doc_count', 'unknown')}")
                print(f"  - 문서 리스트 길이: {len(data.get('documents', []))}")
                print(f"  - 메타데이터 수: {len(data.get('metadata', []))}")
                print(f"  - 임베딩 차원: {data.get('embedding_dim', 'unknown')}")
                
            except Exception as e:
                print(f"  - PKL 로딩 오류: {e}")
        
        # 3. 실제 로딩 테스트
        print(f"\n[3] 실제 로딩 테스트:")
        try:
            kb = KnowledgeBase(embedding_dim=1024)
            print(f"  - 로딩 전 doc_count: {kb.doc_count}")
            
            kb.load(str(kure_index_path))
            print(f"  - 로딩 후 doc_count: {kb.doc_count}")
            print(f"  - 인덱스 ntotal: {kb.index.ntotal}")
            print(f"  - 문서 수: {len(kb.documents)}")
            
            if kb.doc_count > 0:
                print(f"  - 첫 번째 문서: {kb.documents[0][:100]}...")
            
            return kb
            
        except Exception as e:
            print(f"  - 로딩 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return None

def test_search(kb):
    """검색 테스트"""
    if kb is None or kb.doc_count == 0:
        print("\n[SKIP] 검색 테스트 - 지식베이스가 비어있음")
        return
    
    print(f"\n[4] 검색 테스트:")
    
    try:
        from packages.preprocessing.embedder import TextEmbedder
        embedder = TextEmbedder(model_name="nlpai-lab/KURE-v1")
        
        query = "금융 보안"
        query_emb = embedder.embed(query)
        
        print(f"  - 쿼리: {query}")
        print(f"  - 쿼리 임베딩 차원: {query_emb.shape}")
        
        results = kb.search(query_emb, k=3)
        print(f"  - 검색 결과 수: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"    {i+1}. 점수: {result.get('score', 0):.4f}")
            print(f"       내용: {result.get('content', '')[:100]}...")
        
    except Exception as e:
        print(f"  - 검색 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    kb = debug_knowledge_base()
    test_search(kb)