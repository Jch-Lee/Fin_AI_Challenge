"""
KURE-v1 모델 간단 테스트
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from packages.preprocessing.embedder import EmbeddingGenerator

def test_kure_v1():
    print("=== KURE-v1 모델 테스트 ===")
    
    try:
        # 1. 모델 로딩 테스트
        print("\n[1] 모델 로딩 테스트...")
        start = time.time()
        embedder = EmbeddingGenerator(model_name="nlpai-lab/KURE-v1")
        load_time = time.time() - start
        print(f"  - 로딩 시간: {load_time:.2f}초")
        print(f"  - 임베딩 차원: {embedder.embedding_dim}")
        print(f"  - 디바이스: {embedder.device}")
        
        # 2. 임베딩 생성 테스트
        print("\n[2] 임베딩 생성 테스트...")
        test_text = "금융 AI 시스템의 보안 요구사항은 무엇인가요?"
        
        start = time.time()
        embedding = embedder.generate_embedding(test_text)
        gen_time = time.time() - start
        
        print(f"  - 생성 시간: {gen_time*1000:.2f}ms")
        print(f"  - 임베딩 크기: {embedding.shape}")
        print(f"  - 임베딩 타입: {type(embedding)}")
        
        # 3. 배치 테스트
        print("\n[3] 배치 임베딩 테스트...")
        test_texts = [
            "금융 보안 정책은?",
            "AI 시스템 취약점",
            "개인정보 보호 방법",
        ]
        
        start = time.time()
        batch_embeddings = embedder.generate_embeddings_batch(test_texts)
        batch_time = time.time() - start
        
        print(f"  - 배치 생성 시간: {batch_time*1000:.2f}ms")
        print(f"  - 배치 크기: {batch_embeddings.shape}")
        print(f"  - 평균 시간: {(batch_time/len(test_texts))*1000:.2f}ms per text")
        
        # 4. 유사도 테스트
        print("\n[4] 유사도 계산 테스트...")
        query_emb = batch_embeddings[0]
        doc_embs = batch_embeddings[1:]
        
        start = time.time()
        similarities = embedder.compute_similarity(query_emb, doc_embs)
        sim_time = time.time() - start
        
        print(f"  - 유사도 계산 시간: {sim_time*1000:.2f}ms")
        print(f"  - 유사도 점수: {similarities}")
        
        print("\n[SUCCESS] KURE-v1 모델 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    test_kure_v1()