"""
KURE-v1 모델로 업데이트된 RAG 시스템 테스트
"""

import os
import sys
import time
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from packages.preprocessing.embedder import TextEmbedder
from packages.rag.knowledge_base import KnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_kure_rag_system():
    """KURE-v1 기반 RAG 시스템 테스트"""
    print("=== KURE-v1 RAG 시스템 테스트 ===")
    
    try:
        # 1. KURE-v1 임베더 초기화
        print("\n[1] KURE-v1 임베더 초기화...")
        embedder = TextEmbedder(model_name="nlpai-lab/KURE-v1")
        print(f"  - 모델: {embedder.generator.model_name}")
        print(f"  - 임베딩 차원: {embedder.embedding_dim}")
        print(f"  - 디바이스: {embedder.generator.device}")
        
        # 2. 지식베이스 로드
        print("\n[2] 지식베이스 로드...")
        
        # KURE-v1 임베딩 데이터 로드
        kure_index_path = Path("data/kure_embeddings/latest/faiss.index")
        
        if kure_index_path.exists():
            kb = KnowledgeBase.load(str(kure_index_path))
            print(f"  - 로드된 문서 수: {kb.doc_count}")
            print(f"  - 인덱스 크기: {kb.index.ntotal}")
            print(f"  - 인덱스 타입: {type(kb.index)}")
        else:
            print(f"  - 경고: KURE-v1 인덱스 파일이 없습니다: {kure_index_path}")
            return False
        
        # 3. 검색 테스트
        print("\n[3] 검색 성능 테스트...")
        test_queries = [
            "금융 AI 시스템의 보안 요구사항은?",
            "개인정보 보호를 위한 기술적 조치",
            "머신러닝 모델의 취약점과 대응",
            "사이버 보안 위협 대응 방안",
            "AI 모델 공격 방어 전략"
        ]
        
        total_time = 0
        search_results = []
        
        for i, query in enumerate(test_queries):
            print(f"\n  쿼리 {i+1}: {query}")
            
            # 쿼리 임베딩
            start = time.time()
            query_embedding = embedder.embed(query)
            embed_time = time.time() - start
            
            # 검색 수행
            start = time.time()
            results = kb.search(query_embedding, k=3)
            search_time = time.time() - start
            
            total_search_time = embed_time + search_time
            total_time += total_search_time
            
            print(f"    - 임베딩 시간: {embed_time*1000:.2f}ms")
            print(f"    - 검색 시간: {search_time*1000:.2f}ms")
            print(f"    - 총 시간: {total_search_time*1000:.2f}ms")
            print(f"    - 검색 결과 수: {len(results)}")
            
            # 결과 미리보기
            if results:
                top_result = results[0]
                try:
                    if hasattr(top_result, 'content'):
                        content_str = str(top_result.content)
                        content = content_str[:100] if len(content_str) > 100 else content_str
                    else:
                        content = str(top_result)[:100]
                    
                    score = top_result.score if hasattr(top_result, 'score') else 0
                    print(f"    - 최고 점수: {score:.4f}")
                    print(f"    - 내용 미리보기: {content}...")
                except Exception as e:
                    print(f"    - 결과 처리 오류: {e}")
                    print(f"    - 결과 타입: {type(top_result)}")
                    print(f"    - 결과 속성: {dir(top_result) if hasattr(top_result, '__dict__') else 'N/A'}")
            
            search_results.append({
                'query': query,
                'embedding_time': embed_time,
                'search_time': search_time,
                'total_time': total_search_time,
                'results_count': len(results),
                'top_score': results[0].score if results and hasattr(results[0], 'score') else 0
            })
        
        # 4. 성능 요약
        print(f"\n[4] 성능 요약:")
        avg_time = (total_time / len(test_queries)) * 1000
        print(f"  - 전체 쿼리 수: {len(test_queries)}")
        print(f"  - 총 시간: {total_time*1000:.2f}ms")
        print(f"  - 평균 응답 시간: {avg_time:.2f}ms")
        print(f"  - Pipeline.md 목표 (<100ms): {'OK' if avg_time < 100 else 'FAIL'}")
        
        # 5. 품질 평가
        print(f"\n[5] 검색 품질 평가:")
        avg_score = sum(r['top_score'] for r in search_results) / len(search_results)
        successful_searches = sum(1 for r in search_results if r['results_count'] > 0)
        
        print(f"  - 성공한 검색: {successful_searches}/{len(test_queries)}")
        print(f"  - 평균 최고 점수: {avg_score:.4f}")
        print(f"  - 검색 성공률: {(successful_searches/len(test_queries))*100:.1f}%")
        
        # 6. 기존 E5 모델과 비교 (메타데이터에서)
        try:
            import json
            metadata_file = Path("data/kure_embeddings/latest/metadata.json")
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                print(f"\n[6] 모델 마이그레이션 정보:")
                migration = metadata.get('migration_from', {})
                print(f"  - 기존 모델: {migration.get('old_model', 'unknown')}")
                print(f"  - 기존 차원: {migration.get('old_dimension', 'unknown')}")
                print(f"  - 새 모델: {metadata.get('model', 'unknown')}")
                print(f"  - 새 차원: {metadata.get('dimension', 'unknown')}")
                print(f"  - 마이그레이션 일시: {migration.get('migration_date', 'unknown')}")
        except Exception as e:
            print(f"  - 마이그레이션 정보 로드 실패: {e}")
        
        print("\n[SUCCESS] KURE-v1 RAG 시스템 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_old_system():
    """기존 E5 시스템과 비교 (가능한 경우)"""
    print("\n=== 기존 E5 시스템과 성능 비교 ===")
    
    old_index_path = Path("data/e5_embeddings/latest/faiss_index.bin")
    
    if not old_index_path.exists():
        print("기존 E5 인덱스를 찾을 수 없어 비교를 건너뜁니다.")
        return
    
    try:
        # 기존 E5 시스템 테스트는 생략 (차원 불일치로 인한 오류 방지)
        print("기존 E5 시스템과의 직접 비교는 차원 차이(384 vs 1024)로 인해 생략됩니다.")
        print("대신 메타데이터 기반 비교:")
        
        # 메타데이터 비교
        try:
            import json
            
            old_metadata_path = Path("data/e5_embeddings/latest/metadata.json")
            new_metadata_path = Path("data/kure_embeddings/latest/metadata.json")
            
            if old_metadata_path.exists() and new_metadata_path.exists():
                with open(old_metadata_path, 'r', encoding='utf-8') as f:
                    old_meta = json.load(f)
                with open(new_metadata_path, 'r', encoding='utf-8') as f:
                    new_meta = json.load(f)
                
                print(f"  기존 시스템 (E5):")
                print(f"    - 모델: {old_meta.get('model', 'unknown')}")
                print(f"    - 차원: {old_meta.get('dimension', 'unknown')}")
                print(f"    - 청크 수: {old_meta.get('num_chunks', 'unknown')}")
                
                print(f"  새 시스템 (KURE-v1):")
                print(f"    - 모델: {new_meta.get('model', 'unknown')}")
                print(f"    - 차원: {new_meta.get('dimension', 'unknown')}")
                print(f"    - 청크 수: {new_meta.get('num_chunks', 'unknown')}")
                
        except Exception as e:
            print(f"메타데이터 비교 실패: {e}")
            
    except Exception as e:
        print(f"비교 테스트 실패: {e}")


def main():
    """메인 실행 함수"""
    print("KURE-v1 기반 RAG 시스템 종합 테스트")
    print("="*60)
    
    # 1. KURE-v1 RAG 시스템 테스트
    kure_success = test_kure_rag_system()
    
    # 2. 기존 시스템과 비교
    compare_with_old_system()
    
    # 3. 결론 및 권장사항
    print("\n" + "="*60)
    print("테스트 결론 및 권장사항")
    print("="*60)
    
    if kure_success:
        print("\n[OK] KURE-v1 RAG 시스템 정상 작동 확인")
        print("[OK] 한국어 특화 검색 성능 향상 예상")
        print("[OK] 1024차원으로 더 풍부한 의미 표현")
        
        print("\n권장사항:")
        print("1. 기존 코드에서 임베딩 경로를 'data/kure_embeddings/latest'로 변경")
        print("2. 하드코딩된 차원 값을 384에서 1024로 업데이트")
        print("3. 성능 모니터링을 통해 실제 개선 효과 측정")
        print("4. 금융 도메인에서의 검색 품질 평가")
    else:
        print("\n[FAIL] KURE-v1 RAG 시스템 테스트 실패")
        print("추가 디버깅 필요")


if __name__ == "__main__":
    main()