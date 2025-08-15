"""
Qwen3-Reranker-4B 실제 모델 테스트
"""

import torch
import sys
import io
import time
from typing import List, Dict

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_model_loading():
    """모델 다운로드 및 로딩 테스트"""
    print("\n=== Qwen3-Reranker-4B 모델 로딩 테스트 ===")
    
    try:
        from packages.rag.reranking import Qwen3Reranker, RerankerConfig
        
        # GPU 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 모델 설정
        config = RerankerConfig(
            model_name="Qwen/Qwen3-Reranker-4B",
            device=device,
            precision="fp16" if device == "cuda" else "fp32",
            batch_size=8,
            max_length=512,
            cache_enabled=True
        )
        
        print("\n모델 다운로드 및 로딩 중... (처음 실행 시 시간이 걸립니다)")
        start_time = time.time()
        
        # 모델 초기화
        reranker = Qwen3Reranker(
            model_name=config.model_name,
            device=config.device,
            precision=config.precision,
            batch_size=config.batch_size,
            max_length=config.max_length,
            cache_enabled=config.cache_enabled
        )
        
        load_time = time.time() - start_time
        print(f"✅ 모델 로딩 완료! (소요 시간: {load_time:.2f}초)")
        
        # 모델 정보
        print(f"\n모델 정보:")
        print(f"- 모델명: {config.model_name}")
        print(f"- 정밀도: {config.precision}")
        print(f"- 배치 크기: {config.batch_size}")
        print(f"- 최대 길이: {config.max_length}")
        
        return reranker
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return None


def test_korean_financial_reranking(reranker):
    """한국어 금융 질문 리랭킹 테스트"""
    print("\n=== 한국어 금융 질문 리랭킹 테스트 ===")
    
    # 테스트 질문
    query = "금융 기관의 사이버 보안 리스크 관리 방법은?"
    
    # 테스트 문서들 (관련도 순서대로)
    documents = [
        {
            "content": "금융 기관의 사이버 보안 리스크 관리는 다층 방어 체계 구축, 실시간 모니터링, 정기적인 보안 감사, 그리고 직원 교육을 통해 이루어집니다.",
            "metadata": {"source": "doc1"}
        },
        {
            "content": "한국 금융위원회는 금융회사의 정보보호 강화를 위한 가이드라인을 발표했습니다.",
            "metadata": {"source": "doc2"}
        },
        {
            "content": "리스크 관리는 금융 기관 운영의 핵심 요소입니다. VaR 모델을 활용한 시장 리스크 측정이 중요합니다.",
            "metadata": {"source": "doc3"}
        },
        {
            "content": "사이버 공격으로부터 금융 시스템을 보호하기 위해서는 방화벽, 침입 탐지 시스템, 암호화 기술 등이 필요합니다.",
            "metadata": {"source": "doc4"}
        },
        {
            "content": "금융 상품의 수익률은 시장 상황에 따라 변동됩니다.",
            "metadata": {"source": "doc5"}
        }
    ]
    
    print(f"\n질문: {query}")
    print(f"문서 개수: {len(documents)}")
    
    # 리랭킹 수행
    print("\n리랭킹 수행 중...")
    start_time = time.time()
    
    reranked_docs = reranker.rerank(
        query=query,
        documents=documents,
        top_k=3
    )
    
    rerank_time = time.time() - start_time
    print(f"리랭킹 완료! (소요 시간: {rerank_time:.3f}초)")
    
    # 결과 출력
    print("\n=== 리랭킹 결과 (Top 3) ===")
    for i, doc in enumerate(reranked_docs, 1):
        print(f"\n{i}. [{doc['metadata']['source']}] (점수: {doc['score']:.4f})")
        print(f"   내용: {doc['content'][:100]}...")
        if 'rerank_score' in doc:
            print(f"   리랭킹 점수: {doc['rerank_score']:.4f}")
    
    return reranked_docs


def test_batch_processing(reranker):
    """배치 처리 성능 테스트"""
    print("\n=== 배치 처리 성능 테스트 ===")
    
    queries = [
        "금융 보안 규제는?",
        "투자 포트폴리오 리스크 관리",
        "디지털 금융 서비스 보안"
    ]
    
    documents_batch = [
        [
            {"content": "금융보안원은 금융 보안 규제를 관리합니다.", "metadata": {"id": "1-1"}},
            {"content": "금융위원회의 새로운 규제안이 발표되었습니다.", "metadata": {"id": "1-2"}},
        ],
        [
            {"content": "포트폴리오 리스크는 분산 투자로 관리할 수 있습니다.", "metadata": {"id": "2-1"}},
            {"content": "VaR 모델을 통한 리스크 측정이 중요합니다.", "metadata": {"id": "2-2"}},
        ],
        [
            {"content": "디지털 금융의 보안은 암호화가 핵심입니다.", "metadata": {"id": "3-1"}},
            {"content": "핀테크 서비스의 보안 강화가 필요합니다.", "metadata": {"id": "3-2"}},
        ]
    ]
    
    print(f"배치 크기: {len(queries)} 질문")
    
    start_time = time.time()
    results = reranker.batch_rerank(queries, documents_batch, top_k=1)
    batch_time = time.time() - start_time
    
    print(f"배치 처리 완료! (소요 시간: {batch_time:.3f}초)")
    print(f"평균 처리 시간: {batch_time/len(queries):.3f}초/질문")
    
    for i, (query, result) in enumerate(zip(queries, results)):
        if result:
            print(f"\n질문 {i+1}: {query}")
            print(f"최고 문서: {result[0]['content'][:50]}...")


def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("Qwen3-Reranker-4B 실제 모델 테스트")
    print("=" * 60)
    
    # 1. 모델 로딩
    reranker = test_model_loading()
    if not reranker:
        print("\n모델 로딩 실패. 다음을 확인하세요:")
        print("1. 인터넷 연결 확인")
        print("2. Hugging Face 접근 가능 여부")
        print("3. GPU 메모리 충분 여부")
        return False
    
    # 모델 웜업
    print("\n모델 웜업 중...")
    reranker.warmup()
    
    # 2. 한국어 금융 질문 테스트
    test_korean_financial_reranking(reranker)
    
    # 3. 배치 처리 테스트
    test_batch_processing(reranker)
    
    # 4. 캐시 통계
    if hasattr(reranker, 'get_stats'):
        stats = reranker.get_stats()
        print("\n=== 성능 통계 ===")
        print(f"모델: {stats.get('model_name', 'N/A')}")
        print(f"장치: {stats.get('device', 'N/A')}")
        print(f"정밀도: {stats.get('precision', 'N/A')}")
        if 'cache_stats' in stats:
            cache = stats['cache_stats']
            print(f"캐시 적중: {cache.get('hits', 0)}")
            print(f"캐시 미스: {cache.get('misses', 0)}")
            hit_rate = cache['hits'] / (cache['hits'] + cache['misses']) if (cache['hits'] + cache['misses']) > 0 else 0
            print(f"캐시 적중률: {hit_rate:.1%}")
    
    print("\n" + "=" * 60)
    print("✅ 모든 테스트 완료!")
    print("Qwen3-Reranker-4B가 성공적으로 작동합니다.")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n테스트가 사용자에 의해 중단되었습니다.")
        exit(1)
    except Exception as e:
        print(f"\n\n예기치 않은 오류: {e}")
        import traceback
        traceback.print_exc()
        exit(1)