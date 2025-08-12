"""
RAG 시스템 전체 테스트 및 중간 과정 로깅
모든 단계의 출력을 상세히 기록하여 시스템 작동 확인
"""

import sys
import io
import time
import json
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 로깅 설정
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 로그 파일 설정
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"rag_test_{timestamp}.log"

# 콘솔과 파일 모두에 로깅
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 시스템 경로 추가
sys.path.append(str(Path(__file__).parent))

from packages.rag import create_rag_pipeline
from packages.rag.reranking import get_default_config


class RAGSystemTester:
    """RAG 시스템 테스터 - 모든 중간 과정 기록"""
    
    def __init__(self, enable_reranking: bool = True):
        self.enable_reranking = enable_reranking
        self.pipeline = None
        self.test_results = {
            "timestamp": timestamp,
            "system_info": {},
            "pipeline_config": {},
            "test_cases": [],
            "performance_metrics": {}
        }
    
    def log_section(self, title: str, char: str = "="):
        """섹션 구분선 로깅"""
        line = char * 60
        logger.info(f"\n{line}")
        logger.info(f"{title}")
        logger.info(f"{line}")
    
    def check_system(self):
        """시스템 환경 체크"""
        self.log_section("1. 시스템 환경 체크")
        
        # GPU 정보
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ GPU 사용 가능: {gpu_name}")
            logger.info(f"   VRAM: {gpu_memory:.2f} GB")
            device = "cuda"
        else:
            logger.info("⚠️ GPU 사용 불가 - CPU 모드")
            device = "cpu"
        
        # Python 버전
        logger.info(f"Python 버전: {sys.version}")
        
        # PyTorch 버전
        logger.info(f"PyTorch 버전: {torch.__version__}")
        
        self.test_results["system_info"] = {
            "device": device,
            "gpu_name": gpu_name if device == "cuda" else None,
            "gpu_memory_gb": gpu_memory if device == "cuda" else None,
            "python_version": sys.version,
            "pytorch_version": torch.__version__
        }
        
        return device
    
    def initialize_pipeline(self, device: str):
        """RAG 파이프라인 초기화"""
        self.log_section("2. RAG 파이프라인 초기화")
        
        try:
            logger.info("파이프라인 구성 중...")
            
            # Reranker 설정
            reranker_config = None
            if self.enable_reranking:
                logger.info("  - Reranking 활성화")
                reranker_config = get_default_config("qwen3")
                reranker_config.device = device
                logger.info(f"  - Reranker 모델: {reranker_config.model_name}")
            
            # 파이프라인 생성
            start_time = time.time()
            self.pipeline = create_rag_pipeline(
                embedder_type="kure",
                retriever_type="hybrid",
                device=device,
                enable_reranking=self.enable_reranking,
                reranker_config=reranker_config
            )
            init_time = time.time() - start_time
            
            logger.info(f"✅ 파이프라인 초기화 완료 (소요시간: {init_time:.2f}초)")
            
            # 파이프라인 설정 기록
            stats = self.pipeline.get_statistics()
            logger.info(f"파이프라인 구성:")
            logger.info(f"  - Embedder: {stats['embedder_model']}")
            logger.info(f"  - Embedding 차원: {stats['embedding_dim']}")
            logger.info(f"  - Retriever: {stats['retriever_type']}")
            logger.info(f"  - Reranking: {'활성화' if stats['reranking_enabled'] else '비활성화'}")
            
            self.test_results["pipeline_config"] = stats
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 파이프라인 초기화 실패: {e}")
            return False
    
    def add_test_documents(self):
        """테스트 문서 추가"""
        self.log_section("3. 테스트 문서 추가")
        
        # 금융 보안 관련 테스트 문서들
        documents = [
            # 사이버 보안 관련
            "금융 기관의 사이버 보안은 다층 방어 체계를 구축하는 것이 핵심입니다. 방화벽, 침입 탐지 시스템, 엔드포인트 보안 솔루션을 종합적으로 운영해야 합니다.",
            "금융보안원은 24시간 사이버 위협 모니터링 체계를 운영하며, 이상 징후 발견 시 즉시 대응 프로토콜을 가동합니다.",
            "랜섬웨어 공격에 대비하여 금융 기관은 정기적인 백업과 복구 훈련을 실시해야 합니다. 또한 직원 대상 피싱 메일 대응 교육이 필수적입니다.",
            
            # 규제 및 컴플라이언스
            "금융위원회는 전자금융거래법에 따라 금융회사의 IT 보안 수준을 정기적으로 점검합니다. 위반 시 과태료가 부과될 수 있습니다.",
            "개인정보보호법과 신용정보법에 따라 고객 금융 정보는 암호화하여 저장해야 하며, 접근 권한을 엄격히 통제해야 합니다.",
            
            # 리스크 관리
            "운영 리스크 관리를 위해 금융 기관은 BCP(Business Continuity Plan)를 수립하고 연 2회 이상 모의훈련을 실시해야 합니다.",
            "시장 리스크는 VaR(Value at Risk) 모델을 통해 측정하며, 일일 한도를 설정하여 관리합니다.",
            
            # 핀테크 및 디지털 금융
            "오픈뱅킹 API 보안을 위해 OAuth 2.0 기반 인증과 TLS 1.2 이상의 암호화 통신을 사용해야 합니다.",
            "블록체인 기반 금융 서비스는 스마트 컨트랙트 감사를 통해 보안 취약점을 사전에 제거해야 합니다.",
            
            # 인증 및 본인확인
            "금융 거래 시 다중 인증(MFA)을 적용하여 보안을 강화합니다. 생체 인증과 OTP를 조합하는 것이 권장됩니다.",
        ]
        
        metadata = [
            {"category": "사이버보안", "source": "보안가이드", "id": f"doc_{i+1}"}
            for i in range(len(documents))
        ]
        
        logger.info(f"추가할 문서 수: {len(documents)}")
        
        try:
            start_time = time.time()
            num_added = self.pipeline.add_documents(
                texts=documents,
                metadata=metadata,
                batch_size=8
            )
            add_time = time.time() - start_time
            
            logger.info(f"✅ {num_added}개 문서 추가 완료 (소요시간: {add_time:.2f}초)")
            logger.info(f"   평균 처리 시간: {add_time/num_added:.3f}초/문서")
            
            # 지식 베이스 저장
            kb_path = "test_knowledge_base.pkl"
            self.pipeline.save_knowledge_base(kb_path)
            logger.info(f"💾 지식 베이스 저장: {kb_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 문서 추가 실패: {e}")
            return False
    
    def test_retrieval(self, query: str, test_name: str = "Test"):
        """검색 테스트 - 모든 중간 과정 기록"""
        self.log_section(f"4. 검색 테스트: {test_name}")
        
        test_case = {
            "name": test_name,
            "query": query,
            "stages": {},
            "timings": {},
            "results": {}
        }
        
        logger.info(f"질문: {query}")
        logger.info("-" * 40)
        
        try:
            # Stage 1: Query Embedding
            logger.info("\n[Stage 1] 질문 임베딩 생성")
            start = time.time()
            query_embedding = self.pipeline.embedder.embed(query, is_query=True)
            embed_time = time.time() - start
            
            logger.info(f"  - 임베딩 차원: {len(query_embedding)}")
            logger.info(f"  - 임베딩 norm: {torch.norm(torch.tensor(query_embedding)):.4f}")
            logger.info(f"  - 소요 시간: {embed_time:.3f}초")
            
            test_case["stages"]["embedding"] = {
                "dimension": len(query_embedding),
                "norm": float(torch.norm(torch.tensor(query_embedding))),
                "time": embed_time
            }
            
            # Stage 2: Initial Retrieval
            logger.info("\n[Stage 2] 초기 문서 검색")
            start = time.time()
            
            # 많은 후보를 먼저 검색 (reranking을 위해)
            initial_k = 10 if self.enable_reranking else 5
            retrieved_docs = self.pipeline.retrieve(query, top_k=initial_k)
            retrieve_time = time.time() - start
            
            logger.info(f"  - 검색된 문서 수: {len(retrieved_docs)}")
            logger.info(f"  - 소요 시간: {retrieve_time:.3f}초")
            
            # 초기 검색 결과 출력
            logger.info("\n  초기 검색 결과:")
            for i, doc in enumerate(retrieved_docs[:5], 1):
                score = doc.get('score', 0)
                content = doc.get('content', '')[:100]
                logger.info(f"    {i}. [점수: {score:.4f}] {content}...")
            
            test_case["stages"]["retrieval"] = {
                "num_docs": len(retrieved_docs),
                "time": retrieve_time,
                "top_scores": [doc.get('score', 0) for doc in retrieved_docs[:5]]
            }
            
            # Stage 3: Reranking (if enabled)
            if self.enable_reranking and hasattr(self.pipeline, 'reranker') and self.pipeline.reranker:
                logger.info("\n[Stage 3] 리랭킹")
                start = time.time()
                
                # Reranker를 직접 호출
                reranked_docs = self.pipeline.reranker.rerank(
                    query=query,
                    documents=retrieved_docs,
                    top_k=5
                )
                rerank_time = time.time() - start
                
                logger.info(f"  - 리랭킹 후 문서 수: {len(reranked_docs)}")
                logger.info(f"  - 소요 시간: {rerank_time:.3f}초")
                
                # 리랭킹 결과 출력
                logger.info("\n  리랭킹 결과:")
                for i, doc in enumerate(reranked_docs, 1):
                    rerank_score = doc.get('rerank_score', 0)
                    original_score = doc.get('score', 0)
                    content = doc.get('content', '')[:100]
                    logger.info(f"    {i}. [리랭킹: {rerank_score:.4f}, 원본: {original_score:.4f}]")
                    logger.info(f"       {content}...")
                
                test_case["stages"]["reranking"] = {
                    "num_docs": len(reranked_docs),
                    "time": rerank_time,
                    "score_changes": [
                        {
                            "original": doc.get('score', 0),
                            "reranked": doc.get('rerank_score', 0)
                        }
                        for doc in reranked_docs
                    ]
                }
                
                final_docs = reranked_docs
            else:
                final_docs = retrieved_docs[:5]
            
            # Stage 4: Context Generation
            logger.info("\n[Stage 4] 컨텍스트 생성")
            start = time.time()
            context = self.pipeline.generate_context(query, top_k=5, max_length=2000)
            context_time = time.time() - start
            
            logger.info(f"  - 컨텍스트 길이: {len(context)} 문자")
            logger.info(f"  - 소요 시간: {context_time:.3f}초")
            logger.info("\n  생성된 컨텍스트 (첫 500자):")
            logger.info(f"    {context[:500]}...")
            
            test_case["stages"]["context"] = {
                "length": len(context),
                "time": context_time
            }
            
            # 전체 시간 계산
            total_time = embed_time + retrieve_time
            if self.enable_reranking:
                total_time += rerank_time
            total_time += context_time
            
            logger.info(f"\n⏱️ 전체 처리 시간: {total_time:.3f}초")
            
            test_case["timings"] = {
                "embedding": embed_time,
                "retrieval": retrieve_time,
                "reranking": rerank_time if self.enable_reranking else 0,
                "context": context_time,
                "total": total_time
            }
            
            test_case["results"]["success"] = True
            test_case["results"]["final_docs"] = len(final_docs)
            
        except Exception as e:
            logger.error(f"❌ 검색 테스트 실패: {e}")
            test_case["results"]["success"] = False
            test_case["results"]["error"] = str(e)
        
        self.test_results["test_cases"].append(test_case)
        return test_case
    
    def run_test_suite(self):
        """전체 테스트 스위트 실행"""
        self.log_section("RAG 시스템 통합 테스트 시작", "=")
        
        # 1. 시스템 체크
        device = self.check_system()
        
        # 2. 파이프라인 초기화
        if not self.initialize_pipeline(device):
            logger.error("파이프라인 초기화 실패로 테스트 중단")
            return False
        
        # 3. 문서 추가
        if not self.add_test_documents():
            logger.error("문서 추가 실패로 테스트 중단")
            return False
        
        # 4. 다양한 질문으로 테스트
        test_queries = [
            ("사이버 보안 리스크 관리 방법은?", "사이버보안"),
            ("금융 기관의 개인정보 보호 규정은?", "규제준수"),
            ("랜섬웨어 공격 대응 방안은?", "위협대응"),
            ("오픈뱅킹 API 보안 요구사항은?", "핀테크"),
            ("금융 거래 본인 인증 강화 방법은?", "인증"),
        ]
        
        self.log_section("5. 다양한 질문 테스트")
        
        for query, category in test_queries:
            self.test_retrieval(query, test_name=category)
            logger.info("\n" + "="*60 + "\n")
        
        # 6. 성능 통계
        self.calculate_performance_metrics()
        
        # 7. 결과 저장
        self.save_results()
        
        return True
    
    def calculate_performance_metrics(self):
        """성능 메트릭 계산"""
        self.log_section("6. 성능 통계")
        
        if not self.test_results["test_cases"]:
            logger.warning("테스트 케이스가 없어 통계 계산 불가")
            return
        
        # 시간 통계
        timings = {
            "embedding": [],
            "retrieval": [],
            "reranking": [],
            "context": [],
            "total": []
        }
        
        for case in self.test_results["test_cases"]:
            if case["results"]["success"]:
                for key in timings:
                    if key in case["timings"]:
                        timings[key].append(case["timings"][key])
        
        metrics = {}
        for key, values in timings.items():
            if values:
                metrics[f"{key}_avg"] = sum(values) / len(values)
                metrics[f"{key}_min"] = min(values)
                metrics[f"{key}_max"] = max(values)
        
        self.test_results["performance_metrics"] = metrics
        
        # 출력
        logger.info("평균 처리 시간:")
        logger.info(f"  - 임베딩: {metrics.get('embedding_avg', 0):.3f}초")
        logger.info(f"  - 검색: {metrics.get('retrieval_avg', 0):.3f}초")
        if self.enable_reranking:
            logger.info(f"  - 리랭킹: {metrics.get('reranking_avg', 0):.3f}초")
        logger.info(f"  - 컨텍스트: {metrics.get('context_avg', 0):.3f}초")
        logger.info(f"  - 전체: {metrics.get('total_avg', 0):.3f}초")
        
        # 캐시 통계 (if available)
        if hasattr(self.pipeline, 'reranker') and self.pipeline.reranker:
            if hasattr(self.pipeline.reranker, 'get_stats'):
                stats = self.pipeline.reranker.get_stats()
                if 'cache_stats' in stats:
                    cache = stats['cache_stats']
                    logger.info("\n캐시 통계:")
                    logger.info(f"  - 적중: {cache.get('hits', 0)}")
                    logger.info(f"  - 미스: {cache.get('misses', 0)}")
                    hit_rate = cache['hits'] / (cache['hits'] + cache['misses']) if (cache['hits'] + cache['misses']) > 0 else 0
                    logger.info(f"  - 적중률: {hit_rate:.1%}")
    
    def save_results(self):
        """테스트 결과 저장"""
        self.log_section("7. 결과 저장")
        
        # JSON 파일로 저장
        result_file = log_dir / f"rag_test_results_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 테스트 결과 저장: {result_file}")
        logger.info(f"✅ 로그 파일: {log_file}")
        
        # 요약 출력
        logger.info("\n" + "="*60)
        logger.info("테스트 완료 요약")
        logger.info("="*60)
        logger.info(f"총 테스트 케이스: {len(self.test_results['test_cases'])}")
        success_count = sum(1 for case in self.test_results['test_cases'] if case['results']['success'])
        logger.info(f"성공: {success_count}/{len(self.test_results['test_cases'])}")
        
        if self.test_results["performance_metrics"]:
            avg_total = self.test_results["performance_metrics"].get("total_avg", 0)
            logger.info(f"평균 응답 시간: {avg_total:.3f}초")


def main(use_reranking=False):
    """메인 실행 함수
    
    Args:
        use_reranking: Reranking 사용 여부 (기본값: False)
    """
    print("="*60)
    print("RAG 시스템 통합 테스트")
    print("모든 중간 과정을 상세히 기록합니다")
    print("="*60)
    
    # 명령줄 인자 또는 환경 변수로 설정 가능
    import os
    if os.environ.get('USE_RERANKING', '').lower() == 'true':
        use_reranking = True
    
    if use_reranking:
        print("\n⚠️ Reranking 모드 활성화")
        print("- Qwen3-Reranker-4B 모델 사용")
        print("- transformers>=4.51.0 필요")
    else:
        print("\n📌 기본 모드 (Reranking 비활성화)")
        print("- 벡터 유사도 검색만 사용")
    
    # 테스터 생성 및 실행
    tester = RAGSystemTester(enable_reranking=use_reranking)
    success = tester.run_test_suite()
    
    if success:
        print("\n✅ 모든 테스트 완료!")
        print(f"로그 파일을 확인하세요: logs/")
    else:
        print("\n❌ 테스트 중 오류 발생")
    
    return success


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