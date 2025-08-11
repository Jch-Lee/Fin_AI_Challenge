"""
KURE-v1 모델 마이그레이션 테스트 스크립트
기존 모델과 새 모델의 성능 및 호환성 비교
"""

import os
import sys
import time
import logging
import numpy as np
from typing import List, Dict, Any

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from packages.preprocessing.embedder import EmbeddingGenerator, TextEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMigrationTester:
    """모델 마이그레이션 테스트 클래스"""
    
    def __init__(self):
        self.test_texts = [
            "금융 AI 시스템의 보안 요구사항은 무엇인가요?",
            "개인정보 보호를 위한 기술적 조치는 어떤 것들이 있나요?",
            "머신러닝 모델의 취약점과 대응 방안은?",
            "금융 서비스에서 사이버 보안 위협의 종류",
            "AI 모델 공격 방어를 위한 전략",
            "디지털 자산 보호 방법",
            "금융 데이터 암호화 기술",
            "보안 감사 및 모니터링 시스템"
        ]
    
    def test_model_loading(self, model_name: str) -> Dict[str, Any]:
        """모델 로딩 테스트"""
        logger.info(f"Testing model loading: {model_name}")
        
        try:
            start_time = time.time()
            embedder = EmbeddingGenerator(model_name=model_name)
            load_time = time.time() - start_time
            
            return {
                "model_name": model_name,
                "success": True,
                "load_time": load_time,
                "embedding_dim": embedder.embedding_dim,
                "device": embedder.device,
                "error": None
            }
        except Exception as e:
            return {
                "model_name": model_name,
                "success": False,
                "load_time": -1,
                "embedding_dim": -1,
                "device": None,
                "error": str(e)
            }
    
    def test_embedding_generation(self, model_name: str) -> Dict[str, Any]:
        """임베딩 생성 성능 테스트"""
        logger.info(f"Testing embedding generation: {model_name}")
        
        try:
            embedder = EmbeddingGenerator(model_name=model_name)
            
            # 단일 임베딩 테스트
            start_time = time.time()
            single_embedding = embedder.generate_embedding(self.test_texts[0])
            single_time = time.time() - start_time
            
            # 배치 임베딩 테스트
            start_time = time.time()
            batch_embeddings = embedder.generate_embeddings_batch(self.test_texts)
            batch_time = time.time() - start_time
            
            # 유사도 계산 테스트
            start_time = time.time()
            query_emb = batch_embeddings[0]
            doc_embs = batch_embeddings[1:]
            similarities = embedder.compute_similarity(query_emb, doc_embs)
            similarity_time = time.time() - start_time
            
            return {
                "model_name": model_name,
                "success": True,
                "embedding_dim": embedder.embedding_dim,
                "single_embedding_time": single_time * 1000,  # ms
                "batch_embedding_time": batch_time * 1000,  # ms
                "similarity_calculation_time": similarity_time * 1000,  # ms
                "avg_time_per_text": (batch_time / len(self.test_texts)) * 1000,  # ms
                "embedding_shape": single_embedding.shape,
                "batch_shape": batch_embeddings.shape,
                "max_similarity": float(np.max(similarities)),
                "min_similarity": float(np.min(similarities)),
                "avg_similarity": float(np.mean(similarities)),
                "error": None
            }
        except Exception as e:
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def test_memory_usage(self, model_name: str) -> Dict[str, Any]:
        """메모리 사용량 테스트"""
        logger.info(f"Testing memory usage: {model_name}")
        
        try:
            import psutil
            import torch
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 시작 메모리 측정
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_gpu_memory = 0
            
            if torch.cuda.is_available():
                start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            # 모델 로드
            embedder = EmbeddingGenerator(model_name=model_name)
            
            # 임베딩 생성
            embeddings = embedder.generate_embeddings_batch(self.test_texts)
            
            # 종료 메모리 측정
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            end_gpu_memory = 0
            
            if torch.cuda.is_available():
                end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            return {
                "model_name": model_name,
                "success": True,
                "start_cpu_memory_mb": start_memory,
                "end_cpu_memory_mb": end_memory,
                "cpu_memory_usage_mb": end_memory - start_memory,
                "start_gpu_memory_mb": start_gpu_memory,
                "end_gpu_memory_mb": end_gpu_memory,
                "gpu_memory_usage_mb": end_gpu_memory - start_gpu_memory,
                "error": None
            }
        except Exception as e:
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def compare_embedding_quality(self, old_model: str, new_model: str) -> Dict[str, Any]:
        """임베딩 품질 비교"""
        logger.info(f"Comparing embedding quality: {old_model} vs {new_model}")
        
        try:
            old_embedder = EmbeddingGenerator(model_name=old_model)
            new_embedder = EmbeddingGenerator(model_name=new_model)
            
            old_embeddings = old_embedder.generate_embeddings_batch(self.test_texts)
            new_embeddings = new_embedder.generate_embeddings_batch(self.test_texts)
            
            # 내부 일관성 테스트 (동일 모델 내 유사도)
            old_similarities = []
            new_similarities = []
            
            query_text = self.test_texts[0]  # 첫 번째를 쿼리로 사용
            
            for i in range(1, len(self.test_texts)):
                old_sim = old_embedder.compute_similarity(
                    old_embeddings[0], old_embeddings[i:i+1]
                )[0]
                new_sim = new_embedder.compute_similarity(
                    new_embeddings[0], new_embeddings[i:i+1]
                )[0]
                
                old_similarities.append(old_sim)
                new_similarities.append(new_sim)
            
            return {
                "old_model": old_model,
                "new_model": new_model,
                "success": True,
                "old_embedding_dim": old_embeddings.shape[1],
                "new_embedding_dim": new_embeddings.shape[1],
                "old_avg_similarity": float(np.mean(old_similarities)),
                "new_avg_similarity": float(np.mean(new_similarities)),
                "old_std_similarity": float(np.std(old_similarities)),
                "new_std_similarity": float(np.std(new_similarities)),
                "similarity_scores_old": [float(s) for s in old_similarities],
                "similarity_scores_new": [float(s) for s in new_similarities],
                "error": None
            }
        except Exception as e:
            return {
                "old_model": old_model,
                "new_model": new_model,
                "success": False,
                "error": str(e)
            }
    
    def run_full_comparison(self) -> Dict[str, Any]:
        """전체 비교 테스트 실행"""
        logger.info("Starting full model comparison...")
        
        old_model = "jhgan/ko-sroberta-multitask"
        new_model = "nlpai-lab/KURE-v1"
        
        results = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "test_texts_count": len(self.test_texts),
            "old_model_results": {},
            "new_model_results": {},
            "comparison_results": {},
            "recommendation": ""
        }
        
        # 개별 모델 테스트
        for model_name in [old_model, new_model]:
            model_key = "old_model_results" if model_name == old_model else "new_model_results"
            
            results[model_key] = {
                "loading_test": self.test_model_loading(model_name),
                "performance_test": self.test_embedding_generation(model_name),
                "memory_test": self.test_memory_usage(model_name)
            }
        
        # 비교 테스트
        results["comparison_results"] = self.compare_embedding_quality(old_model, new_model)
        
        # 권장사항 생성
        results["recommendation"] = self._generate_recommendation(results)
        
        return results
    
    def _generate_recommendation(self, results: Dict[str, Any]) -> str:
        """테스트 결과를 바탕으로 권장사항 생성"""
        old_results = results["old_model_results"]
        new_results = results["new_model_results"]
        
        recommendation_parts = []
        
        # 성능 비교
        if (new_results["performance_test"]["success"] and 
            old_results["performance_test"]["success"]):
            
            old_time = old_results["performance_test"]["avg_time_per_text"]
            new_time = new_results["performance_test"]["avg_time_per_text"]
            
            if new_time < old_time * 1.2:  # 20% 이내 차이
                recommendation_parts.append("[OK] 성능: KURE-v1이 비슷하거나 더 좋은 성능")
            elif new_time < old_time * 1.5:  # 50% 이내 차이
                recommendation_parts.append("[WARN] 성능: KURE-v1이 약간 느리지만 허용 범위")
            else:
                recommendation_parts.append("[ERROR] 성능: KURE-v1이 현저히 느림")
        
        # 메모리 사용량 비교
        if (new_results["memory_test"]["success"] and 
            old_results["memory_test"]["success"]):
            
            old_memory = old_results["memory_test"]["cpu_memory_usage_mb"]
            new_memory = new_results["memory_test"]["cpu_memory_usage_mb"]
            
            if new_memory < old_memory * 1.3:  # 30% 이내 증가
                recommendation_parts.append("[OK] 메모리: 허용 범위 내 사용량")
            else:
                recommendation_parts.append("[WARN] 메모리: 사용량 증가 주의")
        
        # 차원 및 품질 비교
        comparison = results["comparison_results"]
        if comparison["success"]:
            recommendation_parts.append(
                f"[INFO] 차원 변화: {comparison['old_embedding_dim']} → {comparison['new_embedding_dim']}"
            )
            recommendation_parts.append("[WARN] 기존 임베딩 데이터 재생성 필요")
        
        # 최종 권장사항
        if len([r for r in recommendation_parts if r.startswith("[OK]")]) >= 2:
            recommendation_parts.append("[RECOMMEND] KURE-v1로 마이그레이션 진행")
        elif len([r for r in recommendation_parts if r.startswith("[ERROR]")]) > 0:
            recommendation_parts.append("[WARNING] 마이그레이션 재검토 필요")
        else:
            recommendation_parts.append("[REVIEW] 추가 테스트 후 결정")
        
        return "\n".join(recommendation_parts)


def main():
    """메인 실행 함수"""
    tester = ModelMigrationTester()
    
    print("="*80)
    print("KURE-v1 모델 마이그레이션 테스트")
    print("="*80)
    
    # 전체 비교 테스트 실행
    results = tester.run_full_comparison()
    
    # 결과 출력
    print("\n" + "="*50)
    print("테스트 결과 요약")
    print("="*50)
    
    print(f"\n[TIME] 테스트 시간: {results['timestamp']}")
    print(f"[INFO] 테스트 텍스트 수: {results['test_texts_count']}")
    
    # 기존 모델 결과
    old_results = results["old_model_results"]
    if old_results["loading_test"]["success"]:
        print(f"\n[OLD] 기존 모델 (jhgan/ko-sroberta-multitask):")
        print(f"  - 차원: {old_results['loading_test']['embedding_dim']}")
        print(f"  - 로딩 시간: {old_results['loading_test']['load_time']:.2f}초")
        if old_results["performance_test"]["success"]:
            print(f"  - 평균 임베딩 시간: {old_results['performance_test']['avg_time_per_text']:.2f}ms")
    
    # 새 모델 결과
    new_results = results["new_model_results"]
    if new_results["loading_test"]["success"]:
        print(f"\n[NEW] 새 모델 (nlpai-lab/KURE-v1):")
        print(f"  - 차원: {new_results['loading_test']['embedding_dim']}")
        print(f"  - 로딩 시간: {new_results['loading_test']['load_time']:.2f}초")
        if new_results["performance_test"]["success"]:
            print(f"  - 평균 임베딩 시간: {new_results['performance_test']['avg_time_per_text']:.2f}ms")
    
    # 권장사항
    print(f"\n[RECOMMEND] 권장사항:")
    print(results["recommendation"])
    
    # 상세 결과 저장
    import json
    results_file = f"kure_v1_migration_test_{results['timestamp']}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SAVE] 상세 결과 저장: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()