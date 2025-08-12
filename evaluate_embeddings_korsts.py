"""
KorSTS 벤치마크를 활용한 한국어 임베딩 모델 평가 스크립트
금융보안 도메인 RAG 시스템의 임베딩 품질 검증
"""

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import logging
from typing import Dict, List, Tuple
import json
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KorSTSEvaluator:
    """KorSTS 데이터셋을 이용한 임베딩 모델 평가"""
    
    def __init__(self, model_name: str = "nlpai-lab/KURE-v1", device: str = None):
        """
        Args:
            model_name: 평가할 임베딩 모델
            device: 연산 디바이스
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing evaluator for model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # 모델 로드
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def load_korsts_dataset(self) -> Tuple[List[str], List[str], List[float]]:
        """KorSTS 데이터셋 로드"""
        logger.info("Loading KorSTS dataset...")
        
        try:
            # KorSTS 데이터셋 로드 (KLUE 벤치마크의 일부)
            dataset = load_dataset("klue", "sts", split="validation")
            
            sentences1 = []
            sentences2 = []
            scores = []
            
            for item in dataset:
                sentences1.append(item['sentence1'])
                sentences2.append(item['sentence2'])
                # KorSTS 점수는 0-5 범위, 정규화하여 0-1로 변환
                scores.append(item['labels']['label'] / 5.0)
            
            logger.info(f"Loaded {len(sentences1)} sentence pairs from KorSTS")
            return sentences1, sentences2, scores
            
        except Exception as e:
            logger.warning(f"Failed to load official KorSTS: {e}")
            logger.info("Creating sample test data for evaluation...")
            
            # 대체 샘플 데이터 (실제 테스트용)
            sentences1 = [
                "금융 보안은 매우 중요합니다.",
                "비밀번호는 정기적으로 변경해야 합니다.",
                "은행 계좌를 안전하게 관리하세요.",
                "투자할 때는 위험을 고려해야 합니다.",
                "디지털 자산을 보호하는 것이 중요합니다."
            ]
            sentences2 = [
                "금융 보안의 중요성은 매우 큽니다.",
                "패스워드를 주기적으로 바꿔야 합니다.",
                "오늘 날씨가 매우 좋습니다.",
                "투자 시 리스크 관리가 필요합니다.",
                "암호화폐를 안전하게 보관해야 합니다."
            ]
            scores = [0.9, 0.95, 0.1, 0.85, 0.8]
            
            return sentences1, sentences2, scores
    
    def evaluate_embeddings(self, 
                           sentences1: List[str], 
                           sentences2: List[str], 
                           true_scores: List[float],
                           batch_size: int = 32) -> Dict:
        """임베딩 품질 평가"""
        logger.info(f"Evaluating embeddings for {len(sentences1)} pairs...")
        
        # 임베딩 생성
        embeddings1 = self.model.encode(
            sentences1, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        embeddings2 = self.model.encode(
            sentences2,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 코사인 유사도 계산
        predicted_scores = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            similarity = np.dot(emb1, emb2)  # 이미 정규화됨
            predicted_scores.append(similarity)
        
        # 상관계수 계산
        spearman_corr, spearman_p = spearmanr(true_scores, predicted_scores)
        pearson_corr, pearson_p = pearsonr(true_scores, predicted_scores)
        
        # 추가 메트릭 계산
        mse = np.mean((np.array(true_scores) - np.array(predicted_scores)) ** 2)
        mae = np.mean(np.abs(np.array(true_scores) - np.array(predicted_scores)))
        
        results = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'num_pairs': len(sentences1),
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p),
            'mse': float(mse),
            'mae': float(mae),
            'performance_grade': self._get_performance_grade(spearman_corr)
        }
        
        return results
    
    def _get_performance_grade(self, spearman_corr: float) -> str:
        """성능 등급 판정"""
        if spearman_corr >= 0.8:
            return "Excellent (매우 우수)"
        elif spearman_corr >= 0.7:
            return "Good (우수)"
        elif spearman_corr >= 0.6:
            return "Fair (양호)"
        elif spearman_corr >= 0.5:
            return "Poor (개선 필요)"
        else:
            return "Very Poor (재훈련 필요)"
    
    def evaluate_financial_domain(self) -> Dict:
        """금융 도메인 특화 평가"""
        logger.info("Evaluating financial domain performance...")
        
        # 금융 도메인 테스트 문장 쌍
        financial_pairs = [
            ("금리 인상이 예상됩니다", "이자율이 오를 것으로 보입니다", 0.9),
            ("주식 시장이 상승했습니다", "증권 거래소가 활황입니다", 0.85),
            ("대출 상환이 필요합니다", "융자금을 갚아야 합니다", 0.95),
            ("투자 포트폴리오를 다각화하세요", "분산 투자가 중요합니다", 0.8),
            ("예금 금리가 인하되었습니다", "저축 이자가 낮아졌습니다", 0.9),
            ("비트코인 가격이 급등했습니다", "오늘 날씨가 맑습니다", 0.1),
            ("신용카드 사용을 줄이세요", "카드 결제를 자제하세요", 0.95),
            ("보안 토큰을 생성하세요", "OTP를 만들어야 합니다", 0.85)
        ]
        
        sentences1 = [pair[0] for pair in financial_pairs]
        sentences2 = [pair[1] for pair in financial_pairs]
        true_scores = [pair[2] for pair in financial_pairs]
        
        # 평가 수행
        results = self.evaluate_embeddings(sentences1, sentences2, true_scores)
        results['domain'] = 'financial'
        
        return results
    
    def run_comprehensive_evaluation(self) -> Dict:
        """종합 평가 실행"""
        logger.info("="*60)
        logger.info("Starting comprehensive embedding evaluation")
        logger.info("="*60)
        
        all_results = {}
        
        # 1. KorSTS 벤치마크 평가
        try:
            sentences1, sentences2, scores = self.load_korsts_dataset()
            korsts_results = self.evaluate_embeddings(sentences1, sentences2, scores)
            all_results['korsts'] = korsts_results
            
            logger.info("\n📊 KorSTS Benchmark Results:")
            logger.info(f"  - Spearman Correlation: {korsts_results['spearman_correlation']:.4f}")
            logger.info(f"  - Pearson Correlation: {korsts_results['pearson_correlation']:.4f}")
            logger.info(f"  - Performance Grade: {korsts_results['performance_grade']}")
            
        except Exception as e:
            logger.error(f"KorSTS evaluation failed: {e}")
            all_results['korsts'] = {'error': str(e)}
        
        # 2. 금융 도메인 평가
        try:
            financial_results = self.evaluate_financial_domain()
            all_results['financial'] = financial_results
            
            logger.info("\n💰 Financial Domain Results:")
            logger.info(f"  - Spearman Correlation: {financial_results['spearman_correlation']:.4f}")
            logger.info(f"  - Performance Grade: {financial_results['performance_grade']}")
            
        except Exception as e:
            logger.error(f"Financial domain evaluation failed: {e}")
            all_results['financial'] = {'error': str(e)}
        
        # 3. 종합 판정
        self._print_summary(all_results)
        
        # 결과 저장
        self._save_results(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict):
        """평가 요약 출력"""
        logger.info("\n" + "="*60)
        logger.info("📈 EVALUATION SUMMARY")
        logger.info("="*60)
        
        if 'korsts' in results and 'spearman_correlation' in results['korsts']:
            spearman = results['korsts']['spearman_correlation']
            grade = results['korsts']['performance_grade']
            
            if spearman >= 0.7:
                logger.info(f"✅ 모델 성능: {grade}")
                logger.info(f"   상관계수 {spearman:.4f} ≥ 0.7")
                logger.info("   → 한국어 문서 임베딩이 우수하게 작동합니다!")
            else:
                logger.info(f"⚠️ 모델 성능: {grade}")
                logger.info(f"   상관계수 {spearman:.4f} < 0.7")
                logger.info("   → 추가 학습이 필요할 수 있습니다.")
        
        logger.info("\n💡 권장사항:")
        if 'financial' in results and 'spearman_correlation' in results['financial']:
            fin_score = results['financial']['spearman_correlation']
            if fin_score >= 0.8:
                logger.info("  - 금융 도메인 성능 우수, RAG 시스템에 바로 적용 가능")
            else:
                logger.info("  - 금융 도메인 특화 파인튜닝 고려")
    
    def _save_results(self, results: Dict):
        """평가 결과 저장"""
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"korsts_evaluation_{self.model_name.replace('/', '_')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📁 Results saved to: {output_file}")


def compare_models(model_names: List[str]) -> Dict:
    """여러 모델 비교 평가"""
    logger.info("Comparing multiple models...")
    
    comparison_results = {}
    
    for model_name in model_names:
        logger.info(f"\nEvaluating: {model_name}")
        try:
            evaluator = KorSTSEvaluator(model_name)
            results = evaluator.run_comprehensive_evaluation()
            comparison_results[model_name] = results
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            comparison_results[model_name] = {'error': str(e)}
    
    # 비교 테이블 출력
    logger.info("\n" + "="*60)
    logger.info("📊 MODEL COMPARISON")
    logger.info("="*60)
    
    for model_name, results in comparison_results.items():
        if 'korsts' in results and 'spearman_correlation' in results['korsts']:
            score = results['korsts']['spearman_correlation']
            logger.info(f"{model_name}: {score:.4f}")
    
    return comparison_results


def main():
    """메인 실행 함수"""
    # KURE vs E5 모델 비교 평가
    models_to_evaluate = [
        "nlpai-lab/KURE-v1",  # 현재 메인 모델 (1024차원)
        "dragonkue/multilingual-e5-small-ko",  # 이전 모델 (384차원)
    ]
    
    for model_name in models_to_evaluate:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            evaluator = KorSTSEvaluator(model_name)
            results = evaluator.run_comprehensive_evaluation()
            
            # 성공/실패 판정
            if 'korsts' in results and 'spearman_correlation' in results['korsts']:
                if results['korsts']['spearman_correlation'] >= 0.7:
                    logger.info(f"{model_name}: 성능 기준 통과!")
                else:
                    logger.info(f"{model_name}: 성능 개선 필요")
                    
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
    
    # 전체 모델 비교 실행
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE MODEL COMPARISON")
    logger.info("="*60)
    comparison_results = compare_models(models_to_evaluate)


if __name__ == "__main__":
    main()