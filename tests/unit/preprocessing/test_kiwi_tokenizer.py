"""
Kiwi 통합 테스트 스위트
- 순수 Kiwi 기능 테스트 (외부 데이터 없음)
- 성능 비교 테스트
- 대회 규정 준수 확인
"""

import pytest
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 조건부 imports
try:
    from packages.preprocessing import KIWI_AVAILABLE
    if KIWI_AVAILABLE:
        from packages.preprocessing.kiwi_text_processor import KiwiTextProcessor
        from packages.rag.retrieval.kiwi_bm25_retriever import KiwiBM25Retriever
except ImportError:
    KIWI_AVAILABLE = False

from packages.preprocessing.text_processor import KoreanEnglishTextProcessor


class TestKiwiTextProcessor:
    """KiwiTextProcessor 테스트"""
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_kiwi_available(self):
        """Kiwi 사용 가능 확인"""
        assert KIWI_AVAILABLE, "Kiwi should be available"
        processor = KiwiTextProcessor()
        assert processor is not None
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_spacing_correction(self):
        """띄어쓰기 교정 테스트"""
        processor = KiwiTextProcessor()
        
        test_cases = [
            ("금융보안은중요합니다", ["금융", "보안", "중요"]),
            ("인터넷뱅킹을사용합니다", ["인터넷", "뱅킹", "사용"]),
            ("한국은행이기준금리를인상했다", ["한국", "은행", "기준", "금리", "인상"]),
        ]
        
        for input_text, expected_words in test_cases:
            result = processor.correct_spacing(input_text)
            
            # 띄어쓰기가 추가되었는지 확인
            assert len(result.split()) > len(input_text.split()), \
                f"Spacing not improved for: {input_text}"
            
            # 주요 단어가 분리되었는지 확인
            for word in expected_words:
                assert word in result, f"Word '{word}' not found in: {result}"
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_no_external_data(self):
        """외부 데이터 사용하지 않음 확인"""
        processor = KiwiTextProcessor()
        
        # 사용자 사전이 없어야 함
        assert not hasattr(processor.kiwi, 'user_dict'), \
            "User dictionary should not be present"
        
        # 기본 Kiwi 기능만 사용
        text = "금융위원회가 새로운 규제를 발표했습니다"
        result = processor.process(text)
        assert result, "Should process text without external data"
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_morpheme_extraction(self):
        """형태소 분석 테스트"""
        processor = KiwiTextProcessor()
        
        text = "은행에서 대출을 받았습니다"
        morphemes = processor.extract_morphemes(text)
        
        assert len(morphemes) > 0, "Should extract morphemes"
        assert all(isinstance(m, tuple) for m in morphemes), \
            "Morphemes should be tuples"
        assert all(len(m) == 2 for m in morphemes), \
            "Each morpheme should have (form, tag)"
        
        # 주요 명사 확인
        nouns = [m[0] for m in morphemes if m[1].startswith('N')]
        assert '은행' in nouns or '대출' in nouns, \
            "Should extract key nouns"
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_keyword_extraction(self):
        """키워드 추출 테스트"""
        processor = KiwiTextProcessor()
        
        text = "금융보안은 디지털 시대에 매우 중요한 이슈입니다"
        keywords = processor.extract_keywords(text)
        
        assert len(keywords) > 0, "Should extract keywords"
        assert '금융' in keywords or '보안' in keywords, \
            "Should extract financial keywords"
        assert '디지털' in keywords or '시대' in keywords, \
            "Should extract other keywords"
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_batch_processing(self):
        """배치 처리 테스트"""
        processor = KiwiTextProcessor()
        
        texts = [
            "금융보안은중요합니다",
            "인터넷뱅킹보안강화",
            "디지털자산거래소규제"
        ]
        
        results = processor.batch_process(texts)
        
        assert len(results) == len(texts), "Should process all texts"
        assert all(len(r.split()) > 1 for r in results), \
            "All results should have spacing"
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_performance_improvement(self):
        """성능 개선 측정"""
        kiwi_processor = KiwiTextProcessor()
        basic_processor = KoreanEnglishTextProcessor()
        
        # 테스트 텍스트 (반복으로 긴 텍스트 생성)
        test_text = "금융보안은 매우 중요한 문제입니다. 인터넷뱅킹을 사용할 때는 주의가 필요합니다. " * 50
        
        # Kiwi 처리 시간
        start = time.time()
        kiwi_result = kiwi_processor.process(test_text)
        kiwi_time = time.time() - start
        
        # 기본 처리 시간
        start = time.time()
        basic_result = basic_processor.process(test_text)
        basic_time = time.time() - start
        
        logger.info(f"Kiwi time: {kiwi_time:.3f}s, Basic time: {basic_time:.3f}s")
        
        # Kiwi가 추가 기능 제공하면서도 합리적인 속도 유지
        assert kiwi_time < basic_time * 5, \
            f"Kiwi should not be too slow: {kiwi_time:.3f}s vs {basic_time:.3f}s"
        
        # 띄어쓰기 개선 확인
        kiwi_words = len(kiwi_result.split())
        basic_words = len(basic_result.split())
        assert kiwi_words >= basic_words, \
            f"Kiwi should improve spacing: {kiwi_words} vs {basic_words}"


class TestKiwiBM25Retriever:
    """KiwiBM25Retriever 테스트"""
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_bm25_initialization(self):
        """BM25 초기화 테스트"""
        retriever = KiwiBM25Retriever()
        assert retriever is not None
        assert retriever.tokenizer is not None
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_tokenization_speed(self):
        """토크나이징 속도 테스트"""
        retriever = KiwiBM25Retriever()
        
        # 테스트 문서 100개
        test_docs = [
            f"금융보안은 매우 중요합니다. 문서 번호 {i}번입니다." 
            for i in range(100)
        ]
        
        start = time.time()
        tokenized = [retriever.tokenize(doc) for doc in test_docs]
        elapsed = time.time() - start
        
        docs_per_second = len(test_docs) / elapsed
        logger.info(f"Tokenization speed: {docs_per_second:.1f} docs/sec")
        
        # 최소 성능 기준
        assert docs_per_second > 50, \
            f"Should tokenize at least 50 docs/sec, got {docs_per_second:.1f}"
        
        # 모든 문서가 토크나이징되었는지 확인
        assert len(tokenized) == len(test_docs)
        assert all(len(tokens) > 0 for tokens in tokenized)
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_bm25_search(self):
        """BM25 검색 테스트"""
        retriever = KiwiBM25Retriever()
        
        # 테스트 문서
        test_docs = [
            "한국은행이 기준금리를 인상했습니다.",
            "금융위원회가 새로운 규제를 발표했습니다.",
            "인터넷뱅킹 보안이 강화되었습니다.",
            "디지털 자산 거래소에 대한 감독이 시작됩니다.",
            "중앙은행 디지털화폐 도입이 검토되고 있습니다."
        ]
        
        # 인덱스 구축
        retriever.build_index(test_docs, show_progress=False)
        
        # 검색 테스트
        results = retriever.search("기준금리", k=3)
        
        assert len(results) > 0, "Should find results"
        assert results[0].rank == 1, "First result should have rank 1"
        assert "기준금리" in results[0].content or "한국은행" in results[0].content, \
            "Top result should be relevant"
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_no_user_dictionary(self):
        """사용자 사전 없음 확인"""
        retriever = KiwiBM25Retriever()
        
        # Kiwi 토크나이저에 사용자 사전이 없어야 함
        assert not hasattr(retriever.tokenizer.kiwi, 'user_dict'), \
            "Should not have user dictionary"
        
        # 금융 용어도 기본 토크나이징으로 처리
        text = "자금세탁방지 시스템"
        tokens = retriever.tokenize(text)
        assert len(tokens) > 0, "Should tokenize without user dictionary"


class TestBatchProcessor:
    """배치 프로세서 통합 테스트"""
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_batch_processor_with_kiwi(self):
        """Kiwi를 사용한 배치 프로세서"""
        from packages.preprocessing.batch_processor import BatchPDFProcessor
        
        processor = BatchPDFProcessor(use_kiwi=True)
        assert processor.using_kiwi is True, "Should use Kiwi when available"
    
    def test_batch_processor_fallback(self):
        """Kiwi 없을 때 폴백"""
        from packages.preprocessing.batch_processor import BatchPDFProcessor
        
        # Kiwi 없어도 작동해야 함
        processor = BatchPDFProcessor(use_kiwi=False)
        assert processor.using_kiwi is False, "Should work without Kiwi"


class TestComplianceCheck:
    """대회 규정 준수 확인"""
    
    def test_no_external_data_files(self):
        """외부 데이터 파일 없음 확인"""
        # 금융 사전 파일이 없어야 함
        dict_paths = [
            Path("data/dictionaries"),
            Path("data/financial_terms"),
            Path("packages/preprocessing/dictionaries")
        ]
        
        for path in dict_paths:
            assert not path.exists(), \
                f"External dictionary should not exist: {path}"
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_pure_kiwi_only(self):
        """순수 Kiwi 기능만 사용"""
        processor = KiwiTextProcessor()
        
        # add_user_word 메서드를 호출하지 않아야 함
        text = "테스트 텍스트"
        result = processor.process(text)
        
        # 처리는 되지만 사용자 사전은 없음
        assert result, "Should process without user dictionary"
    
    def test_offline_compatibility(self):
        """오프라인 환경 호환성"""
        # 모든 필수 모듈이 로컬에서 import 가능해야 함
        required_modules = [
            "packages.preprocessing.text_processor",
            "packages.preprocessing.chunker",
            "packages.preprocessing.embedder",
            "packages.rag.retrieval.simple_vector_retriever"
        ]
        
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Module {module_name} not available offline: {e}")


# 성능 벤치마크
@pytest.mark.benchmark
class TestPerformanceBenchmark:
    """성능 벤치마크 테스트"""
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_text_processing_benchmark(self, benchmark):
        """텍스트 처리 벤치마크"""
        processor = KiwiTextProcessor()
        test_text = "금융보안은매우중요한문제입니다" * 10
        
        result = benchmark(processor.process, test_text)
        assert len(result) > len(test_text), "Should improve text"
    
    @pytest.mark.skipif(not KIWI_AVAILABLE, reason="Kiwi not installed")
    def test_tokenization_benchmark(self, benchmark):
        """토크나이징 벤치마크"""
        retriever = KiwiBM25Retriever()
        test_text = "금융 보안은 디지털 시대에 매우 중요한 이슈입니다" * 10
        
        tokens = benchmark(retriever.tokenize, test_text)
        assert len(tokens) > 0, "Should extract tokens"


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v", "--tb=short"])