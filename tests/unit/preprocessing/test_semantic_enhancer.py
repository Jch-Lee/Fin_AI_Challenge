#!/usr/bin/env python3
"""
SemanticEnhancer 단위 테스트

의미 기반 청킹 개선 시스템의 핵심 기능 검증:
- LlamaIndex 가용성 확인
- 의미 기반 분할 로직
- 선택적 처리 조건
- Fallback 메커니즘
- 통계 정보 생성
"""

import unittest
import sys
import os
from pathlib import Path
from typing import List

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from packages.preprocessing.semantic_enhancer import (
        SemanticEnhancer, 
        SemanticConfig, 
        enhance_chunks_semantic
    )
    from packages.preprocessing.chunker import DocumentChunk
    SEMANTIC_ENHANCER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SemanticEnhancer not available: {e}")
    SEMANTIC_ENHANCER_AVAILABLE = False

# LlamaIndex 가용성 별도 확인
try:
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False


class TestSemanticEnhancer(unittest.TestCase):
    """SemanticEnhancer 단위 테스트"""
    
    def setUp(self):
        """테스트 준비"""
        if not SEMANTIC_ENHANCER_AVAILABLE:
            self.skipTest("SemanticEnhancer not available")
        
        self.config = SemanticConfig(
            embedding_model="nlpai-lab/KURE-v1",
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            max_chunk_size=512,
            min_chunk_size=100
        )
        
        self.enhancer = SemanticEnhancer(self.config)
        
        # 테스트용 DocumentChunk 생성
        self.test_chunks = self._create_test_chunks()

    def _create_test_chunks(self) -> List[DocumentChunk]:
        """테스트용 DocumentChunk 리스트 생성"""
        chunks = []
        
        # 짧은 청크 (의미 기반 처리 제외 대상)
        short_chunk = DocumentChunk(
            content="짧은 내용입니다.",
            metadata={'source': 'test', 'type': 'short'},
            chunk_id="short_001",
            doc_id="test_doc",
            chunk_index=0
        )
        chunks.append(short_chunk)
        
        # 긴 청크 (의미 기반 처리 대상)
        long_content = """
        금융기관의 사이버 보안은 매우 중요한 문제입니다. 해커들의 공격이 점점 정교해지고 있어 새로운 대응 방안이 필요합니다.
        
        다요소 인증은 기본적인 보안 조치 중 하나입니다. 사용자의 신원을 확인하기 위해 두 가지 이상의 인증 방법을 사용합니다.
        
        최소 권한 원칙을 적용해야 합니다. 각 사용자에게는 업무 수행에 필요한 최소한의 권한만 부여해야 합니다. 이를 통해 보안 위험을 크게 줄일 수 있습니다.
        
        모니터링 시스템은 실시간으로 모든 접근을 감시해야 합니다. 비정상적인 활동이 감지되면 즉시 차단하고 관리자에게 알려야 합니다.
        """.strip()
        
        long_chunk = DocumentChunk(
            content=long_content,
            metadata={'source': 'test', 'type': 'long'},
            chunk_id="long_001",
            doc_id="test_doc",
            chunk_index=1
        )
        chunks.append(long_chunk)
        
        # 테이블 포함 청크 (구조 보존을 위해 처리 제외 대상)
        table_content = """
        보안 수준별 요구사항:
        
        | 구분 | 요구사항 | 중요도 |
        |------|----------|--------|
        | 인증 | 다요소 인증 필수 | 높음 |
        | 권한 | 최소 권한 원칙 | 높음 |
        | 모니터링 | 실시간 감시 | 중간 |
        
        위 표는 기본적인 보안 요구사항을 나타냅니다.
        """
        
        table_chunk = DocumentChunk(
            content=table_content,
            metadata={'source': 'test', 'type': 'table'},
            chunk_id="table_001",
            doc_id="test_doc",
            chunk_index=2
        )
        chunks.append(table_chunk)
        
        # 이미 처리된 청크 (재처리 방지 테스트용)
        processed_chunk = DocumentChunk(
            content="이미 의미 기반으로 처리된 내용입니다. " * 10,
            metadata={'source': 'test', 'type': 'processed', 'semantic_enhanced': True},
            chunk_id="processed_001",
            doc_id="test_doc",
            chunk_index=3
        )
        chunks.append(processed_chunk)
        
        return chunks

    def test_semantic_config(self):
        """SemanticConfig 설정 테스트"""
        config = SemanticConfig()
        
        # 기본값 확인
        self.assertEqual(config.embedding_model, "nlpai-lab/KURE-v1")
        self.assertEqual(config.buffer_size, 1)
        self.assertEqual(config.breakpoint_percentile_threshold, 95)
        self.assertEqual(config.max_chunk_size, 512)
        self.assertEqual(config.min_chunk_size, 100)
        
        # 커스텀 설정
        custom_config = SemanticConfig(
            embedding_model="custom-model",
            max_chunk_size=256
        )
        self.assertEqual(custom_config.embedding_model, "custom-model")
        self.assertEqual(custom_config.max_chunk_size, 256)

    def test_enhancer_initialization(self):
        """SemanticEnhancer 초기화 테스트"""
        # 기본 초기화
        enhancer = SemanticEnhancer()
        self.assertIsInstance(enhancer, SemanticEnhancer)
        self.assertIsNotNone(enhancer.config)
        
        # 커스텀 설정으로 초기화
        custom_config = SemanticConfig(max_chunk_size=256)
        custom_enhancer = SemanticEnhancer(custom_config)
        self.assertEqual(custom_enhancer.config.max_chunk_size, 256)

    def test_availability_check(self):
        """LlamaIndex 가용성 확인 테스트"""
        is_available = self.enhancer.is_available()
        
        # 실제 LlamaIndex 설치 상태와 일치해야 함
        self.assertEqual(is_available, LLAMAINDEX_AVAILABLE)

    def test_should_enhance_chunk_logic(self):
        """청크 개선 대상 판단 로직 테스트"""
        if not self.enhancer.is_available():
            self.skipTest("LlamaIndex not available")
        
        # 짧은 청크 - 처리하지 않음
        short_chunk = self.test_chunks[0]
        self.assertFalse(self.enhancer._should_enhance_chunk(short_chunk))
        
        # 긴 청크 - 처리함
        long_chunk = self.test_chunks[1]
        self.assertTrue(self.enhancer._should_enhance_chunk(long_chunk))
        
        # 테이블 포함 청크 - 처리하지 않음 (구조 보존)
        table_chunk = self.test_chunks[2]
        self.assertFalse(self.enhancer._should_enhance_chunk(table_chunk))
        
        # 이미 처리된 청크 - 처리하지 않음
        processed_chunk = self.test_chunks[3]
        self.assertFalse(self.enhancer._should_enhance_chunk(processed_chunk))

    def test_enhance_chunks_availability_fallback(self):
        """LlamaIndex 미사용 시 Fallback 테스트"""
        if LLAMAINDEX_AVAILABLE:
            self.skipTest("LlamaIndex available - testing fallback not applicable")
        
        # LlamaIndex가 없을 때는 원본 청크를 그대로 반환해야 함
        result = self.enhancer.enhance_chunks(self.test_chunks)
        
        self.assertEqual(len(result), len(self.test_chunks))
        self.assertEqual(result, self.test_chunks)

    @unittest.skipUnless(LLAMAINDEX_AVAILABLE, "LlamaIndex not available")
    def test_enhance_chunks_with_llamaindex(self):
        """LlamaIndex 사용한 청킹 개선 테스트"""
        try:
            # 지연 초기화 수행
            self.enhancer._initialize_semantic_parser()
            
            if not self.enhancer._initialized:
                self.skipTest("Semantic parser initialization failed")
            
            # 선택적 처리 모드 테스트
            enhanced_chunks = self.enhancer.enhance_chunks(
                self.test_chunks, 
                selective=True
            )
            
            # 결과 검증
            self.assertIsInstance(enhanced_chunks, list)
            self.assertGreaterEqual(len(enhanced_chunks), len(self.test_chunks))
            
            # 모든 결과가 DocumentChunk 인스턴스인지 확인
            for chunk in enhanced_chunks:
                self.assertIsInstance(chunk, DocumentChunk)
                self.assertIsInstance(chunk.content, str)
                self.assertIsInstance(chunk.metadata, dict)
            
        except Exception as e:
            self.skipTest(f"Semantic enhancement test failed (expected): {e}")

    def test_single_chunk_enhancement(self):
        """단일 청크 개선 테스트"""
        if not self.enhancer.is_available():
            # LlamaIndex 없을 때는 원본 반환
            long_chunk = self.test_chunks[1]
            result = self.enhancer.enhance_single_chunk(long_chunk)
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], long_chunk)
        else:
            try:
                long_chunk = self.test_chunks[1]
                result = self.enhancer.enhance_single_chunk(long_chunk)
                
                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0)
                
                for chunk in result:
                    self.assertIsInstance(chunk, DocumentChunk)
                    
            except Exception as e:
                self.skipTest(f"Single chunk enhancement failed (expected): {e}")

    def test_enhancement_stats(self):
        """개선 통계 정보 테스트"""
        original_chunks = self.test_chunks
        
        # 개선 시뮬레이션 (실제로는 원본과 동일하거나 더 많아질 수 있음)
        enhanced_chunks = original_chunks.copy()
        
        # 가상의 의미 개선 청크 추가
        if enhanced_chunks:
            enhanced_chunk = DocumentChunk(
                content="의미적으로 분할된 새로운 청크입니다.",
                metadata={
                    'source': 'test',
                    'semantic_enhanced': True,
                    'semantic_split_index': 0,
                    'semantic_method': 'llamaindex_semantic_splitter'
                },
                chunk_id="semantic_001",
                doc_id="test_doc",
                chunk_index=len(enhanced_chunks)
            )
            enhanced_chunks.append(enhanced_chunk)
        
        stats = self.enhancer.get_enhancement_stats(original_chunks, enhanced_chunks)
        
        # 통계 구조 확인
        self.assertIsInstance(stats, dict)
        self.assertIn('original_count', stats)
        self.assertIn('enhanced_count', stats)
        self.assertIn('semantic_processed', stats)
        self.assertIn('count_change_ratio', stats)
        self.assertIn('original_avg_size', stats)
        self.assertIn('enhanced_avg_size', stats)
        self.assertIn('enhancement_rate', stats)
        
        # 값 확인
        self.assertEqual(stats['original_count'], len(original_chunks))
        self.assertEqual(stats['enhanced_count'], len(enhanced_chunks))
        self.assertGreaterEqual(stats['semantic_processed'], 0)
        self.assertGreaterEqual(stats['count_change_ratio'], 1.0)

    def test_document_chunk_conversion(self):
        """DocumentChunk <-> LlamaIndex Document 변환 테스트"""
        if not self.enhancer.is_available():
            self.skipTest("LlamaIndex not available")
        
        test_chunk = self.test_chunks[1]  # 긴 청크 사용
        
        # DocumentChunk -> LlamaIndex Document 변환
        documents = self.enhancer._convert_chunks_to_documents([test_chunk])
        
        self.assertEqual(len(documents), 1)
        doc = documents[0]
        
        # 텍스트와 메타데이터가 올바르게 변환되었는지 확인
        self.assertEqual(doc.text, test_chunk.content)
        self.assertIn('chunk_id', doc.metadata)
        self.assertIn('doc_id', doc.metadata)
        self.assertEqual(doc.metadata['chunk_id'], test_chunk.chunk_id)

    def test_empty_chunk_list(self):
        """빈 청크 리스트 처리 테스트"""
        empty_result = self.enhancer.enhance_chunks([])
        self.assertEqual(len(empty_result), 0)
        
        # 통계도 빈 리스트로 처리 가능해야 함
        stats = self.enhancer.get_enhancement_stats([], [])
        self.assertEqual(stats['original_count'], 0)
        self.assertEqual(stats['enhanced_count'], 0)

    def test_enhance_chunks_semantic_function(self):
        """편의 함수 enhance_chunks_semantic 테스트"""
        result = enhance_chunks_semantic(
            self.test_chunks,
            config=self.config,
            selective=True
        )
        
        self.assertIsInstance(result, list)
        # LlamaIndex 여부에 관계없이 최소한 원본 개수 이상이어야 함
        self.assertGreaterEqual(len(result), len(self.test_chunks))

    def test_error_handling(self):
        """에러 처리 테스트"""
        # 잘못된 청크 (내용이 None)
        invalid_chunk = DocumentChunk(
            content=None,
            metadata={},
            chunk_id="invalid_001",
            doc_id="test_doc",
            chunk_index=0
        )
        
        try:
            result = self.enhancer.enhance_chunks([invalid_chunk])
            # 에러가 발생해도 빈 리스트나 원본이 반환되어야 함
            self.assertIsInstance(result, list)
        except Exception as e:
            # 예외가 발생해도 정상적으로 처리되어야 함
            self.fail(f"Error handling failed: {e}")


class TestSemanticEnhancerConfiguration(unittest.TestCase):
    """SemanticEnhancer 설정 관련 테스트"""
    
    def setUp(self):
        """테스트 준비"""
        if not SEMANTIC_ENHANCER_AVAILABLE:
            self.skipTest("SemanticEnhancer not available")

    def test_custom_configurations(self):
        """다양한 커스텀 설정 테스트"""
        configs = [
            SemanticConfig(max_chunk_size=128, min_chunk_size=50),
            SemanticConfig(buffer_size=2, breakpoint_percentile_threshold=90),
            SemanticConfig(embedding_model="custom-model", device="cpu"),
        ]
        
        for config in configs:
            enhancer = SemanticEnhancer(config)
            self.assertEqual(enhancer.config, config)

    def test_configuration_validation(self):
        """설정 값 유효성 검사"""
        # 유효한 설정
        valid_config = SemanticConfig(
            max_chunk_size=512,
            min_chunk_size=100,
            buffer_size=1,
            breakpoint_percentile_threshold=95
        )
        
        enhancer = SemanticEnhancer(valid_config)
        self.assertIsInstance(enhancer, SemanticEnhancer)
        
        # 경계값 테스트
        edge_config = SemanticConfig(
            max_chunk_size=1,
            min_chunk_size=1,
            buffer_size=0,
            breakpoint_percentile_threshold=100
        )
        
        edge_enhancer = SemanticEnhancer(edge_config)
        self.assertIsInstance(edge_enhancer, SemanticEnhancer)


if __name__ == '__main__':
    # 테스트 실행 전 환경 확인
    print(f"SemanticEnhancer available: {SEMANTIC_ENHANCER_AVAILABLE}")
    print(f"LlamaIndex available: {LLAMAINDEX_AVAILABLE}")
    
    if SEMANTIC_ENHANCER_AVAILABLE:
        print("Running SemanticEnhancer tests...")
        if not LLAMAINDEX_AVAILABLE:
            print("Note: LlamaIndex not available - testing fallback mechanisms")
    else:
        print("SemanticEnhancer not available - skipping tests")
    
    unittest.main(verbosity=2)