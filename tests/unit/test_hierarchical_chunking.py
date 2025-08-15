#!/usr/bin/env python3
"""
HierarchicalMarkdownChunker 단위 테스트

계층적 마크다운 청킹 시스템의 핵심 기능 검증:
- 마크다운 구조 파싱
- 계층별 청크 크기/오버랩
- 페이지 경계 마커 처리
- DocumentChunk 호환성
- 메타데이터 생성
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
    from packages.preprocessing.hierarchical_chunker import HierarchicalMarkdownChunker
    from packages.preprocessing.chunker import DocumentChunk
    HIERARCHICAL_CHUNKER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: HierarchicalMarkdownChunker not available: {e}")
    HIERARCHICAL_CHUNKER_AVAILABLE = False


class TestHierarchicalMarkdownChunker(unittest.TestCase):
    """HierarchicalMarkdownChunker 단위 테스트"""
    
    def setUp(self):
        """테스트 준비"""
        if not HIERARCHICAL_CHUNKER_AVAILABLE:
            self.skipTest("HierarchicalMarkdownChunker not available")
        
        self.chunker = HierarchicalMarkdownChunker()
        
        # 테스트용 마크다운 문서
        self.test_markdown = """
# 금융보안 가이드라인

## 제1장 총칙

### 제1조 (목적)
이 가이드라인은 금융기관의 정보보호 및 사이버보안 강화를 위한 기본 원칙과 세부 실행방안을 제시함을 목적으로 한다.

금융분야에서 디지털 전환이 가속화되면서 사이버 보안의 중요성이 더욱 커지고 있다.

### 제2조 (정의)
1. "금융기관"이란 은행법에 따른 은행, 금융지주회사법에 따른 금융지주회사를 말한다.
2. "사이버 위협"이란 정보통신망을 통하여 금융기관의 정보자산에 피해를 줄 수 있는 모든 행위를 말한다.
3. "정보보호"란 정보의 수집, 가공, 저장, 검색, 송신, 수신 중에 정보의 훼손, 변조, 유출 등을 방지하기 위한 관리적, 기술적, 물리적 조치를 말한다.

[다음 페이지에 계속]

[이전 페이지에서 계속]

## 제2장 보안 요구사항

### 제3조 (접근통제)

모든 시스템 접근은 다음과 같은 통제 절차를 거쳐야 한다:

| 구분 | 요구사항 | 중요도 |
|------|----------|--------|
| 인증 | 다요소 인증 필수 | 높음 |
| 권한 | 최소 권한 원칙 | 높음 |
| 모니터링 | 실시간 감시 | 중간 |

#### 세부 규정
- 모든 시스템 접근은 로그를 남겨야 한다
- 비정상 접근 시도는 즉시 차단한다
- 권한 변경은 승인 절차를 거쳐야 한다

**중요**: 2024년 1월 1일부터 모든 금융기관은 제로트러스트 보안 모델을 적용해야 한다.
""".strip()

    def test_chunker_initialization(self):
        """청킹기 초기화 테스트"""
        self.assertIsInstance(self.chunker, HierarchicalMarkdownChunker)
        
        # 계층 설정 확인
        self.assertIn(1, self.chunker.hierarchy_configs)
        self.assertIn(2, self.chunker.hierarchy_configs)
        self.assertIn(3, self.chunker.hierarchy_configs)
        
        # 각 레벨별 설정 확인
        level_1 = self.chunker.hierarchy_configs[1]
        self.assertEqual(level_1.chunk_size, 1024)
        self.assertEqual(level_1.chunk_overlap, 100)
        
        level_2 = self.chunker.hierarchy_configs[2]
        self.assertEqual(level_2.chunk_size, 512)
        self.assertEqual(level_2.chunk_overlap, 50)
        
        level_3 = self.chunker.hierarchy_configs[3]
        self.assertEqual(level_3.chunk_size, 256)
        self.assertEqual(level_3.chunk_overlap, 30)

    def test_page_boundary_merging(self):
        """페이지 경계 마커 처리 테스트"""
        # 페이지 마커가 포함된 텍스트
        text_with_markers = """
        첫 번째 문단입니다.
        
        [다음 페이지에 계속]
        
        [이전 페이지에서 계속]
        
        두 번째 문단입니다.
        """
        
        processed = self.chunker._merge_page_boundaries(text_with_markers)
        
        # 페이지 마커가 제거되었는지 확인
        self.assertNotIn('[다음 페이지에 계속]', processed)
        self.assertNotIn('[이전 페이지에서 계속]', processed)
        
        # 내용은 유지되는지 확인
        self.assertIn('첫 번째 문단', processed)
        self.assertIn('두 번째 문단', processed)

    def test_header_level_detection(self):
        """헤더 레벨 감지 테스트"""
        # 다양한 헤더 레벨 테스트
        test_cases = [
            ({'h1': 'Title'}, 1),
            ({'h2': 'Section'}, 2),
            ({'h3': 'Subsection'}, 3),
            ({'h4': 'Sub-subsection'}, 4),
            ({}, 2),  # 헤더 없음 -> 기본 레벨 2
            ({'h1': 'Title', 'h2': 'Section'}, 1),  # 여러 헤더 -> 가장 높은 레벨
        ]
        
        for metadata, expected_level in test_cases:
            level = self.chunker._detect_header_level(metadata)
            self.assertEqual(level, expected_level, f"Failed for metadata: {metadata}")

    def test_hierarchy_path_building(self):
        """계층 경로 생성 테스트"""
        test_cases = [
            ({}, "Root"),
            ({'h1': 'Chapter 1'}, "Chapter 1"),
            ({'h1': 'Chapter 1', 'h2': 'Section 1.1'}, "Chapter 1 > Section 1.1"),
            ({'h1': 'A', 'h2': 'B', 'h3': 'C'}, "A > B > C"),
        ]
        
        for metadata, expected_path in test_cases:
            path = self.chunker._build_hierarchy_path(metadata)
            self.assertEqual(path, expected_path)

    def test_chunk_creation_with_overlap(self):
        """오버랩을 포함한 청킹 테스트"""
        test_text = "이것은 테스트 텍스트입니다. " * 50  # 충분히 긴 텍스트
        
        config = self.chunker.hierarchy_configs[2]  # 레벨 2 설정 (512/50)
        chunks = self.chunker._create_chunks_with_overlap(test_text, config)
        
        self.assertGreater(len(chunks), 1, "충분히 긴 텍스트는 여러 청크로 분할되어야 함")
        
        # 각 청크 크기 확인
        for chunk in chunks:
            self.assertLessEqual(len(chunk), config.chunk_size + 50, "청크가 너무 큼")
            self.assertGreater(len(chunk.strip()), 0, "빈 청크가 생성됨")

    def test_document_chunking(self):
        """전체 문서 청킹 테스트"""
        chunks = self.chunker.chunk_document(self.test_markdown)
        
        # 기본 검증
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0, "청크가 생성되지 않음")
        
        # 모든 청크가 DocumentChunk 인스턴스인지 확인
        for chunk in chunks:
            self.assertIsInstance(chunk, DocumentChunk)
            self.assertIsInstance(chunk.content, str)
            self.assertIsInstance(chunk.metadata, dict)
            self.assertIsInstance(chunk.chunk_id, str)
            self.assertIsInstance(chunk.doc_id, str)
            self.assertIsInstance(chunk.chunk_index, int)

    def test_chunk_metadata(self):
        """청크 메타데이터 검증 테스트"""
        chunks = self.chunker.chunk_document(
            self.test_markdown, 
            metadata={'source': 'test_document'}
        )
        
        for chunk in chunks:
            metadata = chunk.metadata
            
            # 필수 메타데이터 확인
            self.assertIn('hierarchy_level', metadata)
            self.assertIn('hierarchy_path', metadata)
            self.assertIn('chunking_method', metadata)
            self.assertIn('chunk_size', metadata)
            self.assertIn('chunk_overlap', metadata)
            self.assertIn('content_length', metadata)
            self.assertIn('processed_boundaries', metadata)
            
            # 값 타입 확인
            self.assertIsInstance(metadata['hierarchy_level'], int)
            self.assertIsInstance(metadata['hierarchy_path'], str)
            self.assertEqual(metadata['chunking_method'], 'hierarchical_markdown')
            self.assertIsInstance(metadata['chunk_size'], int)
            self.assertIsInstance(metadata['chunk_overlap'], int)
            self.assertEqual(metadata['content_length'], len(chunk.content))
            self.assertTrue(metadata['processed_boundaries'])
            
            # 사용자 제공 메타데이터 확인
            self.assertEqual(metadata['source'], 'test_document')

    def test_page_boundary_removal(self):
        """페이지 경계 마커 완전 제거 확인"""
        chunks = self.chunker.chunk_document(self.test_markdown)
        
        # 모든 청크에서 페이지 마커가 제거되었는지 확인
        for chunk in chunks:
            self.assertNotIn('[다음 페이지에 계속]', chunk.content)
            self.assertNotIn('[이전 페이지에서 계속]', chunk.content)

    def test_hierarchy_stats(self):
        """계층 통계 정보 테스트"""
        chunks = self.chunker.chunk_document(self.test_markdown)
        stats = self.chunker.get_hierarchy_stats(chunks)
        
        # 통계 구조 확인
        self.assertIsInstance(stats, dict)
        self.assertIn('total_chunks', stats)
        self.assertIn('by_level', stats)
        self.assertIn('avg_chunk_size', stats)
        self.assertIn('hierarchy_paths', stats)
        
        # 값 확인
        self.assertEqual(stats['total_chunks'], len(chunks))
        self.assertGreater(stats['avg_chunk_size'], 0)
        self.assertIsInstance(stats['hierarchy_paths'], list)
        
        # 레벨별 통계 확인
        for level, level_stats in stats['by_level'].items():
            self.assertIsInstance(level, int)
            self.assertIn('count', level_stats)
            self.assertIn('avg_size', level_stats)
            self.assertGreater(level_stats['count'], 0)
            self.assertGreater(level_stats['avg_size'], 0)

    def test_empty_document(self):
        """빈 문서 처리 테스트"""
        empty_chunks = self.chunker.chunk_document("")
        self.assertEqual(len(empty_chunks), 0)
        
        whitespace_chunks = self.chunker.chunk_document("   \n\n  \t  ")
        self.assertEqual(len(whitespace_chunks), 0)

    def test_no_header_document(self):
        """헤더 없는 문서 처리 테스트"""
        no_header_text = """
        이것은 마크다운 헤더가 없는 일반 텍스트입니다.
        금융 AI 시스템의 보안은 매우 중요합니다.
        여러 줄의 텍스트가 있지만 구조가 없습니다.
        """
        
        chunks = self.chunker.chunk_document(no_header_text)
        
        self.assertGreater(len(chunks), 0)
        
        # 기본 레벨(2)이 적용되었는지 확인
        for chunk in chunks:
            self.assertEqual(chunk.metadata['hierarchy_level'], 2)
            self.assertEqual(chunk.metadata['hierarchy_path'], 'Root')

    def test_deep_nested_structure(self):
        """깊은 중첩 구조 테스트"""
        deep_nested = """
# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6
이것은 깊은 중첩 구조 테스트입니다.
"""
        
        chunks = self.chunker.chunk_document(deep_nested)
        
        # 레벨 4까지만 지원하므로 최대 레벨 4여야 함
        for chunk in chunks:
            level = chunk.metadata['hierarchy_level']
            self.assertLessEqual(level, 4, f"Level {level} exceeds maximum supported level")

    def test_special_characters(self):
        """특수 문자 포함 문서 테스트"""
        special_text = """
# 특수 문자 테스트

## 코드 예제

```python
def secure_ai_model():
    config = {"encryption": True}
    return config
```

## 수식 예제

A = alpha x beta + gamma

**굵은 텍스트**와 *기울임 텍스트*도 포함
"""
        
        chunks = self.chunker.chunk_document(special_text)
        
        # 특수 문자가 포함된 청크가 정상 처리되는지 확인
        self.assertGreater(len(chunks), 0)
        
        # 코드 블록이나 특수 문자가 깨지지 않는지 확인
        content = ''.join(chunk.content for chunk in chunks)
        self.assertIn('```python', content)
        self.assertIn('alpha x beta', content)
        self.assertIn('**굵은 텍스트**', content)

    def test_chunk_id_generation(self):
        """청크 ID 생성 테스트"""
        chunks = self.chunker.chunk_document(self.test_markdown)
        
        # 모든 청크 ID가 유니크한지 확인
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)), "중복된 청크 ID 발견")
        
        # 청크 ID 형식 확인 (16자 해시)
        for chunk_id in chunk_ids:
            self.assertIsInstance(chunk_id, str)
            self.assertEqual(len(chunk_id), 16)

    def test_batch_document_processing(self):
        """배치 문서 처리 테스트"""
        documents = [
            {
                'content': self.test_markdown,
                'metadata': {'source': 'doc1'}
            },
            {
                'content': "# Simple Document\n\nThis is a longer content for the second document to ensure it passes the chunk cleaner minimum length requirement. This content should be sufficient to create at least one chunk after processing.",
                'metadata': {'source': 'doc2'}
            }
        ]
        
        all_chunks = self.chunker.chunk_documents(documents)
        
        # 여러 문서의 청크가 모두 포함되었는지 확인
        self.assertGreater(len(all_chunks), 2)
        
        # 각 문서의 메타데이터가 보존되었는지 확인
        sources = [chunk.metadata.get('source') for chunk in all_chunks]
        self.assertIn('doc1', sources)
        self.assertIn('doc2', sources)


class TestHierarchicalChunkerEdgeCases(unittest.TestCase):
    """HierarchicalMarkdownChunker 엣지 케이스 테스트"""
    
    def setUp(self):
        """테스트 준비"""
        if not HIERARCHICAL_CHUNKER_AVAILABLE:
            self.skipTest("HierarchicalMarkdownChunker not available")
        
        self.chunker = HierarchicalMarkdownChunker()

    def test_malformed_markdown(self):
        """잘못된 마크다운 문법 처리"""
        malformed_text = """
        # 정상 헤더
        
        ## 정상 섹션
        
        ###잘못된헤더(공백없음)
        
        ####    너무많은공백    ####
        
        일반 텍스트
        """
        
        # 예외 발생 없이 처리되어야 함
        chunks = self.chunker.chunk_document(malformed_text)
        self.assertIsInstance(chunks, list)

    def test_very_long_headers(self):
        """매우 긴 헤더 처리"""
        long_header = "이것은 매우 긴 헤더입니다 " * 20
        long_text = f"# {long_header}\n\n내용입니다."
        
        chunks = self.chunker.chunk_document(long_text)
        
        # 헤더가 메타데이터에 올바르게 저장되는지 확인
        for chunk in chunks:
            if chunk.metadata.get('parent_header'):
                self.assertIn('매우 긴 헤더', chunk.metadata['parent_header'])

    def test_unicode_content(self):
        """유니코드 콘텐츠 처리"""
        unicode_text = """
# 한국어 제목

## 섹션 🔒

다양한 언어: English, 한국어, 日本語, 中文

특수기호: ⚡ ✅ ❌ 🎯 📊
"""
        
        chunks = self.chunker.chunk_document(unicode_text)
        
        # 유니코드가 올바르게 보존되는지 확인
        content = ''.join(chunk.content for chunk in chunks)
        self.assertIn('🔒', content)
        self.assertIn('日本語', content)
        self.assertIn('⚡', content)

    def test_chunk_cleaner_disabled(self):
        """ChunkCleaner 비활성화 테스트"""
        chunker_no_cleaner = HierarchicalMarkdownChunker(use_chunk_cleaner=False)
        
        test_text = "# Test\n\n   \n\n내용"
        chunks = chunker_no_cleaner.chunk_document(test_text)
        
        # ChunkCleaner 없이도 정상 작동하는지 확인
        self.assertGreater(len(chunks), 0)


if __name__ == '__main__':
    # 테스트 실행 전 환경 확인
    if HIERARCHICAL_CHUNKER_AVAILABLE:
        print("HierarchicalMarkdownChunker available - running all tests")
    else:
        print("HierarchicalMarkdownChunker not available - skipping tests")
    
    unittest.main(verbosity=2)