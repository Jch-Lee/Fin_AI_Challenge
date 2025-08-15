#!/usr/bin/env python3
"""Simple test for HierarchicalMarkdownChunker"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'packages'))

def simple_test():
    try:
        # Test LangChain import
        from langchain.text_splitter import MarkdownHeaderTextSplitter
        print("LangChain available")
        
        # Test chunker import
        from packages.preprocessing.hierarchical_chunker import HierarchicalMarkdownChunker
        print("HierarchicalMarkdownChunker imported")
        
        # Simple markdown test
        test_md = """
# Main Title

## Section 1

### Subsection 1.1
Content for subsection 1.1 with some detailed information.

### Subsection 1.2  
Content for subsection 1.2 with more details.

[다음 페이지에 계속]

[이전 페이지에서 계속]

## Section 2

Content for section 2.
"""
        
        # Create chunker and process
        chunker = HierarchicalMarkdownChunker()
        chunks = chunker.chunk_document(test_md)
        
        print(f"Chunks created: {len(chunks)}")
        
        # Check basic functionality
        if chunks:
            first_chunk = chunks[0]
            print(f"First chunk level: {first_chunk.metadata.get('hierarchy_level')}")
            print(f"First chunk header: {first_chunk.metadata.get('parent_header')}")
            print(f"First chunk method: {first_chunk.metadata.get('chunking_method')}")
            print(f"Page boundaries processed: {first_chunk.metadata.get('processed_boundaries')}")
            
            # Check if page boundary markers were removed
            has_markers = any('[다음 페이지에 계속]' in chunk.content or 
                             '[이전 페이지에서 계속]' in chunk.content 
                             for chunk in chunks)
            print(f"Page markers removed: {not has_markers}")
        
        # Get stats
        stats = chunker.get_hierarchy_stats(chunks)
        print(f"Hierarchy levels found: {list(stats['by_level'].keys())}")
        print(f"Average chunk size: {stats['avg_chunk_size']:.1f}")
        
        print("SUCCESS: All tests passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing HierarchicalMarkdownChunker...")
    success = simple_test()
    print(f"Result: {'PASS' if success else 'FAIL'}")