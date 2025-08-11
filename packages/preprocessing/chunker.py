"""
문서 청킹 모듈
RecursiveCharacterTextSplitter를 사용한 효율적인 청킹
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json
import hashlib
from .text_cleaner import ChunkCleaner


@dataclass
class DocumentChunk:
    """문서 청크 데이터 클래스"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    doc_id: str
    chunk_index: int
    
    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'metadata': self.metadata,
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'chunk_index': self.chunk_index
        }


class DocumentChunker:
    """문서 청킹 클래스"""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 separators: Optional[List[str]] = None,
                 use_chunk_cleaner: bool = True):
        """
        Args:
            chunk_size: 청크 최대 크기 (문자 수)
            chunk_overlap: 청크 간 중복 크기
            use_chunk_cleaner: ChunkCleaner 사용 여부
            separators: 분리 기준 문자열 리스트
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_cleaner = ChunkCleaner() if use_chunk_cleaner else None
        
        # 한국어와 영어를 위한 기본 분리자
        if separators is None:
            self.separators = [
                "\n\n",  # 단락
                "\n",    # 줄바꿈
                ". ",    # 영어 문장
                "。",    # 한국어 문장 (일부)
                "! ",    # 느낌표
                "? ",    # 물음표
                "；",    # 세미콜론
                "，",    # 쉼표
                " ",     # 공백
                ""       # 문자
            ]
        else:
            self.separators = separators
    
    def _generate_chunk_id(self, content: str, doc_id: str, index: int) -> str:
        """청크 고유 ID 생성"""
        hash_input = f"{doc_id}_{index}_{content[:50]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _generate_doc_id(self, document: str) -> str:
        """문서 고유 ID 생성"""
        return hashlib.md5(document.encode()).hexdigest()[:16]
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """재귀적으로 텍스트 분할"""
        chunks = []
        
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # 현재 separator로 분할 시도
        separator = separators[0] if separators else ""
        
        if separator:
            splits = text.split(separator)
        else:
            # 마지막 수단: 문자 단위로 분할
            splits = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
            return splits
        
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split) + len(separator)
            
            # 현재 청크에 추가 가능한 경우
            if current_length + split_length <= self.chunk_size:
                current_chunk.append(split)
                current_length += split_length
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    if chunk_text.strip():
                        chunks.append(chunk_text)
                
                # 너무 긴 split은 다시 분할
                if len(split) > self.chunk_size and len(separators) > 1:
                    sub_chunks = self._split_text_recursive(split, separators[1:])
                    chunks.extend(sub_chunks)
                    current_chunk = []
                    current_length = 0
                else:
                    current_chunk = [split]
                    current_length = split_length
        
        # 마지막 청크 처리
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        # 오버랩 적용
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # 이전 청크의 끝부분 추가
                    prev_chunk = chunks[i-1]
                    overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                    chunk = overlap_text + chunk
                overlapped_chunks.append(chunk)
            return overlapped_chunks
        
        return chunks
    
    def chunk_document(self, 
                      document: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """문서를 청크로 분할"""
        if metadata is None:
            metadata = {}
        
        doc_id = self._generate_doc_id(document)
        chunks = self._split_text_recursive(document, self.separators)
        
        # ChunkCleaner로 청크 정제
        if self.chunk_cleaner:
            chunks = self.chunk_cleaner.clean_chunks(chunks)
        
        document_chunks = []
        for i, chunk_content in enumerate(chunks):
            chunk_id = self._generate_chunk_id(chunk_content, doc_id, i)
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_size': len(chunk_content),
                'total_chunks': len(chunks),
                'has_overlap': self.chunk_overlap > 0
            })
            
            chunk = DocumentChunk(
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=i
            )
            document_chunks.append(chunk)
        
        return document_chunks
    
    def chunk_documents(self, 
                       documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """여러 문서를 청크로 분할"""
        all_chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            chunks = self.chunk_document(content, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def save_chunks(self, chunks: List[DocumentChunk], filepath: str):
        """청크를 JSONL 파일로 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                json_line = json.dumps(chunk.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')
    
    def load_chunks(self, filepath: str) -> List[DocumentChunk]:
        """JSONL 파일에서 청크 로드"""
        chunks = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                chunk = DocumentChunk(**data)
                chunks.append(chunk)
        return chunks


if __name__ == "__main__":
    # 테스트
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
    
    test_document = """
    금융보안이란 금융 거래 및 서비스에서 발생할 수 있는 다양한 위협으로부터 
    고객의 자산과 정보를 보호하는 것을 의미합니다. 
    
    최근 디지털 금융 서비스의 확대로 인해 사이버 보안의 중요성이 더욱 커지고 있습니다.
    해커들은 끊임없이 새로운 공격 방법을 개발하고 있으며, 
    금융 기관들은 이에 대응하기 위해 최신 보안 기술을 도입하고 있습니다.
    
    주요 보안 위협으로는 피싱, 파밍, 스미싱, 랜섬웨어 등이 있으며,
    이러한 위협으로부터 보호하기 위해서는 다층적인 보안 체계가 필요합니다.
    """
    
    chunks = chunker.chunk_document(test_document, metadata={'source': 'test'})
    
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_index}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Content: {chunk.content[:100]}...")
        print(f"  Size: {chunk.metadata['chunk_size']}")
        print("-" * 50)