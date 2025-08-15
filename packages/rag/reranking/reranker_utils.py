"""
Utility functions for reranker module.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def preprocess_korean_financial_text(text: str) -> str:
    """
    Preprocess Korean financial text for reranking.
    
    This function handles Korean-specific preprocessing and
    financial domain terminology normalization.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Normalize financial terms
    financial_replacements = {
        # English to Korean normalization
        "risk": "리스크",
        "compliance": "컴플라이언스",
        "portfolio": "포트폴리오",
        "derivative": "파생상품",
        "hedge": "헤지",
        
        # Common abbreviations
        "VaR": "위험가치",
        "ROI": "투자수익률",
        "ROE": "자기자본이익률",
        "P/E": "주가수익비율",
        
        # Number formatting
        r"(\d+),(\d+)": r"\1\2",  # Remove commas from numbers
    }
    
    for pattern, replacement in financial_replacements.items():
        if pattern.startswith(r"("):  # Regex pattern
            text = re.sub(pattern, replacement, text)
        else:  # Simple string replacement
            text = text.replace(pattern, replacement)
    
    # Remove special characters but keep Korean, numbers, and basic punctuation
    text = re.sub(r"[^\w\s가-힣.,!?%-]", " ", text)
    
    # Normalize multiple spaces
    text = " ".join(text.split())
    
    return text.strip()


def normalize_scores(
    scores: List[float],
    method: str = "minmax"
) -> List[float]:
    """
    Normalize scores to [0, 1] range.
    
    Args:
        scores: List of scores to normalize
        method: Normalization method ('minmax', 'zscore', 'sigmoid')
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    scores_array = np.array(scores)
    
    if method == "minmax":
        # Min-max normalization
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        
    elif method == "zscore":
        # Z-score normalization then sigmoid
        mean = scores_array.mean()
        std = scores_array.std()
        
        if std == 0:
            return [0.5] * len(scores)
        
        z_scores = (scores_array - mean) / std
        # Apply sigmoid to map to [0, 1]
        normalized = 1 / (1 + np.exp(-z_scores))
        
    elif method == "sigmoid":
        # Direct sigmoid
        normalized = 1 / (1 + np.exp(-scores_array))
        
    else:
        logger.warning(f"Unknown normalization method: {method}, using minmax")
        return normalize_scores(scores, "minmax")
    
    return normalized.tolist()


def batch_documents(
    documents: List[Dict[str, Any]],
    batch_size: int
) -> List[List[Dict[str, Any]]]:
    """
    Split documents into batches for processing.
    
    Args:
        documents: List of documents
        batch_size: Size of each batch
        
    Returns:
        List of document batches
    """
    batches = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batches.append(batch)
    
    return batches


def extract_financial_terms(text: str) -> List[str]:
    """
    Extract financial terms from Korean text.
    
    Args:
        text: Input text
        
    Returns:
        List of financial terms found
    """
    # Common Korean financial terms
    financial_terms = [
        # Risk management
        "리스크", "위험", "헤지", "변동성", "VaR", "위험가치",
        
        # Investment
        "투자", "수익", "손실", "포트폴리오", "자산", "채권", "주식",
        
        # Compliance
        "규제", "감독", "컴플라이언스", "준수", "법규", "감사",
        
        # Security
        "보안", "사이버", "해킹", "침해", "인증", "암호화",
        
        # Financial metrics
        "이익률", "수익률", "ROI", "ROE", "ROA", "P/E",
        
        # Banking
        "은행", "예금", "대출", "금리", "이자", "신용",
        
        # Market
        "시장", "거래", "매매", "상장", "코스피", "코스닥",
    ]
    
    found_terms = []
    text_lower = text.lower()
    
    for term in financial_terms:
        if term.lower() in text_lower:
            found_terms.append(term)
    
    return found_terms


def calculate_term_boost_score(
    text: str,
    boost_weight: float = 1.2
) -> float:
    """
    Calculate boost score based on financial terms presence.
    
    Args:
        text: Input text
        boost_weight: Weight for term boost
        
    Returns:
        Boost score multiplier
    """
    terms = extract_financial_terms(text)
    
    if not terms:
        return 1.0
    
    # Calculate boost based on number of unique terms
    # Diminishing returns for more terms
    term_count = len(set(terms))
    boost = 1.0 + (boost_weight - 1.0) * min(term_count / 10.0, 1.0)
    
    return boost


def merge_document_metadata(
    original_doc: Dict[str, Any],
    rerank_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge original document with reranking information.
    
    Args:
        original_doc: Original document dictionary
        rerank_info: Reranking information to add
        
    Returns:
        Merged document dictionary
    """
    # Create a copy to avoid modifying original
    merged = original_doc.copy()
    
    # Add reranking information
    merged.update(rerank_info)
    
    # Preserve original score if not already saved
    if "score" in original_doc and "original_score" not in merged:
        merged["original_score"] = original_doc["score"]
    
    return merged


def truncate_document(
    document: str,
    max_length: int,
    strategy: str = "balanced"
) -> str:
    """
    Truncate document to maximum length.
    
    Args:
        document: Document text
        max_length: Maximum character length
        strategy: Truncation strategy ('start', 'end', 'balanced')
        
    Returns:
        Truncated document
    """
    if len(document) <= max_length:
        return document
    
    if strategy == "start":
        # Keep only the beginning
        return document[:max_length]
    
    elif strategy == "end":
        # Keep only the end
        return document[-max_length:]
    
    elif strategy == "balanced":
        # Keep beginning and end
        half_length = max_length // 2
        return (
            document[:half_length] +
            " ... " +
            document[-(max_length - half_length - 5):]
        )
    
    else:
        logger.warning(f"Unknown truncation strategy: {strategy}")
        return document[:max_length]


def rank_fusion(
    rankings: List[List[int]],
    weights: Optional[List[float]] = None,
    k: int = 60
) -> List[int]:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple rankings.
    
    Args:
        rankings: List of rankings (each ranking is a list of document indices)
        weights: Optional weights for each ranking
        k: RRF parameter (default 60)
        
    Returns:
        Fused ranking of document indices
    """
    if not rankings:
        return []
    
    if weights is None:
        weights = [1.0] * len(rankings)
    
    # Calculate RRF scores
    scores = {}
    
    for ranking, weight in zip(rankings, weights):
        for rank, doc_idx in enumerate(ranking):
            if doc_idx not in scores:
                scores[doc_idx] = 0
            scores[doc_idx] += weight / (k + rank + 1)
    
    # Sort by score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [doc_idx for doc_idx, _ in sorted_docs]


def compute_diversity_score(
    documents: List[Dict[str, Any]],
    field: str = "content"
) -> float:
    """
    Compute diversity score for a set of documents.
    
    Higher score means more diverse documents.
    
    Args:
        documents: List of documents
        field: Field to use for diversity calculation
        
    Returns:
        Diversity score between 0 and 1
    """
    if len(documents) < 2:
        return 1.0
    
    # Simple character-level diversity
    texts = [doc.get(field, "") for doc in documents]
    
    # Calculate pairwise Jaccard distances
    distances = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            # Character-level Jaccard distance
            set1 = set(texts[i].split())
            set2 = set(texts[j].split())
            
            if not set1 and not set2:
                distance = 0
            elif not set1 or not set2:
                distance = 1
            else:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                distance = 1 - (intersection / union) if union > 0 else 1
            
            distances.append(distance)
    
    # Average distance is diversity score
    diversity = np.mean(distances) if distances else 0
    
    return float(diversity)


def filter_duplicate_documents(
    documents: List[Dict[str, Any]],
    threshold: float = 0.9,
    field: str = "content"
) -> List[Dict[str, Any]]:
    """
    Filter out near-duplicate documents.
    
    Args:
        documents: List of documents
        threshold: Similarity threshold for duplicates (0-1)
        field: Field to check for duplicates
        
    Returns:
        Filtered list of documents
    """
    if len(documents) <= 1:
        return documents
    
    filtered = [documents[0]]  # Keep first document
    
    for doc in documents[1:]:
        is_duplicate = False
        doc_text = doc.get(field, "")
        doc_words = set(doc_text.split())
        
        for filtered_doc in filtered:
            filtered_text = filtered_doc.get(field, "")
            filtered_words = set(filtered_text.split())
            
            # Calculate Jaccard similarity
            if not doc_words and not filtered_words:
                similarity = 1.0
            elif not doc_words or not filtered_words:
                similarity = 0.0
            else:
                intersection = len(doc_words & filtered_words)
                union = len(doc_words | filtered_words)
                similarity = intersection / union if union > 0 else 0
            
            if similarity >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(doc)
    
    if len(filtered) < len(documents):
        logger.info(f"Filtered {len(documents) - len(filtered)} duplicate documents")
    
    return filtered


def format_rerank_results(
    documents: List[Dict[str, Any]],
    include_fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Format reranking results for output.
    
    Args:
        documents: Reranked documents
        include_fields: Fields to include in output
        
    Returns:
        Formatted documents
    """
    if include_fields is None:
        include_fields = ["content", "score", "rerank_score", "metadata"]
    
    formatted = []
    for i, doc in enumerate(documents):
        result = {"rank": i + 1}
        
        for field in include_fields:
            if field in doc:
                result[field] = doc[field]
        
        formatted.append(result)
    
    return formatted