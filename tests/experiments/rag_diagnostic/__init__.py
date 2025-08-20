"""
RAG Diagnostic Test Package
Tools for analyzing RAG system performance and retrieval quality
"""

from .diagnostic_prompts import (
    create_diagnostic_prompt_mc,
    create_diagnostic_prompt_desc,
    extract_diagnostic_answer,
    create_simple_prompt
)

__version__ = "1.0.0"
__all__ = [
    'create_diagnostic_prompt_mc',
    'create_diagnostic_prompt_desc', 
    'extract_diagnostic_answer',
    'create_simple_prompt'
]