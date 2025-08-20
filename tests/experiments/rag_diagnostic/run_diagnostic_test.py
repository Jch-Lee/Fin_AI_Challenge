#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG System Diagnostic Test
Analyzes retrieval quality and scoring mechanism importance
"""

import os
import sys
import json
import logging
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diagnostic_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import diagnostic prompts
from diagnostic_prompts import (
    create_diagnostic_prompt_mc,
    create_diagnostic_prompt_desc,
    extract_diagnostic_answer,
    create_simple_prompt
)


class RAGDiagnosticTester:
    """RAG System Diagnostic Tester"""
    
    def __init__(self, use_gpu: bool = True):
        """Initialize diagnostic tester"""
        self.use_gpu = use_gpu
        self.rag_system = None
        self.retriever = None
        self.model = None
        self.tokenizer = None
        self.results = []
        
    def setup_environment(self):
        """Setup GPU environment"""
        if self.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            import torch
            if torch.cuda.is_available():
                logger.info(f"GPU Available: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                logger.warning("GPU not available, using CPU")
                self.use_gpu = False
    
    def initialize_rag(self):
        """Initialize RAG system with current configuration"""
        logger.info("Initializing RAG system...")
        
        # Import RAG system
        from scripts.load_rag_v2 import RAGSystemV2
        
        # Load RAG system
        self.rag_system = RAGSystemV2()
        self.rag_system.load_all()
        
        # Create hybrid retriever with current weights (BM25=0.7, Vector=0.3)
        self.retriever = self.rag_system.create_hybrid_retriever()
        
        logger.info("RAG system initialized successfully")
        logger.info(f"BM25 weight: {self.retriever.bm25_weight:.1%}")
        logger.info(f"Vector weight: {self.retriever.vector_weight:.1%}")
    
    def initialize_llm(self):
        """Initialize LLM for answer generation"""
        logger.info("Initializing LLM...")
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Use Qwen2.5-7B-Instruct
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        
        # Load model with appropriate settings
        if self.use_gpu:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("LLM initialized successfully")
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> Tuple[List[str], List[Dict]]:
        """
        Retrieve documents with detailed score information
        
        Returns:
            contexts: List of document contents
            scores: List of score dictionaries with bm25_score, vector_score, hybrid_score
        """
        try:
            # Perform hybrid search
            results = self.retriever.search(query, k=k)
            
            contexts = []
            scores = []
            
            for result in results:
                # Extract content
                content = getattr(result, 'content', '')
                contexts.append(content)
                
                # Extract scores
                score_info = {
                    'bm25_score': getattr(result, 'bm25_score', 0.0),
                    'vector_score': getattr(result, 'vector_score', 0.0),
                    'hybrid_score': getattr(result, 'hybrid_score', 0.0),
                    'retrieval_methods': getattr(result, 'retrieval_methods', [])
                }
                scores.append(score_info)
            
            return contexts, scores
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return [], []
    
    def generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate answer using LLM"""
        try:
            import torch
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000)
            
            if self.use_gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.3,
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""
    
    def analyze_single_question(self, question_id: str, question: str, question_type: str) -> Dict:
        """Analyze a single question comprehensively"""
        logger.info(f"Analyzing {question_id} ({question_type})...")
        
        # Retrieve documents with scores
        contexts, scores = self.retrieve_with_scores(question, k=5)
        
        # Create diagnostic prompt
        if question_type == 'multiple_choice':
            prompt = create_diagnostic_prompt_mc(question, contexts, scores)
        else:
            prompt = create_diagnostic_prompt_desc(question, contexts, scores)
        
        # Generate diagnostic answer
        diagnostic_response = self.generate_answer(prompt)
        
        # Extract structured information
        diagnostic_info = extract_diagnostic_answer(diagnostic_response, question_type)
        
        # Also generate simple answer for comparison
        simple_prompt = create_simple_prompt(question, contexts)
        simple_response = self.generate_answer(simple_prompt, max_new_tokens=128)
        
        # Analyze score dominance
        score_analysis = self.analyze_score_dominance(scores)
        
        # Compile results
        result = {
            'question_id': question_id,
            'question_type': question_type,
            'question': question,
            'retrieval_results': {
                'top_5_documents': [
                    {
                        'content_preview': ctx[:200] + '...' if len(ctx) > 200 else ctx,
                        'bm25_score': score['bm25_score'],
                        'vector_score': score['vector_score'],
                        'hybrid_score': score['hybrid_score'],
                        'retrieval_methods': score['retrieval_methods']
                    }
                    for ctx, score in zip(contexts, scores)
                ],
                'score_analysis': score_analysis
            },
            'generation_process': {
                'thought_process': diagnostic_info['thought_process'],
                'evidence': diagnostic_info['evidence'],
                'document_analysis': diagnostic_info['document_analysis'],
                'score_preference': diagnostic_info['score_preference'],
                'diagnostic_answer': diagnostic_info['final_answer'],
                'simple_answer': simple_response.strip()
            }
        }
        
        return result
    
    def analyze_score_dominance(self, scores: List[Dict]) -> Dict:
        """Analyze which scoring method dominates"""
        if not scores:
            return {'bm25_dominant': False, 'vector_dominant': False, 'balanced': True}
        
        # Calculate average scores
        avg_bm25 = np.mean([s['bm25_score'] for s in scores])
        avg_vector = np.mean([s['vector_score'] for s in scores])
        
        # Determine dominance
        ratio = avg_bm25 / (avg_vector + 1e-10)  # Avoid division by zero
        
        result = {
            'avg_bm25_score': float(avg_bm25),
            'avg_vector_score': float(avg_vector),
            'bm25_to_vector_ratio': float(ratio),
            'bm25_dominant': ratio > 1.5,
            'vector_dominant': ratio < 0.67,
            'balanced': 0.67 <= ratio <= 1.5
        }
        
        return result
    
    def run_diagnostic_test(self, test_file: str = 'test_questions_20.csv'):
        """Run complete diagnostic test"""
        logger.info("Starting diagnostic test...")
        
        # Load test questions
        test_path = Path(__file__).parent / test_file
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded {len(test_df)} test questions")
        
        # Process each question
        for idx, row in test_df.iterrows():
            result = self.analyze_single_question(
                row['ID'],
                row['Question'],
                row['Type']
            )
            self.results.append(result)
            
            # Save intermediate results
            if (idx + 1) % 5 == 0:
                self.save_results()
                logger.info(f"Processed {idx + 1}/{len(test_df)} questions")
        
        # Save final results
        self.save_results()
        logger.info("Diagnostic test completed")
    
    def save_results(self):
        """Save diagnostic results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        # Save detailed JSON results
        json_path = Path(__file__).parent / f'diagnostic_results_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(convert_types(self.results), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {json_path}")
        
        # Generate summary
        self.generate_summary(timestamp)
    
    def generate_summary(self, timestamp: str):
        """Generate summary report"""
        if not self.results:
            return
        
        # Analyze overall patterns
        mc_results = [r for r in self.results if r['question_type'] == 'multiple_choice']
        desc_results = [r for r in self.results if r['question_type'] == 'descriptive']
        
        # Score dominance statistics
        all_score_analyses = [r['retrieval_results']['score_analysis'] for r in self.results]
        bm25_dominant_count = sum(1 for a in all_score_analyses if a['bm25_dominant'])
        vector_dominant_count = sum(1 for a in all_score_analyses if a['vector_dominant'])
        balanced_count = sum(1 for a in all_score_analyses if a['balanced'])
        
        # Average scores
        avg_bm25 = np.mean([a['avg_bm25_score'] for a in all_score_analyses])
        avg_vector = np.mean([a['avg_vector_score'] for a in all_score_analyses])
        
        # Create summary
        summary = f"""# RAG System Diagnostic Report
Generated: {timestamp}

## Overview
- Total Questions: {len(self.results)}
- Multiple Choice: {len(mc_results)}
- Descriptive: {len(desc_results)}

## Score Analysis

### Dominance Distribution
- BM25 Dominant: {bm25_dominant_count} ({bm25_dominant_count/len(self.results)*100:.1f}%)
- Vector Dominant: {vector_dominant_count} ({vector_dominant_count/len(self.results)*100:.1f}%)
- Balanced: {balanced_count} ({balanced_count/len(self.results)*100:.1f}%)

### Average Scores
- Average BM25 Score: {avg_bm25:.4f}
- Average Vector Score: {avg_vector:.4f}
- Overall BM25/Vector Ratio: {avg_bm25/(avg_vector+1e-10):.2f}

## Current Configuration
- BM25 Weight: 70%
- Vector Weight: 30%

## Key Findings
"""
        
        # Add key findings based on results
        if bm25_dominant_count > len(self.results) * 0.5:
            summary += "- **BM25 (keyword matching) is dominant** in most questions\n"
            summary += "- Current 70% BM25 weight appears appropriate\n"
        elif vector_dominant_count > len(self.results) * 0.5:
            summary += "- **Vector (semantic similarity) is dominant** in most questions\n"
            summary += "- Consider increasing vector weight from current 30%\n"
        else:
            summary += "- **Balanced distribution** between BM25 and vector scoring\n"
            summary += "- Current weight distribution (70/30) may need fine-tuning\n"
        
        # Score preference from diagnostic answers
        score_preferences = [r['generation_process'].get('score_preference', '') for r in self.results]
        bm25_preferred = sum(1 for p in score_preferences if 'BM25' in p and '유용' in p)
        vector_preferred = sum(1 for p in score_preferences if 'Vector' in p and '유용' in p)
        
        summary += f"\n### Model's Score Preference (from diagnostic answers)\n"
        summary += f"- BM25 Preferred: {bm25_preferred} times\n"
        summary += f"- Vector Preferred: {vector_preferred} times\n"
        
        # Save summary
        summary_path = Path(__file__).parent / f'diagnostic_summary_{timestamp}.md'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"Summary saved to {summary_path}")
        print("\n" + "="*50)
        print(summary)
        print("="*50)


def main():
    """Main execution"""
    import torch
    
    # Create tester
    tester = RAGDiagnosticTester(use_gpu=torch.cuda.is_available())
    
    # Setup
    tester.setup_environment()
    tester.initialize_rag()
    tester.initialize_llm()
    
    # Run diagnostic test
    tester.run_diagnostic_test()
    
    logger.info("Diagnostic test completed successfully")


if __name__ == "__main__":
    main()