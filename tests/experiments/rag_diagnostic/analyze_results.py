#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze diagnostic test results
Provides detailed insights about RAG system performance
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np
import argparse
from datetime import datetime


class DiagnosticAnalyzer:
    """Analyze RAG diagnostic test results"""
    
    def __init__(self, results_file: str):
        """Initialize analyzer with results file"""
        self.results_file = results_file
        self.results = []
        self.load_results()
    
    def load_results(self):
        """Load results from JSON file"""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        print(f"Loaded {len(self.results)} results from {self.results_file}")
    
    def analyze_retrieval_quality(self) -> Dict:
        """Analyze retrieval quality metrics"""
        analysis = {
            'total_questions': len(self.results),
            'questions_with_retrieval': 0,
            'avg_documents_retrieved': 0,
            'avg_top1_score': 0,
            'avg_top5_score': 0,
            'documents_used_ratio': []
        }
        
        all_top1_scores = []
        all_top5_scores = []
        
        for result in self.results:
            retrieval = result['retrieval_results']
            docs = retrieval['top_5_documents']
            
            if docs:
                analysis['questions_with_retrieval'] += 1
                analysis['avg_documents_retrieved'] += len(docs)
                
                # Top-1 score
                if len(docs) > 0:
                    all_top1_scores.append(docs[0]['hybrid_score'])
                
                # Average of top-5 scores
                top5_scores = [d['hybrid_score'] for d in docs[:5]]
                if top5_scores:
                    all_top5_scores.append(np.mean(top5_scores))
                
                # Check if documents were actually used
                evidence = result['generation_process'].get('evidence', '')
                if '문서' in evidence:
                    analysis['documents_used_ratio'].append(1)
                else:
                    analysis['documents_used_ratio'].append(0)
        
        # Calculate averages
        if analysis['questions_with_retrieval'] > 0:
            analysis['avg_documents_retrieved'] /= analysis['questions_with_retrieval']
            analysis['avg_top1_score'] = np.mean(all_top1_scores) if all_top1_scores else 0
            analysis['avg_top5_score'] = np.mean(all_top5_scores) if all_top5_scores else 0
            analysis['document_usage_rate'] = np.mean(analysis['documents_used_ratio'])
        
        return analysis
    
    def analyze_score_patterns(self) -> Dict:
        """Analyze BM25 vs Vector scoring patterns"""
        patterns = {
            'bm25_dominant': 0,
            'vector_dominant': 0,
            'balanced': 0,
            'by_question_type': {
                'multiple_choice': {'bm25': 0, 'vector': 0, 'balanced': 0},
                'descriptive': {'bm25': 0, 'vector': 0, 'balanced': 0}
            },
            'score_ratios': [],
            'model_preferences': {'bm25': 0, 'vector': 0, 'unclear': 0}
        }
        
        for result in self.results:
            score_analysis = result['retrieval_results']['score_analysis']
            q_type = result['question_type']
            
            # Overall dominance
            if score_analysis['bm25_dominant']:
                patterns['bm25_dominant'] += 1
                patterns['by_question_type'][q_type]['bm25'] += 1
            elif score_analysis['vector_dominant']:
                patterns['vector_dominant'] += 1
                patterns['by_question_type'][q_type]['vector'] += 1
            else:
                patterns['balanced'] += 1
                patterns['by_question_type'][q_type]['balanced'] += 1
            
            # Score ratio
            patterns['score_ratios'].append(score_analysis['bm25_to_vector_ratio'])
            
            # Model's stated preference
            pref = result['generation_process'].get('score_preference', '')
            if 'BM25' in pref and ('유용' in pref or '신뢰' in pref):
                patterns['model_preferences']['bm25'] += 1
            elif 'Vector' in pref and ('유용' in pref or '신뢰' in pref):
                patterns['model_preferences']['vector'] += 1
            else:
                patterns['model_preferences']['unclear'] += 1
        
        # Calculate statistics
        patterns['avg_score_ratio'] = np.mean(patterns['score_ratios'])
        patterns['std_score_ratio'] = np.std(patterns['score_ratios'])
        
        return patterns
    
    def analyze_answer_quality(self) -> Dict:
        """Analyze answer quality and generation patterns"""
        quality = {
            'answers_generated': 0,
            'diagnostic_vs_simple_match': 0,
            'thought_process_provided': 0,
            'evidence_cited': 0,
            'avg_answer_length': {'diagnostic': [], 'simple': []},
            'by_question_type': {
                'multiple_choice': {'generated': 0, 'with_evidence': 0},
                'descriptive': {'generated': 0, 'with_evidence': 0}
            }
        }
        
        for result in self.results:
            gen = result['generation_process']
            q_type = result['question_type']
            
            # Check if answers were generated
            if gen.get('diagnostic_answer'):
                quality['answers_generated'] += 1
                quality['by_question_type'][q_type]['generated'] += 1
                quality['avg_answer_length']['diagnostic'].append(len(gen['diagnostic_answer']))
            
            if gen.get('simple_answer'):
                quality['avg_answer_length']['simple'].append(len(gen['simple_answer']))
            
            # Compare diagnostic vs simple answers
            if gen.get('diagnostic_answer') and gen.get('simple_answer'):
                if q_type == 'multiple_choice':
                    # For MC, check if the number matches
                    if gen['diagnostic_answer'].strip() == gen['simple_answer'].strip():
                        quality['diagnostic_vs_simple_match'] += 1
                else:
                    # For descriptive, check similarity (simplified)
                    if len(set(gen['diagnostic_answer'].split()) & set(gen['simple_answer'].split())) > 5:
                        quality['diagnostic_vs_simple_match'] += 1
            
            # Check for thought process and evidence
            if gen.get('thought_process') and len(gen['thought_process']) > 10:
                quality['thought_process_provided'] += 1
            
            if gen.get('evidence') and len(gen['evidence']) > 10:
                quality['evidence_cited'] += 1
                quality['by_question_type'][q_type]['with_evidence'] += 1
        
        # Calculate averages
        if quality['avg_answer_length']['diagnostic']:
            quality['avg_diagnostic_length'] = np.mean(quality['avg_answer_length']['diagnostic'])
        if quality['avg_answer_length']['simple']:
            quality['avg_simple_length'] = np.mean(quality['avg_answer_length']['simple'])
        
        return quality
    
    def generate_insights(self) -> str:
        """Generate actionable insights from analysis"""
        retrieval = self.analyze_retrieval_quality()
        patterns = self.analyze_score_patterns()
        quality = self.analyze_answer_quality()
        
        insights = []
        
        # Retrieval quality insights
        if retrieval['avg_top1_score'] < 0.5:
            insights.append("⚠️ Low retrieval scores indicate potential issues with document relevance")
        
        if retrieval['document_usage_rate'] < 0.5:
            insights.append("⚠️ Documents are used in less than 50% of answers - may indicate insufficient coverage")
        
        # Score pattern insights
        if patterns['bm25_dominant'] > len(self.results) * 0.6:
            insights.append("✅ BM25 (keyword matching) is dominant - current 70% weight is appropriate")
        elif patterns['vector_dominant'] > len(self.results) * 0.4:
            insights.append("⚡ Consider increasing vector weight from 30% to 40-50%")
        
        if patterns['avg_score_ratio'] > 2.0:
            insights.append("📊 BM25 scores are significantly higher than vector scores")
        elif patterns['avg_score_ratio'] < 0.5:
            insights.append("📊 Vector scores are significantly higher than BM25 scores")
        
        # Model preference vs actual dominance
        if patterns['model_preferences']['bm25'] > patterns['model_preferences']['vector']:
            insights.append("🤖 Model prefers BM25 scoring in its reasoning")
        elif patterns['model_preferences']['vector'] > patterns['model_preferences']['bm25']:
            insights.append("🤖 Model prefers Vector scoring in its reasoning")
        
        # Question type specific insights
        mc_patterns = patterns['by_question_type']['multiple_choice']
        desc_patterns = patterns['by_question_type']['descriptive']
        
        if mc_patterns['bm25'] > mc_patterns['vector'] * 2:
            insights.append("📝 Multiple choice questions strongly favor BM25 retrieval")
        
        if desc_patterns['vector'] > desc_patterns['bm25']:
            insights.append("📝 Descriptive questions benefit more from semantic (vector) search")
        
        # Answer quality insights
        if quality['thought_process_provided'] < len(self.results) * 0.8:
            insights.append("⚠️ Thought process missing in some diagnostic answers")
        
        if quality['diagnostic_vs_simple_match'] < len(self.results) * 0.7:
            insights.append("🔍 Significant differences between diagnostic and simple answers")
        
        return "\n".join(insights)
    
    def generate_report(self, output_file: Optional[str] = None):
        """Generate comprehensive analysis report"""
        retrieval = self.analyze_retrieval_quality()
        patterns = self.analyze_score_patterns()
        quality = self.analyze_answer_quality()
        insights = self.generate_insights()
        
        report = f"""# RAG Diagnostic Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: {self.results_file}

## 📊 Retrieval Quality Analysis

- **Total Questions**: {retrieval['total_questions']}
- **Questions with Retrieval**: {retrieval['questions_with_retrieval']}
- **Average Documents Retrieved**: {retrieval['avg_documents_retrieved']:.1f}
- **Average Top-1 Score**: {retrieval['avg_top1_score']:.4f}
- **Average Top-5 Score**: {retrieval['avg_top5_score']:.4f}
- **Document Usage Rate**: {retrieval.get('document_usage_rate', 0)*100:.1f}%

## 🔍 Score Pattern Analysis

### Overall Dominance
- **BM25 Dominant**: {patterns['bm25_dominant']} ({patterns['bm25_dominant']/len(self.results)*100:.1f}%)
- **Vector Dominant**: {patterns['vector_dominant']} ({patterns['vector_dominant']/len(self.results)*100:.1f}%)
- **Balanced**: {patterns['balanced']} ({patterns['balanced']/len(self.results)*100:.1f}%)

### Score Statistics
- **Average BM25/Vector Ratio**: {patterns['avg_score_ratio']:.2f}
- **Std Dev of Ratio**: {patterns['std_score_ratio']:.2f}

### Model's Stated Preferences
- **Prefers BM25**: {patterns['model_preferences']['bm25']}
- **Prefers Vector**: {patterns['model_preferences']['vector']}
- **Unclear**: {patterns['model_preferences']['unclear']}

### By Question Type
#### Multiple Choice
- BM25 Dominant: {patterns['by_question_type']['multiple_choice']['bm25']}
- Vector Dominant: {patterns['by_question_type']['multiple_choice']['vector']}
- Balanced: {patterns['by_question_type']['multiple_choice']['balanced']}

#### Descriptive
- BM25 Dominant: {patterns['by_question_type']['descriptive']['bm25']}
- Vector Dominant: {patterns['by_question_type']['descriptive']['vector']}
- Balanced: {patterns['by_question_type']['descriptive']['balanced']}

## 📝 Answer Quality Analysis

- **Answers Generated**: {quality['answers_generated']}/{len(self.results)}
- **Diagnostic vs Simple Match**: {quality['diagnostic_vs_simple_match']}/{len(self.results)}
- **Thought Process Provided**: {quality['thought_process_provided']}/{len(self.results)}
- **Evidence Cited**: {quality['evidence_cited']}/{len(self.results)}

### Answer Lengths
- **Average Diagnostic Answer**: {quality.get('avg_diagnostic_length', 0):.0f} characters
- **Average Simple Answer**: {quality.get('avg_simple_length', 0):.0f} characters

## 💡 Key Insights & Recommendations

{insights}

## 🎯 Recommended Actions

1. **Current Configuration**: BM25=70%, Vector=30%
"""
        
        # Add specific recommendations based on analysis
        if patterns['bm25_dominant'] > len(self.results) * 0.6:
            report += "2. **Weight Adjustment**: Keep current weights (BM25=70%, Vector=30%)\n"
        elif patterns['vector_dominant'] > len(self.results) * 0.3:
            report += "2. **Weight Adjustment**: Consider adjusting to BM25=60%, Vector=40%\n"
        else:
            report += "2. **Weight Adjustment**: Current balance appears appropriate\n"
        
        if retrieval.get('document_usage_rate', 0) < 0.5:
            report += "3. **Knowledge Base**: Expand document coverage for better retrieval\n"
        
        if retrieval['avg_top1_score'] < 0.5:
            report += "4. **Chunking Strategy**: Review chunking size and overlap parameters\n"
        
        # Save report
        if output_file:
            output_path = Path(output_file)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(__file__).parent / f'analysis_report_{timestamp}.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n{'='*60}")
        print(report)
        print(f"{'='*60}")
        print(f"\nReport saved to: {output_path}")
        
        return report


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Analyze RAG diagnostic test results')
    parser.add_argument('results_file', help='Path to diagnostic results JSON file')
    parser.add_argument('--output', '-o', help='Output report file path')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = DiagnosticAnalyzer(args.results_file)
    
    # Generate report
    analyzer.generate_report(args.output)


if __name__ == "__main__":
    # If no arguments provided, look for most recent results file
    import sys
    if len(sys.argv) == 1:
        # Find most recent diagnostic results
        results_dir = Path(__file__).parent
        results_files = list(results_dir.glob('diagnostic_results_*.json'))
        
        if results_files:
            most_recent = max(results_files, key=lambda p: p.stat().st_mtime)
            print(f"Using most recent results: {most_recent}")
            analyzer = DiagnosticAnalyzer(str(most_recent))
            analyzer.generate_report()
        else:
            print("No diagnostic results files found. Run diagnostic test first.")
            print("Usage: python analyze_results.py <results_file.json>")
    else:
        main()