#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KURE-v1 Similarity Benchmark Test
Compare performance between NumPy and KURE native similarity methods
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from packages.rag.embeddings.kure_embedder import KUREEmbedder
from packages.rag.embeddings.base_embedder import BaseEmbedder
from packages.rag.knowledge_base import KnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarityBenchmark:
    """Benchmark for comparing similarity computation methods"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Initialize embedder
        self.embedder = KUREEmbedder(
            model_name="nlpai-lab/KURE-v1",
            device=self.device,
            batch_size=32,
            show_progress=False
        )
        
        # Test data sizes
        self.test_sizes = [
            (10, 100),      # 10 queries, 100 documents
            (50, 500),      # 50 queries, 500 documents
            (100, 1000),    # 100 queries, 1000 documents
            (200, 5000),    # 200 queries, 5000 documents
            (500, 10000),   # 500 queries, 10000 documents
        ]
        
        if self.device == 'cuda':
            self.test_sizes.append((1000, 20000))  # Large scale test
    
    def generate_test_data(self, n_queries: int, n_docs: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random embeddings for testing"""
        embedding_dim = self.embedder.embedding_dim
        
        # Generate random embeddings
        query_embeddings = np.random.randn(n_queries, embedding_dim).astype('float32')
        doc_embeddings = np.random.randn(n_docs, embedding_dim).astype('float32')
        
        # Normalize (as KURE does)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        return query_embeddings, doc_embeddings
    
    def benchmark_numpy_similarity(self, query_embeddings: np.ndarray, 
                                  doc_embeddings: np.ndarray,
                                  top_k: int = 5) -> Dict:
        """Benchmark NumPy-based similarity computation"""
        start_time = time.time()
        
        # Compute similarity matrix using NumPy
        similarity_matrix = np.dot(query_embeddings, doc_embeddings.T)
        
        # Get top-k for each query
        top_k_indices = []
        top_k_scores = []
        for i in range(len(query_embeddings)):
            scores = similarity_matrix[i]
            indices = np.argsort(scores)[-top_k:][::-1]
            top_k_indices.append(indices)
            top_k_scores.append(scores[indices])
        
        elapsed_time = time.time() - start_time
        
        return {
            'method': 'NumPy',
            'time': elapsed_time,
            'queries': len(query_embeddings),
            'documents': len(doc_embeddings),
            'top_k': top_k,
            'memory_mb': similarity_matrix.nbytes / (1024 * 1024)
        }
    
    def benchmark_kure_similarity(self, query_embeddings: np.ndarray,
                                 doc_embeddings: np.ndarray,
                                 top_k: int = 5) -> Dict:
        """Benchmark KURE native similarity computation"""
        start_time = time.time()
        
        # Use KURE's batch similarity search
        results = self.embedder.batch_similarity_search(
            query_embeddings,
            doc_embeddings,
            top_k=top_k,
            batch_size=100
        )
        
        elapsed_time = time.time() - start_time
        
        # Estimate memory usage
        if self.device == 'cuda':
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()
        else:
            memory_mb = (query_embeddings.nbytes + doc_embeddings.nbytes) / (1024 * 1024)
        
        return {
            'method': 'KURE',
            'time': elapsed_time,
            'queries': len(query_embeddings),
            'documents': len(doc_embeddings),
            'top_k': top_k,
            'memory_mb': memory_mb
        }
    
    def benchmark_kure_matrix(self, query_embeddings: np.ndarray,
                             doc_embeddings: np.ndarray) -> Dict:
        """Benchmark KURE similarity matrix computation"""
        start_time = time.time()
        
        # Compute full similarity matrix using KURE
        similarity_matrix = self.embedder.compute_similarity_matrix(
            query_embeddings,
            doc_embeddings
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            'method': 'KURE Matrix',
            'time': elapsed_time,
            'queries': len(query_embeddings),
            'documents': len(doc_embeddings),
            'matrix_shape': similarity_matrix.shape,
            'memory_mb': similarity_matrix.nbytes / (1024 * 1024)
        }
    
    def run_benchmarks(self):
        """Run all benchmarks"""
        results = []
        
        print("\n" + "=" * 80)
        print("KURE-v1 Similarity Benchmark")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Embedding dimension: {self.embedder.embedding_dim}")
        print("-" * 80)
        
        for n_queries, n_docs in self.test_sizes:
            print(f"\nTest: {n_queries} queries × {n_docs} documents")
            print("-" * 40)
            
            # Generate test data
            query_emb, doc_emb = self.generate_test_data(n_queries, n_docs)
            
            # NumPy benchmark
            numpy_result = self.benchmark_numpy_similarity(query_emb, doc_emb)
            print(f"NumPy: {numpy_result['time']:.3f}s, Memory: {numpy_result['memory_mb']:.1f}MB")
            results.append(numpy_result)
            
            # KURE batch search benchmark
            kure_result = self.benchmark_kure_similarity(query_emb, doc_emb)
            print(f"KURE Batch: {kure_result['time']:.3f}s, Memory: {kure_result['memory_mb']:.1f}MB")
            results.append(kure_result)
            
            # KURE matrix benchmark (only for smaller sizes)
            if n_queries <= 100 and n_docs <= 1000:
                matrix_result = self.benchmark_kure_matrix(query_emb, doc_emb)
                print(f"KURE Matrix: {matrix_result['time']:.3f}s, Memory: {matrix_result['memory_mb']:.1f}MB")
                results.append(matrix_result)
            
            # Calculate speedup
            speedup = numpy_result['time'] / kure_result['time']
            print(f"Speedup: {speedup:.2f}x")
            
            # Clear memory
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        return results
    
    def test_accuracy(self):
        """Test that both methods produce identical results"""
        print("\n" + "=" * 80)
        print("Accuracy Test")
        print("=" * 80)
        
        # Small test case
        n_queries, n_docs = 5, 20
        query_emb, doc_emb = self.generate_test_data(n_queries, n_docs)
        
        # NumPy computation
        numpy_sim = np.dot(query_emb, doc_emb.T)
        
        # KURE computation
        kure_sim = self.embedder.compute_similarity_matrix(query_emb, doc_emb)
        
        # Compare results
        max_diff = np.max(np.abs(numpy_sim - kure_sim))
        mean_diff = np.mean(np.abs(numpy_sim - kure_sim))
        
        print(f"Maximum difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-5:
            print("✓ Results are identical (within tolerance)")
        else:
            print("✗ Results differ significantly")
        
        return max_diff < 1e-5


def main():
    """Main benchmark execution"""
    benchmark = SimilarityBenchmark()
    
    # Test accuracy first
    accurate = benchmark.test_accuracy()
    
    if accurate:
        # Run performance benchmarks
        results = benchmark.run_benchmarks()
        
        # Summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        
        # Group results by size
        size_groups = {}
        for r in results:
            key = (r['queries'], r['documents'])
            if key not in size_groups:
                size_groups[key] = {}
            size_groups[key][r['method']] = r['time']
        
        print("\n{:<20} {:<15} {:<15} {:<10}".format(
            "Size (Q×D)", "NumPy (s)", "KURE (s)", "Speedup"
        ))
        print("-" * 60)
        
        for (q, d), times in size_groups.items():
            if 'NumPy' in times and 'KURE' in times:
                speedup = times['NumPy'] / times['KURE']
                print(f"{q:>4} × {d:>5}        {times['NumPy']:>8.3f}       {times['KURE']:>8.3f}       {speedup:>6.2f}x")
        
        print("\n" + "=" * 80)
        print("Benchmark Complete")
        print("=" * 80)
    else:
        print("\n✗ Accuracy test failed. Skipping performance benchmarks.")


if __name__ == "__main__":
    main()