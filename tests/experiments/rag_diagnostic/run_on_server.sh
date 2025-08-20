#!/bin/bash
# RAG Diagnostic Test - Server Execution Script

echo "========================================"
echo "RAG Diagnostic Test - Server Execution"
echo "========================================"

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Check if running on server
if [ -z "$HOSTNAME" ]; then
    HOSTNAME=$(hostname)
fi

echo "Running on: $HOSTNAME"
echo "Date: $(date)"
echo ""

# Step 1: Verify files exist
echo "1. Checking required files..."
if [ ! -f "data/rag/chunks_2300.json" ]; then
    echo "[ERROR] RAG data files not found. Please ensure RAG system is built."
    exit 1
fi

if [ ! -f "tests/experiments/rag_diagnostic/test_questions_20.csv" ]; then
    echo "[ERROR] Test questions file not found. Running extraction..."
    python tests/experiments/rag_diagnostic/extract_questions.py
fi

# Step 2: Run diagnostic test
echo ""
echo "2. Running diagnostic test..."
echo "This may take 10-15 minutes for 20 questions."
echo ""

python tests/experiments/rag_diagnostic/run_diagnostic_test.py

# Check if test completed
if [ $? -eq 0 ]; then
    echo ""
    echo "3. Test completed successfully!"
    
    # Find the most recent results file
    LATEST_RESULTS=$(ls -t tests/experiments/rag_diagnostic/diagnostic_results_*.json 2>/dev/null | head -1)
    
    if [ -n "$LATEST_RESULTS" ]; then
        echo "Results saved to: $LATEST_RESULTS"
        
        # Step 3: Run analysis
        echo ""
        echo "4. Running analysis..."
        python tests/experiments/rag_diagnostic/analyze_results.py "$LATEST_RESULTS"
        
        echo ""
        echo "========================================"
        echo "Diagnostic test and analysis complete!"
        echo "========================================"
        
        # List output files
        echo ""
        echo "Output files:"
        ls -la tests/experiments/rag_diagnostic/*.json 2>/dev/null
        ls -la tests/experiments/rag_diagnostic/*.md 2>/dev/null
    else
        echo "[WARNING] No results file found."
    fi
else
    echo "[ERROR] Diagnostic test failed."
    exit 1
fi