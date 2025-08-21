# PowerShell script to prepare files for server deployment
# 서버 배포를 위한 파일 압축 스크립트

Write-Host "========================================" -ForegroundColor Green
Write-Host "Preparing Question Generation Package" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Set paths
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Create temporary directory for package
$tempDir = ".\temp_package"
if (Test-Path $tempDir) {
    Remove-Item $tempDir -Recurse -Force
}
New-Item -ItemType Directory -Path $tempDir | Out-Null

Write-Host "`nCreating directory structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "$tempDir\scripts\utils" -Force | Out-Null
New-Item -ItemType Directory -Path "$tempDir\configs" -Force | Out-Null
New-Item -ItemType Directory -Path "$tempDir\data\rag" -Force | Out-Null

Write-Host "`nCopying script files..." -ForegroundColor Yellow
Copy-Item "scripts\run_question_generation_remote.py" "$tempDir\scripts\" -Force
Copy-Item "scripts\utils\diversity_sampler.py" "$tempDir\scripts\utils\" -Force
Copy-Item "scripts\utils\question_validator.py" "$tempDir\scripts\utils\" -Force

Write-Host "Copying config files..." -ForegroundColor Yellow
Copy-Item "configs\question_generation_config.yaml" "$tempDir\configs\" -Force

Write-Host "Copying RAG data files..." -ForegroundColor Yellow
Copy-Item "data\rag\chunks_2300.json" "$tempDir\data\rag\" -Force
Copy-Item "data\rag\embeddings_2300.npy" "$tempDir\data\rag\" -Force
Copy-Item "data\rag\faiss_index_2300.index" "$tempDir\data\rag\" -Force
Copy-Item "data\rag\bm25_index_2300.pkl" "$tempDir\data\rag\" -Force
Copy-Item "data\rag\metadata_2300.json" "$tempDir\data\rag\" -Force

# Create requirements.txt
Write-Host "`nCreating requirements.txt..." -ForegroundColor Yellow
@"
torch>=2.0.0
transformers>=4.41.0
vllm>=0.5.0
accelerate>=0.30.0
bitsandbytes>=0.43.0
sentence-transformers>=2.7.0
faiss-cpu
rank-bm25
kiwipiepy
numpy
pandas
tqdm
pyyaml
"@ | Out-File -FilePath "$tempDir\requirements.txt" -Encoding UTF8

# Create setup script for server
Write-Host "Creating setup script..." -ForegroundColor Yellow
@'
#!/bin/bash
# Server setup script

echo "Setting up Question Generation environment..."

# Check Python version
python3 --version

# Create virtual environment (optional)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Check GPU availability
echo "Checking GPU..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
if [ $? -eq 0 ]; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
fi

echo "Setup complete!"
'@ | Out-File -FilePath "$tempDir\setup_server.sh" -Encoding UTF8 -NoNewline

# Create run script
Write-Host "Creating run script..." -ForegroundColor Yellow
@'
#!/bin/bash
# Run question generation

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Set CUDA visible devices (adjust as needed)
export CUDA_VISIBLE_DEVICES=0

echo "Starting question generation..."
echo "Model: Qwen/Qwen2.5-32B-Instruct-AWQ"
echo "Questions to generate: 30"
echo ""

# Run with different options based on available resources
if command -v vllm &> /dev/null; then
    echo "Using vLLM for inference..."
    python scripts/run_question_generation_remote.py \
        --model "Qwen/Qwen2.5-32B-Instruct-AWQ" \
        --config configs/question_generation_config.yaml \
        --n-questions 30 \
        --use-vllm
else
    echo "Using transformers for inference..."
    python scripts/run_question_generation_remote.py \
        --model "Qwen/Qwen2.5-32B-Instruct-AWQ" \
        --config configs/question_generation_config.yaml \
        --n-questions 30
fi

echo "Generation complete!"
echo "Results saved in data/synthetic_questions/"
'@ | Out-File -FilePath "$tempDir\run_generation.sh" -Encoding UTF8 -NoNewline

# Create tar.gz archive using 7-Zip if available, otherwise use built-in compression
$archiveName = "question_generation_package.tar.gz"

Write-Host "`nCreating archive: $archiveName" -ForegroundColor Yellow

if (Get-Command 7z -ErrorAction SilentlyContinue) {
    # Use 7-Zip if available
    Set-Location $tempDir
    7z a -ttar temp.tar *
    7z a -tgzip ..\$archiveName temp.tar
    Remove-Item temp.tar
    Set-Location ..
} else {
    # Use PowerShell compression (creates .zip, not .tar.gz)
    $archiveName = "question_generation_package.zip"
    Compress-Archive -Path "$tempDir\*" -DestinationPath $archiveName -Force
    Write-Host "Note: Created .zip file (7-Zip not found for .tar.gz)" -ForegroundColor Yellow
}

# Clean up
Remove-Item $tempDir -Recurse -Force

# Display package info
$packageSize = (Get-Item $archiveName).Length / 1MB
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Package created successfully!" -ForegroundColor Green
Write-Host "File: $archiveName" -ForegroundColor Yellow
Write-Host "Size: $([math]::Round($packageSize, 2)) MB" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green