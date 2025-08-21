# PowerShell script for deploying to remote server

param(
    [Parameter(Mandatory=$true)]
    [string]$ServerAddress,
    
    [Parameter(Mandatory=$false)]
    [string]$UserName = "ubuntu",
    
    [Parameter(Mandatory=$false)]
    [string]$RemoteDir = "/home/$UserName/question_generation"
)

Write-Host "========================================" -ForegroundColor Green
Write-Host "Remote Server Question Generation Deploy" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "Server: $ServerAddress" -ForegroundColor Yellow
Write-Host "User: $UserName" -ForegroundColor Yellow
Write-Host "Remote Directory: $RemoteDir" -ForegroundColor Yellow

# Files to copy
$FilesToCopy = @(
    "scripts\run_question_generation_remote.py",
    "scripts\utils\diversity_sampler.py",
    "scripts\utils\question_validator.py",
    "configs\question_generation_config.yaml",
    "data\rag\chunks_2300.json",
    "data\rag\embeddings_2300.npy",
    "data\rag\faiss_index_2300.index",
    "data\rag\bm25_index_2300.pkl",
    "data\rag\metadata_2300.json"
)

# Create remote directories
Write-Host "`n1. Creating remote directories..." -ForegroundColor Green
$sshCommand = "mkdir -p $RemoteDir/{scripts/utils,configs,data/rag,data/synthetic_questions}"
ssh "$UserName@$ServerAddress" $sshCommand

# Copy files
Write-Host "`n2. Copying files..." -ForegroundColor Green
foreach ($file in $FilesToCopy) {
    Write-Host "  Copying: $file"
    $localPath = Join-Path $PSScriptRoot ".." $file
    $remotePath = "$RemoteDir/" + $file.Replace('\', '/')
    scp $localPath "${UserName}@${ServerAddress}:$remotePath"
}

# Create and copy requirements.txt
Write-Host "`n3. Creating requirements.txt..." -ForegroundColor Green
$requirements = @"
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
"@

$requirementsPath = ".\temp_requirements.txt"
$requirements | Out-File -FilePath $requirementsPath -Encoding UTF8
scp $requirementsPath "${UserName}@${ServerAddress}:$RemoteDir/requirements.txt"
Remove-Item $requirementsPath

# Create run script
Write-Host "`n4. Creating run script..." -ForegroundColor Green
$runScript = @'
#!/bin/bash
cd /home/ubuntu/question_generation

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Install packages
echo "Installing required packages..."
pip install -r requirements.txt

# Run question generation
echo "Starting question generation..."
python scripts/run_question_generation_remote.py \
    --model "Qwen/Qwen2.5-32B-Instruct-AWQ" \
    --config configs/question_generation_config.yaml \
    --n-questions 30 \
    --use-vllm

echo "Generation complete!"
'@

$runScriptPath = ".\temp_run.sh"
$runScript | Out-File -FilePath $runScriptPath -Encoding UTF8 -NoNewline
scp $runScriptPath "${UserName}@${ServerAddress}:$RemoteDir/run_generation.sh"
ssh "$UserName@$ServerAddress" "chmod +x $RemoteDir/run_generation.sh"
Remove-Item $runScriptPath

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run on remote server:" -ForegroundColor Yellow
Write-Host "  ssh $UserName@$ServerAddress"
Write-Host "  cd $RemoteDir"
Write-Host "  ./run_generation.sh"
Write-Host ""
Write-Host "Or run directly:" -ForegroundColor Yellow
Write-Host "  ssh $UserName@$ServerAddress 'cd $RemoteDir && ./run_generation.sh'"