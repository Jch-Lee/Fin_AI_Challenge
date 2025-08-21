# PowerShell script to deploy to remote server
# 원격 서버로 파일 전송 스크립트

param(
    [switch]$SkipPackage,
    [switch]$SkipUpload,
    [switch]$RunOnly
)

# Server configuration
$SERVER_HOST = "108.172.120.126"
$SERVER_PORT = 33374
$SERVER_USER = "root"
$REMOTE_DIR = "/root/question_generation"
$LOCAL_PACKAGE = "question_generation_package.zip"

Write-Host "========================================" -ForegroundColor Green
Write-Host "Deploy to Remote Server" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Server: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT}" -ForegroundColor Yellow
Write-Host "Remote Directory: $REMOTE_DIR" -ForegroundColor Yellow
Write-Host ""

# Step 1: Create package (unless skipped)
if (-not $SkipPackage -and -not $RunOnly) {
    Write-Host "Step 1: Creating package..." -ForegroundColor Green
    & "$PSScriptRoot\prepare_package.ps1"
    
    if (-not (Test-Path $LOCAL_PACKAGE)) {
        Write-Host "Error: Package file not found!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Step 1: Skipping package creation" -ForegroundColor Yellow
}

# Step 2: Upload package (unless skipped)
if (-not $SkipUpload -and -not $RunOnly) {
    Write-Host "`nStep 2: Uploading package to server..." -ForegroundColor Green
    
    # Create remote directory
    Write-Host "Creating remote directory structure..." -ForegroundColor Yellow
    $sshCommand = "mkdir -p $REMOTE_DIR/{scripts/utils,configs,data/rag,data/synthetic_questions,logs}"
    ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} $sshCommand
    
    # Upload package
    Write-Host "Uploading $LOCAL_PACKAGE..." -ForegroundColor Yellow
    scp -P $SERVER_PORT $LOCAL_PACKAGE ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to upload package!" -ForegroundColor Red
        exit 1
    }
    
    # Extract package on server
    Write-Host "Extracting package on server..." -ForegroundColor Yellow
    if ($LOCAL_PACKAGE -like "*.zip") {
        ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} "cd $REMOTE_DIR && unzip -o $LOCAL_PACKAGE"
    } else {
        ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} "cd $REMOTE_DIR && tar -xzf $LOCAL_PACKAGE"
    }
    
    # Make scripts executable
    ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} "cd $REMOTE_DIR && chmod +x *.sh"
    
    Write-Host "Upload complete!" -ForegroundColor Green
} else {
    Write-Host "`nStep 2: Skipping upload" -ForegroundColor Yellow
}

# Step 3: Setup environment on server
Write-Host "`nStep 3: Setting up server environment..." -ForegroundColor Green
Write-Host "Running setup script on server..." -ForegroundColor Yellow

$setupCommand = @"
cd $REMOTE_DIR
if [ -f setup_server.sh ]; then
    echo 'Running setup script...'
    bash setup_server.sh
else
    echo 'Setup script not found, installing packages manually...'
    pip install torch transformers vllm accelerate bitsandbytes
    pip install sentence-transformers faiss-cpu rank-bm25
    pip install kiwipiepy numpy pandas tqdm pyyaml
fi
"@

ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} $setupCommand

# Step 4: Display run instructions
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the experiment on the server:" -ForegroundColor Yellow
Write-Host "  1. Connect to server:" -ForegroundColor Cyan
Write-Host "     ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST}" -ForegroundColor White
Write-Host ""
Write-Host "  2. Navigate to directory:" -ForegroundColor Cyan
Write-Host "     cd $REMOTE_DIR" -ForegroundColor White
Write-Host ""
Write-Host "  3. Run generation:" -ForegroundColor Cyan
Write-Host "     ./run_generation.sh" -ForegroundColor White
Write-Host ""
Write-Host "  4. Or run in background:" -ForegroundColor Cyan
Write-Host "     nohup ./run_generation.sh > generation.log 2>&1 &" -ForegroundColor White
Write-Host ""
Write-Host "  5. Monitor progress:" -ForegroundColor Cyan
Write-Host "     tail -f generation.log" -ForegroundColor White
Write-Host ""

# Optional: Start generation immediately
$response = Read-Host "Do you want to start the generation now? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host "`nStarting generation on server..." -ForegroundColor Green
    ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} "cd $REMOTE_DIR && nohup ./run_generation.sh > generation.log 2>&1 &"
    Write-Host "Generation started in background!" -ForegroundColor Green
    Write-Host "Check progress with: ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} 'tail -f $REMOTE_DIR/generation.log'" -ForegroundColor Yellow
}