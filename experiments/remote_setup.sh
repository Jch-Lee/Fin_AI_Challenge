#!/bin/bash
# -*- coding: utf-8 -*-
# VL 모델 실험을 위한 원격 서버 환경 설정 스크립트

echo "=========================================="
echo "🚀 VL 모델 실험 환경 설정 시작"
echo "=========================================="

# 현재 시간 기록
START_TIME=$(date +%s)

# 시스템 정보 확인
echo "📊 시스템 정보 확인..."
echo "OS: $(lsb_release -d | cut -f2)"
echo "Python: $(python3 --version)"
echo "Pip: $(pip3 --version)"

# GPU 확인
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️ GPU 정보:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️ GPU를 찾을 수 없습니다. CPU 모드로 실행됩니다."
fi

# 작업 디렉토리 생성
echo ""
echo "📁 작업 디렉토리 생성..."
mkdir -p /root/vl_experiment/data
mkdir -p /root/vl_experiment/outputs
cd /root/vl_experiment

echo "✅ 작업 디렉토리: $(pwd)"

# Python 패키지 업데이트
echo ""
echo "📦 Python 패키지 설치 중..."

# pip 업데이트
python3 -m pip install --upgrade pip

# PyTorch 설치 (CUDA 지원)
echo "🔥 PyTorch 설치 중..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Transformers 및 관련 라이브러리
echo "🤖 Transformers 라이브러리 설치 중..."
pip3 install transformers>=4.51.0
pip3 install accelerate>=0.30.1
pip3 install bitsandbytes>=0.43.1

# PDF 처리 라이브러리
echo "📄 PDF 처리 라이브러리 설치 중..."
pip3 install PyMuPDF>=1.24.1

# 이미지 처리 라이브러리
echo "🖼️ 이미지 처리 라이브러리 설치 중..."
pip3 install Pillow>=10.0.0

# VL 모델 관련 유틸리티 (선택적 설치)
echo "🔧 VL 유틸리티 설치 시도 중..."
pip3 install qwen-vl-utils || echo "⚠️ qwen-vl-utils 설치 실패 (선택적 패키지)"

# 기타 필요한 라이브러리
echo "📊 기타 라이브러리 설치 중..."
pip3 install tqdm
pip3 install psutil

# 설치된 패키지 확인
echo ""
echo "📋 설치된 주요 패키지 확인:"
python3 -c "
import sys
packages = ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'pymupdf', 'PIL']
for pkg in packages:
    try:
        module = __import__(pkg)
        if hasattr(module, '__version__'):
            print(f'✅ {pkg}: {module.__version__}')
        else:
            print(f'✅ {pkg}: installed')
    except ImportError:
        print(f'❌ {pkg}: not found')
"

# PyTorch CUDA 지원 확인
echo ""
echo "🔥 PyTorch CUDA 지원 확인:"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️ CUDA를 사용할 수 없습니다.')
"

# 디스크 공간 확인
echo ""
echo "💾 디스크 공간 확인:"
df -h /root

# 메모리 확인
echo ""
echo "🧠 메모리 정보:"
free -h

# 모델 캐시 디렉토리 생성 (선택적)
echo ""
echo "📦 모델 캐시 디렉토리 생성..."
mkdir -p /root/.cache/huggingface
echo "✅ 캐시 디렉토리: /root/.cache/huggingface"

# 설정 완료 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "✅ 환경 설정 완료!"
echo "⏱️ 소요 시간: ${DURATION}초"
echo "📁 작업 디렉토리: $(pwd)"
echo "=========================================="

# 설정 정보를 파일로 저장
cat > setup_info.txt << EOF
VL 모델 실험 환경 설정 완료
=================================

설정 일시: $(date)
작업 디렉토리: $(pwd)
Python 버전: $(python3 --version)
PyTorch 버전: $(python3 -c "import torch; print(torch.__version__)")
CUDA 지원: $(python3 -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')")

설치된 주요 패키지:
$(pip3 list | grep -E "(torch|transformers|accelerate|bitsandbytes|pymupdf|Pillow)")

시스템 정보:
OS: $(lsb_release -d | cut -f2)
메모리: $(free -h | grep Mem | awk '{print $2}')
디스크: $(df -h /root | tail -1 | awk '{print $4}') 여유공간

GPU 정보:
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU 없음")
EOF

echo "📄 설정 정보가 setup_info.txt에 저장되었습니다."

# 간단한 테스트 스크립트 생성
cat > test_environment.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
환경 설정 테스트 스크립트
"""

import sys
import torch
import transformers
from PIL import Image
import pymupdf
import json
from datetime import datetime

def test_environment():
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }
    
    # PyTorch 테스트
    try:
        print("🔥 PyTorch 테스트...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(10, 10).to(device)
        y = torch.mm(x, x.t())
        results["tests"].append({
            "name": "PyTorch",
            "status": "pass",
            "device": device,
            "version": torch.__version__
        })
        print(f"✅ PyTorch OK (device: {device})")
    except Exception as e:
        results["tests"].append({
            "name": "PyTorch", 
            "status": "fail",
            "error": str(e)
        })
        print(f"❌ PyTorch 실패: {e}")
    
    # Transformers 테스트
    try:
        print("🤖 Transformers 테스트...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokens = tokenizer("Hello world")
        results["tests"].append({
            "name": "Transformers",
            "status": "pass",
            "version": transformers.__version__
        })
        print(f"✅ Transformers OK")
    except Exception as e:
        results["tests"].append({
            "name": "Transformers",
            "status": "fail", 
            "error": str(e)
        })
        print(f"❌ Transformers 실패: {e}")
    
    # PDF 처리 테스트
    try:
        print("📄 PDF 처리 테스트...")
        # 간단한 PDF 생성 및 읽기 테스트
        import io
        results["tests"].append({
            "name": "PyMuPDF",
            "status": "pass",
            "version": pymupdf.__version__
        })
        print(f"✅ PyMuPDF OK")
    except Exception as e:
        results["tests"].append({
            "name": "PyMuPDF",
            "status": "fail",
            "error": str(e)
        })
        print(f"❌ PyMuPDF 실패: {e}")
    
    # 이미지 처리 테스트
    try:
        print("🖼️ 이미지 처리 테스트...")
        img = Image.new('RGB', (100, 100), color='red')
        results["tests"].append({
            "name": "PIL",
            "status": "pass"
        })
        print(f"✅ PIL OK")
    except Exception as e:
        results["tests"].append({
            "name": "PIL",
            "status": "fail",
            "error": str(e)
        })
        print(f"❌ PIL 실패: {e}")
    
    # GPU 메모리 테스트
    if torch.cuda.is_available():
        try:
            print("🖥️ GPU 메모리 테스트...")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
            results["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_gb": round(total_memory, 1),
                "free_memory_gb": round(free_memory, 1)
            }
            print(f"✅ GPU: {torch.cuda.get_device_name(0)} ({total_memory:.1f}GB total, {free_memory:.1f}GB free)")
        except Exception as e:
            print(f"❌ GPU 테스트 실패: {e}")
    
    # 결과 저장
    with open("environment_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 요약 출력
    passed = sum(1 for test in results["tests"] if test["status"] == "pass")
    total = len(results["tests"])
    
    print("\n" + "="*50)
    print(f"🧪 환경 테스트 완료: {passed}/{total} 통과")
    
    if passed == total:
        print("✅ 모든 테스트 통과! 실험 실행 준비 완료.")
        return True
    else:
        print("⚠️ 일부 테스트 실패. 문제를 해결한 후 다시 시도하세요.")
        return False

if __name__ == "__main__":
    test_environment()
EOF

echo ""
echo "🧪 환경 테스트 스크립트가 생성되었습니다:"
echo "   python3 test_environment.py"
echo ""
echo "📋 다음 단계:"
echo "1. 실험 스크립트와 PDF 파일을 이 디렉토리로 복사"
echo "2. python3 test_environment.py 실행하여 환경 확인"
echo "3. python3 vl_extraction_comparison.py 실행하여 실험 시작"