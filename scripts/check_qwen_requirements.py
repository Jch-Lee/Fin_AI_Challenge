#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5-7B 실행 환경 검증 스크립트
"""

import torch
import transformers
import sys
import subprocess
import platform

def check_environment():
    """환경 검증"""
    
    print("="*60)
    print(" Qwen2.5-7B 환경 체크")
    print("="*60)
    
    checks = []
    
    # 1. Python 버전
    py_version = sys.version
    py_ok = sys.version_info >= (3, 8)
    checks.append(("Python", py_version.split()[0], py_ok))
    print(f"[CHECK] Python: {py_version.split()[0]} {'OK' if py_ok else 'FAIL'}")
    
    # 2. PyTorch 버전
    torch_version = torch.__version__
    torch_ok = torch_version >= "2.0.0"
    checks.append(("PyTorch", torch_version, torch_ok))
    print(f"[CHECK] PyTorch: {torch_version} {'OK' if torch_ok else 'FAIL'}")
    
    # 3. Transformers 버전
    trans_version = transformers.__version__
    trans_ok = trans_version >= "4.37.0"
    checks.append(("Transformers", trans_version, trans_ok))
    print(f"[CHECK] Transformers: {trans_version} {'OK' if trans_ok else 'FAIL (need >=4.37.0)'}")
    
    # 4. CUDA 확인
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[CHECK] CUDA: {cuda_version} OK")
        print(f"  - GPU: {gpu_name}")
        print(f"  - VRAM: {gpu_memory:.1f}GB")
    else:
        print("[CHECK] CUDA: Not available WARNING (CPU mode will be slow)")
    
    # 5. 필요 패키지 확인
    print("\n필수 패키지 체크:")
    required_packages = {
        "bitsandbytes": "4-bit 양자화",
        "accelerate": "모델 로딩 가속",
        "sentencepiece": "토크나이저",
        "protobuf": "모델 직렬화"
    }
    
    for pkg, desc in required_packages.items():
        try:
            __import__(pkg)
            print(f"  [OK] {pkg}: {desc}")
        except ImportError:
            print(f"  [FAIL] {pkg}: {desc} - pip install {pkg}")
            checks.append((pkg, "Missing", False))
    
    # 6. 메모리 체크
    print("\n시스템 리소스:")
    import psutil
    ram = psutil.virtual_memory().total / 1024**3
    print(f"  - RAM: {ram:.1f}GB")
    
    if cuda_available:
        required_vram = 8.0 if trans_ok else 16.0  # 4-bit: 8GB, fp16: 16GB
        vram_ok = gpu_memory >= required_vram
        print(f"  - VRAM 요구사항: {required_vram:.1f}GB {'OK' if vram_ok else 'FAIL'}")
    
    # 7. 디스크 공간
    import shutil
    disk = shutil.disk_usage("/")
    free_gb = disk.free / 1024**3
    print(f"  - 디스크 여유 공간: {free_gb:.1f}GB")
    
    # 결과 요약
    all_ok = all(check[2] for check in checks)
    
    print("\n" + "="*60)
    if all_ok:
        print("[SUCCESS] 모든 요구사항 충족! Qwen2.5-7B 실행 가능")
    else:
        print("[WARNING] 일부 요구사항 미충족. 위 항목을 확인하세요.")
    print("="*60)
    
    return all_ok

def install_missing():
    """누락된 패키지 설치"""
    
    print("\n누락된 패키지 설치 중...")
    
    packages = [
        "transformers>=4.37.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.30.0",
        "sentencepiece",
        "protobuf"
    ]
    
    for pkg in packages:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    
    print("[SUCCESS] 패키지 설치 완료!")

if __name__ == "__main__":
    if check_environment():
        print("\n다음 명령 실행:")
        print("python scripts/integrate_qwen_llm.py")
    else:
        response = input("\n누락된 패키지를 설치하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            install_missing()
            print("\n환경 설정 완료! 다시 실행해주세요.")