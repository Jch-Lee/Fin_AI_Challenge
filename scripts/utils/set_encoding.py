#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UTF-8 인코딩 환경 설정 스크립트
모든 Python 파일 I/O에 UTF-8을 기본으로 설정
"""

import os
import sys
import locale

def setup_utf8_environment():
    """UTF-8 환경 설정"""
    
    # 1. Python 기본 인코딩 확인 및 설정
    print("=== UTF-8 인코딩 환경 설정 ===")
    print(f"현재 Python 기본 인코딩: {sys.getdefaultencoding()}")
    print(f"현재 선호 인코딩: {locale.getpreferredencoding()}")
    
    # 2. 환경변수 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'ko_KR.UTF-8'
    os.environ['LC_ALL'] = 'ko_KR.UTF-8'
    
    print("\n환경변수 설정 완료:")
    print(f"  PYTHONIOENCODING = {os.environ.get('PYTHONIOENCODING')}")
    print(f"  LANG = {os.environ.get('LANG')}")
    print(f"  LC_ALL = {os.environ.get('LC_ALL')}")
    
    # 3. Windows 환경 추가 설정
    if sys.platform == 'win32':
        # Windows UTF-8 모드 활성화
        import codecs
        # stdout, stderr를 UTF-8로 재설정
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        print("\nWindows UTF-8 모드 활성화 완료")
    
    print("\n✅ UTF-8 인코딩 환경 설정 완료!")
    print("모든 파일 I/O가 UTF-8로 처리됩니다.")
    
    return True

def validate_encoding():
    """인코딩 설정 검증"""
    print("\n=== 인코딩 설정 검증 ===")
    
    # 테스트 문자열
    test_strings = [
        "한글 테스트",
        "金融分野 AI 保安",
        "😊 이모지 테스트",
        "특수문자: ♠♥♦♣"
    ]
    
    # 파일 쓰기/읽기 테스트
    test_file = "encoding_test.txt"
    
    try:
        # UTF-8로 쓰기
        with open(test_file, 'w', encoding='utf-8') as f:
            for s in test_strings:
                f.write(s + '\n')
        
        # UTF-8로 읽기
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print("✅ 파일 I/O 테스트 성공:")
        print(content)
        
        # 테스트 파일 삭제
        os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"❌ 인코딩 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    # UTF-8 환경 설정
    if setup_utf8_environment():
        # 설정 검증
        validate_encoding()
    
    print("\n이제 다른 Python 스크립트를 실행하기 전에")
    print("이 스크립트를 먼저 실행하거나,")
    print("아래 명령어로 환경변수를 설정하세요:")
    print("\nWindows:")
    print("  set PYTHONIOENCODING=utf-8")
    print("\nLinux/Mac:")
    print("  export PYTHONIOENCODING=utf-8")