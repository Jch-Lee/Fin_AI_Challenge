#!/usr/bin/env python3
"""
Kiwi 설치 확인 및 기본 동작 테스트 스크립트
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def verify_kiwi_installation():
    """Kiwi 설치 및 기본 기능 확인"""
    
    # 1. Import 테스트
    try:
        from kiwipiepy import Kiwi
        logger.info("✅ Kiwi import 성공")
    except ImportError as e:
        logger.error(f"❌ Kiwi import 실패: {e}")
        logger.error("설치 필요: pip install kiwipiepy")
        return False
    
    # 2. 기본 초기화 테스트
    try:
        kiwi = Kiwi()
        logger.info("✅ Kiwi 초기화 성공")
    except Exception as e:
        logger.error(f"❌ Kiwi 초기화 실패: {e}")
        return False
    
    # 3. 토크나이징 테스트
    try:
        test_text = "금융보안은 중요합니다"
        tokens = kiwi.tokenize(test_text)
        logger.info(f"✅ 토크나이징 성공: {len(tokens)}개 토큰")
        logger.info(f"   예시: {[(t.form, t.tag) for t in tokens[:3]]}")
    except Exception as e:
        logger.error(f"❌ 토크나이징 실패: {e}")
        return False
    
    # 4. 띄어쓰기 교정 테스트
    try:
        test_text = "금융보안은매우중요합니다"
        spaced = kiwi.space(test_text)
        logger.info(f"✅ 띄어쓰기 교정 성공")
        logger.info(f"   입력: {test_text}")
        logger.info(f"   출력: {spaced}")
    except Exception as e:
        logger.error(f"❌ 띄어쓰기 교정 실패: {e}")
        return False
    
    # 5. 형태소 분석 테스트
    try:
        test_text = "한국은행이 기준금리를 인상했습니다"
        tokens = kiwi.tokenize(test_text)
        morphemes = [(t.form, t.tag) for t in tokens]
        logger.info(f"✅ 형태소 분석 성공")
        logger.info(f"   분석 결과: {morphemes[:5]}...")
    except Exception as e:
        logger.error(f"❌ 형태소 분석 실패: {e}")
        return False
    
    # 6. 성능 테스트
    try:
        import time
        test_text = "금융 보안은 디지털 시대에 매우 중요한 이슈입니다. " * 10
        
        start_time = time.time()
        tokens = kiwi.tokenize(test_text)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ 성능 테스트 성공")
        logger.info(f"   처리 시간: {elapsed:.3f}초")
        logger.info(f"   토큰 수: {len(tokens)}개")
        logger.info(f"   처리 속도: {len(tokens)/elapsed:.0f} tokens/sec")
    except Exception as e:
        logger.error(f"❌ 성능 테스트 실패: {e}")
        return False
    
    logger.info("\n" + "="*50)
    logger.info("🎉 모든 Kiwi 테스트 통과!")
    logger.info("="*50)
    
    return True


def check_version():
    """Kiwi 버전 확인"""
    try:
        import kiwipiepy
        version = getattr(kiwipiepy, '__version__', 'unknown')
        logger.info(f"📦 Kiwi 버전: {version}")
        return True
    except ImportError:
        logger.error("❌ Kiwi가 설치되지 않았습니다")
        return False


if __name__ == "__main__":
    logger.info("="*50)
    logger.info("Kiwi 설치 확인 시작")
    logger.info("="*50 + "\n")
    
    # 버전 확인
    if not check_version():
        sys.exit(1)
    
    logger.info("")
    
    # 기능 테스트
    if not verify_kiwi_installation():
        sys.exit(1)
    
    logger.info("\n✅ Kiwi를 사용할 준비가 완료되었습니다!")