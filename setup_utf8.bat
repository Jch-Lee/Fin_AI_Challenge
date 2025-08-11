@echo off
REM UTF-8 인코딩 환경 설정 배치 파일

echo === UTF-8 인코딩 환경 설정 ===
echo.

REM Python I/O 인코딩 설정
set PYTHONIOENCODING=utf-8

REM 시스템 로케일 설정
set LANG=ko_KR.UTF-8
set LC_ALL=ko_KR.UTF-8

REM Windows 코드 페이지를 UTF-8로 변경 (65001)
chcp 65001 > nul

echo 환경변수 설정 완료:
echo   PYTHONIOENCODING = %PYTHONIOENCODING%
echo   LANG = %LANG%
echo   LC_ALL = %LC_ALL%
echo   Code Page = 65001 (UTF-8)
echo.
echo UTF-8 인코딩 환경이 활성화되었습니다.
echo 이제 Python 스크립트를 실행하세요.
echo.