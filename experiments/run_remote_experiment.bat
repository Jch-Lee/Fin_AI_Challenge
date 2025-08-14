@echo off
chcp 65001 > nul
echo ========================================
echo 🚀 VL 모델 원격 실험 실행 스크립트
echo ========================================
echo.

set SERVER=root@47.186.63.142
set PORT=52283
set REMOTE_DIR=/root/vl_experiment

echo 📋 실행 단계:
echo 1. 파일 전송
echo 2. 환경 설정
echo 3. 실험 실행
echo 4. 결과 회수
echo.

pause

echo.
echo 📤 1단계: 파일 전송 중...
echo ========================================

echo   - 실험 스크립트 전송...
scp -P %PORT% experiments\vl_extraction_comparison.py %SERVER%:%REMOTE_DIR%/
if errorlevel 1 (
    echo ❌ 실험 스크립트 전송 실패
    goto :error
)

echo   - 환경 설정 스크립트 전송...
scp -P %PORT% experiments\remote_setup.sh %SERVER%:%REMOTE_DIR%/
if errorlevel 1 (
    echo ❌ 환경 설정 스크립트 전송 실패
    goto :error
)

echo   - PDF 파일 전송...
scp -P %PORT% "data\raw\금융분야 AI 보안 가이드라인.pdf" %SERVER%:%REMOTE_DIR%/data/
if errorlevel 1 (
    echo ❌ PDF 파일 전송 실패
    goto :error
)

echo ✅ 파일 전송 완료
echo.

echo 🔧 2단계: 환경 설정...
echo ========================================
ssh -p %PORT% %SERVER% "cd %REMOTE_DIR% && chmod +x remote_setup.sh && ./remote_setup.sh"
if errorlevel 1 (
    echo ❌ 환경 설정 실패
    goto :error
)

echo ✅ 환경 설정 완료
echo.

echo 🧪 3단계: 환경 테스트...
echo ========================================
ssh -p %PORT% %SERVER% "cd %REMOTE_DIR% && python3 test_environment.py"
if errorlevel 1 (
    echo ❌ 환경 테스트 실패
    goto :error
)

echo ✅ 환경 테스트 통과
echo.

echo 🚀 4단계: 실험 실행...
echo ========================================
echo VL 모델 실험을 시작합니다. 이 과정은 10-15분 정도 소요됩니다.

ssh -p %PORT% %SERVER% "cd %REMOTE_DIR% && python3 vl_extraction_comparison.py --max-pages 10"
if errorlevel 1 (
    echo ❌ 실험 실행 실패
    goto :error
)

echo ✅ 실험 실행 완료
echo.

echo 📦 5단계: 결과 압축...
echo ========================================
ssh -p %PORT% %SERVER% "cd %REMOTE_DIR% && tar -czf vl_results_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%.tar.gz outputs/vl_comparison_*"

echo.
echo 📥 6단계: 결과 다운로드...
echo ========================================
scp -P %PORT% %SERVER%:%REMOTE_DIR%/vl_results_*.tar.gz experiments\
if errorlevel 1 (
    echo ❌ 결과 다운로드 실패
    goto :error
)

echo   - 압축 파일 해제...
cd experiments
for %%f in (vl_results_*.tar.gz) do (
    tar -xzf "%%f"
)

echo ✅ 결과 다운로드 완료
echo.

echo 📊 7단계: 결과 확인...
echo ========================================
for /d %%d in (outputs\vl_comparison_*) do (
    if exist "%%d\comparison_report.html" (
        echo 🌐 HTML 리포트: %%d\comparison_report.html
        start "" "%%d\comparison_report.html"
    )
    if exist "%%d\summary.json" (
        echo 📊 요약 통계: %%d\summary.json
    )
)

echo.
echo ========================================
echo ✅ VL 모델 실험 완료!
echo ========================================
echo.
echo 📁 결과 위치: experiments\outputs\vl_comparison_*
echo 🌐 HTML 리포트가 브라우저에서 열립니다.
echo 📊 자세한 분석은 RESULT_ANALYSIS_GUIDE.md를 참조하세요.
echo.
pause
goto :end

:error
echo.
echo ========================================
echo ❌ 실험 실행 중 오류 발생
echo ========================================
echo.
echo 🔧 문제 해결 방법:
echo 1. SSH 연결 확인: ssh -p %PORT% %SERVER%
echo 2. 수동 실행 가이드: REMOTE_EXECUTION_GUIDE.md 참조
echo 3. 파일 전송 상태 확인
echo.
pause

:end