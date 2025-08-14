# 🚀 VL 모델 실험 원격 서버 실행 가이드

이 가이드는 VL 모델 텍스트 추출 실험을 원격 서버에서 실행하는 전체 과정을 단계별로 설명합니다.

## 📋 사전 준비

### 로컬 환경 확인
```bash
# 필요한 파일들이 준비되었는지 확인
ls experiments/vl_extraction_comparison.py      # 실험 스크립트
ls experiments/remote_setup.sh                 # 환경 설정 스크립트
ls "data/raw/금융분야 AI 보안 가이드라인.pdf"    # 테스트 PDF
```

### 원격 서버 정보
- **서버 주소**: `47.186.63.142`
- **SSH 포트**: `52283`
- **사용자**: `root`
- **SSH 명령**: `ssh -p 52283 root@47.186.63.142 -L 8080:localhost:8080`

## 🗂️ Phase 1: 파일 전송

### 1. 실험 스크립트 전송
```bash
# 실험 스크립트 전송
scp -P 52283 experiments/vl_extraction_comparison.py root@47.186.63.142:/root/vl_experiment/

# 환경 설정 스크립트 전송
scp -P 52283 experiments/remote_setup.sh root@47.186.63.142:/root/vl_experiment/
```

### 2. PDF 파일 전송
```bash
# PDF 파일 전송 (따옴표 주의)
scp -P 52283 "data/raw/금융분야 AI 보안 가이드라인.pdf" root@47.186.63.142:/root/vl_experiment/data/
```

### 3. 전송 확인
```bash
# SSH 접속
ssh -p 52283 root@47.186.63.142

# 파일 확인
ls -la /root/vl_experiment/
ls -la /root/vl_experiment/data/
```

## ⚙️ Phase 2: 환경 설정

### 1. 작업 디렉토리 이동
```bash
cd /root/vl_experiment
```

### 2. 환경 설정 스크립트 실행
```bash
# 실행 권한 부여
chmod +x remote_setup.sh

# 환경 설정 실행
./remote_setup.sh
```

**예상 소요 시간**: 5-10분 (패키지 다운로드 속도에 따라)

### 3. 환경 테스트
```bash
# 환경 테스트 실행
python3 test_environment.py
```

**성공 시 출력 예시**:
```
🧪 환경 테스트 완료: 4/4 통과
✅ 모든 테스트 통과! 실험 실행 준비 완료.
```

### 4. GPU 상태 확인
```bash
# GPU 메모리 확인
nvidia-smi

# 실시간 모니터링 (선택적)
watch -n 1 nvidia-smi
```

## 🔬 Phase 3: 실험 실행

### 1. 실험 스크립트 실행
```bash
# 기본 실행 (10페이지)
python3 vl_extraction_comparison.py

# 커스텀 설정으로 실행
python3 vl_extraction_comparison.py --pdf "data/금융분야 AI 보안 가이드라인.pdf" --max-pages 10 --output outputs
```

### 2. 실행 상태 모니터링

**새 터미널에서 모니터링** (로컬):
```bash
# 새 SSH 세션 시작
ssh -p 52283 root@47.186.63.142

# GPU 메모리 모니터링
watch -n 2 nvidia-smi

# 디스크 사용량 모니터링
watch -n 5 'df -h /root && du -sh /root/vl_experiment/outputs'
```

### 3. 실행 로그 확인
```bash
# 실시간 로그 확인 (실험 진행 중)
tail -f /root/vl_experiment/outputs/vl_comparison_*/experiment.log
```

**예상 실행 시간**:
- **GPU가 40GB+인 경우**: 약 3-5분 (10페이지)
- **GPU가 20-40GB인 경우**: 약 5-8분 (10페이지)
- **GPU가 8-20GB인 경우**: 약 8-15분 (10페이지)

### 4. 실행 중 문제 해결

**메모리 부족 오류**:
```bash
# 페이지 수를 줄여서 재실행
python3 vl_extraction_comparison.py --max-pages 5
```

**모델 다운로드 느림**:
```bash
# Hugging Face 캐시 확인
du -sh /root/.cache/huggingface/
```

**실행 중단 시**:
```bash
# GPU 메모리 정리
python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
```

## 📊 Phase 4: 결과 확인

### 1. 결과 파일 확인
```bash
# 결과 디렉토리 확인
ls -la outputs/vl_comparison_*/

# 생성된 파일들 확인
find outputs/vl_comparison_* -type f -name "*.html" -o -name "*.json" -o -name "*.txt" | head -20
```

### 2. 요약 결과 확인
```bash
# JSON 요약 보기
cat outputs/vl_comparison_*/summary.json | jq '.statistics'

# 또는 Python으로 보기 좋게 출력
python3 -c "
import json
with open('$(ls outputs/vl_comparison_*/summary.json | head -1)', 'r') as f:
    data = json.load(f)
    stats = data.get('statistics', {})
    print(f'평균 개선율: {stats.get(\"average_improvement_rate\", 0):.1f}%')
    print(f'총 추가 문자: {stats.get(\"total_improvement\", 0):,}')
    print(f'총 실험 시간: {stats.get(\"total_experiment_time\", 0):.1f}초')
"
```

### 3. 페이지별 결과 미리보기
```bash
# 첫 번째 페이지 결과 미리보기
echo "=== PyMuPDF 추출 (첫 200자) ==="
head -c 200 outputs/vl_comparison_*/page_001/pymupdf.txt

echo -e "\n\n=== VL 모델 추출 (첫 200자) ==="
head -c 200 outputs/vl_comparison_*/page_001/vl_model.txt
```

## 📦 Phase 5: 결과 회수

### 1. 결과 압축 (원격 서버)
```bash
# 결과 압축
cd /root/vl_experiment
tar -czf vl_results_$(date +%Y%m%d_%H%M%S).tar.gz outputs/vl_comparison_*

# 압축 파일 확인
ls -lh vl_results_*.tar.gz
```

### 2. 로컬로 전송 (로컬 터미널)
```bash
# 압축 파일 전송
scp -P 52283 root@47.186.63.142:/root/vl_experiment/vl_results_*.tar.gz experiments/

# 압축 해제
cd experiments
tar -xzf vl_results_*.tar.gz

# 결과 확인
ls -la outputs/vl_comparison_*/
```

### 3. HTML 리포트 열기 (로컬)
```bash
# HTML 리포트 열기 (Windows)
start outputs/vl_comparison_*/comparison_report.html

# 또는 브라우저에서 직접 열기
# 파일 경로: experiments/outputs/vl_comparison_YYYYMMDD_HHMMSS/comparison_report.html
```

## 🧹 Phase 6: 정리 (선택적)

### 원격 서버 정리
```bash
# SSH로 원격 서버 접속
ssh -p 52283 root@47.186.63.142

# 작업 파일 정리 (선택적)
rm -rf /root/vl_experiment/outputs/vl_comparison_*  # 결과만 삭제
# 또는
rm -rf /root/vl_experiment  # 전체 삭제
```

## 🐛 문제 해결

### 일반적인 문제들

**1. SSH 연결 실패**
```bash
# 연결 테스트
ssh -p 52283 -o ConnectTimeout=10 root@47.186.63.142 "echo 'Connected successfully'"
```

**2. 파일 전송 실패**
```bash
# 경로 확인
ssh -p 52283 root@47.186.63.142 "mkdir -p /root/vl_experiment/data"

# 파일 크기 확인
ls -lh "data/raw/금융분야 AI 보안 가이드라인.pdf"
```

**3. 환경 설정 실패**
```bash
# 인터넷 연결 확인
ssh -p 52283 root@47.186.63.142 "ping -c 3 google.com"

# 디스크 공간 확인
ssh -p 52283 root@47.186.63.142 "df -h"
```

**4. GPU 메모리 부족**
```bash
# 다른 프로세스 확인
nvidia-smi

# 페이지 수 줄이기
python3 vl_extraction_comparison.py --max-pages 3
```

**5. 모델 다운로드 느림**
```bash
# 캐시 디렉토리 확인
du -sh /root/.cache/huggingface/

# 네트워크 속도 확인
wget -O /dev/null http://speedtest.wdc01.softlayer.com/downloads/test10.zip
```

## 📊 성능 최적화 팁

### GPU 메모리 최적화
- **40GB+ GPU**: FP16 사용으로 최고 품질
- **20-40GB GPU**: 8-bit 양자화로 균형
- **8-20GB GPU**: 4-bit 양자화로 효율성
- **8GB 미만**: 페이지 수를 3-5개로 제한

### 처리 속도 향상
```bash
# 배치 처리 대신 순차 처리 (메모리 효율적)
# DPI 조정으로 이미지 품질/속도 조절
python3 vl_extraction_comparison.py --max-pages 10  # 기본 DPI 150
```

### 디스크 공간 관리
```bash
# 모델 캐시 확인
du -sh /root/.cache/huggingface/

# 이전 실험 결과 정리
find /root/vl_experiment/outputs -name "vl_comparison_*" -type d -mtime +1 -exec rm -rf {} \;
```

## 📈 예상 결과

### 성공적인 실험 시 출력:
```
✅ Experiment completed!
📁 Results saved to: /root/vl_experiment/outputs/vl_comparison_YYYYMMDD_HHMMSS
📊 Statistics:
   - Successful pages: 10/10
   - Average improvement: 45.3%
   - Total additional chars: 15,432
   - Average VL time/page: 8.2s
   - Total experiment time: 127.5s
```

### HTML 리포트 내용:
- 📄 페이지별 원본 이미지
- 🔤 PyMuPDF 추출 텍스트
- 🤖 VL 모델 추출 텍스트
- 📊 비교 통계 및 개선율
- 🎯 시각적 차이점 강조

이 가이드를 따라 실행하면 VL 모델의 이미지 정보 텍스트 복원 능력을 체계적으로 평가할 수 있습니다!