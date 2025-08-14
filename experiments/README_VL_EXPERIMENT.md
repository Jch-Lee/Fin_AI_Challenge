# 🔬 VL 모델 텍스트 추출 품질 검증 실험

이 실험은 **Qwen2.5-VL-7B** 모델을 사용하여 PyMuPDF가 읽지 못하는 이미지 정보까지 텍스트로 복원하는 성능을 검증합니다.

## 🎯 실험 목적

- **목표**: VL 모델이 PyMuPDF 대비 얼마나 많은 추가 정보를 추출하는지 측정
- **초점**: 이미지, 차트, 표, 다이어그램의 텍스트 복원 능력 평가
- **결과**: RAG 시스템 통합 여부 결정을 위한 정량적/정성적 데이터 수집

## 📁 실험 파일 구조

```
experiments/
├── 🚀 run_remote_experiment.bat           # 원클릭 실험 실행 스크립트
├── 🐍 vl_extraction_comparison.py         # 메인 실험 스크립트
├── 🔧 remote_setup.sh                     # 원격 서버 환경 설정
├── 📖 REMOTE_EXECUTION_GUIDE.md          # 상세 실행 가이드
├── 📊 RESULT_ANALYSIS_GUIDE.md           # 결과 분석 가이드
└── 📝 README_VL_EXPERIMENT.md            # 이 파일
```

## ⚡ 빠른 시작

### 방법 1: 자동 실행 (권장)
```cmd
# Windows에서 실행
experiments\run_remote_experiment.bat
```

### 방법 2: 수동 실행
1. **파일 전송**
   ```bash
   scp -P 52283 experiments/vl_extraction_comparison.py root@47.186.63.142:/root/vl_experiment/
   scp -P 52283 experiments/remote_setup.sh root@47.186.63.142:/root/vl_experiment/
   scp -P 52283 "data/raw/금융분야 AI 보안 가이드라인.pdf" root@47.186.63.142:/root/vl_experiment/data/
   ```

2. **원격 서버 접속 및 실행**
   ```bash
   ssh -p 52283 root@47.186.63.142
   cd /root/vl_experiment
   chmod +x remote_setup.sh && ./remote_setup.sh
   python3 vl_extraction_comparison.py --max-pages 10
   ```

3. **결과 다운로드**
   ```bash
   tar -czf vl_results.tar.gz outputs/vl_comparison_*
   scp -P 52283 root@47.186.63.142:/root/vl_experiment/vl_results.tar.gz experiments/
   tar -xzf vl_results.tar.gz
   ```

## 🔧 실험 설정

### 기본 설정
- **처리 페이지**: 10페이지 (조정 가능)
- **이미지 해상도**: 150 DPI
- **VL 모델**: Qwen2.5-VL-7B-Instruct
- **양자화**: GPU 메모리에 따라 자동 선택

### 커스텀 실행
```bash
# 페이지 수 조정
python3 vl_extraction_comparison.py --max-pages 5

# 다른 PDF 파일 사용
python3 vl_extraction_comparison.py --pdf "다른문서.pdf" --max-pages 8
```

## 📊 예상 결과

### 성능 지표
- **평균 개선율**: 45-60% (기존 실험 기준)
- **처리 시간**: 5-15초/페이지 (GPU에 따라)
- **추가 정보**: 15,000+ 추가 문자 (10페이지 기준)

### 생성 파일
```
outputs/vl_comparison_YYYYMMDD_HHMMSS/
├── 📄 comparison_report.html      # 시각적 비교 리포트
├── 📊 summary.json               # 실험 요약 통계
├── 📝 experiment.log             # 실행 로그
└── 📁 page_XXX/                  # 페이지별 결과
    ├── original.png              # 원본 이미지
    ├── pymupdf.txt              # PyMuPDF 추출
    └── vl_model.txt             # VL 모델 추출
```

## 🎯 평가 기준

### 정량적 평가
- **텍스트 증가율**: VL 모델이 추출한 추가 문자 수/비율
- **처리 성능**: 페이지당 처리 시간 및 GPU 활용도
- **안정성**: 성공/실패 페이지 비율

### 정성적 평가 (HTML 리포트)
- **이미지 설명 품질**: 차트/그래프 정보의 텍스트 변환 정확도
- **구조 보존**: 표와 다이어그램의 구조 정보 유지
- **텍스트 완전성**: 원본 텍스트의 완전한 보존 여부

## 🔍 결과 분석

### 1. HTML 리포트 확인
```bash
start outputs/vl_comparison_*/comparison_report.html
```

### 2. 요약 통계 확인
```python
import json
with open('outputs/vl_comparison_*/summary.json', 'r') as f:
    data = json.load(f)
    stats = data['statistics']
    print(f"평균 개선율: {stats['average_improvement_rate']:.1f}%")
    print(f"총 추가 문자: {stats['total_improvement']:,}")
```

### 3. 상세 분석
`RESULT_ANALYSIS_GUIDE.md` 참조하여 심화 분석 수행

## ⚠️ 주의사항

### 시스템 요구사항
- **GPU 메모리**: 8GB+ 권장 (4GB도 가능하지만 느림)
- **디스크 공간**: 20GB+ 여유 공간
- **네트워크**: 모델 다운로드를 위한 안정적 인터넷

### 일반적인 문제
1. **GPU 메모리 부족**: `--max-pages` 줄여서 재실행
2. **네트워크 타임아웃**: 모델 다운로드 재시도
3. **SSH 연결 실패**: 서버 상태 및 네트워크 확인

## 📋 체크리스트

### 실험 전
- [ ] 원격 서버 SSH 접속 확인
- [ ] PDF 파일 준비 완료
- [ ] 로컬 실험 파일 확인

### 실험 중
- [ ] GPU 메모리 모니터링
- [ ] 진행 상황 로그 확인
- [ ] 디스크 공간 모니터링

### 실험 후
- [ ] HTML 리포트 확인
- [ ] 요약 통계 검토
- [ ] 결과 파일 백업

## 🎉 성공 시나리오

실험이 성공적으로 완료되면:

1. **브라우저에서 HTML 리포트 자동 열림**
2. **평균 30-50% 텍스트 증가 확인**
3. **이미지 정보의 상세한 텍스트 변환 확인**
4. **RAG 시스템 통합 여부 결정 가능**

## 🔗 관련 문서

- **상세 실행 가이드**: [REMOTE_EXECUTION_GUIDE.md](REMOTE_EXECUTION_GUIDE.md)
- **결과 분석 가이드**: [RESULT_ANALYSIS_GUIDE.md](RESULT_ANALYSIS_GUIDE.md)
- **기존 실험 결과**: [enhanced_vl_extraction_report.md](enhanced_vl_extraction_report.md)

## 💡 다음 단계

실험 완료 후:
1. 결과 분석 및 품질 평가
2. RAG 시스템 통합 계획 수립
3. 하이브리드 접근법 설계 (PyMuPDF + VL 선택적 사용)
4. 실제 질의응답 성능 테스트

---

**문의사항**: 실험 진행 중 문제가 발생하면 `REMOTE_EXECUTION_GUIDE.md`의 문제 해결 섹션을 참조하세요.