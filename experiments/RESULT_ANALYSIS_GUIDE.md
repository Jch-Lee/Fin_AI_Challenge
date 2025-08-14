# 📊 VL 모델 실험 결과 분석 가이드

이 가이드는 원격 서버에서 실행한 VL 모델 텍스트 추출 실험의 결과를 분석하고 평가하는 방법을 설명합니다.

## 📁 결과 파일 구조

실험 완료 후 다음과 같은 파일 구조가 생성됩니다:

```
experiments/outputs/vl_comparison_YYYYMMDD_HHMMSS/
├── 📄 comparison_report.html          # 시각적 비교 HTML 리포트
├── 📊 summary.json                    # 실험 요약 통계
├── 📝 experiment.log                  # 실행 로그
├── 📁 page_001/
│   ├── 🖼️ original.png              # 원본 페이지 이미지
│   ├── 📝 pymupdf.txt               # PyMuPDF 추출 텍스트
│   └── 🤖 vl_model.txt              # VL 모델 추출 텍스트
├── 📁 page_002/
│   └── ...
└── 📁 page_XXX/
    └── ...
```

## 🎯 1. 핵심 성과 지표 확인

### 요약 통계 확인
```bash
# JSON 요약에서 핵심 지표 추출
python -c "
import json
with open('experiments/outputs/vl_comparison_*/summary.json', 'r') as f:
    data = json.load(f)
    stats = data['statistics']
    
    print('📊 VL 모델 성능 요약')
    print('=' * 40)
    print(f'평균 개선율: {stats[\"average_improvement_rate\"]:.1f}%')
    print(f'총 추가 문자 수: {stats[\"total_improvement\"]:,}')
    print(f'처리된 페이지: {data[\"successful_pages\"]}/{data[\"total_pages_processed\"]}')
    print(f'평균 VL 처리시간: {stats[\"average_vl_time_per_page\"]:.1f}초/페이지')
    print(f'총 실험 시간: {stats[\"total_experiment_time\"]:.0f}초')
    print(f'GPU: {data[\"gpu_info\"]} ({data[\"quantization\"]})')
"
```

### 개선 효과 분석
```python
# 페이지별 개선율 분석 스크립트
import json
import matplotlib.pyplot as plt

# summary.json 로드
with open('experiments/outputs/vl_comparison_*/summary.json', 'r') as f:
    data = json.load(f)

# 페이지별 개선율 추출
pages = []
improvements = []
for result in data['page_results']:
    if result['success']:
        pages.append(result['page'])
        improvements.append(result['improvement_rate'])

# 개선율 분포 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(pages, improvements)
plt.title('페이지별 텍스트 추출 개선율')
plt.xlabel('페이지 번호')
plt.ylabel('개선율 (%)')

plt.subplot(1, 2, 2)
plt.hist(improvements, bins=10, alpha=0.7)
plt.title('개선율 분포')
plt.xlabel('개선율 (%)')
plt.ylabel('페이지 수')

plt.tight_layout()
plt.savefig('experiments/improvement_analysis.png', dpi=300, bbox_inches='tight')
print("📊 개선율 분석 그래프 저장: experiments/improvement_analysis.png")
```

## 👁️ 2. 시각적 품질 평가

### HTML 리포트 분석 체크리스트
웹브라우저에서 `comparison_report.html`을 열고 다음 항목들을 확인하세요:

#### ✅ 텍스트 보존 품질
- [ ] **원본 텍스트 완전성**: VL 모델이 PyMuPDF와 동일한 텍스트를 추출했는가?
- [ ] **텍스트 순서**: 문단과 섹션의 순서가 올바르게 유지되었는가?
- [ ] **특수 문자**: 수식, 기호, 한글 텍스트가 정확히 추출되었는가?
- [ ] **표 구조**: 테이블의 행/열 구조가 텍스트로 적절히 변환되었는가?

#### 🔍 이미지 정보 복원
- [ ] **차트 데이터**: 그래프의 수치와 축 정보가 텍스트로 추출되었는가?
- [ ] **다이어그램 설명**: 플로차트나 구조도의 연결관계가 설명되었는가?
- [ ] **이미지 내 텍스트**: 이미지 안의 텍스트가 OCR로 추출되었는가?
- [ ] **범례 정보**: 차트의 범례와 라벨이 포함되었는가?

#### 📊 품질 평가 기준
| 개선율 | 평가 | 설명 |
|--------|------|------|
| 50%+ | 🟢 우수 | 상당한 추가 정보 추출 |
| 20-50% | 🟡 양호 | 적절한 개선 효과 |
| 10-20% | 🟠 보통 | 제한적 개선 |
| <10% | 🔴 미흡 | 효과 미미 |

### 구체적 페이지 분석 예시

#### 페이지 1 분석
```bash
# 페이지 1 결과 확인
echo "=== 페이지 1 분석 ==="
echo "PyMuPDF 문자 수: $(wc -c < experiments/outputs/vl_comparison_*/page_001/pymupdf.txt)"
echo "VL 모델 문자 수: $(wc -c < experiments/outputs/vl_comparison_*/page_001/vl_model.txt)"

# 추가된 정보 중 주요 키워드 확인
echo -e "\n=== VL 모델이 추가로 추출한 정보 키워드 ==="
diff <(cat experiments/outputs/vl_comparison_*/page_001/pymupdf.txt | tr ' ' '\n' | sort | uniq) \
     <(cat experiments/outputs/vl_comparison_*/page_001/vl_model.txt | tr ' ' '\n' | sort | uniq) \
     | grep "^>" | head -20
```

## 📈 3. 정량적 성능 분석

### 처리 성능 평가
```python
# 성능 분석 스크립트
import json
import numpy as np

with open('experiments/outputs/vl_comparison_*/summary.json', 'r') as f:
    data = json.load(f)

# 성능 지표 계산
page_results = [r for r in data['page_results'] if r['success']]
vl_times = [r['vl_process_time'] for r in page_results]
improvements = [r['improvement'] for r in page_results]
improvement_rates = [r['improvement_rate'] for r in page_results]

print("⚡ 성능 분석 결과")
print("=" * 40)
print(f"VL 처리시간 - 평균: {np.mean(vl_times):.1f}초, 최대: {np.max(vl_times):.1f}초")
print(f"추가 문자 수 - 평균: {np.mean(improvements):.0f}, 최대: {np.max(improvements)}")
print(f"개선율 - 평균: {np.mean(improvement_rates):.1f}%, 표준편차: {np.std(improvement_rates):.1f}%")

# GPU 활용도 (사용 가능한 경우)
if 'gpu_info' in data:
    print(f"GPU: {data['gpu_info']}")
    print(f"양자화: {data['quantization']}")
```

### 비용 효율성 분석
```python
# 시간당 처리량 및 효율성 계산
total_pages = data['successful_pages']
total_time = data['statistics']['total_experiment_time']
total_improvement = data['statistics']['total_improvement']

pages_per_hour = (total_pages / total_time) * 3600
chars_per_second = total_improvement / total_time

print("\n💰 효율성 분석")
print("=" * 40)
print(f"시간당 처리 가능 페이지: {pages_per_hour:.1f}페이지/시간")
print(f"초당 추가 정보 추출: {chars_per_second:.1f}문자/초")
print(f"페이지당 평균 추가 정보: {total_improvement/total_pages:.0f}문자/페이지")
```

## 🎯 4. 구체적 개선 사례 분석

### 이미지 정보 복원 성공 사례 찾기
```bash
# VL 모델에서 "차트", "그래프", "표" 키워드가 추가된 페이지 찾기
echo "📊 시각적 요소 추출 성공 사례:"
for page_dir in experiments/outputs/vl_comparison_*/page_*/; do
    page_num=$(basename "$page_dir" | sed 's/page_0*//')
    
    # VL 모델에서만 발견되는 시각적 키워드 확인
    vl_visual=$(grep -i -c "차트\|그래프\|표\|도표\|이미지\|그림" "$page_dir/vl_model.txt" 2>/dev/null || echo 0)
    pymupdf_visual=$(grep -i -c "차트\|그래프\|표\|도표\|이미지\|그림" "$page_dir/pymupdf.txt" 2>/dev/null || echo 0)
    
    if [ $vl_visual -gt $pymupdf_visual ]; then
        echo "  페이지 $page_num: VL 모델이 $((vl_visual - pymupdf_visual))개 추가 시각적 요소 인식"
    fi
done
```

### 수치 데이터 추출 비교
```bash
# 숫자 패턴 추출 비교
echo -e "\n🔢 수치 데이터 추출 비교:"
for page_dir in experiments/outputs/vl_comparison_*/page_*/; do
    page_num=$(basename "$page_dir" | sed 's/page_0*//')
    
    # 숫자 패턴 카운트 (소수점, 퍼센트, 콤마 포함)
    pymupdf_numbers=$(grep -o -E '[0-9]+(\.[0-9]+)?%?|[0-9]{1,3}(,[0-9]{3})*(\.[0-9]+)?' "$page_dir/pymupdf.txt" 2>/dev/null | wc -l)
    vl_numbers=$(grep -o -E '[0-9]+(\.[0-9]+)?%?|[0-9]{1,3}(,[0-9]{3})*(\.[0-9]+)?' "$page_dir/vl_model.txt" 2>/dev/null | wc -l)
    
    if [ $vl_numbers -gt $pymupdf_numbers ]; then
        diff_numbers=$((vl_numbers - pymupdf_numbers))
        echo "  페이지 $page_num: VL 모델이 $diff_numbers개 추가 수치 데이터 추출"
    fi
done
```

## 📝 5. 결과 보고서 생성

### 자동 요약 보고서 생성
```python
# 종합 분석 보고서 생성
import json
from datetime import datetime

with open('experiments/outputs/vl_comparison_*/summary.json', 'r') as f:
    data = json.load(f)

report = f"""
# VL 모델 텍스트 추출 실험 결과 보고서

**실험 일시**: {data['timestamp']}
**문서**: {data['pdf_name']}
**GPU**: {data['gpu_info']} ({data['quantization']})

## 핵심 성과

### 정량적 성과
- **평균 텍스트 증가율**: {data['statistics']['average_improvement_rate']:.1f}%
- **총 추가 문자 수**: {data['statistics']['total_improvement']:,}문자
- **성공 처리 페이지**: {data['successful_pages']}/{data['total_pages_processed']}페이지

### 처리 성능
- **평균 처리 시간**: {data['statistics']['average_vl_time_per_page']:.1f}초/페이지
- **총 실험 시간**: {data['statistics']['total_experiment_time']:.0f}초
- **시간당 처리량**: {(data['successful_pages'] / data['statistics']['total_experiment_time'] * 3600):.1f}페이지/시간

## 개선 효과 분석

### 텍스트 추출 성능
- **PyMuPDF 총 문자**: {data['statistics']['total_pymupdf_chars']:,}
- **VL 모델 총 문자**: {data['statistics']['total_vl_chars']:,}
- **정보 증가율**: {((data['statistics']['total_vl_chars'] - data['statistics']['total_pymupdf_chars']) / data['statistics']['total_pymupdf_chars'] * 100):.1f}%

### 페이지별 성과 분포
"""

# 페이지별 성과 분석
successful_results = [r for r in data['page_results'] if r['success']]
improvement_rates = [r['improvement_rate'] for r in successful_results]

high_improvement = len([r for r in improvement_rates if r >= 50])
medium_improvement = len([r for r in improvement_rates if 20 <= r < 50])
low_improvement = len([r for r in improvement_rates if r < 20])

report += f"""
- **높은 개선 (50%+)**: {high_improvement}페이지
- **중간 개선 (20-50%)**: {medium_improvement}페이지  
- **낮은 개선 (<20%)**: {low_improvement}페이지

## 결론 및 권장사항

### 성과 평가
"""

avg_improvement = data['statistics']['average_improvement_rate']
if avg_improvement >= 40:
    report += "🟢 **우수**: VL 모델이 상당한 추가 정보를 추출하여 RAG 시스템 성능 향상에 기여할 것으로 예상"
elif avg_improvement >= 20:
    report += "🟡 **양호**: VL 모델이 적절한 개선 효과를 보이며, 특정 페이지에서 유의미한 성과"
else:
    report += "🟠 **보통**: VL 모델의 개선 효과가 제한적이며, 추가 최적화 필요"

report += f"""

### 통합 권장사항
- **메모리 사용**: {data['quantization']} 설정으로 안정적 실행 확인
- **처리 시간**: 페이지당 평균 {data['statistics']['average_vl_time_per_page']:.1f}초로 실용적 수준
- **품질 향상**: 평균 {avg_improvement:.1f}% 정보 증가로 RAG 품질 개선 기대

### 다음 단계
1. 시각적 요소가 많은 페이지에서 VL 모델 활용 권장
2. 하이브리드 접근: PyMuPDF + VL 모델 선택적 적용
3. 실제 RAG 시스템에 통합하여 검색 성능 테스트 수행

**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# 보고서 저장
with open('experiments/VL_EXPERIMENT_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("📄 종합 보고서 생성 완료: experiments/VL_EXPERIMENT_REPORT.md")
```

## 🔄 6. 추가 분석 및 개선

### 실패 페이지 분석 (있는 경우)
```bash
# 실패한 페이지 확인
python -c "
import json
with open('experiments/outputs/vl_comparison_*/summary.json', 'r') as f:
    data = json.load(f)

failed_pages = [r for r in data['page_results'] if not r['success']]
if failed_pages:
    print('❌ 실패한 페이지:')
    for page in failed_pages:
        print(f'  페이지 {page[\"page\"]}: {page.get(\"error\", \"Unknown error\")}')
else:
    print('✅ 모든 페이지 처리 성공')
"
```

### 최적화 제안
```python
# 성능 최적화 분석
import json

with open('experiments/outputs/vl_comparison_*/summary.json', 'r') as f:
    data = json.load(f)

avg_time = data['statistics']['average_vl_time_per_page']
avg_improvement = data['statistics']['average_improvement_rate']

print("🚀 최적화 제안:")

if avg_time > 10:
    print("- 처리 시간 최적화: 더 높은 양자화 또는 DPI 조정 고려")

if avg_improvement < 20:
    print("- 프롬프트 최적화: 도메인 특화 프롬프트 개발")
    
if data['failed_pages'] > 0:
    print("- 안정성 개선: 에러 처리 및 재시도 로직 강화")

print("- 하이브리드 전략: 이미지 밀도에 따른 선택적 VL 모델 적용")
```

이 분석 가이드를 통해 VL 모델 실험 결과를 체계적으로 평가하고, RAG 시스템 통합 여부를 결정할 수 있습니다!