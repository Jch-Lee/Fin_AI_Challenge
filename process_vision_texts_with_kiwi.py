#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vision V2 추출 텍스트를 Kiwi로 정제하여 각 폴더에 저장
사용자 사전 적용하여 최적화된 텍스트 처리
"""

import sys
import io
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from kiwipiepy import Kiwi

def load_custom_dictionary() -> List[Dict]:
    """생성된 사용자 사전 로드"""
    dict_file = "kiwi_custom_dictionary.json"
    if os.path.exists(dict_file):
        with open(dict_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"사용자 사전 파일을 찾을 수 없습니다: {dict_file}")
        return []

def setup_kiwi_with_custom_dict() -> Kiwi:
    """사용자 사전이 적용된 Kiwi 인스턴스 생성"""
    kiwi = Kiwi()
    
    # 사용자 사전 로드
    custom_dict = load_custom_dictionary()
    
    if custom_dict:
        print(f"사용자 사전 {len(custom_dict)}개 항목 적용 중...")
        
        for entry in custom_dict:
            word = entry['word']
            pos = entry['pos']
            score = entry['score']
            
            try:
                kiwi.add_user_word(word, pos, score)
            except Exception as e:
                # 일부 품사 태그가 지원되지 않을 수 있음
                try:
                    kiwi.add_user_word(word, "NNP", score)
                except:
                    print(f"사전 추가 실패: {word}")
        
        print("사용자 사전 적용 완료!")
    else:
        print("기본 Kiwi 사전만 사용합니다.")
    
    return kiwi

def process_text_with_kiwi(kiwi: Kiwi, text: str) -> Dict:
    """Kiwi로 텍스트 처리"""
    if not text or not text.strip():
        return {
            "original_text": text,
            "processed_text": "",
            "spacing_corrected": "",
            "tokens": [],
            "morphemes": [],
            "nouns": [],
            "keywords": [],
            "stats": {
                "original_length": 0,
                "processed_length": 0,
                "token_count": 0,
                "noun_count": 0
            }
        }
    
    # 1. 띄어쓰기 교정
    try:
        spacing_corrected = kiwi.space(text, reset_whitespace=True)
    except:
        spacing_corrected = text
    
    # 2. 형태소 분석
    try:
        tokens = kiwi.tokenize(spacing_corrected)
        
        # 형태소와 품사 추출
        morphemes = [(token.form, token.tag) for token in tokens]
        
        # 명사만 추출
        nouns = [token.form for token in tokens if token.tag.startswith('N')]
        
        # 키워드 추출 (명사, 동사, 형용사, 외국어)
        keywords = []
        for token in tokens:
            if (token.tag.startswith('N') or      # 명사
                token.tag.startswith('V') or      # 동사
                token.tag == 'VA' or              # 형용사
                token.tag.startswith('SL')):      # 외국어
                
                if len(token.form) >= 2 or token.tag.startswith('SL'):
                    keywords.append(token.form)
        
    except Exception as e:
        print(f"형태소 분석 실패: {e}")
        tokens = []
        morphemes = []
        nouns = []
        keywords = []
    
    return {
        "original_text": text,
        "processed_text": spacing_corrected,
        "spacing_corrected": spacing_corrected,
        "tokens": [token.form for token in tokens] if tokens else [],
        "morphemes": morphemes,
        "nouns": nouns,
        "keywords": keywords,
        "stats": {
            "original_length": len(text),
            "processed_length": len(spacing_corrected),
            "token_count": len(tokens) if tokens else 0,
            "noun_count": len(nouns)
        }
    }

def process_single_page(kiwi: Kiwi, page_dir: Path, page_num: int) -> Dict:
    """단일 페이지 처리"""
    vl_v2_file = page_dir / "vl_model_v2.txt"
    
    if not vl_v2_file.exists():
        return {
            "page": page_num,
            "status": "file_not_found",
            "error": f"vl_model_v2.txt not found in {page_dir}"
        }
    
    # Vision V2 텍스트 읽기
    try:
        with open(vl_v2_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
    except Exception as e:
        return {
            "page": page_num,
            "status": "read_error",
            "error": str(e)
        }
    
    # Kiwi로 처리
    processed_result = process_text_with_kiwi(kiwi, original_text)
    
    # 결과 저장할 파일들
    files_to_save = [
        ("kiwi_processed.txt", processed_result["processed_text"]),
        ("kiwi_tokens.txt", " ".join(processed_result["tokens"])),
        ("kiwi_nouns.txt", " ".join(processed_result["nouns"])),
        ("kiwi_keywords.txt", " ".join(processed_result["keywords"])),
    ]
    
    # JSON 결과도 저장
    json_result = {
        "page": page_num,
        "processing_info": {
            "original_length": processed_result["stats"]["original_length"],
            "processed_length": processed_result["stats"]["processed_length"],
            "token_count": processed_result["stats"]["token_count"],
            "noun_count": processed_result["stats"]["noun_count"]
        },
        "morphemes": processed_result["morphemes"][:50],  # 샘플만 저장
        "sample_tokens": processed_result["tokens"][:30],
        "sample_nouns": processed_result["nouns"][:20],
        "sample_keywords": processed_result["keywords"][:25]
    }
    
    files_to_save.append(("kiwi_analysis.json", json.dumps(json_result, ensure_ascii=False, indent=2)))
    
    # 파일 저장
    saved_files = []
    failed_files = []
    
    for filename, content in files_to_save:
        try:
            file_path = page_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            saved_files.append(filename)
        except Exception as e:
            failed_files.append((filename, str(e)))
    
    return {
        "page": page_num,
        "status": "success",
        "saved_files": saved_files,
        "failed_files": failed_files,
        "stats": processed_result["stats"]
    }

def main():
    print("="*80)
    print("Vision V2 텍스트 Kiwi 정제 처리")
    print("="*80)
    
    # 기본 경로 설정
    base_path = Path("data/vision_extraction_benchmark/full_extraction_20250814_052410")
    
    if not base_path.exists():
        print(f"경로를 찾을 수 없습니다: {base_path}")
        return
    
    # 사용자 사전 적용된 Kiwi 설정
    print("\nKiwi 초기화 중...")
    kiwi = setup_kiwi_with_custom_dict()
    
    # 페이지 디렉토리 찾기
    page_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('page_')]
    page_dirs.sort()
    
    print(f"\n총 {len(page_dirs)}개 페이지 발견")
    
    # 전체 처리 결과
    all_results = []
    total_stats = {
        "processed_pages": 0,
        "failed_pages": 0,
        "total_tokens": 0,
        "total_nouns": 0
    }
    
    # 각 페이지 처리
    print("\n페이지별 처리 시작...")
    for page_dir in tqdm(page_dirs, desc="Processing pages"):
        page_num = int(page_dir.name.split('_')[1])
        
        result = process_single_page(kiwi, page_dir, page_num)
        all_results.append(result)
        
        if result["status"] == "success":
            total_stats["processed_pages"] += 1
            total_stats["total_tokens"] += result["stats"]["token_count"]
            total_stats["total_nouns"] += result["stats"]["noun_count"]
            
            print(f"  페이지 {page_num:03d}: ✅ 완료 "
                  f"(토큰: {result['stats']['token_count']}, "
                  f"명사: {result['stats']['noun_count']})")
        else:
            total_stats["failed_pages"] += 1
            print(f"  페이지 {page_num:03d}: ❌ 실패 - {result.get('error', 'Unknown error')}")
    
    # 전체 결과 저장
    summary_result = {
        "processing_date": "2025-01-15",
        "total_pages": len(page_dirs),
        "processed_pages": total_stats["processed_pages"],
        "failed_pages": total_stats["failed_pages"],
        "total_stats": total_stats,
        "page_results": all_results
    }
    
    summary_file = base_path / "kiwi_processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_result, f, ensure_ascii=False, indent=2)
    
    # 결과 출력
    print("\n" + "="*80)
    print("처리 결과 요약")
    print("="*80)
    print(f"총 페이지: {len(page_dirs)}")
    print(f"성공: {total_stats['processed_pages']}")
    print(f"실패: {total_stats['failed_pages']}")
    print(f"총 토큰 수: {total_stats['total_tokens']:,}")
    print(f"총 명사 수: {total_stats['total_nouns']:,}")
    
    if total_stats["processed_pages"] > 0:
        avg_tokens = total_stats["total_tokens"] / total_stats["processed_pages"]
        avg_nouns = total_stats["total_nouns"] / total_stats["processed_pages"]
        print(f"페이지당 평균 토큰: {avg_tokens:.1f}")
        print(f"페이지당 평균 명사: {avg_nouns:.1f}")
    
    print(f"\n요약 결과 저장: {summary_file}")
    
    # 생성된 파일 목록 출력
    print(f"\n각 페이지 폴더에 생성된 파일:")
    print("- kiwi_processed.txt: 띄어쓰기 교정된 텍스트")
    print("- kiwi_tokens.txt: 토큰화된 결과")
    print("- kiwi_nouns.txt: 추출된 명사")
    print("- kiwi_keywords.txt: 추출된 키워드")
    print("- kiwi_analysis.json: 상세 분석 결과")
    
    print("="*80)
    print("처리 완료!")

if __name__ == "__main__":
    main()