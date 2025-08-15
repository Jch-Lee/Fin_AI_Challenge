"""
Vision-Language 기반 텍스트 추출 모듈
Qwen2.5-VL-7B-Instruct 모델을 사용한 PDF 이미지 텍스트 추출
"""

import os
import io
import torch
import logging
import re
from PIL import Image
from typing import Optional, Dict, Any
import pymupdf

logger = logging.getLogger(__name__)


class VisionTextExtractor:
    """Vision-Language 모델 기반 텍스트 추출기"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Args:
            device: 사용할 디바이스 ('cuda', 'cpu', 또는 None for auto)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._initialized = False
        
        # 추가: 페이지 간 컨텍스트 관리
        self.last_page_info = {
            'last_header': None,
            'last_sentence': None,
            'incomplete': False
        }
        
        logger.info(f"VisionTextExtractor initialized with device: {self.device}")
    
    def _ensure_initialized(self):
        """모델이 초기화되지 않았으면 초기화"""
        if not self._initialized:
            self._initialize_model()
    
    def _initialize_model(self):
        """Vision 모델 초기화"""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            
            logger.info("Loading Qwen2.5-VL-7B-Instruct model...")
            
            # 모델 로드
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.cpu()
            
            self._initialized = True
            logger.info("Vision model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}")
            raise
    
    def _get_extraction_prompt(self) -> str:
        """Version 2.5 - 기존 의미론적 추출 + 마크다운 구조화 (41.2% 개선 + 청킹 최적화)"""
        return """<!--PROMPT_START_MARKER_DO_NOT_INCLUDE-->
이 문서 페이지에서 다음 우선순위에 따라 정보를 추출해주세요:

**1순위: 핵심 텍스트 정보**
- 제목, 소제목, 본문 내용을 정확히 추출
- 번호나 기호가 있는 목록 구조 유지
- 각주, 참조 번호 포함

**2순위: 표와 데이터**
- 표 제목과 헤더 정보
- 데이터 값과 단위 (숫자, 퍼센트, 금액 등)
- 행과 열의 관계성 유지

**3순위: 차트와 그래프 해석**
- 차트 제목과 축 레이블
- 주요 데이터 포인트와 추세
- 범례 정보
- 그래프에서 읽을 수 있는 구체적인 수치

**4순위: 구조적 요소**
- 섹션 구분
- 박스나 하이라이트된 내용
- 도표나 다이어그램의 텍스트와 연결관계

**제외사항:**
- 색상, 폰트, 위치 등 시각적 스타일 묘사
- 페이지 레이아웃이나 디자인 설명
- 장식적 요소나 배경

위 우선순위에 따라 실제 정보와 데이터만 추출하여 의미있는 텍스트로 정리해주세요.

---

## 📋 추가 지침: 마크다운 형식화

추출한 내용을 다음 마크다운 규칙에 따라 구조화해주세요:

### 헤더 레벨 지정
- 문서의 주요 제목이나 장(Chapter) 수준: # 사용
- 절(Section) 수준의 중간 제목: ## 사용
- 항(Subsection) 수준의 소제목: ### 사용
- 세부 항목이나 부제목: #### 사용

### 문단과 구조
- 각 문단 사이에 빈 줄 하나 삽입
- 번호 목록: `1.`, `2.`, `3.` 형식 사용
- 글머리 목록: `-` 또는 `*` 사용
- 중첩된 리스트는 들여쓰기(2칸 또는 4칸)로 표현

### 표 형식
표를 발견하면 다음 형식으로 변환:
```
| 헤더1 | 헤더2 | 헤더3 |
|-------|-------|-------|
| 데이터1 | 데이터2 | 데이터3 |
```

### 페이지 경계 표시
- 문장이나 단락이 페이지 끝에서 완료되지 않은 경우: `... [다음 페이지에 계속]`
- 페이지 시작이 이전 내용의 연속인 경우: `[이전 페이지에서 계속] ...`
- 표나 리스트가 페이지를 걸쳐 계속되는 경우도 동일하게 표시

### 특수 콘텐츠
- 코드나 명령어: ` ` (백틱) 사용
- 강조: **굵게** 또는 *기울임*
- 인용문: > 사용

이러한 마크다운 형식을 적용하여 후속 처리(청킹)가 용이하도록 구조화된 텍스트를 생성해주세요.

**중요**: 위의 지침은 참고용이며, 실제 출력에는 이 문서 페이지의 내용만 포함하세요. 만약 추출할 내용이 없다면 "내용 없음"이라고만 출력하세요.
<!--PROMPT_END_MARKER_DO_NOT_INCLUDE-->"""
    
    def _get_contextual_prompt(self, page_num: int = 0) -> str:
        """페이지 컨텍스트를 포함한 프롬프트 생성"""
        
        base_prompt = self._get_extraction_prompt()  # 기본 프롬프트
        
        # 첫 페이지가 아니고 이전 페이지가 불완전한 경우
        if page_num > 0 and self.last_page_info['incomplete']:
            context_hint = f"""

### 📌 이전 페이지 컨텍스트
- 마지막 헤더: {self.last_page_info.get('last_header', '없음')}
- 마지막 문장 일부: ...{self.last_page_info.get('last_sentence', '')[-50:]}

이 페이지가 이전 내용의 연속이라면 [이전 페이지에서 계속] 표시를 추가해주세요.
"""
            return base_prompt + context_hint
        
        return base_prompt
    
    def _update_page_context(self, extracted_text: str):
        """페이지 컨텍스트 업데이트"""
        
        lines = extracted_text.strip().split('\n')
        
        # 마지막 헤더 찾기
        headers = [line for line in lines if re.match(r'^#{1,4}\s', line)]
        if headers:
            self.last_page_info['last_header'] = headers[-1]
        
        # 마지막 문장 저장
        if lines:
            last_line = lines[-1].strip()
            self.last_page_info['last_sentence'] = last_line
            
            # 불완전 여부 판단
            self.last_page_info['incomplete'] = (
                '[다음 페이지에 계속]' in last_line or
                not last_line.endswith(('.', '!', '?', '다.', '요.', '니다.'))
            )
    
    def _filter_prompt_leakage(self, text: str) -> str:
        """프롬프트 누출 제거 및 빈 내용 처리"""
        
        # 1단계: 프롬프트 마커 사이의 내용 제거
        start_marker = "<!--PROMPT_START_MARKER_DO_NOT_INCLUDE-->"
        end_marker = "<!--PROMPT_END_MARKER_DO_NOT_INCLUDE-->"
        
        # 마커가 포함된 경우 제거
        if start_marker in text:
            if end_marker in text:
                # 시작과 끝 마커 사이 모든 내용 제거
                parts = text.split(start_marker)
                if len(parts) > 1:
                    remaining = parts[1].split(end_marker)
                    if len(remaining) > 1:
                        text = parts[0] + remaining[1]
                    else:
                        text = parts[0]
            else:
                # 시작 마커 이후 모든 내용 제거
                text = text.split(start_marker)[0]
        
        # 2단계: 프롬프트 키워드 패턴 제거
        prompt_patterns = [
            r'이 문서 페이지에서.*?추출해주세요[.:]*',
            r'\*\*\d+순위:.*?\*\*',
            r'위 우선순위에 따라.*?정리해주세요[.:]*',
            r'추가 지침:.*?마크다운 형식화',
            r'헤더 레벨 지정',
            r'문단과 구조',
            r'표 형식',
            r'페이지 경계 표시',
            r'특수 콘텐츠',
            r'이러한 마크다운 형식을.*?생성해주세요[.:]*',
            r'\*\*중요\*\*:.*?출력하세요[.:]*'
        ]
        
        for pattern in prompt_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 3단계: 불완전한 마커나 지침 제거
        instruction_patterns = [
            r'###\s*(헤더|문단|표|페이지|특수).*?\n',
            r'-\s*[가-힣\s]*:\s*[#`\*].*?\n',
            r'\|\s*헤더\d+\s*\|.*?\n',
            r'```\s*\n.*?\n```',
            r'위의?\s*지침',
            r'다음\s*규칙',
            r'형식으로\s*변환'
        ]
        
        for pattern in instruction_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 4단계: 정리 및 검증
        text = text.strip()
        
        # 빈 내용이거나 의미없는 텍스트인 경우
        if not text or len(text) < 10:
            return "내용 없음"
        
        # 여전히 프롬프트 키워드가 많이 남아있는 경우
        prompt_keywords = ['우선순위', '추출', '지침', '형식', '변환', '생성해주세요']
        keyword_count = sum(1 for keyword in prompt_keywords if keyword in text)
        
        if keyword_count >= 3 and len(text) < 200:
            return "내용 없음"
        
        # 5단계: 최종 정리
        # 연속된 빈 줄 제거
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 각 줄 앞뒤 공백 제거
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        return text.strip() if text.strip() else "내용 없음"
    
    def pdf_page_to_image(self, pdf_path: str, page_num: int, dpi: int = 150) -> Image.Image:
        """PDF 페이지를 이미지로 변환"""
        try:
            doc = pymupdf.open(pdf_path)
            page = doc[page_num]
            pix = page.get_pixmap(dpi=dpi)
            img_data = pix.tobytes("png")
            doc.close()
            return Image.open(io.BytesIO(img_data))
        except Exception as e:
            logger.error(f"Failed to convert PDF page {page_num + 1} to image: {e}")
            raise
    
    def extract_text_from_image(self, image: Image.Image, page_num: int = 0) -> str:
        """이미지에서 VL 모델을 사용하여 텍스트 추출"""
        self._ensure_initialized()
        
        try:
            # 컨텍스트가 포함된 프롬프트 사용
            prompt = self._get_contextual_prompt(page_num)
            
            # qwen_vl_utils 사용 시도
            try:
                from qwen_vl_utils import process_vision_info
                use_utils = True
            except ImportError:
                process_vision_info = None
                use_utils = False
                logger.debug("qwen_vl_utils not available, using fallback")
            
            # 메시지 구성
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # 프롬프트 템플릿 적용
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 이미지 처리
            if use_utils and process_vision_info:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs = [image]
                video_inputs = []
            
            # 입력 처리
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt"
            )
            
            # 디바이스로 이동
            if self.device == "cuda":
                inputs = {k: v.to("cuda") if hasattr(v, 'to') else v 
                         for k, v in inputs.items()}
            
            # 생성 설정
            generation_config = {
                "max_new_tokens": 2048,
                "temperature": 0.3,  # 일관성을 위해 낮은 temperature
                "do_sample": True,
                "top_p": 0.9,
                "pad_token_id": self.processor.tokenizer.eos_token_id
            }
            
            # 텍스트 생성
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
            # 입력 길이 계산 및 출력 디코딩
            input_ids_tensor = inputs.get('input_ids')
            if input_ids_tensor is not None and len(outputs) > 0:
                input_len = input_ids_tensor.shape[1]
                generated_ids = []
                
                for output_ids in outputs:
                    if len(output_ids) > input_len:
                        generated_ids.append(output_ids[input_len:])
                    else:
                        generated_ids.append(output_ids)
                
                if generated_ids:
                    decoded_outputs = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    
                    if decoded_outputs:
                        result_text = decoded_outputs[0].strip()
                        # 프롬프트 누출 필터링
                        filtered_text = self._filter_prompt_leakage(result_text)
                        # 추출 결과로 컨텍스트 업데이트
                        self._update_page_context(filtered_text)
                        return filtered_text
            
            return "[텍스트 추출 실패]"
            
        except Exception as e:
            logger.error(f"Vision text extraction failed: {e}")
            return f"[VL 모델 추출 실패: {str(e)}]"
    
    def extract_pdf_page(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """PDF의 특정 페이지에서 텍스트 추출"""
        try:
            # PDF 페이지를 이미지로 변환
            image = self.pdf_page_to_image(pdf_path, page_num)
            
            # VL 모델로 텍스트 추출 (페이지 번호 전달)
            extracted_text = self.extract_text_from_image(image, page_num)
            
            # 마크다운 구조 검증
            structure_quality = self._validate_markdown_structure(extracted_text)
            
            return {
                "status": "success",
                "page_num": page_num + 1,
                "text": extracted_text,
                "char_count": len(extracted_text),
                "extraction_method": "vision_v2.5",
                "markdown_quality": structure_quality
            }
            
        except Exception as e:
            logger.error(f"Failed to extract text from page {page_num + 1}: {e}")
            return {
                "status": "error",
                "page_num": page_num + 1,
                "text": f"[페이지 {page_num + 1} 추출 실패: {str(e)}]",
                "char_count": 0,
                "extraction_method": "vision_v2.5",
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """Vision 추출이 사용 가능한지 확인"""
        try:
            # GPU 체크
            if not torch.cuda.is_available():
                logger.info("CUDA not available for vision extraction")
                return False
            
            # 모델 import 체크
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            return True
            
        except ImportError as e:
            logger.warning(f"Vision model dependencies not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Vision extraction availability check failed: {e}")
            return False
    
    def cleanup(self):
        """메모리 정리"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._initialized = False
        logger.info("Vision model cleaned up")
    
    def _validate_markdown_structure(self, text: str) -> Dict[str, Any]:
        """마크다운 구조 품질 검증"""
        
        return {
            'has_headers': bool(re.search(r'^#{1,4}\s', text, re.MULTILINE)),
            'has_lists': bool(re.search(r'^[\-\*\d]+\.?\s', text, re.MULTILINE)),
            'has_tables': '|' in text and '---' in text,
            'has_continuity_markers': '[계속]' in text or '[이어서]' in text,
            'header_count': len(re.findall(r'^#{1,4}\s', text, re.MULTILINE)),
            'paragraph_count': len([line for line in text.split('\n\n') if line.strip()]),
            'avg_paragraph_length': sum(len(p) for p in text.split('\n\n') if p.strip()) / max(len([p for p in text.split('\n\n') if p.strip()]), 1),
            'structure_score': self._calculate_structure_score(text)
        }
    
    def _calculate_structure_score(self, text: str) -> float:
        """구조화 점수 계산 (0.0 ~ 1.0)"""
        
        score = 0.0
        
        # 헤더 존재 (0.3)
        if re.search(r'^#{1,4}\s', text, re.MULTILINE):
            score += 0.3
        
        # 리스트 존재 (0.2)
        if re.search(r'^[\-\*\d]+\.?\s', text, re.MULTILINE):
            score += 0.2
        
        # 적절한 문단 구분 (0.2)
        if '\n\n' in text:
            score += 0.2
        
        # 페이지 연속성 마커 (0.1)
        if '[계속]' in text or '[이어서]' in text:
            score += 0.1
        
        # 표 형식 (0.1)
        if '|' in text and '---' in text:
            score += 0.1
        
        # 텍스트 길이 보정 (0.1)
        if len(text.strip()) > 100:
            score += 0.1
        
        return min(score, 1.0)