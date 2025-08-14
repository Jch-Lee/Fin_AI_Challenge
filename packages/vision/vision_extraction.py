"""
Vision-Language 기반 텍스트 추출 모듈
Qwen2.5-VL-7B-Instruct 모델을 사용한 PDF 이미지 텍스트 추출
"""

import os
import io
import torch
import logging
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
        """Version 2 의미론적 추출 프롬프트 (41.2% 개선 검증됨)"""
        return """이 문서 페이지에서 다음 우선순위에 따라 정보를 추출해주세요:

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

위 우선순위에 따라 실제 정보와 데이터만 추출하여 의미있는 텍스트로 정리해주세요."""
    
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
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """이미지에서 VL 모델을 사용하여 텍스트 추출"""
        self._ensure_initialized()
        
        try:
            prompt = self._get_extraction_prompt()
            
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
                        return decoded_outputs[0].strip()
            
            return "[텍스트 추출 실패]"
            
        except Exception as e:
            logger.error(f"Vision text extraction failed: {e}")
            return f"[VL 모델 추출 실패: {str(e)}]"
    
    def extract_pdf_page(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """PDF의 특정 페이지에서 텍스트 추출"""
        try:
            # PDF 페이지를 이미지로 변환
            image = self.pdf_page_to_image(pdf_path, page_num)
            
            # VL 모델로 텍스트 추출
            extracted_text = self.extract_text_from_image(image)
            
            return {
                "status": "success",
                "page_num": page_num + 1,
                "text": extracted_text,
                "char_count": len(extracted_text),
                "extraction_method": "vision_v2"
            }
            
        except Exception as e:
            logger.error(f"Failed to extract text from page {page_num + 1}: {e}")
            return {
                "status": "error",
                "page_num": page_num + 1,
                "text": f"[페이지 {page_num + 1} 추출 실패: {str(e)}]",
                "char_count": 0,
                "extraction_method": "vision_v2",
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