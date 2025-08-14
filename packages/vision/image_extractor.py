"""
PDF Image Extractor

PDF 문서에서 이미지를 추출하고 전처리하는 모듈
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from io import BytesIO
import json

logger = logging.getLogger(__name__)


class PDFImageExtractor:
    """PDF에서 이미지를 추출하는 클래스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 추출 설정
                - min_width: 최소 이미지 너비 (default: 100)
                - min_height: 최소 이미지 높이 (default: 100)
                - max_images_per_page: 페이지당 최대 이미지 수 (default: 10)
                - enhance_quality: 이미지 품질 향상 여부 (default: True)
                - extract_embedded: 임베디드 이미지 추출 여부 (default: True)
                - extract_rendered: 렌더링된 이미지 추출 여부 (default: True)
                - output_format: 출력 이미지 형식 (default: "PNG")
        """
        self.config = config or {}
        self.min_width = self.config.get("min_width", 100)
        self.min_height = self.config.get("min_height", 100)
        self.max_images_per_page = self.config.get("max_images_per_page", 10)
        self.enhance_quality = self.config.get("enhance_quality", True)
        self.extract_embedded = self.config.get("extract_embedded", True)
        self.extract_rendered = self.config.get("extract_rendered", True)
        self.output_format = self.config.get("output_format", "PNG")
        
        logger.info(f"PDFImageExtractor 초기화 완료: {self.config}")
    
    def extract_from_file(self, pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        PDF 파일에서 이미지를 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            추출된 이미지 정보 리스트
            각 항목: {
                'image': PIL.Image,
                'page': 페이지 번호,
                'index': 페이지 내 이미지 인덱스,
                'bbox': 이미지 위치 (x0, y0, x1, y1),
                'size': 이미지 크기 (width, height),
                'type': 추출 방법 ('embedded' 또는 'rendered'),
                'context': 주변 텍스트 컨텍스트
            }
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        logger.info(f"PDF에서 이미지 추출 시작: {pdf_path}")
        
        extracted_images = []
        
        try:
            with fitz.open(str(pdf_path)) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # 임베디드 이미지 추출
                    if self.extract_embedded:
                        embedded_images = self._extract_embedded_images(page, page_num)
                        extracted_images.extend(embedded_images)
                    
                    # 렌더링된 이미지 추출 (전체 페이지의 특정 영역)
                    if self.extract_rendered:
                        rendered_images = self._extract_rendered_images(page, page_num)
                        extracted_images.extend(rendered_images)
        
        except Exception as e:
            logger.error(f"PDF 이미지 추출 실패: {str(e)}")
            raise
        
        logger.info(f"총 {len(extracted_images)}개 이미지 추출 완료")
        return extracted_images
    
    def _extract_embedded_images(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """페이지에서 임베디드 이미지 추출"""
        images = []
        
        try:
            # 페이지의 이미지 리스트 가져오기
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                if len(images) >= self.max_images_per_page:
                    break
                
                try:
                    # 이미지 정보 가져오기
                    xref = img[0]
                    pix = fitz.Pixmap(page.document, xref)
                    
                    # CMYK 이미지는 RGB로 변환
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                    else:  # CMYK
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix1.tobytes("png")
                        pix1 = None
                    
                    pix = None
                    
                    # PIL Image로 변환
                    pil_image = Image.open(BytesIO(img_data))
                    
                    # 이미지 크기 확인
                    width, height = pil_image.size
                    if width < self.min_width or height < self.min_height:
                        continue
                    
                    # 이미지 향상
                    if self.enhance_quality:
                        pil_image = self._enhance_image(pil_image)
                    
                    # 이미지 위치 정보 (임베디드 이미지의 경우 정확한 위치 찾기 어려움)
                    bbox = (0, 0, width, height)
                    
                    # 주변 텍스트 컨텍스트 추출
                    context = self._extract_context(page, bbox)
                    
                    images.append({
                        'image': pil_image,
                        'page': page_num,
                        'index': img_index,
                        'bbox': bbox,
                        'size': (width, height),
                        'type': 'embedded',
                        'context': context
                    })
                    
                except Exception as e:
                    logger.warning(f"임베디드 이미지 추출 실패 (페이지 {page_num}, 인덱스 {img_index}): {str(e)}")
                    continue
        
        except Exception as e:
            logger.warning(f"페이지 {page_num} 임베디드 이미지 추출 실패: {str(e)}")
        
        return images
    
    def _extract_rendered_images(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """페이지를 렌더링하여 이미지 영역 추출"""
        images = []
        
        try:
            # 페이지의 텍스트 블록과 이미지 영역 분석
            blocks = page.get_text("dict")["blocks"]
            
            # 이미지 블록 찾기
            image_blocks = [block for block in blocks if block.get("type") == 1]  # type 1 = image
            
            for img_index, block in enumerate(image_blocks[:self.max_images_per_page]):
                try:
                    bbox = block["bbox"]
                    x0, y0, x1, y1 = bbox
                    
                    # 이미지 영역 크기 확인
                    width = int(x1 - x0)
                    height = int(y1 - y0)
                    
                    if width < self.min_width or height < self.min_height:
                        continue
                    
                    # 해당 영역을 고해상도로 렌더링
                    mat = fitz.Matrix(2.0, 2.0)  # 2배 확대
                    pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(bbox))
                    
                    img_data = pix.tobytes("png")
                    pix = None
                    
                    # PIL Image로 변환
                    pil_image = Image.open(BytesIO(img_data))
                    
                    # 이미지 향상
                    if self.enhance_quality:
                        pil_image = self._enhance_image(pil_image)
                    
                    # 주변 텍스트 컨텍스트 추출
                    context = self._extract_context(page, bbox)
                    
                    images.append({
                        'image': pil_image,
                        'page': page_num,
                        'index': img_index,
                        'bbox': bbox,
                        'size': pil_image.size,
                        'type': 'rendered',
                        'context': context
                    })
                    
                except Exception as e:
                    logger.warning(f"렌더링된 이미지 추출 실패 (페이지 {page_num}, 인덱스 {img_index}): {str(e)}")
                    continue
        
        except Exception as e:
            logger.warning(f"페이지 {page_num} 렌더링된 이미지 추출 실패: {str(e)}")
        
        return images
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """이미지 품질 향상"""
        try:
            # 선명도 향상
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            # 노이즈 제거 (부드럽게)
            image = image.filter(ImageFilter.SMOOTH_MORE)
            
            return image
            
        except Exception as e:
            logger.warning(f"이미지 향상 실패: {str(e)}")
            return image
    
    def _extract_context(self, page: fitz.Page, bbox: Tuple[float, float, float, float]) -> str:
        """이미지 주변의 텍스트 컨텍스트 추출"""
        try:
            x0, y0, x1, y1 = bbox
            
            # 이미지 주변 영역 확장 (위아래로 50 픽셀)
            context_rect = fitz.Rect(
                max(0, x0 - 50),
                max(0, y0 - 100),
                min(page.rect.width, x1 + 50),
                min(page.rect.height, y1 + 100)
            )
            
            # 해당 영역의 텍스트 추출
            context_text = page.get_text(clip=context_rect)
            
            # 텍스트 정리
            context_lines = []
            for line in context_text.split('\n'):
                line = line.strip()
                if line and len(line) > 2:  # 의미있는 텍스트만
                    context_lines.append(line)
            
            return ' '.join(context_lines[:5])  # 최대 5줄
            
        except Exception as e:
            logger.warning(f"컨텍스트 추출 실패: {str(e)}")
            return ""
    
    def save_images(
        self,
        images: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        prefix: str = "image"
    ) -> List[str]:
        """
        추출된 이미지들을 파일로 저장
        
        Args:
            images: 추출된 이미지 리스트
            output_dir: 출력 디렉토리
            prefix: 파일명 접두사
            
        Returns:
            저장된 파일 경로 리스트
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, img_info in enumerate(images):
            try:
                # 파일명 생성
                filename = f"{prefix}_p{img_info['page']:03d}_i{img_info['index']:02d}.{self.output_format.lower()}"
                file_path = output_dir / filename
                
                # 이미지 저장
                img_info['image'].save(str(file_path), format=self.output_format)
                saved_paths.append(str(file_path))
                
                # 메타데이터 저장
                metadata_path = file_path.with_suffix('.json')
                metadata = {
                    'page': img_info['page'],
                    'index': img_info['index'],
                    'bbox': img_info['bbox'],
                    'size': img_info['size'],
                    'type': img_info['type'],
                    'context': img_info['context']
                }
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logger.error(f"이미지 저장 실패 {i}: {str(e)}")
                continue
        
        logger.info(f"{len(saved_paths)}개 이미지 저장 완료: {output_dir}")
        return saved_paths
    
    def extract_and_save(
        self,
        pdf_path: Union[str, Path],
        output_dir: Union[str, Path],
        prefix: str = "image"
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        PDF에서 이미지를 추출하고 저장
        
        Args:
            pdf_path: PDF 파일 경로
            output_dir: 출력 디렉토리
            prefix: 파일명 접두사
            
        Returns:
            (추출된 이미지 정보, 저장된 파일 경로)
        """
        # 이미지 추출
        images = self.extract_from_file(pdf_path)
        
        # 이미지 저장
        saved_paths = self.save_images(images, output_dir, prefix)
        
        return images, saved_paths


class ImagePreprocessor:
    """이미지 전처리 클래스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 전처리 설정
                - target_size: 타겟 이미지 크기 (width, height) 또는 None
                - maintain_aspect: 종횡비 유지 여부 (default: True)
                - background_color: 배경색 (default: "white")
                - quality_threshold: 품질 임계값 (default: 0.8)
        """
        self.config = config or {}
        self.target_size = self.config.get("target_size", None)
        self.maintain_aspect = self.config.get("maintain_aspect", True)
        self.background_color = self.config.get("background_color", "white")
        self.quality_threshold = self.config.get("quality_threshold", 0.8)
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        """이미지 전처리"""
        try:
            # 복사본 생성
            processed = image.copy()
            
            # RGB 변환 (필요시)
            if processed.mode != 'RGB':
                processed = processed.convert('RGB')
            
            # 크기 조정
            if self.target_size:
                processed = self._resize_image(processed)
            
            # 품질 평가 및 개선
            if self._assess_quality(processed) < self.quality_threshold:
                processed = self._improve_quality(processed)
            
            return processed
            
        except Exception as e:
            logger.warning(f"이미지 전처리 실패: {str(e)}")
            return image
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """이미지 크기 조정"""
        target_width, target_height = self.target_size
        
        if self.maintain_aspect:
            # 종횡비 유지하면서 리사이즈
            image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
            
            # 중앙 정렬로 배경 추가
            new_image = Image.new('RGB', (target_width, target_height), self.background_color)
            
            # 중앙에 배치
            x = (target_width - image.width) // 2
            y = (target_height - image.height) // 2
            new_image.paste(image, (x, y))
            
            return new_image
        else:
            # 강제 리사이즈
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    def _assess_quality(self, image: Image.Image) -> float:
        """이미지 품질 평가 (0.0 ~ 1.0)"""
        try:
            # 간단한 품질 지표들
            img_array = np.array(image)
            
            # 1. 선명도 (라플라시안 분산)
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            laplacian_var = np.var(np.gradient(gray))
            sharpness_score = min(1.0, laplacian_var / 1000.0)
            
            # 2. 대비
            contrast_score = np.std(gray) / 255.0
            
            # 3. 밝기 분포
            hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
            brightness_score = 1.0 - abs(0.5 - np.mean(gray) / 255.0)
            
            # 종합 점수
            quality_score = (sharpness_score * 0.4 + contrast_score * 0.4 + brightness_score * 0.2)
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.warning(f"품질 평가 실패: {str(e)}")
            return 0.5
    
    def _improve_quality(self, image: Image.Image) -> Image.Image:
        """이미지 품질 개선"""
        try:
            # 선명도 향상
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.3)
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            return image
            
        except Exception as e:
            logger.warning(f"품질 개선 실패: {str(e)}")
            return image


def extract_images_from_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, List[str]]:
    """
    디렉토리의 모든 PDF에서 이미지 추출
    
    Args:
        input_dir: PDF 파일들이 있는 입력 디렉토리
        output_dir: 이미지를 저장할 출력 디렉토리
        config: 추출 설정
        
    Returns:
        파일별 추출된 이미지 경로 딕셔너리
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = PDFImageExtractor(config)
    results = {}
    
    # PDF 파일 찾기
    pdf_files = list(input_dir.glob("*.pdf"))
    logger.info(f"{len(pdf_files)}개 PDF 파일 발견: {input_dir}")
    
    for pdf_file in pdf_files:
        try:
            # 파일별 출력 디렉토리
            file_output_dir = output_dir / pdf_file.stem
            
            # 이미지 추출 및 저장
            _, saved_paths = extractor.extract_and_save(
                pdf_file,
                file_output_dir,
                prefix=pdf_file.stem
            )
            
            results[str(pdf_file)] = saved_paths
            
        except Exception as e:
            logger.error(f"PDF 처리 실패 {pdf_file}: {str(e)}")
            results[str(pdf_file)] = []
    
    return results


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트
    test_config = {
        "min_width": 50,
        "min_height": 50,
        "enhance_quality": True,
        "extract_embedded": True,
        "extract_rendered": True
    }
    
    extractor = PDFImageExtractor(test_config)
    print(f"PDFImageExtractor 초기화 완료: {extractor.config}")