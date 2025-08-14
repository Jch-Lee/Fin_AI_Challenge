"""
Vision Processing Pipeline

이미지-텍스트 변환을 위한 통합 파이프라인
"""

import os
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from PIL import Image

from .qwen_vision import QwenVisionProcessor, create_vision_processor
from .image_extractor import PDFImageExtractor, ImagePreprocessor
from .base_processor import BaseVisionProcessor

logger = logging.getLogger(__name__)


class VisionProcessingPipeline:
    """Vision 처리 통합 파이프라인"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 파이프라인 설정
                - vision_model: Vision 모델 설정
                - extractor: 이미지 추출 설정
                - preprocessor: 이미지 전처리 설정
                - batch_size: 배치 크기 (default: 4)
                - max_workers: 최대 워커 수 (default: 2)
                - cache_enabled: 캐싱 활성화 (default: True)
                - output_format: 출력 형식 (default: "json")
        """
        self.config = config or {}
        
        # Vision 프로세서 초기화
        vision_config = self.config.get("vision_model", {})
        self.vision_processor = create_vision_processor(vision_config)
        
        # 이미지 추출기 초기화
        extractor_config = self.config.get("extractor", {})
        self.image_extractor = PDFImageExtractor(extractor_config)
        
        # 이미지 전처리기 초기화
        preprocessor_config = self.config.get("preprocessor", {})
        self.image_preprocessor = ImagePreprocessor(preprocessor_config)
        
        # 파이프라인 설정
        self.batch_size = self.config.get("batch_size", 4)
        self.max_workers = self.config.get("max_workers", 2)
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.output_format = self.config.get("output_format", "json")
        
        # 결과 캐시
        self._results_cache = {}
        
        logger.info(f"VisionProcessingPipeline 초기화 완료")
    
    def process_pdf_document(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_images: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        PDF 문서를 완전히 처리하여 이미지-텍스트 변환
        
        Args:
            pdf_path: 처리할 PDF 파일 경로
            output_dir: 결과 저장 디렉토리
            save_images: 이미지 저장 여부
            save_results: 결과 저장 여부
            
        Returns:
            처리 결과 딕셔너리
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = pdf_path.parent / f"{pdf_path.stem}_vision_output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PDF 문서 처리 시작: {pdf_path}")
        start_time = time.time()
        
        try:
            # 1. 이미지 추출
            logger.info("1. PDF에서 이미지 추출 중...")
            extracted_images = self.image_extractor.extract_from_file(pdf_path)
            
            if not extracted_images:
                logger.warning("추출된 이미지가 없습니다")
                return {
                    "status": "no_images",
                    "pdf_path": str(pdf_path),
                    "processing_time": time.time() - start_time,
                    "images": [],
                    "results": []
                }
            
            # 2. 이미지 전처리
            logger.info(f"2. {len(extracted_images)}개 이미지 전처리 중...")
            for img_info in extracted_images:
                img_info['image'] = self.image_preprocessor.preprocess(img_info['image'])
            
            # 3. Vision 모델로 텍스트 변환
            logger.info("3. 이미지-텍스트 변환 중...")
            vision_results = self._process_images_batch(extracted_images)
            
            # 4. 결과 통합
            processed_results = []
            for i, (img_info, description) in enumerate(zip(extracted_images, vision_results)):
                result = {
                    "image_id": f"img_{img_info['page']:03d}_{img_info['index']:02d}",
                    "page": img_info['page'],
                    "index": img_info['index'],
                    "bbox": img_info['bbox'],
                    "size": img_info['size'],
                    "type": img_info['type'],
                    "context": img_info['context'],
                    "description": description,
                    "timestamp": time.time()
                }
                processed_results.append(result)
            
            # 5. 결과 저장
            if save_images:
                self._save_extracted_images(extracted_images, output_dir)
            
            if save_results:
                self._save_processing_results(processed_results, output_dir, pdf_path.name)
            
            # 최종 결과
            final_result = {
                "status": "success",
                "pdf_path": str(pdf_path),
                "processing_time": time.time() - start_time,
                "total_images": len(extracted_images),
                "output_dir": str(output_dir),
                "results": processed_results
            }
            
            logger.info(f"PDF 문서 처리 완료: {time.time() - start_time:.2f}초")
            return final_result
            
        except Exception as e:
            logger.error(f"PDF 문서 처리 실패: {str(e)}")
            return {
                "status": "error",
                "pdf_path": str(pdf_path),
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def process_image_directory(
        self,
        image_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        이미지 디렉토리의 모든 이미지를 처리
        
        Args:
            image_dir: 이미지 디렉토리 경로
            output_dir: 결과 저장 디렉토리
            
        Returns:
            처리 결과 딕셔너리
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"이미지 디렉토리를 찾을 수 없습니다: {image_dir}")
        
        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = image_dir.parent / f"{image_dir.name}_vision_output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"이미지 디렉토리 처리 시작: {image_dir}")
        start_time = time.time()
        
        try:
            # 이미지 파일 찾기
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_files = [
                f for f in image_dir.iterdir() 
                if f.suffix.lower() in image_extensions
            ]
            
            if not image_files:
                logger.warning("처리할 이미지가 없습니다")
                return {
                    "status": "no_images",
                    "image_dir": str(image_dir),
                    "processing_time": time.time() - start_time,
                    "results": []
                }
            
            logger.info(f"{len(image_files)}개 이미지 파일 발견")
            
            # 이미지 로드 및 처리
            processed_results = []
            
            for i, image_file in enumerate(image_files):
                try:
                    # 이미지 로드
                    pil_image = Image.open(image_file)
                    
                    # 전처리
                    processed_image = self.image_preprocessor.preprocess(pil_image)
                    
                    # Vision 모델로 처리
                    description = self.vision_processor.process_image(
                        processed_image,
                        prompt_type="general"
                    )
                    
                    result = {
                        "image_id": f"img_{i:03d}",
                        "file_path": str(image_file),
                        "file_name": image_file.name,
                        "size": processed_image.size,
                        "description": description,
                        "timestamp": time.time()
                    }
                    
                    processed_results.append(result)
                    logger.info(f"처리 완료: {image_file.name}")
                    
                except Exception as e:
                    logger.error(f"이미지 처리 실패 {image_file}: {str(e)}")
                    continue
            
            # 결과 저장
            self._save_processing_results(processed_results, output_dir, f"{image_dir.name}_results")
            
            final_result = {
                "status": "success",
                "image_dir": str(image_dir),
                "processing_time": time.time() - start_time,
                "total_images": len(processed_results),
                "output_dir": str(output_dir),
                "results": processed_results
            }
            
            logger.info(f"이미지 디렉토리 처리 완료: {time.time() - start_time:.2f}초")
            return final_result
            
        except Exception as e:
            logger.error(f"이미지 디렉토리 처리 실패: {str(e)}")
            return {
                "status": "error",
                "image_dir": str(image_dir),
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _process_images_batch(self, images: List[Dict[str, Any]]) -> List[str]:
        """이미지들을 배치로 처리"""
        results = []
        
        # 배치 단위로 처리
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            batch_images = [img_info['image'] for img_info in batch]
            batch_contexts = [img_info['context'] for img_info in batch]
            
            try:
                # Vision 프로세서로 배치 처리
                batch_results = self.vision_processor.batch_process(
                    batch_images,
                    batch_contexts,
                    self.batch_size
                )
                
                results.extend(batch_results)
                
                logger.info(f"배치 처리 완료: {i//self.batch_size + 1}/{(len(images) + self.batch_size - 1)//self.batch_size}")
                
            except Exception as e:
                logger.error(f"배치 처리 실패: {str(e)}")
                # 개별 처리로 폴백
                for img_info in batch:
                    try:
                        result = self.vision_processor.process_image(
                            img_info['image'],
                            img_info['context'],
                            "general"
                        )
                        results.append(result)
                    except Exception as e2:
                        logger.error(f"개별 이미지 처리 실패: {str(e2)}")
                        results.append(f"처리 실패: {str(e2)}")
        
        return results
    
    def _save_extracted_images(
        self, 
        images: List[Dict[str, Any]], 
        output_dir: Path
    ):
        """추출된 이미지들을 저장"""
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        for img_info in images:
            try:
                filename = f"img_p{img_info['page']:03d}_i{img_info['index']:02d}.png"
                file_path = images_dir / filename
                
                img_info['image'].save(file_path, "PNG")
                
                # 메타데이터 저장
                metadata = {
                    k: v for k, v in img_info.items() 
                    if k != 'image'
                }
                metadata_path = file_path.with_suffix('.json')
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                logger.error(f"이미지 저장 실패: {str(e)}")
    
    def _save_processing_results(
        self,
        results: List[Dict[str, Any]],
        output_dir: Path,
        base_name: str
    ):
        """처리 결과를 저장"""
        if self.output_format == "json":
            output_file = output_dir / f"{base_name}_results.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "total_images": len(results),
                        "timestamp": time.time(),
                        "pipeline_config": self.config
                    },
                    "results": results
                }, f, ensure_ascii=False, indent=2)
        
        elif self.output_format == "txt":
            output_file = output_dir / f"{base_name}_results.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Vision Processing Results\n")
                f.write(f"Total Images: {len(results)}\n")
                f.write(f"Timestamp: {time.ctime()}\n")
                f.write("="*50 + "\n\n")
                
                for result in results:
                    f.write(f"Image ID: {result.get('image_id', 'N/A')}\n")
                    f.write(f"Page: {result.get('page', 'N/A')}\n")
                    f.write(f"Description: {result.get('description', 'N/A')}\n")
                    f.write("-" * 30 + "\n")
        
        logger.info(f"처리 결과 저장 완료: {output_file}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """파이프라인 통계 정보 반환"""
        return {
            "vision_model": self.vision_processor.get_model_info(),
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self._results_cache)
        }


class AsyncVisionPipeline:
    """비동기 Vision 처리 파이프라인"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 파이프라인 설정
        """
        self.sync_pipeline = VisionProcessingPipeline(config)
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 2))
    
    async def process_pdf_async(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """PDF 문서를 비동기로 처리"""
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            self.executor,
            self.sync_pipeline.process_pdf_document,
            pdf_path,
            output_dir
        )
    
    async def process_multiple_pdfs(
        self,
        pdf_paths: List[Union[str, Path]],
        output_base_dir: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """여러 PDF를 동시에 처리"""
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 각 PDF에 대한 비동기 태스크 생성
        tasks = []
        for i, pdf_path in enumerate(pdf_paths):
            pdf_path = Path(pdf_path)
            output_dir = output_base_dir / f"{pdf_path.stem}_output"
            
            task = self.process_pdf_async(pdf_path, output_dir)
            tasks.append(task)
        
        # 모든 태스크 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 정리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "status": "error",
                    "pdf_path": str(pdf_paths[i]),
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def __del__(self):
        """소멸자"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


def create_pipeline(config: Optional[Dict[str, Any]] = None) -> VisionProcessingPipeline:
    """
    Vision 처리 파이프라인 생성 함수
    
    Args:
        config: 파이프라인 설정
        
    Returns:
        설정된 VisionProcessingPipeline 인스턴스
    """
    default_config = {
        "vision_model": {
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "quantization": True,
            "max_tokens": 512,
            "temperature": 0.3
        },
        "extractor": {
            "min_width": 100,
            "min_height": 100,
            "enhance_quality": True,
            "extract_embedded": True,
            "extract_rendered": True
        },
        "preprocessor": {
            "target_size": None,
            "maintain_aspect": True,
            "quality_threshold": 0.8
        },
        "batch_size": 4,
        "max_workers": 2,
        "cache_enabled": True,
        "output_format": "json"
    }
    
    if config:
        # 딥 머지
        def deep_merge(base, override):
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        default_config = deep_merge(default_config, config)
    
    return VisionProcessingPipeline(default_config)


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트
    pipeline = create_pipeline()
    print(f"Vision Processing Pipeline 초기화 완료")
    print(f"Pipeline Stats: {pipeline.get_pipeline_stats()}")