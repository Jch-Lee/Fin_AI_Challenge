#!/usr/bin/env python3
"""
Vision Knowledge Base Builder

PDF 문서에서 이미지를 추출하고 텍스트 설명을 생성하여 지식베이스를 구축하는 스크립트
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from packages.vision.vision_pipeline import create_pipeline, AsyncVisionPipeline
from packages.vision.qwen_vision import test_vision_processor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vision_knowledge_builder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VisionKnowledgeBaseBuilder:
    """Vision 지식베이스 빌더"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 설정 파일 경로 (JSON)
        """
        self.config = self._load_config(config_path)
        self.pipeline = None
        self.async_pipeline = None
        
        logger.info(f"VisionKnowledgeBaseBuilder 초기화")
        logger.info(f"설정: {json.dumps(self.config, indent=2, ensure_ascii=False)}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """설정 파일 로드"""
        default_config = {
            "vision_model": {
                "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
                "quantization": True,
                "max_tokens": 512,
                "temperature": 0.3,
                "cache_dir": "./models/vision_cache"
            },
            "extractor": {
                "min_width": 100,
                "min_height": 100,
                "max_images_per_page": 10,
                "enhance_quality": True,
                "extract_embedded": True,
                "extract_rendered": True
            },
            "preprocessor": {
                "target_size": None,
                "maintain_aspect": True,
                "background_color": "white",
                "quality_threshold": 0.8
            },
            "processing": {
                "batch_size": 4,
                "max_workers": 2,
                "use_async": False,
                "cache_enabled": True
            },
            "output": {
                "format": "json",
                "save_images": True,
                "save_metadata": True,
                "base_dir": "./knowledge_base/vision"
            },
            "data": {
                "input_dirs": ["./data/financial_docs"],
                "file_extensions": [".pdf"],
                "recursive": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # 딥 머지
                def deep_merge(base, override):
                    result = base.copy()
                    for key, value in override.items():
                        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                            result[key] = deep_merge(result[key], value)
                        else:
                            result[key] = value
                    return result
                
                default_config = deep_merge(default_config, user_config)
                logger.info(f"설정 파일 로드됨: {config_path}")
                
            except Exception as e:
                logger.warning(f"설정 파일 로드 실패, 기본 설정 사용: {str(e)}")
        
        return default_config
    
    def initialize_pipeline(self):
        """파이프라인 초기화"""
        try:
            logger.info("Vision 처리 파이프라인 초기화 중...")
            
            pipeline_config = {
                "vision_model": self.config["vision_model"],
                "extractor": self.config["extractor"],
                "preprocessor": self.config["preprocessor"],
                "batch_size": self.config["processing"]["batch_size"],
                "max_workers": self.config["processing"]["max_workers"],
                "cache_enabled": self.config["processing"]["cache_enabled"],
                "output_format": self.config["output"]["format"]
            }
            
            self.pipeline = create_pipeline(pipeline_config)
            
            if self.config["processing"]["use_async"]:
                self.async_pipeline = AsyncVisionPipeline(pipeline_config)
            
            logger.info("파이프라인 초기화 완료")
            logger.info(f"파이프라인 통계: {self.pipeline.get_pipeline_stats()}")
            
        except Exception as e:
            logger.error(f"파이프라인 초기화 실패: {str(e)}")
            raise
    
    def find_input_files(self) -> List[Path]:
        """입력 파일 찾기"""
        input_files = []
        extensions = set(self.config["data"]["file_extensions"])
        
        for input_dir in self.config["data"]["input_dirs"]:
            input_path = Path(input_dir)
            
            if not input_path.exists():
                logger.warning(f"입력 디렉토리가 존재하지 않습니다: {input_path}")
                continue
            
            if self.config["data"]["recursive"]:
                for ext in extensions:
                    pattern = f"**/*{ext}"
                    files = list(input_path.glob(pattern))
                    input_files.extend(files)
            else:
                for ext in extensions:
                    pattern = f"*{ext}"
                    files = list(input_path.glob(pattern))
                    input_files.extend(files)
        
        # 중복 제거
        input_files = list(set(input_files))
        
        logger.info(f"입력 파일 {len(input_files)}개 발견")
        for f in input_files[:10]:  # 처음 10개만 로그
            logger.info(f"  - {f}")
        
        if len(input_files) > 10:
            logger.info(f"  ... 및 {len(input_files) - 10}개 추가")
        
        return input_files
    
    def process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """단일 파일 처리"""
        logger.info(f"파일 처리 시작: {file_path}")
        start_time = time.time()
        
        try:
            # 출력 디렉토리 설정
            output_dir = Path(self.config["output"]["base_dir"]) / file_path.stem
            
            # 파일 처리
            result = self.pipeline.process_pdf_document(
                file_path,
                output_dir,
                save_images=self.config["output"]["save_images"],
                save_results=True
            )
            
            # 처리 시간 기록
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            logger.info(f"파일 처리 완료: {file_path} ({processing_time:.2f}초)")
            
            if result["status"] == "success":
                logger.info(f"  - 추출된 이미지: {result['total_images']}개")
                logger.info(f"  - 출력 디렉토리: {result['output_dir']}")
            else:
                logger.warning(f"  - 처리 상태: {result['status']}")
            
            return result
            
        except Exception as e:
            logger.error(f"파일 처리 실패 {file_path}: {str(e)}")
            return {
                "status": "error",
                "file_path": str(file_path),
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def build_knowledge_base(self, test_mode: bool = False) -> Dict[str, Any]:
        """지식베이스 구축"""
        logger.info("Vision 지식베이스 구축 시작")
        total_start_time = time.time()
        
        try:
            # 파이프라인 초기화
            self.initialize_pipeline()
            
            # 입력 파일 찾기
            input_files = self.find_input_files()
            
            if not input_files:
                logger.warning("처리할 파일이 없습니다")
                return {
                    "status": "no_files",
                    "total_time": time.time() - total_start_time
                }
            
            # 테스트 모드에서는 최대 3개 파일만 처리
            if test_mode:
                input_files = input_files[:3]
                logger.info(f"테스트 모드: {len(input_files)}개 파일만 처리")
            
            # 출력 디렉토리 생성
            output_base_dir = Path(self.config["output"]["base_dir"])
            output_base_dir.mkdir(parents=True, exist_ok=True)
            
            # 파일별 처리
            results = []
            total_images = 0
            successful_files = 0
            
            for i, file_path in enumerate(input_files):
                logger.info(f"진행률: {i+1}/{len(input_files)} - {file_path.name}")
                
                result = self.process_single_file(file_path)
                results.append(result)
                
                if result["status"] == "success":
                    successful_files += 1
                    total_images += result.get("total_images", 0)
            
            # 통합 결과 저장
            summary = {
                "build_info": {
                    "timestamp": time.time(),
                    "total_time": time.time() - total_start_time,
                    "config": self.config
                },
                "statistics": {
                    "total_files": len(input_files),
                    "successful_files": successful_files,
                    "failed_files": len(input_files) - successful_files,
                    "total_images": total_images,
                    "success_rate": successful_files / len(input_files) if input_files else 0
                },
                "results": results
            }
            
            # 요약 저장
            summary_path = output_base_dir / "build_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info("="*60)
            logger.info("Vision 지식베이스 구축 완료")
            logger.info(f"총 처리 시간: {time.time() - total_start_time:.2f}초")
            logger.info(f"처리된 파일: {successful_files}/{len(input_files)}")
            logger.info(f"추출된 이미지: {total_images}개")
            logger.info(f"성공률: {summary['statistics']['success_rate']:.2%}")
            logger.info(f"결과 저장: {output_base_dir}")
            logger.info("="*60)
            
            return summary
            
        except Exception as e:
            logger.error(f"지식베이스 구축 실패: {str(e)}")
            raise
    
    def test_setup(self) -> bool:
        """설정 테스트"""
        logger.info("Vision 시스템 설정 테스트 중...")
        
        try:
            # 1. 모델 로드 테스트
            logger.info("1. Vision 모델 테스트...")
            if not test_vision_processor():
                logger.error("Vision 모델 테스트 실패")
                return False
            
            # 2. 파이프라인 초기화 테스트
            logger.info("2. 파이프라인 초기화 테스트...")
            self.initialize_pipeline()
            
            # 3. 입력 파일 확인
            logger.info("3. 입력 파일 확인...")
            input_files = self.find_input_files()
            if not input_files:
                logger.warning("입력 파일이 없지만 테스트는 성공")
            
            # 4. 출력 디렉토리 생성 테스트
            logger.info("4. 출력 디렉토리 테스트...")
            output_dir = Path(self.config["output"]["base_dir"]) / "test"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("모든 테스트 통과!")
            return True
            
        except Exception as e:
            logger.error(f"설정 테스트 실패: {str(e)}")
            return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Vision Knowledge Base Builder")
    parser.add_argument("--config", type=str, help="설정 파일 경로 (JSON)")
    parser.add_argument("--test", action="store_true", help="테스트 모드 (최대 3개 파일)")
    parser.add_argument("--test-setup", action="store_true", help="설정 테스트만 실행")
    parser.add_argument("--output-dir", type=str, help="출력 디렉토리 경로")
    parser.add_argument("--input-dir", type=str, action="append", help="입력 디렉토리 경로 (여러 개 가능)")
    parser.add_argument("--batch-size", type=int, help="배치 크기")
    parser.add_argument("--workers", type=int, help="워커 수")
    
    args = parser.parse_args()
    
    try:
        # 빌더 초기화
        builder = VisionKnowledgeBaseBuilder(args.config)
        
        # 명령행 인자로 설정 오버라이드
        if args.output_dir:
            builder.config["output"]["base_dir"] = args.output_dir
        
        if args.input_dir:
            builder.config["data"]["input_dirs"] = args.input_dir
        
        if args.batch_size:
            builder.config["processing"]["batch_size"] = args.batch_size
        
        if args.workers:
            builder.config["processing"]["max_workers"] = args.workers
        
        # 설정 테스트만 실행
        if args.test_setup:
            success = builder.test_setup()
            sys.exit(0 if success else 1)
        
        # 지식베이스 구축
        result = builder.build_knowledge_base(test_mode=args.test)
        
        if result["statistics"]["success_rate"] > 0.8:
            logger.info("지식베이스 구축 성공!")
            sys.exit(0)
        else:
            logger.warning("지식베이스 구축이 부분적으로만 성공했습니다.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"실행 실패: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()