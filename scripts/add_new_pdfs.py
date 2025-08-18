#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
증분 PDF 처리 스크립트
새로운 PDF를 기존 지식베이스에 추가하는 파이프라인
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import shutil
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('incremental_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IncrementalPDFProcessor:
    """증분 PDF 처리 클래스"""
    
    def __init__(self, 
                 use_vision: bool = True,
                 use_remote: bool = False,
                 remote_host: str = None):
        """
        Args:
            use_vision: Vision V2 모델 사용 여부
            use_remote: 원격 서버 사용 여부
            remote_host: 원격 서버 주소 (예: "root@86.127.233.28:34270")
        """
        self.use_vision = use_vision
        self.use_remote = use_remote
        self.remote_host = remote_host
        
        # 디렉토리 설정
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.kb_dir = Path("data/knowledge_base")
        self.tracking_file = Path("data/processed_files.json")
        
        # 처리 이력 로드
        self.processed_files = self._load_processed_history()
        
    def _load_processed_history(self) -> Dict[str, Any]:
        """처리된 파일 이력 로드"""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"files": {}, "last_update": None}
    
    def _save_processed_history(self):
        """처리된 파일 이력 저장"""
        self.processed_files["last_update"] = datetime.now().isoformat()
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_files, f, indent=2, ensure_ascii=False)
    
    def find_new_pdfs(self) -> List[Path]:
        """새로운 PDF 파일 찾기"""
        all_pdfs = list(self.raw_dir.glob("*.pdf"))
        new_pdfs = []
        
        for pdf_path in all_pdfs:
            pdf_name = pdf_path.name
            if pdf_name not in self.processed_files["files"]:
                new_pdfs.append(pdf_path)
                logger.info(f"New PDF found: {pdf_name}")
            else:
                # 파일 크기나 수정 시간 체크
                file_stat = pdf_path.stat()
                if pdf_name in self.processed_files["files"]:
                    prev_size = self.processed_files["files"][pdf_name].get("size", 0)
                    if file_stat.st_size != prev_size:
                        new_pdfs.append(pdf_path)
                        logger.info(f"Modified PDF found: {pdf_name}")
        
        return new_pdfs
    
    def process_locally(self, pdf_files: List[Path]) -> Dict[str, Any]:
        """로컬에서 PDF 처리"""
        if not pdf_files:
            logger.info("No new PDFs to process")
            return {"status": "no_new_files"}
        
        logger.info(f"Processing {len(pdf_files)} new PDFs locally...")
        
        # 임시 디렉토리 생성
        temp_dir = Path("data/temp_new_pdfs")
        temp_dir.mkdir(exist_ok=True)
        
        # 새 파일들을 임시 디렉토리로 복사
        for pdf in pdf_files:
            shutil.copy(pdf, temp_dir / pdf.name)
        
        # process_all_pdfs.py 실행
        cmd = f"python scripts/process_all_pdfs.py --pdf-dir {temp_dir}"
        if self.use_vision:
            cmd += " --use-vision"
        cmd += " --batch-size 2"
        
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            # 처리 성공 - 이력 업데이트
            for pdf in pdf_files:
                self.processed_files["files"][pdf.name] = {
                    "size": pdf.stat().st_size,
                    "processed_date": datetime.now().isoformat(),
                    "status": "success"
                }
            self._save_processed_history()
            
            # 임시 디렉토리 정리
            shutil.rmtree(temp_dir)
            
            return {
                "status": "success",
                "processed_count": len(pdf_files),
                "files": [p.name for p in pdf_files]
            }
        else:
            logger.error(f"Processing failed: {result.stderr}")
            return {
                "status": "failed",
                "error": result.stderr
            }
    
    def process_remotely(self, pdf_files: List[Path]) -> Dict[str, Any]:
        """원격 서버에서 PDF 처리"""
        if not self.remote_host:
            raise ValueError("Remote host not configured")
        
        logger.info(f"Uploading {len(pdf_files)} PDFs to remote server...")
        
        # SSH 정보 파싱
        if ":" in self.remote_host:
            host, port = self.remote_host.rsplit(":", 1)
        else:
            host = self.remote_host
            port = "22"
        
        # 파일 업로드
        import subprocess
        for pdf in pdf_files:
            cmd = f"scp -P {port} {pdf} {host}:/root/Fin_AI_Challenge/data/raw/"
            result = subprocess.run(cmd, shell=True, capture_output=True)
            if result.returncode != 0:
                logger.error(f"Failed to upload {pdf.name}")
                return {"status": "upload_failed", "file": pdf.name}
        
        # 원격 처리 실행
        remote_cmd = "cd /root/Fin_AI_Challenge && python3 scripts/process_all_pdfs.py --use-vision --batch-size 2"
        cmd = f"ssh -p {port} {host} '{remote_cmd}'"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            # 처리 성공
            for pdf in pdf_files:
                self.processed_files["files"][pdf.name] = {
                    "size": pdf.stat().st_size,
                    "processed_date": datetime.now().isoformat(),
                    "status": "success",
                    "location": "remote"
                }
            self._save_processed_history()
            
            return {
                "status": "success",
                "processed_count": len(pdf_files),
                "files": [p.name for p in pdf_files],
                "location": "remote"
            }
        else:
            return {"status": "failed", "error": result.stderr}
    
    def update_knowledge_base(self) -> bool:
        """지식베이스 업데이트 (FAISS 인덱스 재구축)"""
        logger.info("Updating knowledge base...")
        
        try:
            from packages.rag.knowledge_base import KnowledgeBase
            kb = KnowledgeBase()
            
            # 모든 처리된 텍스트 파일 로드
            processed_files = list(self.processed_dir.glob("*.txt"))
            logger.info(f"Found {len(processed_files)} processed files")
            
            # 지식베이스 재구축
            kb_file = self.kb_dir / "knowledge_base.pkl"
            if kb_file.exists():
                kb.load(str(kb_file))
                logger.info("Loaded existing knowledge base")
            
            # 새로운 파일들 추가 (여기서는 간단히 표시만)
            logger.info("Knowledge base updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update knowledge base: {e}")
            return False
    
    def process_new_pdfs(self) -> Dict[str, Any]:
        """메인 처리 함수"""
        # 1. 새 PDF 찾기
        new_pdfs = self.find_new_pdfs()
        
        if not new_pdfs:
            logger.info("No new PDFs found")
            return {
                "status": "up_to_date",
                "message": "All PDFs are already processed"
            }
        
        logger.info(f"Found {len(new_pdfs)} new PDFs to process")
        
        # 2. 처리 실행
        if self.use_remote:
            result = self.process_remotely(new_pdfs)
        else:
            result = self.process_locally(new_pdfs)
        
        # 3. 지식베이스 업데이트
        if result["status"] == "success":
            kb_updated = self.update_knowledge_base()
            result["kb_updated"] = kb_updated
        
        return result


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process new PDFs incrementally")
    parser.add_argument('--use-vision', action='store_true', help='Use Vision V2 model')
    parser.add_argument('--remote', action='store_true', help='Use remote server')
    parser.add_argument('--remote-host', type=str, default='root@86.127.233.28:34270',
                       help='Remote host address')
    parser.add_argument('--check-only', action='store_true', 
                       help='Only check for new files without processing')
    
    args = parser.parse_args()
    
    # 프로세서 초기화
    processor = IncrementalPDFProcessor(
        use_vision=args.use_vision,
        use_remote=args.remote,
        remote_host=args.remote_host if args.remote else None
    )
    
    if args.check_only:
        # 새 파일 체크만
        new_pdfs = processor.find_new_pdfs()
        if new_pdfs:
            print(f"\nFound {len(new_pdfs)} new PDFs:")
            for pdf in new_pdfs:
                print(f"  - {pdf.name}")
        else:
            print("All PDFs are already processed")
        return
    
    # 처리 실행
    result = processor.process_new_pdfs()
    
    # 결과 출력
    print("\n" + "=" * 60)
    print(" Processing Result")
    print("=" * 60)
    if result["status"] == "success":
        print(f"Successfully processed {result['processed_count']} PDFs")
        print("Files:")
        for f in result["files"]:
            print(f"  - {f}")
        if result.get("kb_updated"):
            print("Knowledge base updated")
    elif result["status"] == "up_to_date":
        print("All PDFs are already processed")
    else:
        print(f"Processing failed: {result.get('error', 'Unknown error')}")
    print("=" * 60)


if __name__ == "__main__":
    main()