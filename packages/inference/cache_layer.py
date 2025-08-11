"""
Cache Layer Component
Architecture.md의 ICacheLayer 인터페이스 구현
Pipeline.md 3.2.2: 캐시 히트율 ≥ 80% 목표
"""

import json
import hashlib
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict, List
from pathlib import Path
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """캐시 엔트리 데이터 클래스"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: int  # Time to live in seconds
    size_bytes: int


class ICacheLayer(ABC):
    """캐싱 컴포넌트 인터페이스"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """캐시에 값 저장"""
        pass
    
    @abstractmethod
    def invalidate(self, pattern: str) -> int:
        """패턴에 매칭되는 캐시 무효화"""
        pass


class CacheLayer(ICacheLayer):
    """
    LRU 캐시 레이어 구현
    Architecture.md Cache-Aside 패턴
    Pipeline.md 3.2.2: 캐시 히트율 ≥ 80% 목표
    """
    
    def __init__(self, 
                 max_size_mb: int = 512,
                 max_entries: int = 10000,
                 ttl_seconds: int = 3600,
                 persistence_path: Optional[str] = None):
        """
        Args:
            max_size_mb: 최대 캐시 크기 (MB)
            max_entries: 최대 엔트리 수
            ttl_seconds: 기본 TTL (초)
            persistence_path: 캐시 영속화 경로
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl = ttl_seconds
        self.persistence_path = Path(persistence_path) if persistence_path else None
        
        # LRU 캐시 구현을 위한 OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # 통계
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0,
            "total_requests": 0,
            "current_size_bytes": 0,
            "current_entries": 0
        }
        
        # 영속화된 캐시 로드
        if self.persistence_path and self.persistence_path.exists():
            self._load_cache()
        
        logger.info(f"CacheLayer initialized: max_size={max_size_mb}MB, max_entries={max_entries}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값 조회
        
        Args:
            key: 캐시 키
            
        Returns:
            캐시된 값 또는 None
        """
        self.stats["total_requests"] += 1
        
        # 키 정규화
        normalized_key = self._normalize_key(key)
        
        if normalized_key in self._cache:
            entry = self._cache[normalized_key]
            
            # TTL 확인
            if self._is_expired(entry):
                self._remove_entry(normalized_key)
                self.stats["misses"] += 1
                logger.debug(f"Cache miss (expired): {key}")
                return None
            
            # LRU 업데이트 (최근 사용으로 이동)
            self._cache.move_to_end(normalized_key)
            
            # 접근 정보 업데이트
            entry.accessed_at = time.time()
            entry.access_count += 1
            
            self.stats["hits"] += 1
            logger.debug(f"Cache hit: {key}")
            return entry.value
        
        self.stats["misses"] += 1
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        캐시에 값 저장
        
        Args:
            key: 캐시 키
            value: 저장할 값
            ttl: Time to live (초)
            
        Returns:
            저장 성공 여부
        """
        try:
            # 키 정규화
            normalized_key = self._normalize_key(key)
            
            # 값 크기 계산
            value_bytes = self._calculate_size(value)
            
            # 크기 제한 확인
            if value_bytes > self.max_size_bytes:
                logger.warning(f"Value too large to cache: {value_bytes} bytes")
                return False
            
            # 공간 확보 (필요시 eviction)
            self._ensure_space(value_bytes)
            
            # 캐시 엔트리 생성
            current_time = time.time()
            entry = CacheEntry(
                key=normalized_key,
                value=value,
                created_at=current_time,
                accessed_at=current_time,
                access_count=0,
                ttl=ttl or self.default_ttl,
                size_bytes=value_bytes
            )
            
            # 기존 엔트리가 있으면 제거
            if normalized_key in self._cache:
                self._remove_entry(normalized_key)
            
            # 새 엔트리 추가 (LRU: 끝에 추가)
            self._cache[normalized_key] = entry
            self.stats["current_entries"] = len(self._cache)
            self.stats["current_size_bytes"] += value_bytes
            
            logger.debug(f"Cache set: {key} ({value_bytes} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache value: {e}")
            return False
    
    def invalidate(self, pattern: str) -> int:
        """
        패턴에 매칭되는 캐시 무효화
        
        Args:
            pattern: 매칭 패턴 (와일드카드 * 지원)
            
        Returns:
            무효화된 엔트리 수
        """
        import fnmatch
        
        invalidated_count = 0
        keys_to_remove = []
        
        # 패턴 매칭
        for key in self._cache:
            if fnmatch.fnmatch(key, pattern):
                keys_to_remove.append(key)
        
        # 매칭된 키 제거
        for key in keys_to_remove:
            self._remove_entry(key)
            invalidated_count += 1
        
        self.stats["invalidations"] += invalidated_count
        logger.info(f"Invalidated {invalidated_count} cache entries matching pattern: {pattern}")
        
        return invalidated_count
    
    def _normalize_key(self, key: str) -> str:
        """키 정규화 (해시 사용)"""
        # 긴 키는 해시로 변환
        if len(key) > 250:
            return hashlib.sha256(key.encode()).hexdigest()
        return key
    
    def _calculate_size(self, value: Any) -> int:
        """값의 크기 계산 (바이트)"""
        try:
            # pickle을 사용한 크기 추정
            return len(pickle.dumps(value))
        except:
            # 실패 시 기본값
            return 1024
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """TTL 만료 확인"""
        return (time.time() - entry.created_at) > entry.ttl
    
    def _ensure_space(self, required_bytes: int):
        """
        필요한 공간 확보 (LRU eviction)
        """
        # 엔트리 수 제한 확인
        while len(self._cache) >= self.max_entries:
            self._evict_lru()
        
        # 크기 제한 확인
        while (self.stats["current_size_bytes"] + required_bytes) > self.max_size_bytes:
            if not self._evict_lru():
                break  # 더 이상 제거할 수 없음
    
    def _evict_lru(self) -> bool:
        """LRU 엔트리 제거"""
        if not self._cache:
            return False
        
        # OrderedDict의 첫 번째 아이템이 가장 오래된 것
        key, entry = self._cache.popitem(last=False)
        self.stats["current_size_bytes"] -= entry.size_bytes
        self.stats["evictions"] += 1
        self.stats["current_entries"] = len(self._cache)
        
        logger.debug(f"Evicted cache entry: {key}")
        return True
    
    def _remove_entry(self, key: str):
        """캐시 엔트리 제거"""
        if key in self._cache:
            entry = self._cache[key]
            self.stats["current_size_bytes"] -= entry.size_bytes
            del self._cache[key]
            self.stats["current_entries"] = len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 반환
        Pipeline.md 3.2.2: 캐시 히트율 모니터링
        """
        stats = self.stats.copy()
        
        # 히트율 계산
        if stats["total_requests"] > 0:
            stats["hit_rate"] = stats["hits"] / stats["total_requests"] * 100
        else:
            stats["hit_rate"] = 0.0
        
        # 크기 정보
        stats["size_mb"] = stats["current_size_bytes"] / (1024 * 1024)
        stats["max_size_mb"] = self.max_size_bytes / (1024 * 1024)
        
        # 사용률
        stats["memory_usage_percent"] = (stats["current_size_bytes"] / self.max_size_bytes * 100 
                                        if self.max_size_bytes > 0 else 0)
        stats["entry_usage_percent"] = (stats["current_entries"] / self.max_entries * 100 
                                       if self.max_entries > 0 else 0)
        
        return stats
    
    def clear(self):
        """캐시 전체 삭제"""
        self._cache.clear()
        self.stats["current_size_bytes"] = 0
        self.stats["current_entries"] = 0
        logger.info("Cache cleared")
    
    def _save_cache(self):
        """캐시 영속화"""
        if not self.persistence_path:
            return
        
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 만료되지 않은 엔트리만 저장
            valid_cache = {}
            for key, entry in self._cache.items():
                if not self._is_expired(entry):
                    valid_cache[key] = entry
            
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(valid_cache, f)
            
            logger.info(f"Cache persisted: {len(valid_cache)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _load_cache(self):
        """영속화된 캐시 로드"""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        
        try:
            with open(self.persistence_path, 'rb') as f:
                loaded_cache = pickle.load(f)
            
            # 만료되지 않은 엔트리만 복원
            restored_count = 0
            for key, entry in loaded_cache.items():
                if not self._is_expired(entry):
                    self._cache[key] = entry
                    self.stats["current_size_bytes"] += entry.size_bytes
                    restored_count += 1
            
            self.stats["current_entries"] = len(self._cache)
            logger.info(f"Cache restored: {restored_count} entries")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def __del__(self):
        """소멸자: 캐시 영속화"""
        if hasattr(self, 'persistence_path'):
            self._save_cache()


class QueryCache(CacheLayer):
    """
    쿼리 전용 캐시 (RAG 시스템용)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query_normalizer = self._create_query_normalizer()
    
    def _create_query_normalizer(self):
        """쿼리 정규화 함수"""
        def normalize(query: str) -> str:
            # 소문자 변환, 공백 정규화
            normalized = query.lower().strip()
            normalized = ' '.join(normalized.split())
            return normalized
        return normalize
    
    def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """캐시된 쿼리 응답 조회"""
        normalized_query = self.query_normalizer(query)
        cache_key = f"query:{hashlib.md5(normalized_query.encode()).hexdigest()}"
        return self.get(cache_key)
    
    def cache_response(self, query: str, response: Dict[str, Any], ttl: int = 3600) -> bool:
        """쿼리 응답 캐싱"""
        normalized_query = self.query_normalizer(query)
        cache_key = f"query:{hashlib.md5(normalized_query.encode()).hexdigest()}"
        return self.set(cache_key, response, ttl)


def main():
    """테스트 및 데모"""
    # 캐시 레이어 생성
    cache = CacheLayer(
        max_size_mb=10,
        max_entries=100,
        ttl_seconds=60,
        persistence_path="cache/test_cache.pkl"
    )
    
    # 데이터 캐싱
    test_data = {
        "question": "금융 AI 시스템의 보안 요구사항은?",
        "answer": "암호화, 접근 제어, 감사 로그 등이 필요합니다.",
        "confidence": 0.95
    }
    
    # 캐시 저장
    cache.set("test_query_1", test_data)
    
    # 캐시 조회
    cached_data = cache.get("test_query_1")
    print(f"Cached data: {cached_data}")
    
    # 통계 출력
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Hit rate: {stats['hit_rate']:.1f}%")
    print(f"  Entries: {stats['current_entries']}/{cache.max_entries}")
    print(f"  Size: {stats['size_mb']:.2f}/{stats['max_size_mb']:.2f} MB")
    
    # 쿼리 캐시 테스트
    query_cache = QueryCache(max_size_mb=5, max_entries=50)
    
    # 쿼리 응답 캐싱
    query = "What are the security requirements for financial AI?"
    response = {"answer": "Encryption, access control, audit logs", "score": 0.9}
    
    query_cache.cache_response(query, response)
    cached_response = query_cache.get_cached_response(query)
    
    print(f"\nQuery Cache Test:")
    print(f"  Original: {response}")
    print(f"  Cached: {cached_response}")
    
    # 패턴 무효화 테스트
    cache.set("api_v1_users", {"data": "users"})
    cache.set("api_v1_products", {"data": "products"})
    cache.set("api_v2_users", {"data": "v2_users"})
    
    invalidated = cache.invalidate("api_v1_*")
    print(f"\nInvalidated {invalidated} entries matching 'api_v1_*'")
    
    # 최종 통계
    final_stats = cache.get_stats()
    print(f"\nFinal hit rate: {final_stats['hit_rate']:.1f}%")
    
    # Pipeline.md 3.2.2 목표 확인
    if final_stats['hit_rate'] >= 80:
        print("✅ Cache hit rate target (≥80%) achieved!")
    else:
        print(f"⚠️ Cache hit rate {final_stats['hit_rate']:.1f}% below target (80%)")


if __name__ == "__main__":
    main()