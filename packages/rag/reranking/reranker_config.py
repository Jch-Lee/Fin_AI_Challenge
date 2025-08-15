"""
Configuration management for reranker models.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """
    Configuration class for reranker models.
    
    This class holds all configuration parameters for rerankers,
    with special optimizations for Korean financial domain.
    """
    
    # Model configuration
    model_name: str = "Qwen/Qwen3-Reranker-4B"
    model_type: str = "qwen3"  # qwen3, bge, jina, etc.
    
    # Device configuration
    device: str = "cuda"
    precision: str = "fp16"  # fp16, fp32, bf16
    
    # Inference configuration
    batch_size: int = 8
    max_length: int = 512
    
    # Caching configuration
    cache_enabled: bool = True
    cache_size: int = 10000
    cache_ttl: int = 3600  # seconds
    
    # Score combination
    use_original_scores: bool = True
    rerank_weight: float = 0.7  # Weight for reranking scores (1-weight for original)
    
    # Korean financial domain specific
    korean_preprocessing: bool = True
    financial_terms_boost: bool = True
    financial_terms_weight: float = 1.2
    
    # Performance optimization
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    compile_model: bool = False  # torch.compile for PyTorch 2.0+
    
    # Fallback configuration
    fallback_enabled: bool = True
    fallback_retriever: str = "hybrid"  # hybrid, vector, bm25
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_interval: int = 100  # Log metrics every N queries
    
    # Additional model-specific parameters
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RerankerConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> "RerankerConfig":
        """Create configuration from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid
        """
        errors = []
        
        # Validate device
        if self.device not in ["cuda", "cpu", "mps"]:
            errors.append(f"Invalid device: {self.device}")
        
        # Validate precision
        if self.precision not in ["fp16", "fp32", "bf16"]:
            errors.append(f"Invalid precision: {self.precision}")
        
        # Validate batch size
        if self.batch_size <= 0:
            errors.append(f"Batch size must be positive: {self.batch_size}")
        
        # Validate max length
        if self.max_length <= 0:
            errors.append(f"Max length must be positive: {self.max_length}")
        
        # Validate weights
        if not 0 <= self.rerank_weight <= 1:
            errors.append(f"Rerank weight must be between 0 and 1: {self.rerank_weight}")
        
        if errors:
            for error in errors:
                logger.error(error)
            return False
        
        return True
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model-specific configuration.
        
        Returns:
            Dictionary of model configuration parameters
        """
        config = {
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "cache_enabled": self.cache_enabled,
        }
        
        # Add model-specific parameters
        config.update(self.model_kwargs)
        
        return config


def get_default_config(model_type: str = "qwen3") -> RerankerConfig:
    """
    Get default configuration for a specific model type.
    
    Args:
        model_type: Type of reranker model
        
    Returns:
        RerankerConfig with model-specific defaults
    """
    base_config = RerankerConfig(model_type=model_type)
    
    if model_type == "qwen3":
        base_config.model_name = "Qwen/Qwen3-Reranker-4B"
        base_config.batch_size = 8
        base_config.max_length = 512
        base_config.precision = "fp16"
        
    elif model_type == "bge":
        base_config.model_name = "BAAI/bge-reranker-v2-m3"
        base_config.batch_size = 16
        base_config.max_length = 512
        base_config.precision = "fp16"
        
    elif model_type == "jina":
        base_config.model_name = "jinaai/jina-reranker-v2-base-multilingual"
        base_config.batch_size = 12
        base_config.max_length = 512
        base_config.precision = "fp16"
    
    else:
        logger.warning(f"Unknown model type: {model_type}, using default configuration")
    
    return base_config


def get_competition_config() -> RerankerConfig:
    """
    Get optimized configuration for the competition environment.
    
    This configuration is specifically tuned for:
    - RTX 4090 24GB VRAM
    - Korean financial security questions
    - Offline environment
    - 4.5 hours inference time limit
    
    Returns:
        Optimized RerankerConfig for competition
    """
    config = RerankerConfig(
        # Model
        model_name="Qwen/Qwen3-Reranker-4B",
        model_type="qwen3",
        
        # Device - RTX 4090
        device="cuda",
        precision="fp16",  # FP16 for memory efficiency
        
        # Inference - Optimized for RTX 4090
        batch_size=16,  # Larger batch for RTX 4090
        max_length=512,  # Balance between context and speed
        
        # Caching - Important for offline environment
        cache_enabled=True,
        cache_size=20000,  # Larger cache for repeated queries
        cache_ttl=7200,  # 2 hours
        
        # Score combination
        use_original_scores=True,
        rerank_weight=0.75,  # Higher weight for reranker
        
        # Korean financial domain
        korean_preprocessing=True,
        financial_terms_boost=True,
        financial_terms_weight=1.3,
        
        # Performance optimization for RTX 4090
        use_flash_attention=False,  # Not needed for 4B model
        gradient_checkpointing=False,  # Not needed for inference
        compile_model=False,  # Can enable if using PyTorch 2.0+
        
        # Fallback
        fallback_enabled=True,
        fallback_retriever="hybrid",
        
        # Monitoring
        log_level="INFO",
        enable_metrics=True,
        metrics_interval=50,
    )
    
    return config


# Pre-defined configurations
CONFIGS = {
    "default": get_default_config("qwen3"),
    "competition": get_competition_config(),
    "qwen3": get_default_config("qwen3"),
    "bge": get_default_config("bge"),
    "jina": get_default_config("jina"),
    "debug": RerankerConfig(
        batch_size=1,
        cache_enabled=False,
        log_level="DEBUG",
        enable_metrics=True,
        metrics_interval=1,
    ),
    "fast": RerankerConfig(
        batch_size=32,
        max_length=256,
        cache_enabled=True,
        cache_size=50000,
        precision="fp16",
    ),
}


def load_config(config_name: str = "competition") -> RerankerConfig:
    """
    Load a pre-defined configuration.
    
    Args:
        config_name: Name of the configuration to load
        
    Returns:
        RerankerConfig instance
    """
    if config_name not in CONFIGS:
        logger.warning(f"Unknown config: {config_name}, using default")
        return CONFIGS["default"]
    
    config = CONFIGS[config_name]
    if config.validate():
        logger.info(f"Loaded configuration: {config_name}")
        return config
    else:
        logger.error(f"Invalid configuration: {config_name}, using default")
        return CONFIGS["default"]