"""
Model Configuration for Financial Security AI Competition
Updated: 2025-08-12
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    """Base configuration for models"""
    model_name: str
    model_type: str  # "teacher", "student", "synthetic"
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    max_length: int = 2048
    quantization: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transformers"""
        config = {
            "model_name": self.model_name,
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "max_length": self.max_length
        }
        if self.quantization:
            config.update(self.quantization)
        return config


# =====================================
# Production Models (for final system)
# =====================================

TEACHER_MODEL_CONFIG = ModelConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    model_type="teacher",
    device_map="auto",
    torch_dtype=torch.float16,
    max_length=4096,
    quantization={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True
    }
)

STUDENT_MODEL_CONFIG = ModelConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    model_type="student",
    device_map="cuda",
    torch_dtype=torch.float16,
    max_length=2048,
    quantization=None  # Small enough to run without quantization
)

# =====================================
# Synthetic Data Generation Model
# =====================================

SYNTHETIC_DATA_MODEL_CONFIG = ModelConfig(
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
    model_type="synthetic",
    device_map="auto",
    torch_dtype=torch.float16,
    max_length=8192,  # Longer context for document processing
    quantization={
        "load_in_4bit": True,  # Essential for local usage
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True
    }
)
# Note: Qwen3-30B is eligible (released July 30, 2025, before Aug 1, 2025 deadline)

# =====================================
# Alternative Configurations
# =====================================

# If memory is insufficient for synthetic data generation
SYNTHETIC_DATA_FALLBACK_CONFIG = ModelConfig(
    model_name="Qwen/Qwen2.5-14B-Instruct",
    model_type="synthetic",
    device_map="auto",
    torch_dtype=torch.float16,
    max_length=4096,
    quantization={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True
    }
)

# =====================================
# Model Selection Logic
# =====================================

def get_model_config(model_type: str, use_fallback: bool = False) -> ModelConfig:
    """
    Get model configuration by type
    
    Args:
        model_type: "teacher", "student", or "synthetic"
        use_fallback: Use fallback configuration if available
    
    Returns:
        ModelConfig object
    """
    configs = {
        "teacher": TEACHER_MODEL_CONFIG,
        "student": STUDENT_MODEL_CONFIG,
        "synthetic": SYNTHETIC_DATA_FALLBACK_CONFIG if use_fallback else SYNTHETIC_DATA_MODEL_CONFIG
    }
    
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return configs[model_type]


# =====================================
# Memory Estimation
# =====================================

def estimate_memory_usage(config: ModelConfig) -> Dict[str, float]:
    """
    Estimate memory usage for a model configuration
    
    Returns:
        Dictionary with memory estimates in GB
    """
    model_params = {
        "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
        "Qwen/Qwen2.5-7B-Instruct": 7,
        "Qwen/Qwen2.5-14B-Instruct": 14,
        "Qwen/Qwen3-30B-A3B-Instruct-2507": 30.5  # Total params, 3.3B activated
    }
    
    base_params = model_params.get(config.model_name, 7)  # Default 7B
    
    # Memory calculation
    if config.quantization and config.quantization.get("load_in_4bit"):
        # 4-bit quantization: ~0.5 bytes per parameter
        model_memory = base_params * 0.5
    elif config.quantization and config.quantization.get("load_in_8bit"):
        # 8-bit quantization: ~1 byte per parameter
        model_memory = base_params * 1
    else:
        # FP16: ~2 bytes per parameter
        model_memory = base_params * 2
    
    # Add overhead for activations and KV cache
    overhead = model_memory * 0.3
    
    return {
        "model_memory_gb": model_memory,
        "overhead_gb": overhead,
        "total_estimated_gb": model_memory + overhead,
        "fits_in_24gb": (model_memory + overhead) < 24
    }


# =====================================
# Competition Compliance Check
# =====================================

def check_competition_compliance(config: ModelConfig) -> Dict[str, bool]:
    """
    Check if model configuration meets competition requirements
    
    Returns:
        Dictionary with compliance status
    """
    # Known compliant models (Apache 2.0 or similar licenses)
    compliant_models = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct", 
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen3-30B-A3B-Instruct-2507"
    ]
    
    memory_estimate = estimate_memory_usage(config)
    
    return {
        "open_source": config.model_name in compliant_models,
        "memory_constraint": memory_estimate["fits_in_24gb"],
        "single_model": True,  # All configs use single models
        "license_compliant": True  # All Qwen models use Apache 2.0
    }


if __name__ == "__main__":
    # Test configurations
    print("Model Configurations Summary")
    print("=" * 50)
    
    for model_type in ["teacher", "student", "synthetic"]:
        config = get_model_config(model_type)
        memory = estimate_memory_usage(config)
        compliance = check_competition_compliance(config)
        
        print(f"\n{model_type.upper()} Model: {config.model_name}")
        print(f"  Memory: {memory['total_estimated_gb']:.1f}GB")
        print(f"  Fits in 24GB: {memory['fits_in_24gb']}")
        print(f"  Competition Compliant: {all(compliance.values())}")
        
        if model_type == "synthetic" and not memory['fits_in_24gb']:
            print("\n  Note: Synthetic data generation model exceeds 24GB limit.")
            print("  Options:")
            print("  1. Use cloud GPU for data generation phase only")
            print("  2. Use fallback model (14B)")
            print("  3. Use aggressive quantization (may impact quality)")
            
            # Test fallback
            fallback_config = get_model_config(model_type, use_fallback=True)
            fallback_memory = estimate_memory_usage(fallback_config)
            print(f"\n  Fallback Model: {fallback_config.model_name}")
            print(f"  Fallback Memory: {fallback_memory['total_estimated_gb']:.1f}GB")
            print(f"  Fallback Fits: {fallback_memory['fits_in_24gb']}")