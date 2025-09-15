
#!/usr/bin/env python3
"""Utility functions for fine-tuning pipeline"""

import os
import json
import torch
import psutil
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def check_gpu_memory():
    if torch.cuda.is_available():
        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1e9
            total = gpu.total_memory / 1e9
            
            gpu_info[f"GPU_{i}"] = {
                "name": gpu.name,
                "total_memory_gb": total,
                "allocated_memory_gb": allocated,
                "free_memory_gb": total - allocated,
                "utilization_percent": (allocated / total) * 100
            }
        return gpu_info
    return {"error": "No CUDA GPUs available"}

def check_system_resources():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "cpu_percent": cpu_percent,
        "memory_total_gb": memory.total / 1e9,
        "memory_used_gb": memory.used / 1e9,
        "memory_percent": memory.percent,
        "disk_total_gb": disk.total / 1e9,
        "disk_used_gb": disk.used / 1e9,
        "disk_percent": (disk.used / disk.total) * 100
    }

def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_parameters_millions": total_params / 1e6,
        "trainable_parameters_millions": trainable_params / 1e6,
        "trainable_percent": (trainable_params / total_params) * 100
    }

def estimate_memory_usage(model_name: str, batch_size: int = 1, sequence_length: int = 512):
    model_sizes = {
        "gpt2": 124e6,
        "gpt2-medium": 355e6,
        "gpt2-large": 774e6,
        "gpt2-xl": 1.5e9,
    }
    
    params = model_sizes.get(model_name.lower(), 124e6)
    model_memory_fp16 = params * 2 / 1e9
    gradient_memory = model_memory_fp16
    optimizer_memory = model_memory_fp16 * 2
    activation_memory = batch_size * sequence_length * 1024 * 2 / 1e9
    total_memory = model_memory_fp16 + gradient_memory + optimizer_memory + activation_memory
    
    return {
        "model_memory_gb": model_memory_fp16,
        "gradient_memory_gb": gradient_memory,
        "optimizer_memory_gb": optimizer_memory,
        "activation_memory_gb": activation_memory,
        "total_estimated_gb": total_memory,
        "recommended_gpu_memory_gb": total_memory * 1.2
    }

def save_training_config(config: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Training config saved to {output_path}")

def load_training_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Training config loaded from {config_path}")
    return config

def create_sample_data(output_path: str, num_samples: int = 100):
    sample_texts = [
        "Artificial intelligence is transforming the world of technology.",
        "Machine learning algorithms can process vast amounts of data.",
        "Deep learning models achieve breakthrough performance in many tasks.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision systems can identify and classify objects in images.",
        "Reinforcement learning teaches AI agents through trial and error.",
        "Transfer learning allows models to apply knowledge across domains.",
        "Data science combines statistics, programming, and domain expertise.",
    ]
    
    texts = (sample_texts * (num_samples // len(sample_texts) + 1))[:num_samples]
    data = {"texts": texts}
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Created {num_samples} sample texts in {output_path}")

def validate_environment():
    issues = []
    
    import sys
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA not available")
    except ImportError:
        issues.append("PyTorch not installed")
    
    try:
        import transformers
        if hasattr(transformers, '__version__'):
            version = transformers.__version__
            if version < "4.20.0":
                issues.append(f"Transformers {version} may be outdated")
    except ImportError:
        issues.append("Transformers not installed")
    
    memory = psutil.virtual_memory()
    if memory.total < 16 * 1e9:
        issues.append("Less than 16GB RAM detected")
    
    if issues:
        logger.warning("Environment issues detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Environment validation passed")
    
    return len(issues) == 0

def benchmark_model_speed(model, tokenizer, device="cuda", num_iterations=10):
    model.eval()
    model.to(device)
    
    test_prompt = "The future of artificial intelligence"
    inputs = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(inputs, max_length=50, do_sample=False)
    
    import time
    times = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=50, do_sample=False)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    tokens_generated = outputs.shape[1] - inputs.shape[1]
    tokens_per_second = tokens_generated / avg_time
    
    return {
        "avg_generation_time_seconds": avg_time,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
        "iterations": num_iterations
    }

if __name__ == "__main__":
    print("System Diagnostics")
    print("=" * 50)
    
    print("\nSystem Resources:")
    system_info = check_system_resources()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    print("\nGPU Information:")
    gpu_info = check_gpu_memory()
    for key, value in gpu_info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("\nEnvironment Validation:")
    validate_environment()
    
    print("\nMemory Estimates for Popular Models:")
    models = ["gpt2", "gpt2-medium", "gpt2-large"]
    for model in models:
        estimate = estimate_memory_usage(model, batch_size=2)
        print(f"  {model}: ~{estimate['total_estimated_gb']:.1f}GB")
