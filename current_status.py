#!/usr/bin/env python3
"""
Quick test to see what's working in terminal now
"""

print("🧪 Current Status Test...")

# Test 1: Basic imports
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✅ Basic imports work!")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    exit(1)

# Test 2: CUDA
print(f"✅ CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# Test 3: Trainer imports (the problematic one)
try:
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    print("✅ Trainer imports work!")
except Exception as e:
    print(f"❌ Trainer imports failed: {e}")

print("\n🎯 Summary: Basic functionality works, training setup needs fixing")
