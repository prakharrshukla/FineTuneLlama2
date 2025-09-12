#!/usr/bin/env python3
"""
Test training imports to see what's failing
"""

print("🧪 Testing training imports...")

# Test basic imports first
try:
    import torch
    print("✅ torch imported")
except Exception as e:
    print(f"❌ torch error: {e}")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ Basic transformers imported")
except Exception as e:
    print(f"❌ Basic transformers error: {e}")

# Test training imports
try:
    from transformers import TrainingArguments
    print("✅ TrainingArguments imported")
except Exception as e:
    print(f"❌ TrainingArguments error: {e}")

try:
    from transformers import Trainer
    print("✅ Trainer imported")
except Exception as e:
    print(f"❌ Trainer error: {e}")

try:
    from transformers import DataCollatorForLanguageModeling
    print("✅ DataCollatorForLanguageModeling imported")
except Exception as e:
    print(f"❌ DataCollatorForLanguageModeling error: {e}")

print("\n🎉 Import testing complete!")
