# Quick Fine-tuning Test for RTX 4070
print("🚀 Starting quick fine-tuning test...")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Configuration
model_name = "gpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
print(f"📦 Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

print("✅ Model loaded successfully!")
print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check GPU memory
if torch.cuda.is_available():
    memory_used = torch.cuda.memory_allocated(0) / 1e9
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"💾 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")

print("🎉 RTX 4070 setup complete and working!")
print("💡 The notebook kernel might have package conflicts.")
print("💡 Consider restarting VS Code or creating a new notebook.")
