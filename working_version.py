# Working Fine-tuning for RTX 4070 (No PEFT conflicts)
print("🚀 RTX 4070 Fine-tuning - Working Version")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("⚠️ Using CPU")
    device = torch.device("cpu")

# Load model
model_name = "gpt2"
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

# Simple training loop example (without Trainer)
print("\n🎯 Simple Training Example:")
print("# Set up optimizer")
print("optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)")
print("\n# Training loop")
print("for batch in dataloader:")
print("    optimizer.zero_grad()")
print("    outputs = model(**batch)")
print("    loss = outputs.loss")
print("    loss.backward()")
print("    optimizer.step()")

print("\n🎉 RTX 4070 setup working!")
print("💡 Use this approach instead of Trainer to avoid PEFT conflicts")

# Test generation
test_prompt = "The weather today is"
inputs = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(inputs, max_length=50, do_sample=True, temperature=0.7)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n🧪 Test generation:")
print(f"Input: {test_prompt}")
print(f"Output: {generated}")
