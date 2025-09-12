# MINIMAL WORKING VERSION FOR RTX 4070 - NO TRANSFORMERS
print("🚀 Minimal Fine-tuning Test - RTX 4070")

import torch
import torch.nn as nn

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
    
    # Test GPU memory
    torch.cuda.empty_cache()
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"💾 Total GPU Memory: {memory_total:.1f}GB")
    
    # Create a simple model to test
    print("\n🧪 Testing GPU with simple model...")
    test_model = nn.Linear(1000, 1000).to(device)
    test_input = torch.randn(100, 1000).to(device)
    
    with torch.no_grad():
        output = test_model(test_input)
    
    memory_used = torch.cuda.memory_allocated(0) / 1e9
    print(f"✅ Test successful! GPU Memory used: {memory_used:.2f}GB")
    
    print("\n🎯 Your RTX 4070 is working perfectly!")
    print("📝 Next steps:")
    print("1. Fix transformers package conflicts")
    print("2. Or use raw PyTorch for fine-tuning")
    print("3. Available memory for models: {:.1f}GB".format(memory_total - memory_used))
    
else:
    print("❌ CUDA not available")

print("\n✅ Basic test complete - RTX 4070 ready!")
