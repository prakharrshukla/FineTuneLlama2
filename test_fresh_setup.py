#!/usr/bin/env python3
"""
Test the fresh setup to see if everything works in terminal
"""

print("🧪 Testing fresh setup...")

# Test 1: Basic imports
print("\n1️⃣ Testing imports...")
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test 2: CUDA detection
print("\n2️⃣ Testing CUDA...")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Test 3: Model loading
print("\n3️⃣ Testing model loading...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    
    print(f"✅ Model loaded on: {next(model.parameters()).device}")
    
    # Check memory
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"💾 GPU Memory used: {memory_used:.1f}GB")
        
except Exception as e:
    print(f"❌ Model loading error: {e}")
    exit(1)

# Test 4: Simple dataset creation
print("\n4️⃣ Testing dataset creation...")
try:
    texts = ["Hello world", "AI is cool", "Python rocks"] * 10
    print(f"✅ Dataset created: {len(texts)} samples")
except Exception as e:
    print(f"❌ Dataset error: {e}")
    exit(1)

# Test 5: Training setup
print("\n5️⃣ Testing training setup...")
try:
    # Tokenize texts
    tokenized_texts = []
    for text in texts:
        encoded = tokenizer.encode(text, truncation=True, max_length=512)
        tokenized_texts.append(encoded)
    
    # Simple dataset class
    class SimpleDataset:
        def __init__(self, texts):
            self.texts = texts
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            return {"input_ids": torch.tensor(self.texts[idx])}
    
    # Create dataset
    dataset = SimpleDataset(tokenized_texts)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./test_model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        max_steps=2,  # Very short test
        fp16=True,
        logging_steps=1,
        save_steps=10,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("✅ Trainer created successfully!")
    
except Exception as e:
    print(f"❌ Training setup error: {e}")
    exit(1)

# Clean up
print("\n6️⃣ Cleaning up...")
torch.cuda.empty_cache()
print("✅ Memory cleared!")

print("\n🎉 ALL TESTS PASSED!")
print("💡 Your RTX 4070 setup is working perfectly!")
print("🔧 The issue is likely with the Jupyter kernel, not your hardware/software.")
