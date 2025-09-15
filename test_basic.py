#!/usr/bin/env python3
"""Basic functionality test for the fine-tuning project"""

import sys
import importlib

def test_imports():
    """Test basic imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test basic model loading"""
    print("\nTesting model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load a small model for testing
        model_name = "gpt2"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print(f"✓ {model_name} loaded successfully")
        
        # Test tokenization
        text = "Hello world"
        tokens = tokenizer(text, return_tensors="pt")
        print(f"✓ Tokenization works: '{text}' -> {tokens['input_ids'].shape[1]} tokens")
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_sample_generation():
    """Test text generation"""
    print("\nTesting text generation...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Generate text
        prompt = "The future of AI is"
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generation works!")
        print(f"   Input: '{prompt}'")
        print(f"   Output: '{generated_text}'")
        
        return True
    except Exception as e:
        print(f"✗ Text generation failed: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\nTesting utility functions...")
    
    try:
        sys.path.append('.')
        from scripts.utils import check_system_resources, validate_environment
        
        # Test system check
        resources = check_system_resources()
        print(f"✓ System resources check works")
        print(f"   CPU: {resources['cpu_percent']}%")
        print(f"   Memory: {resources['memory_percent']}%")
        
        # Test environment validation
        is_valid = validate_environment()
        print(f"✓ Environment validation works (valid: {is_valid})")
        
        return True
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False

def main():
    print("Fine-Tuning Project Basic Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_model_loading,
        test_sample_generation,
        test_utils
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic functionality tests passed!")
        print("The project is working correctly!")
    else:
        print("⚠️  Some tests failed - check dependencies")
    
    return passed == total

if __name__ == "__main__":
    import torch
    success = main()
    sys.exit(0 if success else 1)