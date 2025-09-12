#!/usr/bin/env python3
"""
Simple test to identify what's causing the notebook cells to hang
"""

print("🧪 Testing simple operations...")

# Test 1: Simple list creation (what the hanging cell does)
print("\n1️⃣ Testing simple list creation...")
try:
    texts = ["Hello world", "AI is cool", "Python rocks"] * 10
    print(f"✅ Dataset ready: {len(texts)} samples")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Print with emojis
print("\n2️⃣ Testing emoji printing...")
try:
    print("📊 Creating instant dataset...")
    print("🔄 Next: Training setup")
    print("✅ Success!")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🎉 All simple tests passed!")
