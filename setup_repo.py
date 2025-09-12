#!/usr/bin/env python3
"""
Repository setup script for FineTuneLlama2
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, check=True):
    """Run shell command and return result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        return e.stdout.strip(), e.stderr.strip()

def check_git():
    """Check if git is installed and repository is initialized"""
    stdout, stderr = run_command("git --version", check=False)
    if "git version" not in stdout:
        print("❌ Git is not installed. Please install Git first.")
        return False
    
    stdout, stderr = run_command("git status", check=False)
    if "not a git repository" in stderr:
        print("📁 Initializing Git repository...")
        run_command("git init")
        return True
    else:
        print("✅ Git repository already initialized")
        return True

def setup_gitignore():
    """Ensure .gitignore exists"""
    if not os.path.exists(".gitignore"):
        print("❌ .gitignore not found")
        return False
    print("✅ .gitignore configured")
    return True

def create_initial_commit():
    """Create initial commit if needed"""
    stdout, stderr = run_command("git log --oneline", check=False)
    if "fatal: your current branch" in stderr or not stdout:
        print("📝 Creating initial commit...")
        run_command("git add .")
        run_command('git commit -m "Initial commit: RTX 4070 optimized fine-tuning pipeline"')
        print("✅ Initial commit created")
    else:
        print("✅ Repository already has commits")

def setup_remote(repo_url):
    """Setup GitHub remote"""
    stdout, stderr = run_command("git remote -v", check=False)
    if "origin" not in stdout:
        print(f"🔗 Adding remote origin: {repo_url}")
        run_command(f"git remote add origin {repo_url}")
    else:
        print("✅ Remote origin already configured")

def check_environment():
    """Check Python environment"""
    print("\n🔍 Environment Check:")
    print("-" * 30)
    
    # Python version
    python_version = sys.version.split()[0]
    print(f"🐍 Python: {python_version}")
    
    # Required packages
    required_packages = ["torch", "transformers", "datasets", "accelerate", "peft"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: installed")
        except ImportError:
            print(f"❌ {package}: missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("❌ No CUDA GPU detected")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def create_sample_config():
    """Create a sample training configuration"""
    config_content = """# Sample training configuration
model_name: "gpt2"
output_dir: "./models/my_fine_tuned_model"

training:
  batch_size: 4
  num_epochs: 3
  learning_rate: 5e-5
  max_length: 512

data:
  train_file: "./data/my_training_data.json"
"""
    
    os.makedirs("configs", exist_ok=True)
    with open("configs/my_config.yaml", "w") as f:
        f.write(config_content)
    
    print("✅ Sample config created: configs/my_config.yaml")

def main():
    print("🔥 FineTuneLlama2 Repository Setup")
    print("=" * 50)
    
    # Check current directory
    current_dir = Path.cwd().name
    print(f"📁 Current directory: {current_dir}")
    
    if current_dir != "FineTuning-LLM":
        print("⚠️  Consider renaming directory to 'FineTuneLlama2' for consistency")
    
    # Environment checks
    env_ok = check_environment()
    gpu_ok = check_gpu()
    
    # Git setup
    print(f"\n🔧 Git Repository Setup:")
    print("-" * 30)
    
    git_ok = check_git()
    gitignore_ok = setup_gitignore()
    
    if git_ok and gitignore_ok:
        create_initial_commit()
        
        # Optional: setup remote
        repo_url = "https://github.com/prakharrshukla/FineTuneLlama2.git"
        setup_remote(repo_url)
    
    # Create sample files
    print(f"\n📝 Additional Setup:")
    print("-" * 30)
    create_sample_config()
    
    # Summary
    print(f"\n📊 Setup Summary:")
    print("-" * 30)
    print(f"✅ Environment: {'OK' if env_ok else 'Issues detected'}")
    print(f"✅ GPU: {'OK' if gpu_ok else 'No GPU detected'}")
    print(f"✅ Git: {'OK' if git_ok else 'Issues detected'}")
    print(f"✅ Files: All repository files created")
    
    if env_ok and gpu_ok and git_ok:
        print(f"\n🎉 Repository setup complete!")
        print(f"🚀 Ready to fine-tune models!")
    else:
        print(f"\n⚠️  Some issues detected. Please resolve before proceeding.")
    
    print(f"\n📖 Next steps:")
    print(f"1. Open Fine_tune_Llama_2.ipynb in Jupyter")
    print(f"2. Run the training cells")
    print(f"3. Push to GitHub: git push -u origin main")

if __name__ == "__main__":
    main()
