# Fine-Tune GPT Models

Simple fine-tuning for GPT-2, Llama 2, and other language models. Optimized for RTX 4070.

## What this does

- Fine-tune language models on your own data
- Works well on RTX 4070 (8GB VRAM)
- Includes Jupyter notebook tutorial
- Ready-to-use Python scripts

## Quick setup

```bash
git clone https://github.com/prakharrshukla/FineTuneLlama2.git
cd FineTuneLlama2
pip install torch transformers datasets accelerate
```

## Usage

1. Open the Jupyter notebook: `Fine_tune_Llama_2.ipynb`
2. Follow the step-by-step guide
3. Train in 5-15 minutes on RTX 4070

Or use the Python scripts:
```bash
python scripts/train.py --data_path your_data.txt --model_name gpt2
python scripts/inference.py --model_path ./fine_tuned_model --prompt "Hello"
```

## Requirements

- Python 3.8+
- NVIDIA GPU with 6GB+ VRAM
- CUDA 11.8+

## Training times on RTX 4070

- GPT-2 (124M): 2-5 minutes
- GPT-2 Medium (355M): 10-15 minutes  
- Llama 2 7B (with LoRA): 30-60 minutes

## Common issues

**Out of memory?** Reduce batch size or use shorter sequences
**Slow training?** Enable mixed precision with fp16=True
**Package errors?** Create a fresh virtual environment

## License

MIT License - feel free to use and modify.