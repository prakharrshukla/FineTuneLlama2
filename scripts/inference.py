#!/usr/bin/env python3
"""Inference script for fine-tuned models"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Use device_map only if accelerate is available and CUDA is available
    try:
        import accelerate
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, num_return_sequences=1):
    device = next(model.parameters()).device
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs, max_length=max_length, temperature=temperature,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_return_sequences, repetition_penalty=1.1
        )
    
    generated_texts = []
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts

def interactive_mode(model, tokenizer):
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            responses = generate_text(model, tokenizer, prompt)
            print(f"\nBot: {responses[0]}")
        except KeyboardInterrupt:
            break
    print("\nGoodbye!")

def batch_inference(model, tokenizer, prompts_file, output_file):
    logger.info(f"Processing prompts from {prompts_file}")
    
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    results = []
    for i, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        responses = generate_text(model, tokenizer, prompt)
        results.append({"prompt": prompt, "response": responses[0]})
    
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--prompt", type=str, help="Single prompt for generation")
    parser.add_argument("--prompts_file", type=str, help="File containing multiple prompts")
    parser.add_argument("--output_file", type=str, default="results.json", help="Output file for batch results")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model_path)
    
    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.prompts_file:
        batch_inference(model, tokenizer, args.prompts_file, args.output_file)
    elif args.prompt:
        responses = generate_text(
            model, tokenizer, args.prompt, args.max_length, args.temperature
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {responses[0]}")
    else:
        print("Please provide --prompt, --prompts_file, or use --interactive mode")

if __name__ == "__main__":
    main()
