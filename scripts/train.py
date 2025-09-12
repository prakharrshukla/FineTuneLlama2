#!/usr/bin/env python3
"""
Fine-tuning script for language models
Optimized for RTX 4070 and similar GPUs
"""

import os
import json
import time
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Custom dataset for text data"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def load_data(data_path):
    """Load training data from file"""
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data['texts'] if 'texts' in data else data
    else:
        with open(data_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

def setup_model_and_tokenizer(model_name, use_lora=False):
    """Setup model and tokenizer"""
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Apply LoRA if requested
    if use_lora:
        logger.info("Applying LoRA configuration")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def train_model(model, dataloader, num_epochs=3, learning_rate=5e-5):
    """Train the model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    num_training_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    total_loss = 0
    step = 0
    start_time = time.time()
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        valid_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss
            
            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning(f"NaN loss at epoch {epoch}, batch {batch_idx}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Track metrics
            epoch_loss += loss.item()
            total_loss += loss.item()
            step += 1
            valid_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        # Log epoch results
        if valid_batches > 0:
            avg_epoch_loss = epoch_loss / valid_batches
            logger.info(f"Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.4f}")
    
    # Training summary
    end_time = time.time()
    duration = (end_time - start_time) / 60
    avg_loss = total_loss / step if step > 0 else float('inf')
    
    logger.info(f"Training completed in {duration:.1f} minutes")
    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"Total training steps: {step}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune language models")
    parser.add_argument("--model_name", type=str, default="gpt2", 
                       help="Model name from Hugging Face")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data file")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model",
                       help="Output directory for saved model")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for parameter-efficient fine-tuning")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    texts = load_data(args.data_path)
    logger.info(f"Loaded {len(texts)} training samples")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.use_lora)
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Train model
    model = train_model(
        model, 
        dataloader, 
        args.num_epochs, 
        args.learning_rate
    )
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
