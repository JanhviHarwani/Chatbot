import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Set device for MPS or CPU
device = torch.device("cpu")  # Use CPU explicitly for MPS compatibility issues
print(f"Using device: {device}")

# Model and Tokenizer
model_name = "lmsys/vicuna-7b-v1.5"  
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for CPU compatibility
    low_cpu_mem_usage=True,  # Optimize CPU memory usage
    device_map="auto",  # Offload some layers to CPU automatically
)
model = model.to(device)  # Send model explicitly to CPU

# LoRA Configuration
lora_config = LoraConfig(
    r=8,  # Low-rank updates
    lora_alpha=16,  # Reduced scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention heads
    lora_dropout=0.1,  # Slight dropout for regularization
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Dataset Loading and Tokenization
print("Loading dataset...")
dataset = load_dataset("json", data_files="./dataset.json")

# Set tokenizer padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"]
    return tokens

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",  # Disable evaluation to save memory
    learning_rate=1e-5,  # Lower learning rate
    per_device_train_batch_size=1,  # Minimize memory usage
    gradient_accumulation_steps=16,  # Simulate larger batch sizes
    num_train_epochs=1,  # Fewer epochs to test feasibility
    save_strategy="no",  # Avoid saving during training
    logging_dir="./logs",
)

# Trainer
print("Setting up Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

print("Starting training...")
trainer.train()

print("Saving fine-tuned model...")
trainer.save_model("./fine_tuned_vicuna")
print("Fine-tuning complete!")
