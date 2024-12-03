import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Step 1: Model Name and Dataset
model_name = "lmsys/vicuna-7b-v1.5"
dataset_path = "./dataset.json"

# Step 2: Check Device (MPS for macOS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Step 3: Load Tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Step 4: Load Model and Prepare for LoRA
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model = model.to(device)

# Prepare the model for LoRA fine-tuning
print("Preparing model for LoRA...")
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=2,  # LoRA rank
    lora_alpha=16,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target layers
    lora_dropout=0.1,  # Dropout rate
    bias="none",  # How biases are handled
    task_type="CAUSAL_LM"  # Task type for causal language modeling
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)

# Step 5: Load Dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files=dataset_path)

# Step 6: Tokenize Dataset
def preprocess_function(examples):
    inputs = [
        f"{instruction.strip()} {input_text.strip()}"
        for instruction, input_text in zip(examples["instruction"], examples["input"])
    ]
    outputs = [output_text.strip() for output_text in examples["output"]]
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(outputs, max_length=512, padding="max_length", truncation=True)["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Step 7: Training Arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save model and logs
    eval_strategy="no",  # Disable evaluation during training
    learning_rate=2e-5,  # Learning rate
    per_device_train_batch_size=1,  # Training batch size
    gradient_accumulation_steps=8,  # Gradients accumulation steps
    num_train_epochs=3,  # Number of epochs
    save_strategy="epoch",  # Save model at each epoch
    save_total_limit=2,  # Keep only the last 2 checkpoints
    logging_dir="./logs",  # Directory for logs
    bf16=True,  # Use bfloat16 (supported by MPS)
    push_to_hub=False,  # Do not push to Hugging Face Hub
    report_to="none"  # Disable logging platforms
)

# Step 8: Set Up Trainer
print("Setting up Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],  # Use training dataset
)

# Step 9: Start Training
print("Starting fine-tuning...")
trainer.train()

# Step 10: Save Model
print("Saving fine-tuned model...")
trainer.save_model("./fine_tuned_model")
print("Fine-tuning complete!")
