import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Step 1: Set Device (MPS, CUDA, or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Load the Tokenizer
model_name = "meta-llama/Llama-3.2-3B"  # Base model name
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Load the Fine-Tuned Model
print("Loading fine-tuned model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # Automatically map model to available devices
).to(device)

# Load the LoRA-adapted model
fine_tuned_model_path = "./fine_tuned_llama"  # Path to your fine-tuned LoRA model
model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
model.to(device)

# Step 4: Generate Text
def generate_text(prompt, max_length=1000, temperature=0.7, top_p=0.9):
    """
    Generate text from the fine-tuned model.
    Args:
        prompt (str): Input text prompt.
        max_length (int): Maximum length of the generated sequence.
        temperature (float): Sampling temperature for diversity.
        top_p (float): Nucleus sampling probability.
    Returns:
        str: Generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example Usage
if __name__ == "__main__":
    # Provide a prompt for inference
    prompt = "How do I make a document accessible to partially blind students? \n Answer: "
    print("Generating text...")
    output = generate_text(prompt)
    
    # Print and save the output
    print(f"Prompt: {prompt}")
    print(f"Generated Text: {output}")
    
    # Save the output to a text file
    output_file = "generated_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Generated Text: {output}\n")
    
    print(f"Output saved to {output_file}")