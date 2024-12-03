import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Set Device (MPS, CUDA, or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Tokenizer
model_name = "meta-llama/Llama-3.2-3B"  # Base model name
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the Fine-Tuned Model
print("Loading fine-tuned model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
).to(device)

# Load the LoRA-adapted model
fine_tuned_model_path = "./fine_tuned_llama"  # Path to your fine-tuned LoRA model
model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
model.to(device)

# Function to Generate Responses
def generate_text(prompt, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.2, max_length=None):
    """
    Generate text from the fine-tuned model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length if max_length else len(inputs["input_ids"][0]) + 300,  # Extend the limit
        min_length=len(inputs["input_ids"][0]) + 50,  # Ensure meaningful output
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
# Gradio Interface
def chatbot_interface(user_input):
    """
    Interface function for chatbot.
    Args:
        user_input (str): Input from the user.
    Returns:
        str: Response from the chatbot model.
    """
    return generate_text(user_input)

# Define Gradio UI
interface = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(label="Your Question", placeholder="Ask something..."),
    outputs=gr.Textbox(label="Bot's Response"),
    title="Chatbot Interface",
    description="Ask questions to the chatbot powered by a fine-tuned model.",
)

# Launch the Chatbot
if __name__ == "__main__":
    interface.launch()
