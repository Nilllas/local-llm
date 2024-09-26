import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import os

def generate_text(prompt, model_dir='~/localllm/models/gpt-neo', max_length=100):
    # Expand the '~' to the full home directory path
    model_dir = os.path.expanduser(model_dir)

    # Load GPT-Neo model and tokenizer from the specified directory
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPTNeoForCausalLM.from_pretrained(model_dir)

    # Set the pad token to the same as eos token
    tokenizer.pad_token = tokenizer.eos_token

    # Ensure the model is set to run on CPU
    device = torch.device("cpu")
    model.to(device)

    # Encode the input text and send the tensors to the CPU
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate text with adjusted sampling parameters
    with torch.no_grad():  # Disable gradient calculations for faster inference
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=max_length,
            attention_mask=inputs['attention_mask'],  # Pass the attention mask
            pad_token_id=tokenizer.pad_token_id,  # Use the pad token id
            temperature=0.7,  # Lower value for more deterministic outputs
            top_k=50,         # Consider only the top k tokens for sampling
            top_p=0.95,       # Cumulative probability for nucleus sampling
            do_sample=True    # Enable sampling
        )

    # Decode the generated output into text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # You can modify this prompt for your demo
    prompt = "Once upon a time"
    print(f"Prompt: {prompt}")

    # Call the generate_text function and print the result
    generated_text = generate_text(prompt)
    print(f"Generated text: {generated_text}")
