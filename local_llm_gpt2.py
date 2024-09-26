import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

def generate_text(prompt, model_dir='~/localllm/models/gpt2', max_length=100):

    # Expand the '~' to the full home directory path
    model_dir = os.path.expanduser(model_dir)

    # Load GPT-2 model and tokenizer from the specified directory
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    # Ensure the model is set to run on CPU
    device = torch.device("cpu")
    model.to(device)

    # Encode the input text and send the tensors to the CPU
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # GPT-2 does not have a padding token by default, so we set it manually
    pad_token_id = tokenizer.eos_token_id  # Set pad token to eos_token_id (50256)
    
    # Generate text with attention mask and the pad_token_id set
    with torch.no_grad():  # Disable gradient calculations for faster inference
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=max_length,
            attention_mask=inputs['attention_mask'],  # Set attention mask
            pad_token_id=pad_token_id  # Set pad token
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
