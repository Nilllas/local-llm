import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # You can replace this with your preferred local model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Main interaction loop
print("Chat with the local LLM (type 'quit' to exit):")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = generate_response(user_input)
    print("LLM:", response)

print("Chat ended.")