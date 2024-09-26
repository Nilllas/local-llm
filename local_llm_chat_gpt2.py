import os
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model and tokenizer from your local directory
model_path = "models/gpt2/"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set the padding token to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
model.config.pad_token_id = tokenizer.eos_token_id  # Set pad_token_id in model config

# Initialize chat history
chat_history = []

def generate_response(user_input):
    # Append the user input to the chat history
    chat_history.append(f"You: {user_input}")
    input_text = "\n".join(chat_history)  # Combine chat history for context

    # Tokenize input and create attention masks
    inputs = tokenizer.encode_plus(input_text, return_tensors="pt", truncation=True, padding=True)

    # Get input ids and attention mask
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate response using input_ids and attention_mask
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        num_return_sequences=1,
        temperature=0.7,  # Controls randomness (higher is more random)
        top_k=50,         # Limits the sampling pool to the top k tokens
        top_p=0.95,       # Cumulative probability for sampling
        do_sample=True    # Enables sampling instead of greedy decoding
    )
    
    # Decode the output and update chat history
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    chat_history.append(f"LLM: {response}")
    
    return "\n".join(chat_history)

# Create the Gradio interface using the new syntax
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Enter your message:"),
    outputs=gr.Textbox(label="Chat History"),
    title="Local GPT-2 Chat",
    description="Chat with your local GPT-2 model."
)

iface.launch()
