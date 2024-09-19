import transformers
import torch

model_id = "meta-llama/Llama-2-7b-chat-hf"  # Using a smaller model

device = torch.device("cpu")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    device=device
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"])