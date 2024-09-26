from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, LlamaForCausalLM, LlamaTokenizer
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def model_exists(model_dir):
    model_files = ['pytorch_model.bin', 'config.json', 'tokenizer.json', 'vocab.json']
    return all(os.path.exists(os.path.join(model_dir, f)) for f in model_files)

def download_and_save_model(model_class, tokenizer_class, model_name, model_dir):
    if model_exists(model_dir):
        logging.info(f"Model already exists at {model_dir}. Skipping download.")
        return
    
    os.makedirs(model_dir, exist_ok=True)

    try:
        logging.info(f"Downloading {model_name} model...")
        tokenizer = tokenizer_class.from_pretrained(model_name)

        # Load model on CPU in 8-bit format
        model = model_class.from_pretrained(model_name, load_in_8bit=True, device_map="cpu")

        logging.info(f"Saving {model_name} model to {model_dir}...")
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

        logging.info(f"{model_name} model saved to {model_dir} successfully.\n")
    except Exception as e:
        logging.error(f"Failed to download and save model {model_name}: {e}")

if __name__ == "__main__":
    base_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(base_dir, exist_ok=True)

    gpt2_dir = os.path.join(base_dir, 'gpt2')
    download_and_save_model(GPT2LMHeadModel, GPT2Tokenizer, 'gpt2', gpt2_dir)

    gpt_neo_dir = os.path.join(base_dir, 'gpt-neo')
    download_and_save_model(GPTNeoForCausalLM, GPT2Tokenizer, 'EleutherAI/gpt-neo-125M', gpt_neo_dir)

    llama2_dir = os.path.join(base_dir, 'llama2')
    download_and_save_model(LlamaForCausalLM, LlamaTokenizer, 'meta-llama/Llama-2-7b-hf', llama2_dir)
