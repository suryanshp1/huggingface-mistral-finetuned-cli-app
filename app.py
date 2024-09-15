from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv
import torch
import os

load_dotenv()  # take environment variables from .env.

login(token=os.getenv("HF_TOKEN"))


def main(prompt: str):
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", quantization_config=quantization_config, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = [
        {"role": "user", "content": "You are a psycologist named John. You are expert in human psycology and behaviour. As a psycologist try to detect the anomly and it solution."},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": prompt}
    ]

    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    generated_ids = model.generate(model_inputs, max_new_tokens=1024, do_sample=True)
    result = tokenizer.batch_decode(generated_ids)[0]
    return result


if __name__ == "__main__":
    result = main("How to tackle a stressed situation ?")
    print(result)