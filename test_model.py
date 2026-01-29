#!/usr/bin/env python3
"""Quick test script for the abliterated model."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading abliterated model...")
tokenizer = AutoTokenizer.from_pretrained("./output", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load base model architecture, then load our modified weights
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-1_8B-Chat",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load our abliterated weights
state_dict = torch.load("./output/weights.pt", map_location="cuda")
model.load_state_dict(state_dict, strict=False)

print("Model loaded! Type prompts to test (type 'quit' to exit)\n")

while True:
    prompt = input("📝 Prompt: ").strip()
    if prompt.lower() in ['quit', 'exit', 'q']:
        break
    if not prompt:
        continue
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n🤖 Response: {response}\n")
