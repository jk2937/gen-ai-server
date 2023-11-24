from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "PygmalionAI/pygmalion-2-13b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name) 

def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=500, do_sample=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

user_input = ""
while user_input != "exit":
    user_input = input("User: ")
    prompt = f"<|user|>{user_input}<|model|>"
    response = generate_response(prompt)
    print(f"PygmalionAI: {response}")
