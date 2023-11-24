from transformers import pipeline

model_name = "PygmalionAI/pygmalion-2-13b"
generator = pipeline("text-generation", model=model_name)

conversation_history = []

while True:
    user_input = input("User: ")
    conversation_history.append(f"<|user|>{user_input}<|model|>")
    response = generator(conversation_history, max_length=100)[0]["generated_text"]
    conversation_history.append(f"<|system|>{response}")
    print(f"Model: {response}")
