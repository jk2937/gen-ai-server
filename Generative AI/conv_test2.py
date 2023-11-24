from transformers import pipeline, Conversation

model_path = 'openlm-research/open_llama_3b'
conversational_pipeline = pipeline("conversational", model=model_path)

conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
conversation_2 = Conversation("What's the last book you have read?")

conversational_pipeline([conversation_1, conversation_2])

conversation_1.add_user_input("Is it an action movie?")
conversation_2.add_user_input("What is the genre of this book?")

conversational_pipeline([conversation_1, conversation_2])
