'''
https://www.kaggle.com/code/georgesaavedra/blenderbot-2-0-vs-dialogpt
'''

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

models = [
        "facebook/blenderbot-400M-distill",
        "PygmalionAI/pygmalion-2-13b"
        ]

model_blender = BlenderbotForConditionalGeneration.from_pretrained(models[1])
tokenizer_blender = BlenderbotTokenizer.from_pretrained(models[1])

print("Type \"q\" to quit")
while True:
        message = input("MESSAGE: ")
        if message in ["", "q", "quit"]:
            break
        inputs = tokenizer_blender([message], return_tensors='pt')
        reply_ids = model_blender.generate(**inputs)
        print(f"Blenderbot 2.0 response:     {tokenizer_blender.batch_decode(reply_ids, skip_special_tokens=True)[0]}")

