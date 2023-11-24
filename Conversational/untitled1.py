from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

mname = "facebook/blenderbot_small-90M"
modeldl = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizerdl = BlenderbotTokenizer.from_pretrained(mname)

modeldl.save_pretrained("./facebook--blenderbot_small-90M/")
tokenizerdl.save_pretrained("./facebook--blenderbot_small-90M/")

model = BlenderbotForConditionalGeneration.from_pretrained("./facebook--blenderbot_small-90M/")
tokenizer = BlenderbotTokenizer.from_pretrained("./facebook--blenderbot_small-90M/")

UTTERANCE = "My friends are cool but they eat too many carbs."
inputs = tokenizer([UTTERANCE], return_tensors="pt")
reply_ids = model.generate(**inputs)
print(tokenizer.batch_decode(reply_ids))
["<s> That's unfortunate. Are they trying to lose weight or are they just trying to be healthier?</s>"]