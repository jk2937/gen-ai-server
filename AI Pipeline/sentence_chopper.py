from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

import os
import datetime

import imgen

randfolder = "./pipeline-" + str(datetime.datetime.now().strftime("%H%M%S%f"))
os.mkdir(randfolder)

text = '''A man walks into a bar and orders a drink. He notices a sign on the wall that says "Free Beer Tomorrow". He thinks to himself, "Wow, that's a great deal. I'll come back tomorrow and get some free beer."

The next day, he returns to the bar and orders a drink. He sees the same sign on the wall that says "Free Beer Tomorrow". He asks the bartender, "Hey, what's the deal? I thought you had free beer today."

The bartender smiles and says, "Sorry, buddy. The sign says free beer tomorrow. Come back tomorrow and you'll get some free beer."

The man is confused and annoyed, but he decides to give it another try. He comes back the following day and orders a drink. He looks at the sign on the wall that still says "Free Beer Tomorrow". He confronts the bartender, "Hey, this is ridiculous. You've been saying free beer tomorrow for the past two days. When are you going to give me some free beer?"

The bartender laughs and says, "You don't get it, do you? The sign always says free beer tomorrow. That means you'll never get any free beer. It's a joke, a prank, a hoax. Get it?"

The man is furious and says, "That's not funny. That's false advertising. That's a scam. That's a rip-off. I want to speak to your manager."

The bartender says, "Sure, no problem. He'll be here tomorrow."'''

sentences = text.split('.')
print(sentences)

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

imgen.init()

for i in range(len(sentences)):
    print(i)
    print(sentences[i])
    speech = synthesiser(sentences[i], forward_params={"speaker_embeddings": speaker_embedding})

    sf.write(randfolder + "/speech" + str(i) + ".wav", speech["audio"], samplerate=speech["sampling_rate"])
    imgen.startGeneration(sentences[i], path=randfolder + "/image" + str(i) + ".png")
