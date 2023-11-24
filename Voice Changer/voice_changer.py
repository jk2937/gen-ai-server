from transformers import AutoModelForCausalLM

models = [
        "0x3e9/Biden_RVC",
        "sail-rvc/SpongeBob_SquarePants__RVC_v2_"
        ]

model = AutoModelForCausalLM.from_pretrained(models[1])

from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Convert your audio input into a tensor
audio_input = processor(audio_input, sampling_rate=16_000, return_tensors="pt").input_values

# Pass the tensor through the model
outputs = model(input_values=audio_input)
# Convert the tensor back into an audio file
audio_output = processor.decode(outputs.logits.squeeze(0))

# Save the audio file
with open("audio_output.wav", "wb") as f:
        f.write(audio_output)

