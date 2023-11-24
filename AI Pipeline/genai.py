from diffusers import DiffusionPipeline
import torch
import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
dtype = torch.float16
if device == "gpu":
    dtype = torch.float32
pipe = DiffusionPipeline.from_pretrained(model_fold, safety_checker=True, torch_dtype=dtype) # set to float16 if gpu
pipe = pipe.to(device)

prompts = []
prompts.append("cringe chatgpt ai nerd hype monster")
prompts.append("the ai hype train")

for i in range(138, 500):
    print("i:", i)
    for j in range(len(prompts)):
        print("j:", j)
        image = pipe(prompts[j], width=504 * 1, height=896 * 1, negative_prompt="lowres, bad anatomy").images[0]
        image.save(save_fold + str(datetime.datetime.now().strftime("%H%M%S%f")) + "--" + str(j) + "--" + ".png")
