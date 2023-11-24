from diffusers import DiffusionPipeline
import torch
import datetime

defmodel = "dreamlike-art/dreamlike-photoreal-2.0"  # "stabilityai/stable-diffusion-xl-base-1.0"

def init(force_cpu=False, model=defmodel):
    global pipe
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if force_cpu:
        device = 'cpu'
    print(device)
    
    if device == "cuda":
        pipe = DiffusionPipeline.from_pretrained(model, safety_checker=None, torch_dtype=torch.float16)
    else:
        pipe = DiffusionPipeline.from_pretrained(model, safety_checker=None, torch_dtype=torch.float32)
    pipe = pipe.to(device)

    print("initialized")

def startGeneration(prompt, x_res=504, y_res=896, path="./image.png"):
    global pipe
    image = pipe(prompt, width=x_res, height=y_res, negative_prompt="lowres, bad anatomy").images[0]
    image.save(path)
