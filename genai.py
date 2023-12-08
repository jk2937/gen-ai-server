from diffusers import DiffusionPipeline
import torch
import datetime

pipe = None

def init():
    global pipe
    model = "dreamlike-art/dreamlike-anime-1.0"
    save_fold = "./img/"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    if device == "cuda":
        pipe = DiffusionPipeline.from_pretrained(model, safety_checker=None, torch_dtype=torch.float16)
    else:
        pipe = DiffusionPipeline.from_pretrained(model, safety_checker=None, torch_dtype=torch.float32)
    pipe = pipe.to(device)

    print("initialized")

def startGeneration(prompt, x_res, y_res, path):
    global pipe
    image = pipe(prompt, width=x_res, height=y_res, negative_prompt="lowres, bad anatomy").images[0]
    image.save(path)
