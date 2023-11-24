from diffusers import DiffusionPipeline
import torch
import datetime

model = "dreamlike-art/dreamlike-anime-1.0"
model_fold = "./dreamlike-art--dreamlike-anime-1-0--saved/"

save_fold = "./"

pipedl = DiffusionPipeline.from_pretrained(model)
pipedl.save_pretrained(model_fold)

