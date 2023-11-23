#!/bin/bash
sudo apt install -y git 
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib torch torchvision opencv-python albumentations tqdm seaborn timm

