#!/bin/bash
cd "$(dirname "$0")"
apt install -y python3
python3 -m venv venv
source venv/bin/activate
python3 -m pip install numpy pandas matplotlib torch torchvision opencv-python albumentations tqdm seaborn timm

