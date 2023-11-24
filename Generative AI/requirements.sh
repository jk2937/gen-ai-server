sudo apt install python3-pip
sudo apt install python3-venv
mkdir gen-ai-env
sudo python3 -m venv ./gen-ai-env
source ./gen-ai-env/bin/activate
sudo ./gen-ai-env/bin/python3 -m pip install diffusers torch transformers accelerate
sudo apt install nvidia-driver
sudo gen-ai-env/bin/python3 -m pip install SentencePiece protobuf
sudo apt-add-repository --component non-free
sudo apt install nvidia-detect
