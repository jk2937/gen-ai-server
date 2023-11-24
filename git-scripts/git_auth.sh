#!/bin/bash

# Create an ssh key
clear
ssh-keygen -t ed25519 -C "kaschakjonathan+github@gmail.com" -f ./gitkey
chmod 600 ./gitkey

sudo apt -y install keychain
keychain

eval $(ssh-agent -s)
ssh-add ./gitkey

# Display the ssh key so the user can copy it
clear
echo "Please copy this key: (Press enter to continue)"
cat ./gitkey.pub
read -p ""

# Instruct the user to paste it into github
clear
read -p "Please paste the key into github (Profile Image -> Settings -> SSH and GPG keys -> New SSH key) (Press enter to continue)"

# Configure Git to use the SSH key
git config --global user.email "kaschakjonathan+github@gmail.com"
git config --global user.name "jk2937"

ssh -vT git@github.com

git clone git@github.com:jk2937/hello-world.git
