#!/bin/bash

# Configure Git to use the SSH key
git config --global user.email "kaschakjonathan+github@gmail.com"
git config --global user.name "jk2937"

ssh -vT git@github.com

