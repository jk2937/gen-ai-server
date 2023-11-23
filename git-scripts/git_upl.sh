#!/bin/bash
cd "$(dirname "$0")"
cd hello-world
git add .
git commit
git push
git status
