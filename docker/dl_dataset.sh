#!/bin/bash
git clone https://github.com/smaranjitghose/Big_Cat_Classifier.git
mkdir Images Images/Cheetahs Images/Lions
cp -r Big_Cat_Classifier/data/cheetah/* Images/Cheetahs/
cp -r Big_Cat_Classifier/data/lion/* Images/Lions/
rm Images/Cheetahs/114.jpg Images/Cheetahs/135.jpg Images/Cheetahs/125.jpg
