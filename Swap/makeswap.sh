sudo fallocate -l 30G ~/swapfile2
sudo chmod 600 ~/swapfile22
sudo mkswap ~/swapfile2
sudo swapon ~/swapfile2
sudo swapon --show
