# Profiling FSDP with T5 and GPT2-xl

There are four branches:
* `fsdp-nightly`
* `fsdp-1.13`
* `gpt2-nightly`
* `gpt2-1.13`

Because we want to profile ColossialAIs (CAI) memory manager and general implementation, we need to make an apples-to-apples comparison with the same 1.13 PyTorch version, as CAI does not support PyTorch > 1.13 at the time of writing.

Due to the slightly differing implementation of FSDP between the PyTorch versions, we have special branches for each version.

# HowTo

* Install the same or higher CUDA driver for each experiment on the node.
```bash
sudo apt update
sudo apt install wget libxml2 build-essential psmisc file rsync tmux git linux-headers-`uname -r` -y
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
sudo sh cuda_12.2.2_535.104.05_linux.run --silent
```
```bash
sudo apt update
sudo apt install wget libxml2 build-essential psmisc file rsync tmux git linux-headers-`uname -r` -y
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run --silent
```

* Install miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b
miniconda3/bin/conda init
source ~/.bashrc
```
* Clone the repository and setup the first environment
```bash
cd fsdp-profiling
conda create -n nightly python=3.9 -y
conda activate nightly
pip install -r requirements-nightly.txt
./install-nightly-torch.sh
```
* Do the same for 1.13 environment
```bash
conda create -n stable-1.13 python=3.7 -y
conda activate stable-1.13
pip install -r requirements-1.13.txt
./install-1.13-torch.sh
```




