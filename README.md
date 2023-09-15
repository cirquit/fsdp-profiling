# Profiling FSDP with T5 and GPT2-xl

There are four branches:
* fsdp-nightly
* fsdp-1.13
* gpt2-nightly
* gpt2-1.13

Because we want to profile ColossialAIs (CAI) memory manager and general implementation, we need to make an apples-to-apples comparison with the same 1.13 PyTorch version, as CAI does not support PyTorch > 1.13 at the time of writing.

Due to the slightly differing implementation of FSDP between the PyTorch versions, we have special branches for each version.

# HowTo

* Install the same or higher CUDA driver for each experiment on the node
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
conda create -n fsdp-nightly python=3.9 -y
pip install 


