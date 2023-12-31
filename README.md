# Profiling FSDP with T5 and GPT2-XL

Code to reproduce the findings from this blogpost.

Because we want to profile ColossialAIs (CAI) memory manager and the PyTorch implementation, we need to make an apples-to-apples comparison with the same 1.13 PyTorch version, as CAI does not support PyTorch > 1.13 at the time of writing. Due to the slightly differing implementation of FSDP between the PyTorch versions, we have special branches for each version.

* [Original FSDP-T5 code](https://github.com/pytorch/workshops/tree/master/FSDP_Workshop)
* [Original GPT2 CAI code](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt)

The repository is organized in five branches for each configuration:
* `t5-nightly`
* `t5-1.13`
    - disabled `use_orig_params=True` in FSDP as it's not available
    - some gpu metrics are missing, like `requested_bytes.all.current`
* `gpt2-nightly`
* `gpt2-1.13`
    - disabled `use_orig_params=True` in FSDP as it's not available
    - some gpu metrics are missing, like `requested_bytes.all.current`
* `cai`
    - also uses `torch==1.13`

### Prepare the environment

* Install the same or higher CUDA driver for each experiment on the node.
  - PyTorch 1.13 -> CUDA 11.7
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
```
  - PyTorch Nightly (2.x)  -> CUDA 12.2
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
```

```bash
sudo apt update
sudo apt install wget libxml2 build-essential psmisc file rsync tmux git linux-headers-`uname -r` -y
sudo sh cuda_XXX_XXX.run --silent
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

* And lastly for the `cai` environment
```bash
conda create -n cai python=3.7 -y
conda activate cai
cd cai
pip install -r requirements.txt
```

* And finally, let it rip with the configuration that you want in `cfg/benchmark.py` for all FSDP runs, or with `cai` you have to modify the bash runner
* `./run_benchmark.sh`
