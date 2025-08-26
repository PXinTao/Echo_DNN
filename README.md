<a href="https://discrete-distribution-networks.github.io/"><img src="https://img.shields.io/static/v1?label=Page&message=github.io&color=blue"></a>
<a href="https://arxiv.org/abs/2401.00036"><img src="https://img.shields.io/badge/arXiv-2401.00036-b31b1b.svg"></a>
<a href="https://openreview.net/forum?id=xNsIfzlefG"><img src="https://img.shields.io/badge/Accepted-ICLR%202025-brightgreen.svg"></a>
<a href="https://ddn-coloring-demo.diyer22.com/"><img src="https://img.shields.io/static/v1?label=Online&message=Demo&color=orange"></a>
<a href="https://huggingface.co/diyer22/ddn_asset/tree/main"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=Models&color=yellow"></a>
<a href="README_cn.md"><img src="https://img.shields.io/badge/Language-中文-lightgrey.svg"></a>
<a href="https://github.com/Discrete-Distribution-Networks/Discrete-Distribution-Networks.github.io/blob/main/blog_en.md"><img src="https://img.shields.io/badge/Blog-English-00ab6c.svg"></a>




<div align="center">

<!-- <p style="font-size: 2em; font-weight: bold; margin-top: 20px; margin-bottom: 7px; line-height: 1;">离散分布网络</p> -->

# Discrete Distribution Networks
**A novel generative model with simple principles and unique properties**

<div style="margin-top:px;font-size:px"> 
  <a target="_blank" href="https://www.stepfun.com/">
    <img src="https://discrete-distribution-networks.github.io/img/logo-StepFun.png" style="height:20px">
  </a>
    &nbsp;
  <a target="_blank" href="https://en.megvii.com/megvii_research">
    <img src="https://discrete-distribution-networks.github.io/img/logo-Megvii.png" style="height:20px">
  </a>
</div>


<br>
<div align="center">
  <a target="_blank" href="https://discrete-distribution-networks.github.io/2d-density-estimation-gif-with-10000-nodes-ddn.html">
    <img src="https://discrete-distribution-networks.github.io/img/frames_bin100_k2000_itern1800_batch40_framen96_2d-density-estimation-DDN.gif" style="height:">
  </a>
  <small><br>2D density estimation <a target="_blank" href="https://discrete-distribution-networks.github.io/2d-density-estimation-gif-with-10000-nodes-ddn.html"><small>[details]</small></a><br>Left: All samples; Right: GT density</small>
</div>
<br>
</div>

<!-- ![SVG](docs/draft/header.svg) -->

This code repository is the official PyTorch implementation of the ICLR 2025 paper.
## ▮ Introduction

<div align="center">
  <a target="_blank" href="https://discrete-distribution-networks.github.io/img/ddn-intro.png">
    <img src="https://discrete-distribution-networks.github.io/img/ddn-intro.png" style="height:250px;width:auto">
  </a>
  <br>
  <small>DDN reconstruction process diagram</small>
</div>
<br>

We introduce a novel generative model: Discrete Distribution Networks (DDN).

DDN employs a simple yet unique approach to model the target distribution, distinctly different from mainstream generative models:
1. The model generates multiple outputs simultaneously in a single forward pass, rather than just one output.
2. It utilizes these simultaneously generated outputs to fit the target distribution of the training data.
3. These outputs collectively represent a discrete distribution, which is the origin of the name "Discrete Distribution Networks."

Every generative model has its unique characteristics, and DDN is no exception. We will highlight three key features of DDN:
- Zero-Shot Conditional Generation without gradient computation.
- Tree-Structured 1D Discrete Latent Variables.
- Fully End-to-End Differentiable.

See:
- Paper: https://arxiv.org/abs/2401.00036  
- Page: https://discrete-distribution-networks.github.io/


![](https://discrete-distribution-networks.github.io/img/zscg.png)  
*DDN's feature: Zero-Shot Conditional Generation*

---

# DDN Code Tutorial
The core algorithm implementation of DDN is separately housed in the library [**sddn**](https://github.com/diyer22/sddn) for isolation and reuse. Additionally, **`sddn`** includes simple experiments (2D toy data generation and MNIST example).

To run more complex DDN experiments (CIFAR, FFHQ), we integrated DDN with the [NVlabs/EDM](https://github.com/NVlabs/edm) codebase, resulting in this repository. Therefore, the usage of this repo is almost identical to `NVlabs/EDM`.


## ▮ Preparing
We provide two environment setup options: `pip` and `docker`

### pip
Please install the appropriate [PyTorch](https://pytorch.org/get-started/locally/) version according to your CUDA version first
```bash
git clone https://github.com/DIYer22/discrete_distribution_networks.git
cd discrete_distribution_networks
pip install -r requirements.txt
```

### docker
First install [docker](https://docs.docker.com/get-started/) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
```bash
git clone https://github.com/DIYer22/discrete_distribution_networks.git
cd discrete_distribution_networks
docker build --network=host -t diyer22/ddn .
# Enter the docker environment
docker run -it --gpus all --net=host -v `pwd`:/workspace --user $(id -u):$(id -g) diyer22/ddn bash
```


## ▮ Inference
```bash
# cd discrete_distribution_networks
# Download pre-trained CIFAR weights
wget -O weights/cifar-ddn.pkl http://out.diyer22.com:9000/ddn/weights/v15-00035-cifar10-32x32-cifar_blockn32_outputk64_chain.dropout0.05_fp32_goon.v15.22-shot-087808.pkl
# Backup URL: http://113.44.140.251:9000/ddn/weights/v15-00035-cifar10-32x32-cifar_blockn32_outputk64_chain.dropout0.05_fp32_goon.v15.22-shot-087808.pkl

# Generate images using inference
python generate.py --debug 0 --batch=10 --seeds=0-99 --network weights/cifar-ddn.pkl
# Generating 100 images to "weights/generate"...
# Save vis to: weights/cifar-ddn-vis.png
```
More pre-trained weights download links are available in [weights/README.md](weights/README.md)


## ▮ Train
The dataset preparation process is the same as `NVlabs/edm`. Please follow [NVlabs/edm#preparing-datasets](https://github.com/NVlabs/edm?tab=readme-ov-file#preparing-datasets) to prepare training datasets and fid-refs

### Training and evaluation

```bash
# train CIFAR10 DDN on 8 x NVIDIA A100(80GB)
torchrun --standalone --nproc_per_node=8 train.py --data datasets/cifar10-32x32.zip \
  --outdir training-runs --batch-gpu=256 --batch=2048 --desc=task_name \
  --max-blockn=32 --chain-dropout=0.05 --max-outputk=64

# evaluation using 2 GPUs, if len(seeds)==50000 will auto calculating FID.
torchrun --standalone --nproc_per_node=2 generate.py --seeds=0-49999 --subdirs \
  --batch 128 --fid_ref fid-refs/cifar10-32x32.npz \
  --network training-runs/00000-cifar10-32x32-task_name/shot-200000.pkl
# Calculating FID...
# 51.856
# Saving example images tar to: xxx/sample-example.tar
# Save vis to: xxx/vis.png
```
View parameter descriptions via `python train.py --help` and `python generate.py --help`

### Conditional training for coloring task
```bash
torchrun --standalone --nproc_per_node=8 train.py --data=datasets/ffhq-256x256.zip \
  --lr=2e-4 --outdir training-runs --batch-gpu=64 --batch=512 --desc=ffhq256_cond.color \
  --chain-dropout=0.05 --max-outputk=64 --condition=color
```

### DDN coloring demo
```bash
# The weights will be automatically downloaded from HuggingFace
python zero_condition/gradio_coloring_demo.py
```
The following demo will be deployed locally:

<div align="center">
  <a target="_blank" href="https://ddn-coloring-demo.diyer22.com/">
    <img src="https://discrete-distribution-networks.github.io/img/astronaut_coloring.gif" style="height:">
  </a>
  <br>DDN coloring demo <a target="_blank" href="https://ddn-coloring-demo.diyer22.com/">[online]</a>
</div>

## ▮ Misc
### TODO
- [x] Dockerfile
- [x] Inference
- [x] Train
- [x] [MNIST and 2D toy data](https://github.com/DIYer22/sddn?tab=readme-ov-file#-toy-example-for-2d-density-estimation)
- [x] Zero-Shot Conditional Generation
- [x] Upload weights to HF
- [x] Online demo of ZSCG on coloring task
<!-- - [ ] Support CPU inference -->
### Citation
```bibtex
@inproceedings{yang2025discrete,
  title     = {Discrete Distribution Networks},
  author    = {Lei Yang},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=xNsIfzlefG}}
``` 