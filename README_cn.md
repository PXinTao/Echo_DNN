<a href="https://discrete-distribution-networks.github.io/"><img src="https://img.shields.io/static/v1?label=Page&message=github.io&color=blue"></a>
<a href="https://arxiv.org/abs/2401.00036"><img src="https://img.shields.io/badge/arXiv-2401.00036-b31b1b.svg"></a>
<a href="https://openreview.net/forum?id=xNsIfzlefG"><img src="https://img.shields.io/badge/Accepted-ICLR%202025-brightgreen.svg"></a>
<a href="https://ddn-coloring-demo.diyer22.com/"><img src="https://img.shields.io/static/v1?label=Online&message=Demo&color=orange"></a>
<a href="https://huggingface.co/diyer22/ddn_asset/tree/main"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=Models&color=yellow"></a>
<a href="https://zhuanlan.zhihu.com/p/1935903948990047972"><img src="https://img.shields.io/badge/Blog-知乎-blue.svg"></a>

<!-- <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg"></a> -->


<div align="center">

<!-- <p style="font-size: 2em; font-weight: bold; margin-top: 20px; margin-bottom: 7px; line-height: 1;">离散分布网络</p> -->

# ☯ 离散分布网络

**全新的生成模型，带来独一无二的能力**

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
  <small><br>DDN 做二维概率密度估计 <a target="_blank" href="https://discrete-distribution-networks.github.io/2d-density-estimation-gif-with-10000-nodes-ddn.html"><small>[详情]</small></a><br>左：生成样本；右：概率密度GT</small>
</div>
<br>
</div>

<!-- ![SVG](docs/draft/header.svg) -->

本代码仓库为 ICLR 2025 论文的官方 PyTorch 实现.
## ▮ Introduction

<div align="center">
  <a target="_blank" href="https://discrete-distribution-networks.github.io/img/ddn-intro.png">
    <img src="https://discrete-distribution-networks.github.io/img/ddn-intro.png" style="height:250px;width:auto">
  </a>
  <br>
  <small>DDN 重建过程示意图</small>
</div>
<br>
  

我们提出了一种全新的生成模型：离散分布网络（Discrete Distribution Networks），简称 DDN。

DDN 采用一种简洁且独特的方法来建模目标分布，与主流生成模型截然不同：
1. 模型在一次前向过程中同时生成多个输出，而不仅仅是一个输出。
2. 利用这些一次性生成的多个输出来拟合训练数据的目标分布。
3. 这些输出共同表示一个离散分布，这也是“离散分布网络”名称的由来。

每个生成模型都有其独特的特性，DDN 也不例外。我们将重点介绍 DDN 的三个特性：
- 零样本条件生成 (Zero-Shot Conditional Generation, ZSCG)
- 树状结构的一维离散潜变量 (Tree-Structured 1D Discrete Latent)
- 完全的端到端可微分 (Fully End-to-End Differentiable)


参见：
- Paper: https://arxiv.org/abs/2401.00036  
- Page: https://discrete-distribution-networks.github.io/


![](https://discrete-distribution-networks.github.io/img/zscg.png)  
*DDN 的特色：零样本条件生成*

---

# DDN 代码教程
DDN 核心算法的代码实现单独放在了库 [**sddn**](https://github.com/diyer22/sddn) 中，以方便隔离和复用。此外，**`sddn`** 也包括了简单的实验 (2D toy data generation and MNIST example).

为了跑更复杂的 DDN 实验 (CIFAR、FFHQ)，我们在 [NVlabs/EDM](https://github.com/NVlabs/edm) 的 codebase 上整合了 DDN，从而诞生本 repo。所以本 repo 的用法和 `NVlabs/EDM` 几乎一致。


## ▮ Preparing
我们提供两种环境安装方案: `pip` 和 `docker`

### pip
请先根据你的 CUDA 版本安装对应的 [PyTorch](https://pytorch.org/get-started/locally/)
```bash
git clone https://github.com/DIYer22/discrete_distribution_networks.git
cd discrete_distribution_networks
pip install -r requirements.txt
```

### docker
首先安装好 [docker](https://docs.docker.com/get-started/) 和 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
```bash
git clone https://github.com/DIYer22/discrete_distribution_networks.git
cd discrete_distribution_networks
docker build --network=host -t diyer22/ddn .
# 进入 docker 环境
docker run -it --gpus all --net=host -v `pwd`:/workspace --user $(id -u):$(id -g) diyer22/ddn bash
```


## ▮ Inference
```bash
# cd discrete_distribution_networks
# 下载训练好的 CIFAR 权重
wget -O weights/cifar-ddn.pkl http://113.44.140.251:9000/ddn/weights/v15-00035-cifar10-32x32-cifar_blockn32_outputk64_chain.dropout0.05_fp32_goon.v15.22-shot-087808.pkl

# Inference 生成图片
python generate.py --debug 0 --batch=10 --seeds=0-99 --network weights/cifar-ddn.pkl
# Generating 100 images to "weights/generate"...
# Save vis to: weights/cifar-ddn-vis.png
```
更多 pre-trained 权重下载地址在 [weights/README.md](weights/README.md)


## ▮ Train
数据集准备流程和 `NVlabs/edm` 一样, 请根据 [NVlabs/edm#preparing-datasets](https://github.com/NVlabs/edm?tab=readme-ov-file#preparing-datasets) 来准备 training datasets 和 fid-refs

### 训练和测试
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
通过 `python train.py --help` 和 `python generate.py --help` 查看参数说明

### Conditional training for coloring task
```bash
torchrun --standalone --nproc_per_node=8 train.py --data=datasets/ffhq-256x256.zip \
  --lr=2e-4 --outdir training-runs --batch-gpu=64 --batch=512 --desc=ffhq256_cond.color \
  --chain-dropout=0.05 --max-outputk=64 --condition=color
```


### DDN coloring demo
```bash
# 会从 HuggingFace 自动下载权重（可能需要代理）
python zero_condition/gradio_coloring_demo.py
```
将会在本地部署如下 demo:

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

