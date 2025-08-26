import ddn_utils
from ddn_utils import *
import boxx
import sddn
import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

# 从原generate.py导入采样器
from generate import ddn_sampler, StackedRandomGenerator, parse_int_list

@click.command()
@click.option("--network", "network_pkl", help="Network pickle filename", metavar="PATH|URL", type=str, required=True)
@click.option("--outdir", help="Where to save the output images", metavar="DIR", type=str)
@click.option("--seeds", help="Random seeds (e.g. 1,2,5-10)", metavar="LIST", type=parse_int_list, default="0-99", show_default=True)
@click.option("--batch", "max_batch_size", help="Maximum batch size", metavar="INT", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--class", "class_idx", help="Class label  [default: random]", metavar="INT", type=click.IntRange(min=0), default=None)
@click.option("--enable-annealed", help="Enable annealed selector for generation", metavar="BOOL", type=bool, default=True, show_default=True)

def enhanced_generate(network_pkl, outdir, seeds, max_batch_size, class_idx=None, 
                     enable_annealed=True, device=torch.device("cuda")):
    """Enhanced generation with annealed selector"""
    
    if outdir is None:
        outdir = os.path.abspath(os.path.join(network_pkl, "..", "enhanced_generate"))
        os.makedirs(outdir, exist_ok=True)
        
    # 可视化图像路径
    visp = network_pkl.replace(".pkl", "-enhanced-vis.png")
    if outdir.endswith("/enhanced_generate"):
        visp = visp
    else:
        visp = os.path.abspath(outdir) + "-enhanced-vis.png"

    dist.init()
    
    # 检查是否已存在
    skip_exist = len(seeds) in [100, 50000]
    if skip_exist and os.path.exists(visp) and len(seeds) == 100:
        if torch.distributed.get_rank() == 0:
            print("Enhanced vis exists:", visp)
        return

    # 加载训练配置
    dirr = os.path.dirname(network_pkl)
    training_options_json = os.path.join(dirr, "enhanced_training_options.json")
    if os.path.exists(training_options_json):
        train_kwargs = boxx.loadjson(training_options_json)
        boxx.cf.kwargs = train_kwargs.get("kwargs", {})
        learn_res = train_kwargs.get("kwargs", {}).get("learn_res", True)
        sddn.DiscreteDistributionOutput.learn_residual = learn_res

    # 计算批次
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 先行
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # 加载增强网络
    dist.print0(f'Loading enhanced network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        if network_pkl.endswith(".pkl"):
            net = pickle.load(f)["ema"].to(device)
        elif network_pkl.endswith(".pt"):
            net = torch.load(f)["net"].to(device)
        net = net.eval()

    # 其他rank跟随
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # 生成循环
    dist.print0(f'Generating {len(seeds)} enhanced images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # 选择潜在变量和标签
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
        )
        
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[batch_size], device=device)
            ]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # 生成图像
        sampler_kwargs = {"batch_seeds": batch_seeds}
        
        # 如果启用退火选择器，设置为推理模式
        if enable_annealed and hasattr(net, 'enable_annealed_selector'):
            # 在推理时可以禁用退火选择器，使用确定性选择
            for name in getattr(net, 'module_names', []):
                module = getattr(net, name)
                if hasattr(module, 'enable_annealed_selector'):
                    module.enable_annealed_selector = False

        images = ddn_sampler(net, latents, class_labels, **sampler_kwargs)
        
        if isinstance(images, dict):
            d, images = images, images["predict"]

        # 保存图像
        images_np = (
            (images * 127.5 + 128)
            .clip(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"enhanced_{seed:06d}.png")
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
            else:
                PIL.Image.fromarray(image_np, "RGB").save(image_path)

    # 完成
    torch.distributed.barrier()
    if dist.get_rank() == 0 and len(seeds) >= 9:
        example_paths = sorted(glob(outdir + "/*.??g"))[:100]
        make_vis_img(example_paths, visp)

    dist.print0("Enhanced generation done.")

if __name__ == "__main__":
    enhanced_generate()

# ==================================================
# File: setup_enhanced_ddn.py (完整设置脚本)
# ==================================================
"""Complete setup script for enhanced DDN with annealed selector"""

import os
import sys
import shutil

def setup_enhanced_ddn():
    """设置增强DDN环境"""
    
    current_dir = os.getcwd()
    print(f"Setting up Enhanced DDN in: {current_dir}")
    
    # 1. 创建ddn_annealed目录
    annealed_dir = os.path.join(current_dir, "ddn_annealed")
    os.makedirs(annealed_dir, exist_ok=True)
    print(f"Created directory: {annealed_dir}")
    
    # 2. 创建必要的文件
    files_to_create = {
        "__init__.py": '''from .selector import AnnealedSelector, AnnealedSelectorConfig
from .temperature import TemperatureScheduler, LayerTemperaturePolicy  
from .fitness import MultiObjectiveFitness, FitnessWeights, DiversityConfig
from .distributed import ddp_all_reduce_mean_std, broadcast_scalar, set_global_seed
''',
        
        "README.md": '''# DDN Annealed Selector

Enhanced Discrete Distribution Networks with Multi-Objective Annealed Selector for Cardiac Ultrasound Generation.

## Features
- Multi-objective fitness function (quality + diversity + minority)
- Layer-wise temperature scheduling
- Simulated annealing optimization
- Minority class protection for imbalanced medical data

## Usage
```python
from ddn_annealed import AnnealedSelector, AnnealedSelectorConfig

# Configure for cardiac ultrasound
config = AnnealedSelectorConfig(
    weights=FitnessWeights(quality=1.0, diversity=0.3, minority=0.7),
    temp_policy=LayerTemperaturePolicy(init_T=2.0, min_T=0.01)
)

selector = AnnealedSelector(config)
```
''',
    }
    
    for filename, content in files_to_create.items():
        filepath = os.path.join(annealed_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created: {filepath}")
    
    # 3. 提示用户下一步
    print("\n" + "="*60)
    print("Setup completed! Next steps:")
    print("1. Copy your fitness.py, selector.py, temperature.py, distributed.py to ddn_annealed/")
    print("2. Run enhanced training:")
    print("   python enhanced_train.py --data=path/to/cardiac_data --outdir=./cardiac_runs")
    print("3. Generate enhanced images:")
    print("   python enhanced_generate.py --network=cardiac_runs/xxxxx/enhanced-shot-xxxxxx.pkl")
    print("="*60)

if __name__ == "__main__":
    setup_enhanced_ddn()