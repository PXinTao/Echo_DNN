import ddn_utils
import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training.enhanced_training_loop import enhanced_training_loop
from ddn_annealed import AnnealedSelectorConfig, FitnessWeights, DiversityConfig, LayerTemperaturePolicy

import warnings
warnings.filterwarnings("ignore", "Grad strides do not match bucket view strides")

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges

@click.command()
# 主要选项（与原train.py相同）
@click.option("--outdir", help="Where to save the results", metavar="DIR", type=str, required=True)
@click.option("--data", help="Path to the dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--cond", help="Train class-conditional model", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--condition", help="Train conditional model with condition type", metavar="class|color|edge|resize32|resize16", type=str, default=None, show_default=True)
@click.option("--arch", help="Network architecture", metavar="ddn|ddpmpp|ncsnpp|adm", type=click.Choice(["ddn", "ddpmpp", "ncsnpp", "adm"]), default="ddn", show_default=True)
@click.option("--precond", help="Preconditioning & loss function", metavar="vp|ve|edm", type=click.Choice(["vp", "ve", "edm"]), default="edm", show_default=True)

# 超参数
@click.option("--duration", help="Training duration", metavar="MIMG", type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option("--batch", help="Total batch size", metavar="INT", type=click.IntRange(min=1), default=512, show_default=True)
@click.option("--batch-gpu", help="Limit batch size per GPU", metavar="INT", type=click.IntRange(min=1))
@click.option("--max-blockn", help="max block num per resolution", metavar="INT", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--max-outputk", help="max output k of DDN", metavar="INT", type=click.IntRange(min=2), default=8, show_default=True)
@click.option("--learn-res", help="learn_residual in SDDNOutput", metavar="BOOL", type=bool, default=True, show_default=True)
@click.option("--chain-dropout", help="Chain Dropout of DDN", metavar="FLOAT", type=click.FloatRange(min=0, max=1), default=0.05, show_default=True)
@click.option("--lr", help="Learning rate", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option("--fp16", help="Enable mixed-precision training", metavar="BOOL", type=bool, default=True, show_default=True)

# 新增退火选择器参数
@click.option("--enable-annealed", help="Enable annealed selector", metavar="BOOL", type=bool, default=True, show_default=True)
@click.option("--quality-weight", help="Quality fitness weight", metavar="FLOAT", type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option("--diversity-weight", help="Diversity fitness weight", metavar="FLOAT", type=click.FloatRange(min=0), default=0.4, show_default=True)
@click.option("--minority-weight", help="Minority fitness weight", metavar="FLOAT", type=click.FloatRange(min=0), default=0.6, show_default=True)
@click.option("--init-temp", help="Initial temperature", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=1.5, show_default=True)
@click.option("--min-temp", help="Minimum temperature", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=0.01, show_default=True)
@click.option("--max-temp", help="Maximum temperature", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=3.0, show_default=True)
@click.option("--sa-steps", help="Simulated annealing steps", metavar="INT", type=click.IntRange(min=0), default=0, show_default=True)

# I/O相关
@click.option("--desc", help="String to include in result dir name", metavar="STR", type=str)
@click.option("--tick", help="How often to print progress", metavar="KIMG", type=click.IntRange(min=1), default=50, show_default=True)
@click.option("--snap", help="How often to save snapshots", metavar="TICKS", type=click.IntRange(min=1), default=50, show_default=True)
@click.option("--seed", help="Random seed  [default: random]", metavar="INT", type=int)
@click.option("--transfer", help="Transfer learning from network pickle", metavar="PKL|URL", type=str)
@click.option("--resume", help="Resume from previous training state", metavar="PT", type=str)
@click.option("-n", "--dry-run", help="Print training options and exit", is_flag=True)

def main(**kwargs):
    """Enhanced DDN training with annealed selector for cardiac ultrasound generation"""
    
    # 设置全局配置
    boxx.cf.kwargs = kwargs
    
    # 导入并配置sddn
    import sddn
    sddn.DiscreteDistributionOutput.learn_residual = kwargs.get("learn_res")
    sddn.DiscreteDistributionOutput.chain_dropout = kwargs.get("chain_dropout")

    if kwargs.get("condition") == "class":
        kwargs["cond"] = True
        
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method("spawn", force=True)
    dist.init()

    # 初始化配置字典
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        path=opts.data,
        use_labels=opts.cond,
        xflip=False,  # 医学图像通常不做翻转
        cache=True,
    )
    c.data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=True, num_workers=1, prefetch_factor=2
    )
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(
        class_name="torch.optim.Adam", lr=opts.lr, betas=[0.9, 0.999], eps=1e-8
    )

    # 验证数据集选项
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution
        c.dataset_kwargs.max_size = len(dataset_obj)
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException("--cond=True requires labels specified in dataset.json")
        del dataset_obj
    except IOError as err:
        raise click.ClickException(f"--data: {err}")

    # 网络架构配置
    if opts.arch == "ddn":
        c.network_kwargs.class_name = "training.enhanced_networks.EnhancedPHDDN"
        # c.network_kwargs.model_type = "EnhancedPHDDN"
        c.loss_kwargs.class_name = "training.enhanced_loss.EnhancedDDNLoss"
    else:
        # 其他架构保持原样
        if opts.arch == "ddpmpp":
            c.network_kwargs.update(
                model_type="SongUNet",
                embedding_type="positional",
                encoder_type="standard",
                decoder_type="standard",
            )
        # ... 其他架构配置

    # 预条件和损失函数
    if opts.precond == "edm":
        if opts.arch != "ddn":
            c.network_kwargs.class_name = "training.networks.EDMPrecond"
            c.loss_kwargs.class_name = "training.loss.EDMLoss"
    # ... 其他预条件配置

    # 配置退火选择器
    if opts.enable_annealed:
        annealed_selector_cfg = AnnealedSelectorConfig(
            weights=FitnessWeights(
                quality=opts.quality_weight,
                diversity=opts.diversity_weight,
                minority=opts.minority_weight,
            ),
            diversity=DiversityConfig(
                mode="centroid",
                pool_size=8,
                sample_m=16
            ),
            temp_policy=LayerTemperaturePolicy(
                init_T=opts.init_temp,
                min_T=opts.min_temp,
                max_T=opts.max_temp,
                global_decay=1e-4,
                first_layer_boost=1.6,
                last_layer_factor=0.2
            ),
            sa_steps=opts.sa_steps
        )
        
        # 创建可序列化的配置字典用于保存
        annealed_config_dict = {
            "enable_annealed_selector": True,
            "weights": {
                "quality": opts.quality_weight,
                "diversity": opts.diversity_weight,
                "minority": opts.minority_weight,
            },
            "diversity": {
                "mode": "centroid",
                "pool_size": 8,
                "sample_m": 16
            },
            "temp_policy": {
                "init_T": opts.init_temp,
                "min_T": opts.min_temp,
                "max_T": opts.max_temp,
                "global_decay": 1e-4,
                "first_layer_boost": 1.6,
                "last_layer_factor": 0.2
            },
            "sa_steps": opts.sa_steps
        }
    else:
        annealed_selector_cfg = None
        annealed_config_dict = {"enable_annealed_selector": False}

    # 网络选项
    c.network_kwargs.update(
        dropout=0,  # 医学图像通常不用dropout
        # use_fp16=opts.fp16,
        enable_annealed_selector=opts.enable_annealed,
        annealed_selector_cfg=annealed_selector_cfg,
    )
    
    # 添加可序列化的退火选择器配置到主配置中
    c.annealed_selector_cfg = annealed_config_dict

    # 训练选项
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = 500  # 医学图像用较长的EMA
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=1, cudnn_benchmark=True)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=500)

    # 随机种子
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device("cuda"))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # 迁移学习和恢复
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException("--transfer and --resume cannot be specified at the same time")
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r"enhanced-training-state-(\d+).pt", os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException("--resume must point to enhanced-training-state-*.pt from a previous training run")
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f"enhanced-shot-{match.group(1)}.pkl")
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # 描述字符串
    cond_str = "cond" if c.dataset_kwargs.use_labels else "uncond"
    desc = f"{dataset_name:s}-enhanced"
    if opts.desc is not None:
        desc += f"-{opts.desc}"
    if opts.enable_annealed:
        desc += f"-annealed"
    boxx.cf.desc = desc

    # 选择输出目录
    if dist.get_rank() != 0:
        c.run_dir = None
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [
                x for x in os.listdir(opts.outdir)
                if os.path.isdir(os.path.join(opts.outdir, x))
            ]
        prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f"{cur_run_id:05d}-{desc}")
        assert not os.path.exists(c.run_dir)

    # 打印配置（修复JSON序列化问题）
    dist.print0()
    dist.print0("Enhanced DDN Training options:")
    
    # 创建可序列化的配置副本
    c_printable = dnnlib.EasyDict()
    for key, value in c.items():
        if key == 'network_kwargs':
            # 过滤掉不能序列化的对象
            c_printable[key] = {k: v for k, v in value.items() 
                              if k != 'annealed_selector_cfg'}
        else:
            try:
                # 测试是否可以序列化
                json.dumps(value)
                c_printable[key] = value
            except (TypeError, ValueError):
                c_printable[key] = str(value)  # 转换为字符串
    
    dist.print0(json.dumps(c_printable, indent=2))
    dist.print0()
    dist.print0(f"Output directory:        {c.run_dir}")
    dist.print0(f"Dataset path:            {c.dataset_kwargs.path}")
    dist.print0(f"Class-conditional:       {c.dataset_kwargs.use_labels}")
    dist.print0(f"Network architecture:    {opts.arch}")
    dist.print0(f"Preconditioning & loss:  {opts.precond}")
    dist.print0(f"Annealed selector:       {opts.enable_annealed}")
    if opts.enable_annealed:
        dist.print0(f"Fitness weights:         Q={opts.quality_weight}, D={opts.diversity_weight}, M={opts.minority_weight}")
        dist.print0(f"Temperature range:       {opts.min_temp} - {opts.max_temp} (init: {opts.init_temp})")
    dist.print0(f"Number of GPUs:          {dist.get_world_size()}")
    dist.print0(f"Batch size:              {c.batch_size}")
    dist.print0(f"Mixed-precision:         {opts.fp16}")
    dist.print0()

    # 干运行检查
    if opts.dry_run:
        dist.print0("Dry run; exiting.")
        return

    # 创建输出目录
    dist.print0("Creating output directory...")
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "enhanced_training_options.json"), "wt") as f:
            # 🔧 修复：创建完全可序列化的配置字典
            c_dump = {"kwargs": kwargs, "annealed_selector_enabled": opts.enable_annealed}
            
            # 过滤掉不可序列化的项目
            for key, value in c.items():
                if key == 'network_kwargs':
                    # 过滤网络配置中的不可序列化对象
                    filtered_network_kwargs = {}
                    for k, v in value.items():
                        if k == 'annealed_selector_cfg':
                            # 跳过不可序列化的配置对象
                            continue
                        try:
                            json.dumps(v)  # 测试是否可序列化
                            filtered_network_kwargs[k] = v
                        except (TypeError, ValueError):
                            filtered_network_kwargs[k] = str(v)
                    c_dump[key] = filtered_network_kwargs
                else:
                    try:
                        json.dumps(value)  # 测试是否可序列化
                        c_dump[key] = value
                    except (TypeError, ValueError):
                        c_dump[key] = str(value)  # 转换为字符串
            
            # 添加退火选择器的配置（已经是可序列化的字典）
            if hasattr(c, 'annealed_selector_config'):
                c_dump['annealed_selector_config'] = c.annealed_selector_config
                
            json.dump(c_dump, f, indent=2)
        dnnlib.util.Logger(
            file_name=os.path.join(c.run_dir, "enhanced_log.txt"),
            file_mode="a",
            should_flush=True,
        )

    # 开始增强训练
    enhanced_training_loop(**c)

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath("."))
    import boxx
    from boxx.ylth import *
    from ddn_utils import debug, argkv

    if debug:
        boxx.cf.debug = True
        main([
            "--data=datasets/cardiac_ultrasound.zip",  # 假设的心脏超声数据集
            "--outdir=/tmp/enhanced-cardiac-ddn",
            "--duration=100",  # 较短的训练用于测试
            "--batch=32",
            "--batch-gpu=8",
            "--max-blockn=16",
            "--max-outputk=32",
            "--tick=10",
            "--enable-annealed=1",
            "--quality-weight=1.0",
            "--diversity-weight=0.3", 
            "--minority-weight=0.7",  # 心脏疾病中罕见病很重要
            "--init-temp=2.0",
            "--condition=class",  # 或者其他医学相关的条件
        ])
    else:
        main()