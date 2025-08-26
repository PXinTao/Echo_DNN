import boxx
import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

def enhanced_training_loop(
    run_dir=".",
    dataset_kwargs={},
    data_loader_kwargs={},
    network_kwargs={},
    loss_kwargs={},
    optimizer_kwargs={},
    augment_kwargs=None,
    seed=0,
    batch_size=512,
    batch_gpu=None,
    total_kimg=200000,
    ema_halflife_kimg=500,
    ema_rampup_ratio=0.05,
    lr_rampup_kimg=10000,
    loss_scaling=1,
    kimg_per_tick=50,
    snapshot_ticks=50,
    state_dump_ticks=500,
    resume_pkl=None,
    resume_state_dump=None,
    resume_kimg=0,
    cudnn_benchmark=True,
    device=torch.device("cuda"),
    # 新增退火选择器参数
    enable_annealed_selector=True,
    annealed_selector_cfg=None,
):
    """增强的训练循环，支持退火选择器"""
    
    # 初始化（与原代码相同）
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # 批处理大小计算
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # 加载数据集
    dist.print0("Loading dataset...")
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs,
        )
    )

    # 构建增强网络
    dist.print0("Constructing enhanced network with annealed selector...")
    interface_kwargs = dict(
        img_resolution=dataset_obj.resolution,
        img_channels=dataset_obj.num_channels,
        label_dim=dataset_obj.label_dim,
    )
    
    # 添加退火选择器参数
    network_kwargs.update(
        enable_annealed_selector=enable_annealed_selector,
        annealed_selector_cfg=annealed_selector_cfg,
    )
    
    # 使用增强的网络类
    if network_kwargs.get('model_type') == 'PHDDN':
        network_kwargs['class_name'] = 'training.enhanced_networks.EnhancedPHDDN'
    
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs).to(device)

    # 网络摘要
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros(
                [batch_gpu, net.img_channels, net.img_resolution, net.img_resolution],
                device=device,
            )
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net.eval(), [images, sigma, labels], max_nesting=2)
            
            # 显示增强网络的表格
            if hasattr(net, 'table'):
                print("Enhanced DDN Architecture:")
                print(net.table())
                
    net.train().requires_grad_(True)

    # 设置优化器和损失函数
    dist.print0("Setting up enhanced optimizer and loss...")
    
    # 使用增强的损失函数
    if loss_kwargs.get('class_name') == 'training.loss.DDNLoss':
        loss_kwargs['class_name'] = 'training.enhanced_loss.EnhancedDDNLoss'
        
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    augment_pipe = (
        dnnlib.util.construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )
    
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[device], broadcast_buffers=False
    )
    
    if ema_halflife_kimg:
        ema = copy.deepcopy(net).eval().requires_grad_(False)
    else:
        ema = net

    # 恢复训练（与原代码相同的逻辑）
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()
        misc.copy_params_and_buffers(src_module=data["ema"], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data["ema"], dst_module=ema, require_all=False)
        del data
        
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device("cpu"))
        misc.copy_params_and_buffers(src_module=data["net"], dst_module=net, require_all=True)
        optimizer.load_state_dict(data["optimizer_state"])
        del data

    # 主训练循环
    dist.print0(f"Training enhanced DDN for {total_kimg} kimg...")
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    
    while True:
        # 累积梯度
        optimizer.zero_grad(set_to_none=True)
        
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                
                # 设置损失函数的全局步数
                if hasattr(loss_fn, 'set_global_step'):
                    loss_fn.set_global_step(cur_nimg // 1000)
                
                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                training_stats.report("Loss/loss", loss)
                loss_ = loss.sum().mul(loss_scaling / batch_gpu_total)
                loss_.backward()

        # 更新权重
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # DDN特有的分割操作
        from sddn import DiscreteDistributionOutput
        DiscreteDistributionOutput.try_split_all(optimizer)

        # 更新EMA
        if ema_halflife_kimg:
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
            for p_ema, p_net in zip(ema.parameters(), net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # 维护任务
        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000
        
        if ((not done) and (cur_tick != 0) and 
            (cur_nimg < tick_start_nimg + kimg_per_tick * 1000)):
            continue

        # 打印状态
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):.1f}/{total_kimg}({round(cur_nimg / 1e3/total_kimg*100, 1)}%)"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        
        torch.cuda.reset_peak_memory_stats()
        
        # 添加退火选择器的状态信息
        status_msg = boxx.cf.get("desc", "enhanced") + ": " + " ".join(fields) + f" loss {round(loss_.tolist(),3)}" + f"/mean {boxx.strnum(loss.sum().tolist()/images.numel())}"
        
        # 如果有退火选择器信息，添加到状态消息
        if hasattr(loss_fn, 'loss_history') and len(loss_fn.loss_history) > 5:
            perf_trend = loss_fn._calculate_perf_trend()
            status_msg += f" trend {perf_trend:.4f}"
        
        dist.print0(status_msg)

        # 检查中止
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        # 保存网络快照
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            torch.distributed.barrier()
            data = dict(
                ema=ema,
                loss_fn=loss_fn,
                augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs),
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f"enhanced-shot-{cur_nimg//1000:06d}.pkl"), "wb") as f:
                    pickle.dump(data, f)
            del data

        # 保存训练状态
        if ((state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and 
            cur_tick != 0 and dist.get_rank() == 0):
            torch.save(
                dict(net=net, optimizer_state=optimizer.state_dict()),
                os.path.join(run_dir, f"enhanced-training-state-{cur_nimg//1000:06d}.pt"),
            )

        # 更新日志
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "enhanced_stats.jsonl"), "at")
            log_data = dict(training_stats.default_collector.as_dict(), timestamp=time.time())
            
            # 添加退火选择器的统计信息
            if hasattr(loss_fn, 'loss_history'):
                log_data['annealed_selector'] = {
                    'perf_trend': loss_fn._calculate_perf_trend(),
                    'loss_window_size': len(loss_fn.loss_history)
                }
            
            stats_jsonl.write(json.dumps(log_data) + "\n")
            stats_jsonl.flush()
            
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # 更新状态
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        
        if boxx.cf.debug and cur_nimg >= 12:
            boxx.g()
            done = True
        if done:
            break

    # 完成
    dist.print0()
    dist.print0("Enhanced training completed!")