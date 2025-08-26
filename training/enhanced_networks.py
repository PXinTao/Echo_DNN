import boxx
import numpy as np
import torch
import random
import math
import sys
import os

# 添加当前目录到路径，以便导入 ddn_annealed
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ddn_annealed import AnnealedSelector, AnnealedSelectorConfig, FitnessWeights, DiversityConfig, LayerTemperaturePolicy

# 导入原有的模块
from torch_utils import persistence, misc
from torch.nn.functional import silu
import torch_utils
import dnnlib
import sddn
from sddn import DiscreteDistributionOutput

# 从原networks.py导入所有基础类，如果导入失败则直接复制定义
try:
    from .networks import (
        weight_init, Linear, Conv2d, GroupNorm, AttentionOp, UNetBlock, 
        PositionalEmbedding, FourierEmbedding, SongUNet, UNetBlockWoEmb,
        WarpDictIO, SongUNetInputDict, UpBlock, get_channeln, get_outputk, get_blockn,
        ClassEmbeding, ConditionProcess, DiscreteDistributionBlock
    )
except ImportError:
    # 如果导入失败，直接从原代码复制 DiscreteDistributionBlock 定义
    @persistence.persistent_class
    class DiscreteDistributionBlock(torch.nn.Module):
        short_plus = True

        def __init__(
            self,
            block,
            k=64,
            output_size=None,
            in_c=None,
            out_c=None,
            predict_c=3,
            loss_func=None,
            distance_func=None,
            leak_choice=True,
            input_dict=False,
        ):
            super().__init__()
            self.block = block
            block_first, block_last = (
                (block[0], block[-1])
                if isinstance(block, torch.nn.Sequential)
                else (block, block)
            )
            self.in_c = (
                in_c
                or getattr(block_first, "in_c", None)
                or getattr(block_first, "in_channels", None)
            )
            self.out_c = (
                out_c
                or getattr(block_last, "out_c", None)
                or getattr(block_last, "out_channels", self.in_c)
            )
            self.predict_c = predict_c
            self.leak_choice = leak_choice
            self.input_dict = input_dict

            if not self.short_plus:
                self.choice_conv1x1 = sddn.Conv2dMixedPrecision(
                    predict_c, self.in_c, (1, 1), bias=False
                )
                if leak_choice:
                    self.leak_conv1x1 = sddn.Conv2dMixedPrecision(
                        predict_c, self.in_c, (1, 1), bias=False
                    )
            self.ddo = DiscreteDistributionOutput(
                k,
                last_c=self.out_c,
                predict_c=predict_c,
                size=output_size,
                loss_func=loss_func,
                distance_func=distance_func,
                leak_choice=leak_choice,
            )
            self.output_size = output_size

        def forward(self, d=None, condition_process=None):
            d = d if isinstance(d, dict) else {"batch_size": 1 if d is None else len(d)}
            if "target" in d:
                batch_size = len(d["target"])
            else:
                batch_size = d.get("batch_size", 1)
            inp = d.get("feat_last")
            predict = d.get("predict")
            feat_leak = d.get("feat_leak")
            if inp is None:
                inp = sddn.build_init_feature(
                    (batch_size, self.in_c, self.output_size, self.output_size)
                ).cuda()
                predict = torch.zeros(
                    (batch_size, self.predict_c, self.output_size, self.output_size)
                ).cuda()
                if boxx.cf.get("kwargs", {}).get("fp16"):
                    inp, predict = inp.half(), predict.half()
                feat_leak = predict
            d["feat_last"] = inp
            b, c, h, w = inp.shape
            if not hasattr(self, "choice_conv1x1"):
                inp = inp + torch.nn.functional.pad(
                    predict, (0, 0, 0, 0, 0, c - self.predict_c)
                )
                if self.leak_choice:
                    inp = inp + torch.nn.functional.pad(
                        feat_leak, (0, 0, 0, 0, c - self.predict_c, 0)
                    )
                if condition_process:
                    stage_condition = condition_process(d)
                    if stage_condition is not None:
                        condc = stage_condition.shape[1]
                        cond_start = c // 2 - condc // 2
                        inp = inp + torch.nn.functional.pad(
                            stage_condition,
                            (0, 0, 0, 0, cond_start, c - cond_start - condc),
                        )
            else:
                inp = inp + self.choice_conv1x1(predict)
                if self.leak_choice:
                    inp = inp + self.leak_conv1x1(feat_leak)
            if self.input_dict:
                d["feat_last"] = inp
                d = self.block(d)
            else:
                d["feat_last"] = self.block(inp)
            d = self.ddo(d)
            return d
    
    # 同样需要导入其他必要的类，这里只是简化示例
    # 实际使用时，建议直接从原networks.py文件导入

@persistence.persistent_class
class EnhancedDiscreteDistributionBlock(torch.nn.Module):
    """增强的DDN块，集成了退火选择器"""
    short_plus = True

    def __init__(
        self,
        block,
        k=64,
        output_size=None,
        in_c=None,
        out_c=None,
        predict_c=3,
        loss_func=None,
        distance_func=None,
        leak_choice=True,
        input_dict=False,
        # 新增参数
        annealed_selector_cfg: AnnealedSelectorConfig = None,
        enable_annealed_selector: bool = True,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self.enable_annealed_selector = enable_annealed_selector
        
        block_first, block_last = (
            (block[0], block[-1])
            if isinstance(block, torch.nn.Sequential)
            else (block, block)
        )
        self.in_c = (
            in_c
            or getattr(block_first, "in_c", None)
            or getattr(block_first, "in_channels", None)
        )
        self.out_c = (
            out_c
            or getattr(block_last, "out_c", None)
            or getattr(block_last, "out_channels", self.in_c)
        )
        self.predict_c = predict_c
        self.leak_choice = leak_choice
        self.input_dict = input_dict

        # 原有的选择和泄露卷积层
        if not self.short_plus:
            self.choice_conv1x1 = sddn.Conv2dMixedPrecision(
                predict_c, self.in_c, (1, 1), bias=False
            )
            if leak_choice:
                self.leak_conv1x1 = sddn.Conv2dMixedPrecision(
                    predict_c, self.in_c, (1, 1), bias=False
                )
                
        self.ddo = DiscreteDistributionOutput(
            k,
            last_c=self.out_c,
            predict_c=predict_c,
            size=output_size,
            loss_func=loss_func,
            distance_func=distance_func,
            leak_choice=leak_choice,
        )
        self.output_size = output_size
        
        # 初始化退火选择器
        if enable_annealed_selector:
            if annealed_selector_cfg is None:
                # 为心脏超声优化的默认配置
                annealed_selector_cfg = AnnealedSelectorConfig(
                    weights=FitnessWeights(quality=1.0, diversity=0.4, minority=0.6),  # 更重视少数类
                    diversity=DiversityConfig(mode="centroid", pool_size=8, sample_m=16),
                    temp_policy=LayerTemperaturePolicy(
                        init_T=1.5,  # 稍高的初始温度用于医学图像
                        min_T=0.01,
                        max_T=2.5,
                        first_layer_boost=1.6,  # 早期层更高探索
                        last_layer_factor=0.2   # 后期层更精确
                    )
                )
            self.annealed_selector = AnnealedSelector(annealed_selector_cfg)
        else:
            self.annealed_selector = None

    def forward(self, d=None, condition_process=None):
        # 初始化字典
        d = d if isinstance(d, dict) else {"batch_size": 1 if d is None else len(d)}
        if "target" in d:
            batch_size = len(d["target"])
        else:
            batch_size = d.get("batch_size", 1)
            
        inp = d.get("feat_last")
        predict = d.get("predict")
        feat_leak = d.get("feat_leak")
        
        if inp is None:  # 初始化特征
            inp = sddn.build_init_feature(
                (batch_size, self.in_c, self.output_size, self.output_size)
            ).cuda()
            predict = torch.zeros(
                (batch_size, self.predict_c, self.output_size, self.output_size)
            ).cuda()
            if boxx.cf.get("kwargs", {}).get("fp16"):
                inp, predict = inp.half(), predict.half()
            feat_leak = predict
            
        d["feat_last"] = inp
        b, c, h, w = inp.shape
        
        # 特征融合逻辑（与原代码相同）
        if not hasattr(self, "choice_conv1x1"):
            inp = inp + torch.nn.functional.pad(
                predict, (0, 0, 0, 0, 0, c - self.predict_c)
            )
            if self.leak_choice:
                inp = inp + torch.nn.functional.pad(
                    feat_leak, (0, 0, 0, 0, c - self.predict_c, 0)
                )
            if condition_process:
                stage_condition = condition_process(d)
                if stage_condition is not None:
                    condc = stage_condition.shape[1]
                    cond_start = c // 2 - condc // 2
                    inp = inp + torch.nn.functional.pad(
                        stage_condition,
                        (0, 0, 0, 0, cond_start, c - cond_start - condc),
                    )
        else:
            inp = inp + self.choice_conv1x1(predict)
            if self.leak_choice:
                inp = inp + self.leak_conv1x1(feat_leak)
                
        # 通过网络块
        if self.input_dict:
            d["feat_last"] = inp
            d = self.block(d)
        else:
            d["feat_last"] = self.block(inp)
            
        # 调用增强的DDO（集成退火选择器）
        d = self._enhanced_ddo_forward(d)
        return d

    def _enhanced_ddo_forward(self, d):
        """增强的DDO前向传播，集成退火选择器"""
        # 获取DDO的输出（多个候选）
        d = self.ddo(d)
        
        # 如果启用退火选择器且在训练模式
        if (self.enable_annealed_selector and 
            hasattr(self.ddo, 'output') and 
            self.training and 
            'target' in d):
            
            try:
                # 准备退火选择器的输入
                candidates = d.get('output')  # [K, B, C, H, W]
                target = d.get('target')
                class_labels = d.get('class_labels')
                
                # 获取当前训练状态信息
                global_step = d.get('global_step', 0)
                total_layers = d.get('total_layers', self.layer_idx + 1)
                
                # 计算性能趋势（简化版本）
                perf_trend = d.get('perf_trend', None)
                
                # 少数类保护
                minority_guard = d.get('minority_guard', False)
                
                if candidates is not None and candidates.dim() == 5:
                    # 使用退火选择器进行选择
                    selected, selected_idx, selector_info = self.annealed_selector.select(
                        candidates=candidates,
                        target=target,
                        class_labels=class_labels,
                        layer_idx=self.layer_idx,
                        total_layers=total_layers,
                        global_step=global_step,
                        perf_trend=perf_trend,
                        minority_guard=minority_guard
                    )
                    
                    # 更新字典
                    d['predict'] = selected
                    d['selected_idx'] = selected_idx
                    d['selector_info'] = selector_info
                    
                    # 添加调试信息
                    if boxx.cf.get("debug", False):
                        print(f"Layer {self.layer_idx}: T={selector_info['T']:.3f}, "
                              f"entropy={selector_info['entropy']:.3f}, "
                              f"fitness_mean={selector_info['fitness_mean']:.3f}")
                
            except Exception as e:
                print(f"退火选择器出错，回退到原选择方式: {e}")
                # 如果退火选择器失败，回退到原来的选择方式
                pass
        
        return d

@persistence.persistent_class
class EnhancedPHDDN(torch.nn.Module):
    """增强的PHDDN，支持退火选择器"""
    
    def __init__(
        self,
        img_resolution=32,
        img_channels=3,      # 改为 img_channels（之前是 in_channels）
        out_channels=3,
        label_dim=0,
        augment_dim=0,
        model_channels=128,
        channel_mult=[1, 2, 2, 2],
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=[16],
        dropout=0.10,
        label_dropout=0,
        embedding_type="positional",
        channel_mult_noise=1,
        encoder_type="standard",
        decoder_type="standard",
        resample_filter=[1, 1],
        # 新增退火选择器参数
        enable_annealed_selector: bool = True,
        annealed_selector_cfg: AnnealedSelectorConfig = None,
    ):
        super().__init__()
        
        # 将 img_channels 赋值给内部使用的变量
        in_channels = img_channels  # 这样你的其余代码不用改
        
        self.label_dropout = label_dropout
        self.label_dim = label_dim
        self.enable_annealed_selector = enable_annealed_selector
        
        # 其余代码保持完全不变...
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        # 条件处理
        condition_type = boxx.cf.get("kwargs", {}).get("condition")
        if condition_type:
            if condition_type == "class":
                assert condition_type and label_dim, (condition_type, label_dim)
                condition_type += str(label_dim)
            self.condition_process = ConditionProcess(condition_type)
        self.condition_type = condition_type

        self.scalen = int(np.log2(img_resolution))

        # 初始化模块名称和缩放映射
        self.module_names = []
        self.scale_to_module_names = {}
        self.scale_to_repeatn = {}
        
        # 为心脏超声优化的退火选择器配置
        if annealed_selector_cfg is None:
            annealed_selector_cfg = AnnealedSelectorConfig(
                weights=FitnessWeights(
                    quality=1.0,    # 医学图像质量很重要
                    diversity=0.4,  # 适度的多样性
                    minority=0.6    # 重视罕见病例
                ),
                diversity=DiversityConfig(
                    mode="centroid",
                    pool_size=8,
                    sample_m=16
                ),
                temp_policy=LayerTemperaturePolicy(
                    init_T=1.5,             # 医学图像需要更高初始温度
                    min_T=0.01,
                    max_T=3.0,              # 更大的温度范围
                    global_decay=2e-4,      # 更慢的全局衰减
                    first_layer_boost=1.8,  # 早期层高探索
                    last_layer_factor=0.1   # 后期层高精度
                )
            )

        def set_enhanced_block(name, block, layer_idx):
            """设置增强的DDN块"""
            enhanced_block = EnhancedDiscreteDistributionBlock(
                block=block.block,
                k=block.ddo.k,
                output_size=block.output_size,
                in_c=block.in_c,
                out_c=block.out_c,
                predict_c=block.predict_c,
                loss_func=getattr(block.ddo, 'loss_func', None),
                distance_func=getattr(block.ddo, 'distance_func', None),
                leak_choice=block.leak_choice,
                input_dict=block.input_dict,
                annealed_selector_cfg=annealed_selector_cfg,
                enable_annealed_selector=self.enable_annealed_selector,
                layer_idx=layer_idx,
            )
            self.module_names.append(name)
            setattr(self, name, enhanced_block)
            self.scale_to_module_names[scalei] = self.scale_to_module_names.get(scalei, []) + [name]
            return enhanced_block

        # 构建网络结构（基于原PHDDN设计）
        start_size = boxx.cf.get("kwargs", {}).get("start_size", 1)
        blockn_times = boxx.cf.get("kwargs", {}).get("blockn_times", 1)
        self.scalis = range(int(math.log2(start_size)), self.scalen + 1)
        
        layer_count = 0
        last_scalei = self.scalis[0]
        
        for scalei in self.scalis:
            size = 2**scalei
            channeln = get_channeln(scalei)
            last_channeln = get_channeln(scalei - 1)
            k = get_outputk(scalei)
            
            if last_scalei != scalei:
                # 上采样块
                block_up = UNetBlockWoEmb(
                    in_channels=last_channeln,
                    out_channels=channeln,
                    up=True,
                    **block_kwargs,
                )
                # 创建原始的DDN块，然后用增强版本替换
                original_block = DiscreteDistributionBlock(block_up, k, output_size=size)
                set_enhanced_block(f"block_{size}x{size}_0_up", original_block, layer_count)
                layer_count += 1
            else:  # scale0只有一个块
                block = UNetBlockWoEmb(channeln, channeln, **block_kwargs)
                original_block = DiscreteDistributionBlock(block, k, output_size=size)
                set_enhanced_block(f"block_{size}x{size}_0", original_block, layer_count)
                layer_count += 1
                if not scalei:  # scale0的特殊处理
                    continue
                    
            cin = channeln
            blockn = int(round(get_blockn(scalei) * blockn_times))
            
            for block_count in range(1, blockn):
                block = UNetBlockWoEmb(cin, channeln, **block_kwargs)
                original_block = DiscreteDistributionBlock(block, k, output_size=size)
                set_enhanced_block(f"block_{size}x{size}_{block_count}", original_block, layer_count)
                layer_count += 1
                cin = channeln
                
        # 记录总层数，用于温度调度
        self.total_layers = layer_count
        
        # Refiner网络（如果需要）
        self.refiner_repeatn = (
            3 if boxx.cf.debug else boxx.cf.get("kwargs", {}).get("refinern", 0)
        )
        if self.refiner_repeatn:
            refiner_outputk = 4
            unet = SongUNetInputDict(
                img_resolution=img_resolution,
                in_channels=channeln,
                out_channels=channeln,
                label_dim=label_dim,
                augment_dim=augment_dim,
                model_channels=model_channels,
                channel_mult=channel_mult,
                channel_mult_emb=channel_mult_emb,
                num_blocks=num_blocks,
                attn_resolutions=attn_resolutions,
                dropout=dropout,
                label_dropout=label_dropout,
                embedding_type=embedding_type,
                channel_mult_noise=channel_mult_noise,
                encoder_type=encoder_type,
                decoder_type=decoder_type,
                resample_filter=resample_filter,
            )
            original_refiner = DiscreteDistributionBlock(
                unet,
                refiner_outputk,
                output_size=img_resolution,
                in_c=channeln,
                out_c=channeln,
                predict_c=out_channels,
                input_dict=True,
            )
            self.refiner = EnhancedDiscreteDistributionBlock(
                block=original_refiner.block,
                k=refiner_outputk,
                output_size=img_resolution,
                in_c=channeln,
                out_c=channeln,
                predict_c=out_channels,
                input_dict=True,
                annealed_selector_cfg=annealed_selector_cfg,
                enable_annealed_selector=self.enable_annealed_selector,
                layer_idx=layer_count,
            )

    def forward(self, d=None, _sigma=None, labels=None):
        if isinstance(d, torch.Tensor):
            d = {"target": d}
        elif d is None:
            d = {"batch_size": 1}
        assert isinstance(d, dict), d

        # 添加训练状态信息到字典
        d["total_layers"] = self.total_layers
        if self.training:
            # 从全局状态获取training step（需要在training_loop中设置）
            d["global_step"] = getattr(self, '_current_global_step', 0)
            d["perf_trend"] = getattr(self, '_current_perf_trend', None)
            d["minority_guard"] = getattr(self, '_minority_guard', False)

        if self.label_dim and labels is not None:
            d["class_labels"] = labels
            
        # 主要的前向传播循环
        for scalei in self.scalis:
            for repeati in range(self.scale_to_repeatn.get(scalei, 1)):
                for module_idx, name in enumerate(self.scale_to_module_names[scalei]):
                    if module_idx == 0 and repeati != 0:
                        # 重复时跳过第一个模块（上采样）
                        continue
                    module = getattr(self, name)
                    d = module(d, condition_process=getattr(self, "condition_process", None))
                    
        # Refiner处理
        feat = d["feat_last"]
        batch_size = feat.shape[0]
        for repeati in range(self.refiner_repeatn):
            d["noise_labels"] = torch.Tensor(
                [(repeati / max(self.refiner_repeatn - 1, 1)) * 2 - 1] * batch_size
            ).to(feat)
            d = self.refiner(d, condition_process=getattr(self, "condition_process", None))
            
        return d

    def set_training_state(self, global_step: int, perf_trend: float = None, minority_guard: bool = False):
        """设置当前训练状态，用于退火选择器"""
        self._current_global_step = global_step
        self._current_perf_trend = perf_trend
        self._minority_guard = minority_guard
        
        # 递归设置所有子模块的状态
        for name in self.module_names:
            module = getattr(self, name)
            if hasattr(module, 'annealed_selector'):
                # 可以在这里设置模块特定的状态
                pass

    def table(self):
        """显示网络结构表格"""
        times = 1
        mds = []
        for name in self.module_names:
            m = getattr(self, name)
            k = m.ddo.k if hasattr(m, "ddo") else 1
            c = (m.in_c, m.out_c) if hasattr(m, "in_c") else (None, None)
            size = m.output_size if hasattr(m, "output_size") else 32
            repeat = (
                self.scale_to_repeatn.get(int(np.log2(size)), 1)
                if hasattr(m, "output_size")
                else 1
            )
            times *= k * repeat
            log2 = math.log2(times)
            
            # 添加退火选择器信息
            annealed_info = "✓" if (hasattr(m, 'enable_annealed_selector') and m.enable_annealed_selector) else "✗"
            
            row = dict(
                name=name,
                size=size,
                c=c,
                k=k,
                repeat=repeat,
                annealed=annealed_info,
                log2=log2,
                log10=math.log10(times),
            )
            mds.append(row)
        return boxx.Markdown(mds)