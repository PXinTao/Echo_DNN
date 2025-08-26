import torch
import boxx
import numpy as np
from cardiac_ultrasound_config import CardiacUltrasoundConfig

def demo_cardiac_enhanced():
    """演示增强心脏超声生成"""
    
    print("=== Enhanced DDN for Cardiac Ultrasound Demo ===\n")
    
    # 模拟心脏超声数据
    B, C, H, W = 4, 1, 128, 128  # 通常心脏超声是单通道
    K = 32  # DDN的候选数量
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模拟数据
    candidates = torch.randn(K, B, C, H, W, device=device)
    target = torch.randn(B, C, H, W, device=device)
    
    # 模拟不平衡的心脏疾病标签
    # 0: 正常, 1: 心肌梗死, 2: 心律不齐, 3: 罕见心脏病
    class_labels = torch.tensor([0, 0, 1, 3], device=device)  # 大部分正常，少数罕见疾病
    
    print(f"Input shape: candidates={candidates.shape}, target={target.shape}")
    print(f"Class distribution: {class_labels.tolist()}")
    
    # 测试不同配置
    configs = {
        "default": CardiacUltrasoundConfig.get_default_config(),
        "rare_disease": CardiacUltrasoundConfig.get_rare_disease_config(), 
        "high_quality": CardiacUltrasoundConfig.get_high_quality_config(),
    }
    
    for config_name, config in configs.items():
        print(f"\n--- Testing {config_name} configuration ---")
        
        from ddn_annealed import AnnealedSelector
        selector = AnnealedSelector(config)
        
        # 执行选择
        selected, idx, info = selector.select(
            candidates=candidates,
            target=target,
            class_labels=class_labels,
            layer_idx=2,          # 假设在第3层
            total_layers=8,       # 总共8层
            global_step=1000,     # 训练1000步
            perf_trend=-0.02,     # 性能在提升
            minority_guard=(config_name == "rare_disease")  # 罕见疾病配置启用保护
        )
        
        print(f"Selected shape: {selected.shape}")
        print(f"Selected indices: {idx.tolist()}")
        print(f"Temperature: {info['T']:.4f}")
        print(f"Entropy: {info['entropy']:.4f}")  
        print(f"Fitness mean±std: {info['fitness_mean']:.4f}±{info['fitness_std']:.4f}")
        
        # 分析选择分布
        unique_idx, counts = idx.unique(return_counts=True)
        print(f"Index distribution: {dict(zip(unique_idx.tolist(), counts.tolist()))}")
    
    print("\n=== Demo completed successfully! ===")

if __name__ == "__main__":
    demo_cardiac_enhanced()