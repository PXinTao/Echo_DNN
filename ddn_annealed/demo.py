\
import torch
from ddn_annealed import AnnealedSelector, AnnealedSelectorConfig, FitnessWeights, DiversityConfig

B, C, H, W = 4, 3, 32, 32
K = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

candidates = torch.randn(K, B, C, H, W, device=device)
target = torch.randn(B, C, H, W, device=device)
class_labels = torch.randint(0, 10, (B,), device=device)

selector = AnnealedSelector(
    AnnealedSelectorConfig(
        weights=FitnessWeights(quality=1.0, diversity=0.4, minority=0.4),
        diversity=DiversityConfig(mode="centroid", pool_size=8, sample_m=16),
    )
)
selected, idx, info = selector.select(
    candidates=candidates,
    target=target,
    class_labels=class_labels,
    layer_idx=0,
    total_layers=8,
    global_step=1000,
    perf_trend=0.0,
    minority_guard=False
)

print("Selected:", selected.shape, "Idx:", idx.shape, "Info:", info)
