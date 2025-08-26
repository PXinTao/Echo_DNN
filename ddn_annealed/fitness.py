from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

@dataclass
class FitnessWeights:
    quality: float = 1.0
    diversity: float = 0.3
    minority: float = 0.5

@dataclass
class DiversityConfig:
    mode: str = "centroid"
    pool_size: int = 8
    sample_m: int = 16
    eps: float = 1e-6

class MultiObjectiveFitness:
    def __init__(self, weights: FitnessWeights = FitnessWeights(),
                 diversity_cfg: DiversityConfig = DiversityConfig()):
        self.w = weights
        self.cfg = diversity_cfg

    def _ensure_tensor5(self, candidates):
        if isinstance(candidates, (list, tuple)):
            cand = torch.stack(candidates, dim=0)
        else:
            cand = candidates
            assert cand.dim() == 5, "candidates must be [K,B,C,H,W]"
        return cand

    @torch.no_grad()
    def quality(self, candidates: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
        K, B = candidates.shape[:2]
        if target is None:
            return torch.zeros(K, B, device=candidates.device, dtype=candidates.dtype)
        diff = (candidates - target.unsqueeze(0))
        l2 = diff.pow(2).flatten(2).mean(dim=2)
        l1 = diff.abs().flatten(2).mean(dim=2)
        score = -(0.5 * l2 + 0.5 * l1)
        return score

    @torch.no_grad()
    def diversity(self, candidates: torch.Tensor) -> torch.Tensor:
        K, B, C, H, W = candidates.shape
        if self.cfg.pool_size and (H > self.cfg.pool_size or W > self.cfg.pool_size):
            cand_low = F.adaptive_avg_pool2d(candidates.flatten(0,1), (self.cfg.pool_size, self.cfg.pool_size))
            cand_low = cand_low.flatten(2).mean(dim=2)
            cand_low = cand_low.view(K, B, C)
        else:
            cand_low = candidates.flatten(3).mean(dim=3)
        centroid = cand_low.mean(dim=0, keepdim=True).detach()
        dist = (cand_low - centroid).norm(dim=2)
        return dist

    @torch.no_grad()
    def minority(self, class_labels: Optional[torch.Tensor], B: int) -> torch.Tensor:
        if class_labels is None:
            return torch.zeros(1, B, device='cpu')
        device = class_labels.device
        unique, counts = class_labels.unique(return_counts=True)
        freq = torch.zeros(int(class_labels.max().item())+1, device=device).float()
        freq[unique] = counts.float()
        inv_freq = 1.0 / (freq.clamp_min(1.0))
        sample_w = inv_freq[class_labels]
        sample_w = (sample_w / (sample_w.mean().clamp_min(1e-6)))
        return sample_w.unsqueeze(0).expand(-1, B)

    @staticmethod
    def _zscore(x: torch.Tensor, dim=0, eps: float = 1e-6) -> torch.Tensor:
        mu = x.mean(dim=dim, keepdim=True)
        sd = x.std(dim=dim, keepdim=True).clamp_min(eps)
        return (x - mu) / sd

    @torch.no_grad()
    def __call__(self, candidates, target=None, class_labels=None) -> torch.Tensor:
        cand = self._ensure_tensor5(candidates)
        q = self.quality(cand, target)
        d = self.diversity(cand)
        K, B = cand.shape[:2]
        m = self.minority(class_labels, B).to(device=cand.device, dtype=cand.dtype)

        qz = self._zscore(q, dim=0)
        dz = self._zscore(d, dim=0)
        mz = self._zscore(m.expand_as(q), dim=0) if m.numel() > 1 else torch.zeros_like(qz)

        fitness = self.w.quality * qz + self.w.diversity * dz + self.w.minority * mz
        return fitness

# ==================================================
# File: ddn_annealed/temperature.py
# ==================================================
from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class LayerTemperaturePolicy:
    init_T: float = 1.2
    min_T: float = 0.02
    max_T: float = 2.0
    global_decay: float = 1e-4
    first_layer_boost: float = 1.4
    last_layer_factor: float = 0.3

class TemperatureScheduler:
    def __init__(self, policy: LayerTemperaturePolicy = LayerTemperaturePolicy()):
        self.p = policy

    def get_T(self, layer_idx: int, total_layers: int, global_step: int,
              perf_trend: Optional[float] = None, minority_guard: Optional[bool] = None) -> float:
        T_global = self.p.init_T * math.exp(-self.p.global_decay * global_step)
        if total_layers <= 1:
            layer_factor = 1.0
        else:
            progress = layer_idx / (total_layers - 1.0)
            layer_factor = self.p.first_layer_boost + (self.p.last_layer_factor - self.p.first_layer_boost) * progress
        T = T_global * layer_factor
        if perf_trend is not None:
            if perf_trend > 0.01:
                T *= 1.15
            elif perf_trend < -0.05:
                T *= 0.9
        if minority_guard:
            T = max(T, 0.6)
        T = max(min(T, self.p.max_T), self.p.min_T)
        return float(T)