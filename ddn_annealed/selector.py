\
from dataclasses import dataclass
from typing import Optional, Union, List
import torch
import torch.nn.functional as F

from .fitness import MultiObjectiveFitness, FitnessWeights, DiversityConfig
from .temperature import TemperatureScheduler, LayerTemperaturePolicy
from .distributed import broadcast_scalar, set_global_seed

TensorOrList = Union[torch.Tensor, List[torch.Tensor]]

@dataclass
class AnnealedSelectorConfig:
    weights: FitnessWeights = FitnessWeights()
    diversity: DiversityConfig = DiversityConfig()
    temp_policy: LayerTemperaturePolicy = LayerTemperaturePolicy()
    sa_steps: int = 0
    seed_base: int = 20250811

class AnnealedSelector:
    def __init__(self, cfg: AnnealedSelectorConfig = AnnealedSelectorConfig()):
        self.cfg = cfg
        self.fitness_fn = MultiObjectiveFitness(weights=cfg.weights, diversity_cfg=cfg.diversity)
        self.temp_sched = TemperatureScheduler(cfg.temp_policy)

    @torch.no_grad()
    def _softmax_sample(self, fitness: torch.Tensor, T: float, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            set_global_seed(seed)
        Tt = torch.tensor(T, device=fitness.device, dtype=fitness.dtype).clamp_min(1e-6)
        logits = (fitness - fitness.max(dim=0, keepdim=True).values) / Tt
        probs = torch.softmax(logits, dim=0)
        cat = torch.distributions.Categorical(probs.T.contiguous())
        idx = cat.sample()
        return idx

    @torch.no_grad()
    def _gather_selected(self, candidates: TensorOrList, idx: torch.Tensor) -> torch.Tensor:
        if isinstance(candidates, (list, tuple)):
            B = candidates[0].shape[0]
            out = torch.stack([candidates[idx[b]][b] for b in range(B)], dim=0)
            return out
        else:
            K, B = candidates.shape[:2]
            one_hot = torch.nn.functional.one_hot(idx, num_classes=K).T.view(K, B, 1, 1, 1)
            selected = (candidates * one_hot).sum(dim=0)
            return selected

    @torch.no_grad()
    def _metropolis_step(self, candidates: torch.Tensor, current_idx: torch.Tensor, fitness: torch.Tensor, T: float) -> torch.Tensor:
        K, B = fitness.shape
        device = fitness.device
        prop_j = torch.randint(low=0, high=K, size=(B,), device=device)
        f_cur = fitness[current_idx, torch.arange(B, device=device)]
        f_new = fitness[prop_j, torch.arange(B, device=device)]
        accept_logprob = (f_new - f_cur) / max(T, 1e-6)
        u = torch.rand(B, device=device).log()
        accept = (u < accept_logprob)
        out_idx = current_idx.clone()
        out_idx[accept] = prop_j[accept]
        return out_idx

    @torch.no_grad()
    def select(self,
               candidates: TensorOrList,
               target: Optional[torch.Tensor],
               class_labels: Optional[torch.Tensor],
               layer_idx: int,
               total_layers: int,
               global_step: int,
               perf_trend: Optional[float] = None,
               minority_guard: Optional[bool] = None):
        fitness = self.fitness_fn(candidates, target=target, class_labels=class_labels)
        T = self.temp_sched.get_T(layer_idx, total_layers, global_step,
                                  perf_trend=perf_trend, minority_guard=minority_guard)
        T = broadcast_scalar(T, src=0)

        seed = self.cfg.seed_base + (global_step * 1315423911 + layer_idx * 2654435761) % (2**31 - 1)
        idx = self._softmax_sample(fitness, T=T, seed=seed)

        for _ in range(max(0, int(self.cfg.sa_steps))):
            idx = self._metropolis_step(
                candidates if not isinstance(candidates, list) else torch.stack(candidates, dim=0),
                idx, fitness, T
            )

        selected = self._gather_selected(candidates, idx)

        logits = (fitness - fitness.max(dim=0, keepdim=True).values) / max(T, 1e-6)
        probs = torch.softmax(logits, dim=0)
        entropy = -(probs * (probs.clamp_min(1e-9)).log()).sum(dim=0).mean().item()
        info = {"T": float(T), "entropy": float(entropy),
                "fitness_mean": float(fitness.mean().item()),
                "fitness_std": float(fitness.std().item())}
        return selected, idx, info
