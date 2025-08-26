\
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
