"""
src/model.py
============
PrunableLinear layer and SelfPruningNet definition.

Each weight is multiplied by a learnable gate value in (0, 1):
    gate  = sigmoid(gate_score)
    out   = x @ (W ⊙ gate)ᵀ + b

An L1 penalty on gate values during training drives them toward 0,
effectively pruning less-useful connections.
"""

import math
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class PrunableLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a per-weight sigmoid gate.

    Why sparsity emerges
    --------------------
    The sparsity loss  λ · mean(sigmoid(gate_scores))  penalises every
    gate proportionally to its magnitude.  Its gradient always pushes
    gate_scores toward −∞  →  gate → 0  →  weight is masked out.
    The task loss resists this for weights that aid classification;
    the network settles with a bimodal gate distribution:
        cluster near 0  → pruned connections
        cluster near 1  → active connections
    λ controls how aggressively the pruning pressure is applied.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Init at +3.0  →  σ(3) ≈ 0.95  (all gates start nearly open)
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), 3.0)
        )

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        logger.debug(
            "PrunableLinear(%d → %d): %d gate parameters",
            in_features, out_features, in_features * out_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        return F.linear(x, self.weight * gates, self.bias)

    # ── Inspection helpers ────────────────────────────────────────────────
    def get_gates(self) -> torch.Tensor:
        """Flattened gate values, detached on CPU."""
        return torch.sigmoid(self.gate_scores).detach().cpu().flatten()

    def sparsity_ratio(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below *threshold* (effectively pruned)."""
        return (self.get_gates() < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ─────────────────────────────────────────────────────────────────────────────
class SelfPruningNet(nn.Module):
    """
    Fully-connected network using PrunableLinear throughout.
    Default architecture:  3072 → 1024 → 512 → 256 → 10
    """

    def __init__(
        self,
        input_dim:   int             = 3 * 32 * 32,
        hidden_dims: Tuple[int, ...] = (1024, 512, 256),
        num_classes: int             = 10,
    ) -> None:
        super().__init__()

        dims   = [input_dim] + list(hidden_dims) + [num_classes]
        layers: List[nn.Module] = []

        for i in range(len(dims) - 1):
            layers.append(PrunableLinear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(*layers)

        total_gates = sum(
            l.in_features * l.out_features for l in self.prunable_layers()
        )
        logger.info(
            "SelfPruningNet built | architecture: %s | total gates: %s",
            "-".join(str(d) for d in dims),
            f"{total_gates:,}",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.view(x.size(0), -1))

    def prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 sparsity penalty = mean of ALL gate values.

        .mean() (not .sum()) makes the scale architecture-independent,
        so the same λ range transfers across network sizes.
        """
        all_gates = [
            torch.sigmoid(l.gate_scores).flatten()
            for l in self.prunable_layers()
        ]
        return torch.cat(all_gates).mean()

    def global_sparsity(self, threshold: float = 1e-2) -> float:
        """Percentage of gates below *threshold* across the whole model."""
        all_gates = torch.cat([l.get_gates() for l in self.prunable_layers()])
        return (all_gates < threshold).float().mean().item() * 100.0

    def all_gate_values(self) -> torch.Tensor:
        return torch.cat([l.get_gates() for l in self.prunable_layers()])

    def layer_sparsities(self, threshold: float = 1e-2) -> dict:
        return {
            f"layer_{i}": l.sparsity_ratio(threshold)
            for i, l in enumerate(self.prunable_layers())
        }
