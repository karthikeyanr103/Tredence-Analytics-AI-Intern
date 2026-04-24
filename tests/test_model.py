"""
tests/test_model.py
===================
Unit tests for PrunableLinear and SelfPruningNet.
Run with:  pytest tests/ -v
"""

import math
import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PrunableLinear, SelfPruningNet


# ─────────────────────────────────────────────────────────────────────────────
class TestPrunableLinear:

    def test_output_shape(self):
        layer = PrunableLinear(64, 32)
        x = torch.randn(8, 64)
        assert layer(x).shape == (8, 32)

    def test_gates_in_unit_interval(self):
        layer = PrunableLinear(16, 8)
        gates = layer.get_gates()
        assert gates.min() >= 0.0
        assert gates.max() <= 1.0

    def test_initial_gates_near_095(self):
        """sigmoid(3.0) ≈ 0.9526 — gates should start almost fully open."""
        layer = PrunableLinear(32, 16)
        gates = layer.get_gates()
        assert gates.mean().item() == pytest.approx(0.9526, abs=1e-3)

    def test_gradients_flow_through_gates(self):
        layer = PrunableLinear(8, 4)
        x = torch.randn(2, 8)
        out = layer(x).sum()
        out.backward()
        assert layer.gate_scores.grad is not None
        assert layer.weight.grad is not None

    def test_sparsity_ratio_zero_initially(self):
        """No gates should be pruned at initialisation."""
        layer = PrunableLinear(64, 32)
        assert layer.sparsity_ratio(threshold=1e-2) == pytest.approx(0.0, abs=1e-6)

    def test_sparsity_ratio_after_forced_close(self):
        """Force all gate_scores very negative → all gates ≈ 0."""
        layer = PrunableLinear(16, 8)
        with torch.no_grad():
            layer.gate_scores.fill_(-100.0)
        assert layer.sparsity_ratio(threshold=1e-2) == pytest.approx(1.0, abs=1e-4)

    def test_masked_weight_effect(self):
        """With gates = 0 the output should depend only on bias."""
        layer = PrunableLinear(4, 2)
        with torch.no_grad():
            layer.gate_scores.fill_(-100.0)  # gates → 0
            layer.bias.fill_(3.0)
        x = torch.randn(5, 4)
        out = layer(x)
        expected = torch.full((5, 2), 3.0)
        assert torch.allclose(out, expected, atol=1e-4)

    def test_extra_repr(self):
        layer = PrunableLinear(128, 64)
        assert "128" in repr(layer)
        assert "64"  in repr(layer)


# ─────────────────────────────────────────────────────────────────────────────
class TestSelfPruningNet:

    @pytest.fixture
    def model(self):
        return SelfPruningNet(input_dim=32, hidden_dims=(16,), num_classes=4)

    def test_forward_shape(self, model):
        x = torch.randn(10, 32)
        assert model(x).shape == (10, 4)

    def test_forward_from_image_batch(self):
        """Accepts CIFAR-10-style (B, C, H, W) input."""
        model = SelfPruningNet()
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_prunable_layers_count(self):
        model = SelfPruningNet(input_dim=32, hidden_dims=(16, 8), num_classes=4)
        assert len(model.prunable_layers()) == 3  # 3 linear layers

    def test_sparsity_loss_is_scalar(self, model):
        loss = model.sparsity_loss()
        assert loss.shape == torch.Size([])

    def test_sparsity_loss_in_unit_interval(self, model):
        loss = model.sparsity_loss().item()
        assert 0.0 < loss < 1.0

    def test_global_sparsity_zero_initially(self, model):
        assert model.global_sparsity() == pytest.approx(0.0, abs=1e-2)

    def test_global_sparsity_100_after_force_close(self, model):
        for layer in model.prunable_layers():
            with torch.no_grad():
                layer.gate_scores.fill_(-100.0)
        assert model.global_sparsity() == pytest.approx(100.0, abs=1e-2)

    def test_sparsity_loss_gradient(self, model):
        """Gradient of sparsity_loss w.r.t. gate_scores must be non-zero."""
        loss = model.sparsity_loss()
        loss.backward()
        for layer in model.prunable_layers():
            assert layer.gate_scores.grad is not None
            assert layer.gate_scores.grad.abs().sum().item() > 0

    def test_layer_sparsities_keys(self, model):
        spars = model.layer_sparsities()
        assert set(spars.keys()) == {"layer_0", "layer_1", "layer_2"}

    def test_all_gate_values_length(self, model):
        gates = model.all_gate_values()
        expected = sum(
            l.in_features * l.out_features
            for l in model.prunable_layers()
        )
        assert len(gates) == expected

    def test_total_loss_backward(self):
        """End-to-end: task + sparsity loss; all grads flow."""
        import torch.nn.functional as F
        model = SelfPruningNet(input_dim=32, hidden_dims=(16,), num_classes=4)
        x     = torch.randn(8, 32)
        y     = torch.randint(0, 4, (8,))
        loss  = F.cross_entropy(model(x), y) + 0.5 * model.sparsity_loss()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
