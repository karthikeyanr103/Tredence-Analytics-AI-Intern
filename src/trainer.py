"""
src/trainer.py
==============
Training loop, evaluation, and single-experiment orchestration.

Total Loss  =  CrossEntropy(logits, labels)  +  λ · mean(gate_values)
              ─────────────────────────────    ────────────────────────
              pushes correct classification     pushes all gates → 0
"""

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .model import SelfPruningNet

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: SelfPruningNet,
    loader,
    optimizer: torch.optim.Optimizer,
    lambda_sparse: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Single pass through the training data.

    Returns
    -------
    (avg_total_loss, avg_ce_loss, avg_sparsity_loss)
    """
    model.train()
    total_sum = ce_sum = spar_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits      = model(images)
        ce_loss     = F.cross_entropy(logits, labels)
        spar_loss   = model.sparsity_loss()
        total_loss  = ce_loss + lambda_sparse * spar_loss

        total_loss.backward()
        optimizer.step()

        total_sum += total_loss.item()
        ce_sum    += ce_loss.item()
        spar_sum  += spar_loss.item()

    n = len(loader)
    return total_sum / n, ce_sum / n, spar_sum / n


@torch.no_grad()
def evaluate(model: SelfPruningNet, loader, device: torch.device) -> float:
    """Top-1 test accuracy (%)."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds    = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    acc = 100.0 * correct / total
    logger.debug("Evaluation  accuracy=%.2f%%  (%d/%d)", acc, correct, total)
    return acc


# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(
    lambda_sparse: float,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    checkpoint_dir: str | None = None,
    wandb_run: Any | None = None,
) -> tuple[float, float, Any, list]:
    """
    Train one SelfPruningNet with a given λ value.

    Parameters
    ----------
    lambda_sparse   : sparsity penalty weight
    train_loader    : DataLoader for training split
    test_loader     : DataLoader for test/val split
    device          : torch.device
    epochs          : number of training epochs
    lr              : base learning rate (gate params use lr × 10)
    checkpoint_dir  : if set, saves best model checkpoint here
    wandb_run       : active wandb run (or None to skip logging)

    Returns
    -------
    (test_accuracy, sparsity_pct, gate_tensor, history_list)
    """
    model = SelfPruningNet().to(device)

    # Gate scores adapt at 10× the weight learning rate so pruning
    # decisions emerge before the weights fully overfit to them.
    weight_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]
    gate_params   = [p for n, p in model.named_parameters() if "gate_scores"     in n]

    optimizer = torch.optim.Adam([
        {"params": weight_params, "lr": lr,      "weight_decay": 1e-4},
        {"params": gate_params,   "lr": lr * 10, "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info(
        "Experiment start  λ=%.4f  epochs=%d  lr=%.0e  device=%s",
        lambda_sparse, epochs, lr, device,
    )

    history: list[dict] = []
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        total_l, ce_l, spar_l = train_one_epoch(
            model, train_loader, optimizer, lambda_sparse, device)
        scheduler.step()

        sparsity      = model.global_sparsity()
        layer_spars   = model.layer_sparsities()
        gates_snapshot = model.all_gate_values().numpy().copy()
        elapsed       = time.perf_counter() - t0

        row = {
            "epoch":              epoch,
            "loss/total":         total_l,
            "loss/cross_entropy": ce_l,
            "loss/sparsity":      spar_l,
            "sparsity/global_%":  sparsity,
            "lr":                 scheduler.get_last_lr()[0],
            "epoch_time_s":       elapsed,
            "gates_snapshot":     gates_snapshot,
            **{f"sparsity/{k}": v for k, v in layer_spars.items()},
        }
        history.append(row)

        if wandb_run:
            wandb_run.log({k: v for k, v in row.items()
                           if k != "gates_snapshot"})

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "  λ=%.2f  epoch %3d/%d  "
                "total=%.4f  CE=%.4f  spar=%.4f  "
                "sparsity=%.1f%%  %.1fs",
                lambda_sparse, epoch, epochs,
                total_l, ce_l, spar_l, sparsity, elapsed,
            )

        # Optional checkpoint: save when accuracy improves
        if checkpoint_dir and epoch % 5 == 0:
            test_acc_mid = evaluate(model, test_loader, device)
            if test_acc_mid > best_acc:
                best_acc = test_acc_mid
                ckpt_path = (
                    Path(checkpoint_dir)
                    / f"lambda_{lambda_sparse:.2f}_best.pt"
                )
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(),
                     "accuracy": best_acc, "sparsity": sparsity},
                    ckpt_path,
                )
                logger.info("  ✓ Checkpoint saved → %s  (acc=%.2f%%)", ckpt_path, best_acc)

    # ── Final evaluation ──────────────────────────────────────────────────
    test_acc = evaluate(model, test_loader, device)
    sparsity = model.global_sparsity()
    gates    = model.all_gate_values()

    if wandb_run:
        import wandb as _wandb
        wandb_run.log({
            "final/test_accuracy_%": test_acc,
            "final/sparsity_%":      sparsity,
        })
        wandb_run.log({
            "gates/distribution": _wandb.Histogram(gates.numpy(), num_bins=100)
        })

    logger.info(
        "Experiment done   λ=%.4f  acc=%.2f%%  sparsity=%.2f%%",
        lambda_sparse, test_acc, sparsity,
    )
    return test_acc, sparsity, gates, history
