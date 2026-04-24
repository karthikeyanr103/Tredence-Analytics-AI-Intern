"""
train.py
========
Entry point for running self-pruning experiments on CIFAR-10.

Usage examples
--------------
# Quick run (no W&B)
python train.py --no-wandb

# Full run with W&B
python train.py --wandb-key YOUR_KEY --epochs 30 --lambdas 0.1 0.5 1.0

# Custom lambdas + faster GIF
python train.py --lambdas 0.05 0.3 0.8 --gif-fps 6 --gif-step 2 --no-wandb
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

from src.logging_setup import setup_logging
from src.data          import get_loaders
from src.trainer       import run_experiment
from src.visualize     import plot_gate_distributions, make_training_gif

# ── optional W&B ──────────────────────────────────────────────────────────────
try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """CUDA if available and functional, else CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1).cuda()
        return torch.device("cuda")
    except Exception as exc:
        logger.warning("CUDA unusable (%s) – falling back to CPU.", exc)
        return torch.device("cpu")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-Pruning Neural Network on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--lambdas",     type=float, nargs="+", default=[0.1, 0.5, 1.0],
                   help="Sparsity penalty weights to sweep")
    p.add_argument("--data-dir",    type=str,   default="./data")
    p.add_argument("--output-dir",  type=str,   default="./outputs")
    p.add_argument("--log-dir",     type=str,   default="./logs")
    p.add_argument("--checkpoint",  action="store_true",
                   help="Save best model checkpoints during training")
    p.add_argument("--gif-fps",     type=int,   default=4)
    p.add_argument("--gif-step",    type=int,   default=1,
                   help="Sample every N epochs for GIF (reduces size)")
    p.add_argument("--wandb-key",   type=str,   default=None)
    p.add_argument("--no-wandb",    action="store_true",
                   help="Disable W&B logging entirely")
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> None:
    args = parse_args(argv)

    # ── Logging ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(
        log_dir=args.log_dir,
        console_level=getattr(logging, args.log_level),
        run_name=f"lambdas_{'_'.join(str(l) for l in args.lambdas)}",
    )

    logger.info("=" * 60)
    logger.info("Self-Pruning Neural Network  —  CIFAR-10")
    logger.info("=" * 60)
    logger.info("Config: epochs=%d  lr=%.0e  batch=%d  λ=%s",
                args.epochs, args.lr, args.batch_size, args.lambdas)

    # ── Output directories ────────────────────────────────────────────────
    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints" if args.checkpoint else None
    out_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Device + data ─────────────────────────────────────────────────────
    device = get_device()
    logger.info("Device: %s", device)

    train_loader, test_loader = get_loaders(
        args.data_dir, args.batch_size)

    # ── W&B setup ─────────────────────────────────────────────────────────
    use_wandb   = _WANDB_OK and not args.no_wandb
    project_name = f"self-pruning-nn-{ts}"
    if use_wandb and args.wandb_key:
        wandb.login(key=args.wandb_key)
    if use_wandb:
        logger.info("W&B project: %s", project_name)
    else:
        logger.info("W&B logging disabled")

    # ── Experiments ───────────────────────────────────────────────────────
    results:       list[tuple] = []
    gate_data:     list[tuple] = []
    all_histories: list[list]  = []

    for lam in args.lambdas:
        wandb_run = None
        if use_wandb:
            wandb_run = wandb.init(
                project=project_name,
                name=f"lambda_{lam:.0e}",
                config={
                    "lambda_sparse": lam,
                    "epochs":   args.epochs,
                    "lr":       args.lr,
                    "gate_lr":  args.lr * 10,
                    "architecture": "3072-1024-512-256-10",
                    "optimizer":    "Adam",
                    "scheduler":    "CosineAnnealingLR",
                    "sparsity_loss": "mean(sigmoid(gate_scores))",
                },
                reinit=True,
            )

        acc, spar, gates, history = run_experiment(
            lambda_sparse  = lam,
            train_loader   = train_loader,
            test_loader    = test_loader,
            device         = device,
            epochs         = args.epochs,
            lr             = args.lr,
            checkpoint_dir = str(ckpt_dir) if ckpt_dir else None,
            wandb_run      = wandb_run,
        )

        if wandb_run:
            wandb_run.finish()

        results.append((lam, acc, spar))
        gate_data.append((lam, acc, spar, gates))
        all_histories.append(history)

    # ── Summary table ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 55)
    logger.info("  %-10s  %-14s  %-14s", "Lambda", "Test Acc (%)", "Sparsity (%)")
    logger.info("  " + "-" * 51)
    for lam, acc, spar in results:
        logger.info("  %-10.4f  %-14.2f  %-14.2f", lam, acc, spar)
    logger.info("=" * 55)

    # ── Plots ─────────────────────────────────────────────────────────────
    dist_path = str(out_dir / "gate_distributions.png")
    gif_path  = str(out_dir / "training_progress.gif")

    plot_gate_distributions(gate_data, save_path=dist_path)
    make_training_gif(
        all_histories, args.lambdas,
        save_path=gif_path,
        frame_step=args.gif_step,
        fps=args.gif_fps,
    )

    logger.info("All outputs written to: %s", out_dir)


if __name__ == "__main__":
    main()
