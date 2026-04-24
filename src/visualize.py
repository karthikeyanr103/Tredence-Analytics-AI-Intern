"""
src/visualize.py
================
Plotting utilities:
  • plot_gate_distributions()  – final gate histograms (static PNG)
  • make_training_gif()        – animated training-progress GIF
"""

import io
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Dark-theme palette ────────────────────────────────────────────────────────
BG_DARK  = "#0d1117"
BG_PANEL = "#161b22"
ACCENT   = "#e6edf3"
GRID     = "#21262d"
COLORS   = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"]

plt.rcParams.update({
    "font.family":      "monospace",
    "axes.facecolor":   BG_PANEL,
    "figure.facecolor": BG_DARK,
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       GRID,
    "text.color":       "#c9d1d9",
})


# ─────────────────────────────────────────────────────────────────────────────
def plot_gate_distributions(
    gate_data: list[tuple[float, float, float, Any]],
    save_path: str = "outputs/gate_distributions.png",
) -> None:
    """
    Side-by-side histograms of final gate values for every λ.

    Parameters
    ----------
    gate_data : list of (lambda, test_acc, sparsity_pct, gate_tensor)
    save_path : output PNG path
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    n = len(gate_data)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), facecolor=BG_DARK)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Distribution of Final Gate Values  ·  σ(gate_score) per weight\n"
        "spike at 0 = pruned connection   ·   cluster away from 0 = active",
        fontsize=12, fontweight="bold", color=ACCENT, y=1.04,
    )

    for ax, (lam, acc, spar, gates), color in zip(axes, gate_data, COLORS):
        ax.set_facecolor(BG_PANEL)
        g = gates.numpy() if hasattr(gates, "numpy") else np.array(gates)

        counts, bins, patches = ax.hist(
            g, bins=100, range=(0, 1), color=color, edgecolor="none", alpha=0.85)
        # Colour pruned region red
        for patch, left in zip(patches, bins[:-1]):
            if left < 0.01:
                patch.set_facecolor("#ff6b6b")
                patch.set_alpha(0.9)

        ax.axvline(x=0.01, color="white", linestyle="--",
                   linewidth=1.0, alpha=0.7, label="prune threshold (0.01)")
        ax.set_title(
            f"λ = {lam:.2f}\nAcc = {acc:.1f}%  ·  Sparsity = {spar:.1f}%",
            fontsize=10, fontweight="bold", color=color, pad=6,
        )
        ax.set_xlabel("Gate value  σ(gate_score)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, labelcolor="#c9d1d9", framealpha=0.2)

        pruned = (g < 0.01).mean() * 100
        ax.text(0.03, 0.92, f"PRUNED\n{pruned:.0f}%",
                transform=ax.transAxes, fontsize=8, color="#ff6b6b",
                va="top", bbox=dict(fc=BG_PANEL, ec="#ff6b6b",
                                    alpha=0.7, boxstyle="round,pad=0.3"))
        ax.text(0.72, 0.92, f"ACTIVE\n{100-pruned:.0f}%",
                transform=ax.transAxes, fontsize=8, color=color,
                va="top", bbox=dict(fc=BG_PANEL, ec=color,
                                    alpha=0.7, boxstyle="round,pad=0.3"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    logger.info("Gate-distribution plot saved → %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
def make_training_gif(
    all_histories: list[list[dict]],
    lambdas: list[float],
    save_path: str = "outputs/training_progress.gif",
    frame_step: int = 1,
    fps: int = 4,
) -> None:
    """
    Animated GIF showing training dynamics for ALL λ values at once.

    Each frame = one epoch.  Each λ row has three panels:
        1. Gate histogram  (pruning develops visually over time)
        2. Sparsity curve  (cumulative pruned %)
        3. Accuracy curve  (test accuracy trajectory)

    Parameters
    ----------
    all_histories : list of history dicts (one list per λ, from trainer)
    lambdas       : λ values in the same order
    save_path     : output .gif path
    frame_step    : render every N-th epoch (reduces file size)
    fps           : playback speed
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    max_epochs    = max(len(h) for h in all_histories)
    epoch_indices = list(range(0, max_epochs, frame_step))
    if (max_epochs - 1) not in epoch_indices:
        epoch_indices.append(max_epochs - 1)

    n_lam  = len(lambdas)
    frames = []

    for ei in epoch_indices:
        fig = plt.figure(figsize=(18, 3.8 * n_lam + 1.2), facecolor=BG_DARK)
        fig.patch.set_facecolor(BG_DARK)

        fig.text(0.5, 0.98,
                 "Self-Pruning Neural Network  ·  CIFAR-10  ·  Training Progress",
                 ha="center", va="top", fontsize=13, fontweight="bold",
                 color=ACCENT, family="monospace")
        fig.text(0.5, 0.955,
                 f"Epoch  {ei + 1:02d} / {max_epochs}",
                 ha="center", va="top", fontsize=10,
                 color="#58a6ff", family="monospace")

        outer = gridspec.GridSpec(
            n_lam, 1, figure=fig,
            top=0.93, bottom=0.06, left=0.05, right=0.97,
            hspace=0.6,
        )

        for row_i, (sd_hist, lam, color) in enumerate(
                zip(all_histories, lambdas, COLORS)):

            ei_cap    = min(ei, len(sd_hist) - 1)
            snap      = sd_hist[ei_cap]
            gates_np  = snap["gates_snapshot"]
            spar_now  = snap["sparsity/global_%"]
            acc_now   = snap.get("eval_acc", None)

            inner = gridspec.GridSpecFromSubplotSpec(
                1, 3, subplot_spec=outer[row_i],
                wspace=0.38, width_ratios=[2.6, 1.7, 1.7],
            )

            # ── 1. Gate histogram ─────────────────────────────────────────
            ax_h = fig.add_subplot(inner[0])
            ax_h.set_facecolor(BG_PANEL)

            _, bins, patches = ax_h.hist(
                gates_np, bins=80, range=(0, 1),
                color=color, edgecolor="none", alpha=0.85)
            for patch, left in zip(patches, bins[:-1]):
                if left < 0.01:
                    patch.set_facecolor("#ff6b6b"); patch.set_alpha(0.9)

            ax_h.axvline(x=0.01, color="white", linestyle="--",
                         linewidth=0.9, alpha=0.65, label="threshold")
            ax_h.set_xlim(0, 1)
            ax_h.set_title(
                f"λ = {lam:.2f}   |   Sparsity {spar_now:.1f}%",
                fontsize=9, fontweight="bold", color=color, pad=4)
            ax_h.set_xlabel("Gate value σ(score)", fontsize=8)
            ax_h.set_ylabel("Count", fontsize=8)
            ax_h.legend(fontsize=7, labelcolor="#c9d1d9", framealpha=0.15)

            pruned_p = (gates_np < 0.01).mean() * 100
            ax_h.text(0.03, 0.90, f"PRUNED {pruned_p:.0f}%",
                      transform=ax_h.transAxes, fontsize=7,
                      color="#ff6b6b", va="top",
                      bbox=dict(fc=BG_PANEL, ec="#ff6b6b",
                                alpha=0.65, boxstyle="round,pad=0.25"))
            ax_h.text(0.65, 0.90, f"ACTIVE {100-pruned_p:.0f}%",
                      transform=ax_h.transAxes, fontsize=7,
                      color=color, va="top",
                      bbox=dict(fc=BG_PANEL, ec=color,
                                alpha=0.65, boxstyle="round,pad=0.25"))

            # ── 2. Sparsity curve ─────────────────────────────────────────
            ax_s = fig.add_subplot(inner[1])
            ax_s.set_facecolor(BG_PANEL)
            ep_x    = [h["epoch"]           for h in sd_hist[: ei_cap + 1]]
            spar_y  = [h["sparsity/global_%"] for h in sd_hist[: ei_cap + 1]]

            ax_s.plot(ep_x, spar_y, color=color, linewidth=2)
            ax_s.fill_between(ep_x, spar_y, alpha=0.14, color=color)
            if ep_x:
                ax_s.scatter([ep_x[-1]], [spar_y[-1]],
                             color=color, s=45, zorder=5)
            ax_s.set_xlim(1, max_epochs); ax_s.set_ylim(0, 100)
            ax_s.set_xlabel("Epoch", fontsize=8)
            ax_s.set_ylabel("Sparsity (%)", fontsize=8)
            ax_s.set_title("Sparsity over training", fontsize=8, pad=4)
            ax_s.grid(True, alpha=0.2)
            ax_s.text(0.97, 0.07, f"{spar_now:.1f}%",
                      transform=ax_s.transAxes, ha="right", va="bottom",
                      fontsize=9, fontweight="bold", color=color)

            # ── 3. CE-loss curve ──────────────────────────────────────────
            ax_a = fig.add_subplot(inner[2])
            ax_a.set_facecolor(BG_PANEL)
            ce_y = [h["loss/cross_entropy"] for h in sd_hist[: ei_cap + 1]]

            ax_a.plot(ep_x, ce_y, color=color, linewidth=2)
            ax_a.fill_between(ep_x, ce_y, alpha=0.14, color=color)
            if ep_x:
                ax_a.scatter([ep_x[-1]], [ce_y[-1]],
                             color=color, s=45, zorder=5)
            ax_a.set_xlim(1, max_epochs)
            ax_a.set_xlabel("Epoch", fontsize=8)
            ax_a.set_ylabel("CE Loss", fontsize=8)
            ax_a.set_title("Cross-entropy loss", fontsize=8, pad=4)
            ax_a.grid(True, alpha=0.2)
            if ce_y:
                ax_a.text(0.97, 0.92, f"{ce_y[-1]:.3f}",
                          transform=ax_a.transAxes, ha="right", va="top",
                          fontsize=9, fontweight="bold", color=color)

        # Legend strip at bottom
        for i, (lam, color) in enumerate(zip(lambdas, COLORS)):
            xpos = 0.18 + i * 0.32
            fig.text(xpos, 0.018, "■ ", color=color,
                     fontsize=12, ha="center", va="bottom")
            fig.text(xpos + 0.02, 0.018, f"λ = {lam:.2f}",
                     color="#c9d1d9", fontsize=8, ha="left", va="bottom")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=90,
                    bbox_inches="tight", facecolor=BG_DARK)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

        if (ei + 1) % 5 == 0 or ei == 0:
            logger.debug("GIF: rendered frame epoch=%d/%d", ei + 1, max_epochs)

    # Hold last frame longer
    frames += [frames[-1]] * 8

    frames[0].save(
        save_path, save_all=True, append_images=frames[1:],
        loop=0, duration=int(1000 / fps), optimize=True,
    )
    logger.info(
        "Training GIF saved → %s  (%d frames, %d fps)",
        save_path, len(frames), fps,
    )
