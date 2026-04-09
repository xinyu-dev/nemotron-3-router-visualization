"""
visualize_router.py
~~~~~~~~~~~~~~~~~~~
Generate visualizations from a RouterData .npz file produced by
extract_router_data.py.

Outputs (written to the output/ folder by default)
-------
  router_viz_1_expert_frequency.png  — bar chart of expert selection frequency
  router_viz_2_avg_score.png         — avg router score heatmap (MoE Layer × Expert)
  router_viz_3_top1_expert.png       — top-1 expert routed (MoE Layer × Token)

Usage
-----
  python visualize_router.py
  python visualize_router.py --input output/router_data.npz
"""

import argparse
import colorsys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_expert_frequency(topk_indices, n_experts, subtitle, out_dir):
    """Bar chart: how often each expert is selected across all tokens and layers."""
    expert_counts = np.zeros(n_experts, dtype=int)
    for idx in topk_indices.reshape(-1):
        expert_counts[idx] += 1

    top10 = np.argsort(expert_counts)[-10:]
    colors = ["#e05c5c" if i in top10 else "#6baed6" for i in range(n_experts)]

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle(f"Nemotron-3 MoE Router Analysis — {subtitle}", fontsize=11, fontweight="bold")

    ax.bar(range(n_experts), expert_counts, color=colors, width=1.0, linewidth=0)
    ax.set_xlabel("Expert ID", fontsize=11)
    ax.set_ylabel("Selection count", fontsize=11)
    ax.set_title("Expert Selection Frequency (all tokens × all MoE layers)", fontsize=12)
    ax.set_xlim(-1, n_experts)

    for i in top10:
        ax.text(i, expert_counts[i] + 0.3, str(i),
                ha="center", va="bottom", fontsize=7, color="#c0392b", fontweight="bold")

    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#e05c5c", label="Top-10 experts"),
            plt.Rectangle((0, 0), 1, 1, color="#6baed6", label="Other experts"),
        ],
        loc="upper right", fontsize=9,
    )

    plt.tight_layout()
    save(fig, out_dir / "router_viz_1_expert_frequency.png")


def plot_avg_score(scores, layer_labels, subtitle, out_dir):
    """Heatmap: avg sigmoid router score per MoE layer × expert."""
    avg_scores = scores.mean(axis=0)  # (n_layers, n_experts)
    n_layers, n_experts = avg_scores.shape

    fig_h = max(5, n_layers * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.suptitle(f"Nemotron-3 MoE Router Analysis — {subtitle}", fontsize=11, fontweight="bold")

    im = ax.imshow(avg_scores, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Expert ID", fontsize=10)
    ax.set_ylabel("MoE Layer", fontsize=10)
    ax.set_title("Avg Router Score per MoE Layer × Expert (averaged across tokens)", fontsize=11)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_labels, fontsize=8)
    ax.set_xticks(range(0, n_experts, 16))
    plt.colorbar(im, ax=ax, label="Avg sigmoid score")

    plt.tight_layout()
    save(fig, out_dir / "router_viz_2_avg_score.png")


def plot_top1_expert(topk_indices, token_labels, layer_labels, n_experts, subtitle, out_dir):
    """Heatmap: top-1 expert per (MoE layer × token), discrete color per expert."""
    top1_expert = topk_indices[:, :, 0]  # (n_tokens, n_layers)
    n_tokens, n_layers = top1_expert.shape

    # One unique color per expert that appears in the data
    unique_experts = np.unique(top1_expert)
    n_unique = len(unique_experts)
    hues = np.linspace(0, 1, n_unique, endpoint=False)
    palette = [colorsys.hsv_to_rgb(h, 0.75, 0.9) for h in hues]

    expert_to_idx = {e: i for i, e in enumerate(unique_experts)}
    indexed = np.vectorize(expert_to_idx.get)(top1_expert)  # (n_tokens, n_layers)

    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(-0.5, n_unique), n_unique)

    # Transpose: rows = MoE layers (y), cols = tokens (x)
    fig_h = max(8, n_layers * 0.6)
    fig_w = max(16, n_tokens * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.suptitle(f"Nemotron-3 MoE Router Analysis — {subtitle}", fontsize=11, fontweight="bold")

    ax.imshow(indexed.T, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlabel("Token", fontsize=10)
    ax.set_ylabel("MoE Layer", fontsize=10)
    ax.set_title("Top-1 Expert Routed (MoE layer × token)  |  cell label = expert ID", fontsize=11)
    ax.set_xticks(range(n_tokens))
    ax.set_xticklabels(token_labels, fontsize=7, rotation=90)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_labels, fontsize=8)

    # Annotate each cell with its expert ID
    cell_h_in = fig_h / n_layers
    fontsize = max(4, min(8, cell_h_in * 8))
    for ti in range(n_tokens):
        for li in range(n_layers):
            ax.text(ti, li, str(top1_expert[ti, li]),
                    ha="center", va="center", fontsize=fontsize,
                    color="white", fontweight="bold")

    plt.tight_layout()
    save(fig, out_dir / "router_viz_3_top1_expert.png", dpi=200)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Nemotron-3 MoE router data from a .npz file."
    )
    parser.add_argument(
        "--input",
        default="output/router_data.npz",
        help="Path to the .npz file produced by extract_router_data.py (default: output/router_data.npz).",
    )
    args = parser.parse_args()

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    data = np.load(args.input, allow_pickle=True)
    scores       = data["scores"]             # (N, num_moe_layers, n_routed_experts)
    topk_indices = data["topk_indices"]       # (N, num_moe_layers, top_k)
    tokens       = data["tokens"]             # (N,)
    moe_layers   = data["moe_layer_indices"]  # (num_moe_layers,)

    n_tokens, n_layers, n_experts = scores.shape
    n_topk = topk_indices.shape[2]

    token_labels = [f"{t!r}" for t in tokens]
    layer_labels = [f"L{l}" for l in moe_layers]
    subtitle = f"({n_tokens} tokens · {n_layers} MoE layers · {n_experts} experts · top-{n_topk})"

    print(f"Loaded {args.input}  —  {subtitle}")

    plot_expert_frequency(topk_indices, n_experts, subtitle, out_dir)
    plot_avg_score(scores, layer_labels, subtitle, out_dir)
    plot_top1_expert(topk_indices, token_labels, layer_labels, n_experts, subtitle, out_dir)


if __name__ == "__main__":
    main()
