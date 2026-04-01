import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ── data sources ──────────────────────────────────────────────────────────────
CONFIGS = {
    "Empty-8x8": {
        "Baseline": "../results/minigrid_empty_8x8_v0_baseline.csv",
        "LTL":      "../results/empty_ltl.csv",
        "TLTL":     "../results/empty_tltl.csv",
    },
    "LavaGap-S5": {
        "Baseline": "../results/minigrid_lavagaps5_v0_baseline.csv",
        "LTL":      "../results/lava_ltl.csv",
        "TLTL":     "../results/lava_tltl.csv",
    },
    "DoorKey-8x8": {
        "Baseline": "../results/minigrid_doorkey_8x8_v0_baseline.csv",
        "LTL":      "../results/doorkey_ltl.csv",
        "TLTL":     "../results/doorkey_tltl.csv",
    },
}

COLORS  = {"Baseline": "#555555", "LTL": "#2196F3", "TLTL": "#FF5722"}
STYLES  = {"Baseline": "--",      "LTL": "-",        "TLTL": "-"}
SMOOTH  = 15   # rolling-average window

os.makedirs("../plots", exist_ok=True)

def smooth(series, w):
    return series.rolling(window=w, min_periods=1, center=True).mean()

# ── Figure 1: Episode Reward (ep_rew_mean) ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
fig.suptitle("Episode Reward vs Timesteps", fontsize=14, fontweight="bold")

for ax, (env_name, methods) in zip(axes, CONFIGS.items()):
    for method, path in methods.items():
        df = pd.read_csv(path)
        df = df.drop_duplicates(subset="timesteps")
        y  = smooth(df["ep_rew_mean"], SMOOTH)
        ax.plot(df["timesteps"], y,
                label=method,
                color=COLORS[method],
                linestyle=STYLES[method],
                linewidth=1.8)
    ax.set_title(env_name)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("ep_rew_mean (smoothed)")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../plots/reward_curves.png", dpi=150)
plt.close()
print("Saved ../plots/reward_curves.png")

# ── Figure 2: Episode Length (ep_len_mean) ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
fig.suptitle("Episode Length vs Timesteps", fontsize=14, fontweight="bold")

for ax, (env_name, methods) in zip(axes, CONFIGS.items()):
    for method, path in methods.items():
        df = pd.read_csv(path)
        df = df.drop_duplicates(subset="timesteps")
        y  = smooth(df["ep_len_mean"], SMOOTH)
        ax.plot(df["timesteps"], y,
                label=method,
                color=COLORS[method],
                linestyle=STYLES[method],
                linewidth=1.8)
    ax.set_title(env_name)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("ep_len_mean (smoothed)")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../plots/length_curves.png", dpi=150)
plt.close()
print("Saved ../plots/length_curves.png")

# ── Figure 3: Final performance bar chart ─────────────────────────────────────
envs    = list(CONFIGS.keys())
methods = ["Baseline", "LTL", "TLTL"]
x       = np.arange(len(envs))
width   = 0.25

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Final Performance Comparison (last 5% of training)",
             fontsize=13, fontweight="bold")

final_rew = {m: [] for m in methods}
final_len = {m: [] for m in methods}

for env_name, method_paths in CONFIGS.items():
    for method, path in method_paths.items():
        df   = pd.read_csv(path)
        tail = df.tail(max(1, len(df) // 20))  # last 5%
        final_rew[method].append(tail["ep_rew_mean"].mean())
        final_len[method].append(tail["ep_len_mean"].mean())

for i, method in enumerate(methods):
    axes[0].bar(x + i * width, final_rew[method],
                width, label=method, color=COLORS[method], alpha=0.85)
    axes[1].bar(x + i * width, final_len[method],
                width, label=method, color=COLORS[method], alpha=0.85)

for ax, ylabel, title in zip(axes,
    ["Mean Episode Reward", "Mean Episode Length"],
    ["Final Episode Reward", "Final Episode Length"]):
    ax.set_xticks(x + width)
    ax.set_xticklabels(envs)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("../plots/final_comparison.png", dpi=150)
plt.close()
print("Saved ../plots/final_comparison.png")

print("\nAll plots saved to ../plots/")