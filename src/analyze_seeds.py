import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from ltl_wrappers import LTLEmptyWrapper, LTLLavaGapWrapper, LTLDoorKeyWrapper
from tltl_wrappers import TLTLEmptyWrapper, TLTLLavaGapWrapper, TLTLDoorKeyWrapper

N_RUNS = 30
N_EVAL_EPISODES = 100
TIMESTEPS = 100_000
SMOOTH = 15
SAMPLE_EFF_THRESHOLD = 0.8

os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

CONFIGS = [
    {"name": "empty_baseline",   "env_id": "MiniGrid-Empty-8x8-v0",   "env_label": "Empty-8x8",   "method": "Baseline", "wrapper": None},
    {"name": "lava_baseline",    "env_id": "MiniGrid-LavaGapS5-v0",   "env_label": "LavaGap-S5",  "method": "Baseline", "wrapper": None},
    {"name": "doorkey_baseline", "env_id": "MiniGrid-DoorKey-8x8-v0", "env_label": "DoorKey-8x8", "method": "Baseline", "wrapper": None},
    {"name": "empty_ltl",        "env_id": "MiniGrid-Empty-8x8-v0",   "env_label": "Empty-8x8",   "method": "LTL",      "wrapper": LTLEmptyWrapper},
    {"name": "lava_ltl",         "env_id": "MiniGrid-LavaGapS5-v0",   "env_label": "LavaGap-S5",  "method": "LTL",      "wrapper": LTLLavaGapWrapper},
    {"name": "doorkey_ltl",      "env_id": "MiniGrid-DoorKey-8x8-v0", "env_label": "DoorKey-8x8", "method": "LTL",      "wrapper": LTLDoorKeyWrapper},
    {"name": "empty_tltl",       "env_id": "MiniGrid-Empty-8x8-v0",   "env_label": "Empty-8x8",   "method": "TLTL",     "wrapper": TLTLEmptyWrapper},
    {"name": "lava_tltl",        "env_id": "MiniGrid-LavaGapS5-v0",   "env_label": "LavaGap-S5",  "method": "TLTL",     "wrapper": TLTLLavaGapWrapper},
    {"name": "doorkey_tltl",     "env_id": "MiniGrid-DoorKey-8x8-v0", "env_label": "DoorKey-8x8", "method": "TLTL",     "wrapper": TLTLDoorKeyWrapper},
]

COLORS  = {"Baseline": "#555555", "LTL": "#2196F3", "TLTL": "#FF5722"}
STYLES  = {"Baseline": "--",      "LTL": "-",        "TLTL": "-"}
ENVS    = ["Empty-8x8", "LavaGap-S5", "DoorKey-8x8"]
METHODS = ["Baseline", "LTL", "TLTL"]

# ── helpers ───────────────────────────────────────────────────────────────────
def load_and_interpolate(csv_path, col, grid):
    df = pd.read_csv(csv_path).drop_duplicates(subset="timesteps")
    return np.interp(grid, df["timesteps"], df[col])

def smooth(arr, w):
    return pd.Series(arr).rolling(window=w, min_periods=1, center=True).mean().values

def make_env(env_id, WrapperClass):
    env = gym.make(env_id, max_steps=640)
    env = FlatObsWrapper(env)
    if WrapperClass is not None:
        env = WrapperClass(env)
    return env

def is_success(env, terminated, truncated):
    if not terminated or truncated:
        return False
    unwrapped = env.unwrapped
    cell = unwrapped.grid.get(*unwrapped.agent_pos)
    if cell is not None and cell.type == "lava":
        return False
    return True

def compute_sample_efficiency(csv_path, threshold):
    df = pd.read_csv(csv_path).drop_duplicates(subset="timesteps")
    df["smoothed"] = df["ep_len_mean"].rolling(10, min_periods=1).mean()
    max_len = df["ep_len_mean"].max()
    target  = max_len * (1 - threshold)
    hits    = df[df["smoothed"] < target]
    return int(hits.iloc[0]["timesteps"]) if not hits.empty else None

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Learning curves with confidence bands (clipped at 0)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("PART 1: Learning curves with confidence bands")
print("=" * 55)

grid = np.linspace(0, TIMESTEPS, 500)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
fig.suptitle("Episode Length vs Timesteps (mean ± std over 30 runs)",
             fontsize=13, fontweight="bold")

for ax, env_label in zip(axes, ENVS):
    for method in METHODS:
        config = next(c for c in CONFIGS
                      if c["env_label"] == env_label and c["method"] == method)
        curves = []
        for run in range(1, N_RUNS + 1):
            path = f"results/seeds30/{config['name']}_run{run}.csv"
            curves.append(load_and_interpolate(path, "ep_len_mean", grid))

        curves  = np.array(curves)
        mean    = smooth(curves.mean(axis=0), SMOOTH)
        std     = smooth(curves.std(axis=0),  SMOOTH)
        lower   = np.clip(mean - std, 0, None)  # clip at 0
        upper   = mean + std

        ax.plot(grid, mean,
                label=method,
                color=COLORS[method],
                linestyle=STYLES[method],
                linewidth=1.8)
        ax.fill_between(grid, lower, upper,
                        color=COLORS[method], alpha=0.15)

    ax.set_title(env_label)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Length (smoothed)")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/seeds_length_curves.png", dpi=150)
plt.close()
print("  Saved plots/seeds_length_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Evaluate all 45 models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("PART 2: Evaluating all 45 models")
print("=" * 55)

eval_rows = []

for config in CONFIGS:
    name      = config["name"]
    env_id    = config["env_id"]
    env_label = config["env_label"]
    method    = config["method"]
    Wrapper   = config["wrapper"]

    sat_rates    = []
    mean_times   = []
    sample_effs  = []

    for run in range(1, N_RUNS + 1):
        model_path = f"models/seeds30/{name}_run{run}"
        csv_path   = f"results/seeds30/{name}_run{run}.csv"
        print(f"  Evaluating {name} run {run}...")
        model = PPO.load(model_path, device='cpu')

        successes       = 0
        success_lengths = []

        for _ in range(N_EVAL_EPISODES):
            env = make_env(env_id, Wrapper)
            obs, _ = env.reset()
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=False)
                obs, _, terminated, truncated, _ = env.step(action)
                steps += 1
            if is_success(env, terminated, truncated):
                successes += 1
                success_lengths.append(steps)
            env.close()

        sat_rates.append(successes / N_EVAL_EPISODES)
        mean_times.append(np.mean(success_lengths) if success_lengths else np.nan)
        sample_effs.append(compute_sample_efficiency(csv_path, SAMPLE_EFF_THRESHOLD))

    # convert None to nan for sample efficiency
    sample_effs_clean = [s if s is not None else np.nan for s in sample_effs]

    eval_rows.append({
        "env":         env_label,
        "method":      method,
        "sat_mean":    np.mean(sat_rates),
        "sat_std":     np.std(sat_rates),
        "time_mean":   np.nanmean(mean_times),
        "time_std":    np.nanstd(mean_times),
        "seff_mean":   np.nanmean(sample_effs_clean),
        "seff_std":    np.nanstd(sample_effs_clean),
    })

    print(f"    sat:  {np.mean(sat_rates):.3f} ± {np.std(sat_rates):.3f}")
    print(f"    time: {np.nanmean(mean_times):.1f} ± {np.nanstd(mean_times):.1f}")
    print(f"    seff: {np.nanmean(sample_effs_clean):.0f} ± {np.nanstd(sample_effs_clean):.0f}"
          if not np.isnan(np.nanmean(sample_effs_clean)) else "    seff: N/A")

df_eval = pd.DataFrame(eval_rows)
df_eval.to_csv("results/seed_evaluation_metrics.csv", index=False)
print("\nSaved results/seed_evaluation_metrics.csv")
print(df_eval.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Three evaluation bar charts with error bars
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("PART 3: Plotting evaluation metrics")
print("=" * 55)

x     = np.arange(len(ENVS))
width = 0.25

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Evaluation Metrics (mean ± std over 30 runs × 100 episodes)",
             fontsize=12, fontweight="bold")

for i, method in enumerate(METHODS):
    sat_means  = [df_eval[(df_eval.env == e) & (df_eval.method == method)]["sat_mean"].values[0]  for e in ENVS]
    sat_stds   = [df_eval[(df_eval.env == e) & (df_eval.method == method)]["sat_std"].values[0]   for e in ENVS]
    t_means    = [df_eval[(df_eval.env == e) & (df_eval.method == method)]["time_mean"].values[0]  for e in ENVS]
    t_stds     = [df_eval[(df_eval.env == e) & (df_eval.method == method)]["time_std"].values[0]   for e in ENVS]
    se_means   = [df_eval[(df_eval.env == e) & (df_eval.method == method)]["seff_mean"].values[0]  for e in ENVS]
    se_stds    = [df_eval[(df_eval.env == e) & (df_eval.method == method)]["seff_std"].values[0]   for e in ENVS]

    # replace nan with 0 for plotting, track which are nan
    t_means_plot  = [v if not np.isnan(v) else 0 for v in t_means]
    t_stds_plot   = [v if not np.isnan(v) else 0 for v in t_stds]
    se_means_plot = [v if not np.isnan(v) else 0 for v in se_means]
    se_stds_plot  = [v if not np.isnan(v) else 0 for v in se_stds]

    axes[0].bar(x + i * width, sat_means, width,
                yerr=sat_stds, capsize=4,
                label=method, color=COLORS[method], alpha=0.85)
    axes[1].bar(x + i * width, t_means_plot, width,
                yerr=t_stds_plot, capsize=4,
                label=method, color=COLORS[method], alpha=0.85)
    axes[2].bar(x + i * width, se_means_plot, width,
                yerr=se_stds_plot, capsize=4,
                label=method, color=COLORS[method], alpha=0.85)

# Labels and formatting
axes[0].set_title("Specification Satisfaction Rate")
axes[0].set_ylabel("Satisfaction Rate")
axes[0].set_ylim(0, 1.1)

axes[1].set_title("Mean Time to Satisfaction")
axes[1].set_ylabel("Steps")

axes[2].set_title("Sample Efficiency")
axes[2].set_ylabel("Timesteps to 80% length\nreduction")
axes[2].annotate("0 = threshold never reached",
                 xy=(0.98, 0.97), xycoords="axes fraction",
                 ha="right", va="top", fontsize=7, color="gray")

for ax in axes:
    ax.set_xticks(x + width)
    ax.set_xticklabels(ENVS)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/seed_evaluation_metrics.png", dpi=150)
plt.close()
print("  Saved plots/seed_evaluation_metrics.png")

print("\nAll done!")
print("  plots/seeds_length_curves.png     — learning curves with confidence bands")
print("  plots/seed_evaluation_metrics.png — all three metrics with error bars")
print("  results/seed_evaluation_metrics.csv — full numbers table")