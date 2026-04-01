import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from ltl_wrappers import LTLEmptyWrapper, LTLLavaGapWrapper, LTLDoorKeyWrapper
from tltl_wrappers import TLTLEmptyWrapper, TLTLLavaGapWrapper, TLTLDoorKeyWrapper

N_EVAL_EPISODES = 200
SAMPLE_EFF_THRESHOLD = 0.8  # 80% satisfaction rate threshold
os.makedirs("../plots", exist_ok=True)

# ── model configs ─────────────────────────────────────────────────────────────
CONFIGS = [
    {
        "env_id":   "MiniGrid-Empty-8x8-v0",
        "env_name": "Empty-8x8",
        "models": {
            "Baseline": ("../ppo_minigrid_empty_8x8_v0",    None),
            "LTL":      ("../ppo_empty_ltl",                LTLEmptyWrapper),
            "TLTL":     ("../ppo_empty_tltl",               TLTLEmptyWrapper),
        },
        "csv": {
            "Baseline": "../results/minigrid_empty_8x8_v0_baseline.csv",
            "LTL":      "../results/empty_ltl.csv",
            "TLTL":     "../results/empty_tltl.csv",
        }
    },
    {
        "env_id":   "MiniGrid-LavaGapS5-v0",
        "env_name": "LavaGap-S5",
        "models": {
            "Baseline": ("../ppo_minigrid_lavagaps5_v0",    None),
            "LTL":      ("../ppo_lava_ltl",                 LTLLavaGapWrapper),
            "TLTL":     ("../ppo_lava_tltl",                TLTLLavaGapWrapper),
        },
        "csv": {
            "Baseline": "../results/minigrid_lavagaps5_v0_baseline.csv",
            "LTL":      "../results/lava_ltl.csv",
            "TLTL":     "../results/lava_tltl.csv",
        }
    },
    {
        "env_id":   "MiniGrid-DoorKey-8x8-v0",
        "env_name": "DoorKey-8x8",
        "models": {
            "Baseline": ("../ppo_minigrid_doorkey_8x8_v0",  None),
            "LTL":      ("../ppo_doorkey_ltl",              LTLDoorKeyWrapper),
            "TLTL":     ("../ppo_doorkey_tltl",             TLTLDoorKeyWrapper),
        },
        "csv": {
            "Baseline": "../results/minigrid_doorkey_8x8_v0_baseline.csv",
            "LTL":      "../results/doorkey_ltl.csv",
            "TLTL":     "../results/doorkey_tltl.csv",
        }
    },
]

COLORS = {"Baseline": "#555555", "LTL": "#2196F3", "TLTL": "#FF5722"}

# ── helper: check success from underlying env ─────────────────────────────────
def is_success(env, terminated, truncated):
    """
    Success = reached goal without truncation.
    For LavaGap: also check agent isn't on lava.
    """
    if not terminated or truncated:
        return False
    unwrapped = env.unwrapped
    agent_pos = unwrapped.agent_pos
    cell = unwrapped.grid.get(*agent_pos)
    # If standing on lava, it's a failure
    if cell is not None and cell.type == "lava":
        return False
    return True

# ── helper: make env with correct wrapper stack ───────────────────────────────
def make_env(env_id, WrapperClass):
    env = gym.make(env_id, max_steps=640)
    env = FlatObsWrapper(env)
    if WrapperClass is not None:
        env = WrapperClass(env)
    return env

# ── helper: compute sample efficiency from CSV ────────────────────────────────
def compute_sample_efficiency(csv_path, threshold, env_id):
    """
    Find the timestep where ep_len_mean first drops below a threshold
    as a proxy for sample efficiency (lower ep_len = faster solving).
    We use episode length rather than reward since reward scales differ.
    For DoorKey baseline which never solves, return None.
    """
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="timesteps")

    # Use a smoothed version to avoid noise
    df["smoothed_len"] = df["ep_len_mean"].rolling(10, min_periods=1).mean()

    # Find max episode length (used as "unsolved" baseline)
    max_len = df["ep_len_mean"].max()

    # Threshold: episode length drops to below threshold * max_len
    target = max_len * (1 - threshold)
    hits = df[df["smoothed_len"] < target]

    if hits.empty:
        return None  # never reached threshold
    return int(hits.iloc[0]["timesteps"])

# ── main evaluation loop ──────────────────────────────────────────────────────
results = []

for config in CONFIGS:
    env_id   = config["env_id"]
    env_name = config["env_name"]
    print(f"\n{'='*50}")
    print(f"Evaluating: {env_name}")
    print(f"{'='*50}")

    for method, (model_path, WrapperClass) in config["models"].items():
        print(f"\n  [{method}] Loading {model_path}...")
        model = PPO.load(model_path, device="cpu")

        successes  = 0
        success_lengths = []

        for ep in range(N_EVAL_EPISODES):
            env = make_env(env_id, WrapperClass)
            obs, _ = env.reset()
            terminated = truncated = False
            steps = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1

            if is_success(env, terminated, truncated):
                successes += 1
                success_lengths.append(steps)
            env.close()

        sat_rate = successes / N_EVAL_EPISODES
        mean_time = np.mean(success_lengths) if success_lengths else None
        sample_eff = compute_sample_efficiency(
            config["csv"][method], SAMPLE_EFF_THRESHOLD, env_id
        )

        print(f"    Satisfaction rate:        {sat_rate:.3f} ({successes}/{N_EVAL_EPISODES})")
        print(f"    Mean time to satisfaction: {mean_time:.1f} steps" if mean_time else "    Mean time to satisfaction: N/A (never solved)")
        print(f"    Sample efficiency:         {sample_eff} timesteps" if sample_eff else "    Sample efficiency:         N/A (threshold not reached)")

        results.append({
            "env":         env_name,
            "method":      method,
            "sat_rate":    sat_rate,
            "mean_time":   mean_time,
            "sample_eff":  sample_eff,
        })

# ── save results table ────────────────────────────────────────────────────────
df_results = pd.DataFrame(results)
df_results.to_csv("../results/evaluation_metrics.csv", index=False)
print("\n\nFull results saved to ../results/evaluation_metrics.csv")
print(df_results.to_string(index=False))

# ── Figure: three bar charts side by side ────────────────────────────────────
envs    = [c["env_name"] for c in CONFIGS]
methods = ["Baseline", "LTL", "TLTL"]
x       = np.arange(len(envs))
width   = 0.25

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Evaluation Metrics by Environment and Method",
             fontsize=13, fontweight="bold")

# Satisfaction rate
ax = axes[0]
for i, method in enumerate(methods):
    vals = [df_results[(df_results.env == e) & (df_results.method == method)]["sat_rate"].values[0]
            for e in envs]
    ax.bar(x + i * width, vals, width, label=method,
           color=COLORS[method], alpha=0.85)
ax.set_title("Specification Satisfaction Rate")
ax.set_ylabel("Satisfaction Rate (0–1)")
ax.set_xticks(x + width); ax.set_xticklabels(envs)
ax.set_ylim(0, 1.1); ax.legend(); ax.grid(axis="y", alpha=0.3)

# Mean time to satisfaction
ax = axes[1]
for i, method in enumerate(methods):
    vals = []
    for e in envs:
        v = df_results[(df_results.env == e) & (df_results.method == method)]["mean_time"].values[0]
        vals.append(v if v is not None else 0)
    ax.bar(x + i * width, vals, width, label=method,
           color=COLORS[method], alpha=0.85)
ax.set_title("Mean Time to Satisfaction (steps)")
ax.set_ylabel("Steps (lower = better)")
ax.set_xticks(x + width); ax.set_xticklabels(envs)
ax.legend(); ax.grid(axis="y", alpha=0.3)

# Sample efficiency
ax = axes[2]
for i, method in enumerate(methods):
    vals = []
    for e in envs:
        v = df_results[(df_results.env == e) & (df_results.method == method)]["sample_eff"].values[0]
        vals.append(v if v is not None else 100_000)
    ax.bar(x + i * width, vals, width, label=method,
           color=COLORS[method], alpha=0.85)
ax.set_title("Sample Efficiency\n(timesteps to 80% length reduction)")
ax.set_ylabel("Timesteps (lower = better)")
ax.set_xticks(x + width); ax.set_xticklabels(envs)
ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.annotate("100k = threshold\nnever reached",
            xy=(0.98, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=7, color="gray")

plt.tight_layout()
plt.savefig("../plots/evaluation_metrics.png", dpi=150)
plt.close()
print("\nPlot saved to ../plots/evaluation_metrics.png")