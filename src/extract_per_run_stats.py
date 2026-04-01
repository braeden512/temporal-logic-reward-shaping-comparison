import numpy as np
import pandas as pd
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from ltl_wrappers import LTLEmptyWrapper, LTLLavaGapWrapper, LTLDoorKeyWrapper
from tltl_wrappers import TLTLEmptyWrapper, TLTLLavaGapWrapper, TLTLDoorKeyWrapper

N_RUNS = 30
N_EVAL_EPISODES = 100

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

rows = []

for config in CONFIGS:
    name      = config["name"]
    env_id    = config["env_id"]
    env_label = config["env_label"]
    method    = config["method"]
    Wrapper   = config["wrapper"]

    for run in range(1, N_RUNS + 1):
        model_path = f"../models/seeds30/{name}_run{run}"
        print(f"  Evaluating {name} run {run}...")
        model = PPO.load(model_path, device='cpu')

        successes = 0
        for _ in range(N_EVAL_EPISODES):
            env = make_env(env_id, Wrapper)
            obs, _ = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=False)
                obs, _, terminated, truncated, _ = env.step(action)
            if is_success(env, terminated, truncated):
                successes += 1
            env.close()

        sat_rate = successes / N_EVAL_EPISODES
        rows.append({
            "env":     env_label,
            "method":  method,
            "run":     run,
            "sat_rate": sat_rate,
        })
        print(f"    run {run} sat_rate: {sat_rate:.3f}")

df = pd.DataFrame(rows)
df.to_csv("../results/per_run_sat_rates.csv", index=False)
print("\nSaved ../results/per_run_sat_rates.csv")
print(df.to_string(index=False))