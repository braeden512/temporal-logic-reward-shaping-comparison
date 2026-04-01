import gymnasium as gym
import csv
import os
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from ltl_wrappers import LTLEmptyWrapper, LTLLavaGapWrapper, LTLDoorKeyWrapper
from tltl_wrappers import TLTLEmptyWrapper, TLTLLavaGapWrapper, TLTLDoorKeyWrapper

N_RUNS = 30
TIMESTEPS = 100_000

class MetricsCallback(BaseCallback):
    def __init__(self, filepath, verbose=0):
        super().__init__(verbose)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timesteps", "ep_rew_mean", "ep_len_mean"])
        self.ep_rewards = []
        self.ep_lengths = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                if len(self.ep_rewards) >= 10:
                    mean_rew = sum(self.ep_rewards[-10:]) / 10
                    mean_len = sum(self.ep_lengths[-10:]) / 10
                    self.writer.writerow([
                        self.num_timesteps,
                        round(mean_rew, 4),
                        round(mean_len, 2)
                    ])
                    self.file.flush()
        return True

    def _on_training_end(self):
        self.file.close()

CONFIGS = [
    {"name": "empty_baseline",   "env_id": "MiniGrid-Empty-8x8-v0",   "wrapper": None},
    {"name": "lava_baseline",    "env_id": "MiniGrid-LavaGapS5-v0",   "wrapper": None},
    {"name": "doorkey_baseline", "env_id": "MiniGrid-DoorKey-8x8-v0", "wrapper": None},
    {"name": "empty_ltl",        "env_id": "MiniGrid-Empty-8x8-v0",   "wrapper": LTLEmptyWrapper},
    {"name": "lava_ltl",         "env_id": "MiniGrid-LavaGapS5-v0",   "wrapper": LTLLavaGapWrapper},
    {"name": "doorkey_ltl",      "env_id": "MiniGrid-DoorKey-8x8-v0", "wrapper": LTLDoorKeyWrapper},
    {"name": "empty_tltl",       "env_id": "MiniGrid-Empty-8x8-v0",   "wrapper": TLTLEmptyWrapper},
    {"name": "lava_tltl",        "env_id": "MiniGrid-LavaGapS5-v0",   "wrapper": TLTLLavaGapWrapper},
    {"name": "doorkey_tltl",     "env_id": "MiniGrid-DoorKey-8x8-v0", "wrapper": TLTLDoorKeyWrapper},
]

total     = len(CONFIGS) * N_RUNS
completed = 0

for run in range(1, N_RUNS + 1):
    for config in CONFIGS:
        completed += 1
        name    = config["name"]
        env_id  = config["env_id"]
        Wrapper = config["wrapper"]

        print(f"\n{'='*55}")
        print(f"[{completed}/{total}] Run {run}/30 — {name}")
        print(f"{'='*55}")

        def make_env():
            env = gym.make(env_id, max_steps=640)
            env = FlatObsWrapper(env)
            if Wrapper is not None:
                env = Wrapper(env)
            return env

        env = make_vec_env(make_env, n_envs=1)

        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            device='cpu',
        )

        csv_path   = f"../results/seeds30/{name}_run{run}.csv"
        model_path = f"../models/seeds30/{name}_run{run}"
        os.makedirs("../models/seeds30", exist_ok=True)

        callback = MetricsCallback(csv_path)
        model.learn(total_timesteps=TIMESTEPS, callback=callback)
        model.save(model_path)

        print(f"  Saved: {model_path}")
        print(f"  CSV:   {csv_path}")

print(f"\n{'='*55}")
print("All 270 runs complete!")
print(f"{'='*55}")