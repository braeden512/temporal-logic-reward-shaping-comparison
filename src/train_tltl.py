import gymnasium as gym
import csv
import os
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from tltl_wrappers import TLTLEmptyWrapper, TLTLLavaGapWrapper, TLTLDoorKeyWrapper

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
                    self.writer.writerow([self.num_timesteps, round(mean_rew, 4), round(mean_len, 2)])
                    self.file.flush()
        return True

    def _on_training_end(self):
        self.file.close()

ENVS = [
    ("MiniGrid-Empty-8x8-v0",   TLTLEmptyWrapper,   "empty"),
    ("MiniGrid-LavaGapS5-v0",   TLTLLavaGapWrapper, "lava"),
    ("MiniGrid-DoorKey-8x8-v0", TLTLDoorKeyWrapper, "doorkey"),
]

for env_id, WrapperClass, name in ENVS:
    print(f"\n{'='*50}")
    print(f"Training on: {env_id} with TLTL wrapper")
    print(f"{'='*50}")

    def make_env():
        env = gym.make(env_id)
        env = FlatObsWrapper(env)
        env = WrapperClass(env)
        return env

    env = make_vec_env(make_env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        device='cpu',
    )

    callback = MetricsCallback(f"results/{name}_tltl.csv")
    model.learn(total_timesteps=100_000, callback=callback)
    model.save(f"ppo_{name}_tltl")
    print(f"Done! Saved ppo_{name}_tltl")

print("\nAll TLTL environments complete!")