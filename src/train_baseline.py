import gymnasium as gym
import csv
import os
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

class MetricsCallback(BaseCallback):
    def __init__(self, filepath, verbose=0):
        super().__init__(verbose)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
        self.file = open(filepath, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timesteps", "ep_rew_mean", "ep_len_mean"])
        self.ep_rewards = []
        self.ep_lengths = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])

                # Only write when we have enough episodes, and only once per episode
                if len(self.ep_rewards) >= 10:
                    mean_rew = sum(self.ep_rewards[-10:]) / 10
                    mean_len = sum(self.ep_lengths[-10:]) / 10
                    self.writer.writerow([self.num_timesteps, round(mean_rew, 4), round(mean_len, 2)])
                    self.file.flush()

        return True

    def _on_training_end(self):
        self.file.close()

ENVS = [
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-DoorKey-8x8-v0",
]

for env_id in ENVS:
    print(f"\n{'='*50}")
    print(f"Training on: {env_id}")
    print(f"{'='*50}")

    env = make_vec_env(env_id, n_envs=1, wrapper_class=FlatObsWrapper)

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

    save_name = env_id.replace("/", "_").replace("-", "_").lower()
    callback = MetricsCallback(f"../results/{save_name}_baseline.csv")
    model.learn(total_timesteps=100_000, callback=callback)
    model.save(f"../ppo_{save_name}")
    print(f"Done! Saved ppo_{save_name}")

print("\nAll environments complete!")