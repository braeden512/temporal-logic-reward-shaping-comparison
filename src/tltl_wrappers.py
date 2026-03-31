import gymnasium as gym
import numpy as np

def normalize(value, min_val, max_val):
    """Normalize a value to [-1, 1] range."""
    return 2 * (value - min_val) / (max_val - min_val) - 1

class TLTLEmptyWrapper(gym.Wrapper):
    """
    Finite-horizon TLTL wrapper for MiniGrid-Empty-8x8-v0.
    Specification: F[0,H](goal)
    Robustness: normalized negative distance to goal.
    Reward is the change in robustness at each step (dense signal).
    """

    def __init__(self, env, horizon=200):
        super().__init__(env)
        self.horizon = horizon
        self.prev_robustness = None
        self.grid_size = 8
        self.max_dist = np.sqrt(2) * self.grid_size

        original_obs_space = env.observation_space
        low = np.append(original_obs_space.low, -1.0)
        high = np.append(original_obs_space.high, 1.0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_robustness = self._compute_robustness()
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        robustness = self._compute_robustness()
        # Dense reward: change in robustness at each step
        shaped_reward = robustness - self.prev_robustness
        # Bonus on goal completion
        if terminated:
            shaped_reward += 1.0
        self.prev_robustness = robustness
        return self._augment_obs(obs), shaped_reward, terminated, truncated, info

    def _compute_robustness(self):
        env = self.env.unwrapped
        agent_pos = np.array(env.agent_pos)
        # Find goal position
        goal_pos = None
        for x in range(env.grid.width):
            for y in range(env.grid.height):
                cell = env.grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    goal_pos = np.array([x, y])
                    break
            if goal_pos is not None:
                break
        if goal_pos is None:
            return 0.0
        dist = np.linalg.norm(agent_pos - goal_pos)
        return normalize(dist, 0, self.max_dist) * -1  # closer = higher robustness

    def _augment_obs(self, obs):
        return np.append(obs, self.prev_robustness).astype(np.float32)


class TLTLLavaGapWrapper(gym.Wrapper):
    """
    Finite-horizon TLTL wrapper for MiniGrid-LavaGapS5-v0.
    Specification: G[0,H](!lava) & F[0,H](goal)
    Robustness: min(lava_safety_margin, goal_progress)
    The min operator encodes the conjunction — both must be satisfied.
    """

    def __init__(self, env, horizon=200):
        super().__init__(env)
        self.horizon = horizon
        self.prev_robustness = None
        self.grid_size = 8
        self.max_dist = np.sqrt(2) * self.grid_size

        original_obs_space = env.observation_space
        low = np.append(original_obs_space.low, -1.0)
        high = np.append(original_obs_space.high, 1.0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_robustness = self._compute_robustness()
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        robustness = self._compute_robustness()
        shaped_reward = robustness - self.prev_robustness
        if terminated:
            agent_pos = np.array(self.env.unwrapped.agent_pos)
            cell = self.env.unwrapped.grid.get(*agent_pos)
            if cell is not None and cell.type == "lava":
                shaped_reward -= 1.0  # lava penalty
            else:
                shaped_reward += 1.0  # goal bonus
        self.prev_robustness = robustness
        return self._augment_obs(obs), shaped_reward, terminated, truncated, info

    def _compute_robustness(self):
        env = self.env.unwrapped
        agent_pos = np.array(env.agent_pos)

        # Lava safety: min distance to any lava cell (higher = safer)
        min_lava_dist = float('inf')
        goal_pos = None
        for x in range(env.grid.width):
            for y in range(env.grid.height):
                cell = env.grid.get(x, y)
                if cell is not None:
                    if cell.type == "lava":
                        d = np.linalg.norm(agent_pos - np.array([x, y]))
                        min_lava_dist = min(min_lava_dist, d)
                    elif cell.type == "goal":
                        goal_pos = np.array([x, y])

        lava_robustness = normalize(min_lava_dist, 0, self.max_dist)

        if goal_pos is None:
            goal_robustness = -1.0
        else:
            dist_to_goal = np.linalg.norm(agent_pos - goal_pos)
            goal_robustness = normalize(dist_to_goal, 0, self.max_dist) * -1

        # Conjunction: min of both robustness values
        return min(lava_robustness, goal_robustness)

    def _augment_obs(self, obs):
        return np.append(obs, self.prev_robustness).astype(np.float32)


class TLTLDoorKeyWrapper(gym.Wrapper):
    """
    Finite-horizon TLTL wrapper for MiniGrid-DoorKey-8x8-v0.
    Specification: F[0,H](key & F[0,H](door & F[0,H](goal)))
    Robustness: weighted progress through subtasks as continuous value.
    """

    def __init__(self, env, horizon=640):
        super().__init__(env)
        self.horizon = horizon
        self.prev_robustness = None
        self.max_dist = np.sqrt(2) * 8

        original_obs_space = env.observation_space
        low = np.append(original_obs_space.low, -1.0)
        high = np.append(original_obs_space.high, 1.0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_robustness = self._compute_robustness()
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        robustness = self._compute_robustness()
        shaped_reward = robustness - self.prev_robustness
        if terminated:
            shaped_reward += 1.0
        self.prev_robustness = robustness
        return self._augment_obs(obs), shaped_reward, terminated, truncated, info

    def _compute_robustness(self):
        env = self.env.unwrapped
        agent_pos = np.array(env.agent_pos)
        carrying = env.carrying

        # Find key, door, goal positions
        key_pos = door_pos = goal_pos = None
        door_open = False
        for x in range(env.grid.width):
            for y in range(env.grid.height):
                cell = env.grid.get(x, y)
                if cell is not None:
                    if cell.type == "key" and key_pos is None:
                        key_pos = np.array([x, y])
                    elif cell.type == "door":
                        door_pos = np.array([x, y])
                        door_open = cell.is_open
                    elif cell.type == "goal":
                        goal_pos = np.array([x, y])

        has_key = carrying is not None and carrying.type == "key"

        if has_key and door_open and goal_pos is not None:
            # Phase 3: get to goal
            dist = np.linalg.norm(agent_pos - goal_pos)
            return 0.67 + 0.33 * (normalize(dist, 0, self.max_dist) * -1)
        elif has_key and door_pos is not None:
            # Phase 2: get to door
            dist = np.linalg.norm(agent_pos - door_pos)
            return 0.33 + 0.33 * (normalize(dist, 0, self.max_dist) * -1)
        elif key_pos is not None:
            # Phase 1: get to key
            dist = np.linalg.norm(agent_pos - key_pos)
            return 0.0 + 0.33 * (normalize(dist, 0, self.max_dist) * -1)
        else:
            return -1.0

    def _augment_obs(self, obs):
        return np.append(obs, self.prev_robustness).astype(np.float32)