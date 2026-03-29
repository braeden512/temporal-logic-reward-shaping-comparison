import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

# Automaton states
Q_RUNNING = 0
Q_ACCEPT  = 1
Q_DEAD    = 2

class LTLLavaGapWrapper(gym.Wrapper):
    """
    Infinite-horizon LTL wrapper for MiniGrid-LavaGapS5-v0.
    Specification: G(!lava) & F(goal)
    Automaton states:
        Q_RUNNING (0): agent is alive and has not reached goal
        Q_ACCEPT  (1): agent reached goal without hitting lava
        Q_DEAD    (2): agent stepped on lava (absorbing failure state)

    Reward shaping based on automaton transitions:
        Q_RUNNING -> Q_ACCEPT : +1.0 (task complete)
        Q_RUNNING -> Q_DEAD   : -1.0 (safety violation)
        staying in Q_RUNNING  :  0.0 (no progress signal)
    """

    def __init__(self, env):
        super().__init__(env)
        self.automaton_state = Q_RUNNING

        # Extend observation space to include automaton state
        original_obs_space = env.observation_space
        low = np.append(original_obs_space.low, 0)
        high = np.append(original_obs_space.high, 2)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=original_obs_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.automaton_state = Q_RUNNING
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Read automaton state from environment
        new_automaton_state = self._get_automaton_state(terminated, info)

        # Shape reward based on automaton transition
        shaped_reward = self._shape_reward(self.automaton_state, new_automaton_state)
        self.automaton_state = new_automaton_state

        return self._augment_obs(obs), shaped_reward, terminated, truncated, info

    def _get_automaton_state(self, terminated, info):
        if self.automaton_state == Q_DEAD:
            return Q_DEAD  # absorbing state

        agent_pos = self.env.unwrapped.agent_pos
        grid = self.env.unwrapped.grid
        cell = grid.get(*agent_pos)

        if cell is not None and cell.type == "lava":
            return Q_DEAD
        if terminated and info.get("success", False):
            return Q_ACCEPT
        # Also check if agent reached goal via termination with positive reward
        if terminated and not info.get("success", False):
            # MiniGrid marks success via termination + positive reward
            return Q_ACCEPT if self.env.unwrapped.carrying is None else Q_RUNNING

        return Q_RUNNING

    def _shape_reward(self, prev_state, new_state):
        if prev_state == Q_RUNNING and new_state == Q_ACCEPT:
            return 1.0
        if prev_state == Q_RUNNING and new_state == Q_DEAD:
            return -1.0
        return 0.0

    def _augment_obs(self, obs):
        return np.append(obs, self.automaton_state)


class LTLEmptyWrapper(gym.Wrapper):
    """
    Infinite-horizon LTL wrapper for MiniGrid-Empty-8x8-v0.
    Specification: F(goal)
    Automaton states:
        Q_RUNNING (0): has not reached goal
        Q_ACCEPT  (1): reached goal
    """

    def __init__(self, env):
        super().__init__(env)
        self.automaton_state = Q_RUNNING
        original_obs_space = env.observation_space
        low = np.append(original_obs_space.low, 0)
        high = np.append(original_obs_space.high, 2)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=original_obs_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.automaton_state = Q_RUNNING
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_automaton_state = Q_ACCEPT if terminated else Q_RUNNING
        shaped_reward = 1.0 if (self.automaton_state == Q_RUNNING and new_automaton_state == Q_ACCEPT) else 0.0
        self.automaton_state = new_automaton_state
        return self._augment_obs(obs), shaped_reward, terminated, truncated, info

    def _augment_obs(self, obs):
        return np.append(obs, self.automaton_state)


class LTLDoorKeyWrapper(gym.Wrapper):
    """
    Infinite-horizon LTL wrapper for MiniGrid-DoorKey-8x8-v0.
    Specification: F(key & F(door & F(goal)))
    Automaton states:
        Q_RUNNING (0): no key yet
        Q_HAS_KEY (1): has key, door not yet opened
        Q_DOOR_OPEN (2): door opened, goal not yet reached
        Q_ACCEPT (3): goal reached
    """

    Q_NO_KEY    = 0
    Q_HAS_KEY   = 1
    Q_DOOR_OPEN = 2
    Q_ACCEPT    = 3

    def __init__(self, env):
        super().__init__(env)
        self.automaton_state = self.Q_NO_KEY
        original_obs_space = env.observation_space
        low = np.append(original_obs_space.low, 0)
        high = np.append(original_obs_space.high, 3)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=original_obs_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.automaton_state = self.Q_NO_KEY
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_automaton_state = self._get_automaton_state(terminated)
        shaped_reward = self._shape_reward(self.automaton_state, new_automaton_state)
        self.automaton_state = new_automaton_state
        return self._augment_obs(obs), shaped_reward, terminated, truncated, info

    def _get_automaton_state(self, terminated):
        env = self.env.unwrapped
        carrying = env.carrying

        if self.automaton_state == self.Q_ACCEPT:
            return self.Q_ACCEPT

        # Check goal reached
        if terminated:
            return self.Q_ACCEPT

        # Check door opened
        if self.automaton_state >= self.Q_HAS_KEY:
            # Look for an open door in the grid
            for x in range(env.grid.width):
                for y in range(env.grid.height):
                    cell = env.grid.get(x, y)
                    if cell is not None and cell.type == "door" and cell.is_open:
                        return self.Q_DOOR_OPEN

        # Check if carrying key
        if carrying is not None and carrying.type == "key":
            return self.Q_HAS_KEY

        return self.automaton_state

    def _shape_reward(self, prev_state, new_state):
        if new_state > prev_state:
            # Reward each automaton transition
            return 1.0
        return 0.0

    def _augment_obs(self, obs):
        return np.append(obs, self.automaton_state)