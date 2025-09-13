import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    """Causal t→t+1 trading environment with transaction cost + slippage."""
    def __init__(self, returns, cost_bps=10, slippage_bps=5):
        super().__init__()
        self.returns = returns
        self.n_steps = len(returns)
        self.cost = cost_bps / 10000
        self.slippage = slippage_bps / 10000
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # exposure [0,1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.t = 0
        self.equity = 1.0
        return np.array([self.returns[self.t]]), {}

    def step(self, action):
        prev_eq = self.equity
        r = self.returns[self.t]
        effective_r = action[0] * (r - self.cost - self.slippage)
        self.equity *= (1 + effective_r)
        self.t += 1
        done = self.t >= self.n_steps - 1
        return np.array([self.returns[self.t]]), self.equity - prev_eq, done, False, {}
