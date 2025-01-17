# -*- coding: utf-8 -*-
import numpy as np


# Ornstein-Ulhenbeck Process
# Source: https://github.com/vitchyr/rlkit/blob/5274672e9ff6481def0ffed61cd1b1c52210a840/rlkit/exploration_strategies/ou_strategy.py#L7  # noqa: E501
class OUNoise:
    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        max_sigma: float = 0.3,
        min_sigma: float = 0.3,
        decay_period: int = 100000,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.low = -1
        self.high = 1
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.clip(action + ou_state, self.low, self.high)


# Plotting utilities from OpenAI baselines package
# Source: https://github.com/openai/baselines/blob/master/baselines/common/plot_util.py

# Plotting utility to smooth out a noisy series
def smooth(y, radius, mode='two_sided', valid_only=False):
    """
    Smooth signal y, where radius is determines the size of the window
    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    """
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / np.convolve(
            np.ones_like(y), convkernel, mode='same'
        )
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / np.convolve(
            np.ones_like(y), convkernel, mode='full'
        )
        out = out[: -radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out
