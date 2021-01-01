# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from collections import deque, namedtuple


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, num_agents, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal buffer (deque)
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.experience = namedtuple(
            "Experience",
            field_names=["states", "actions", "rewards", "next_states", "dones"],
        )
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to buffer."""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """Randomly sample a batch of experiences from buffer."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states_list = [
            torch.from_numpy(
                np.vstack([e.states[index] for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
            for index in range(self.num_agents)
        ]
        actions_list = [
            torch.from_numpy(
                np.vstack([e.actions[index] for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
            for index in range(self.num_agents)
        ]
        next_states_list = [
            torch.from_numpy(
                np.vstack([e.next_states[index] for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
            for index in range(self.num_agents)
        ]
        rewards = (
            torch.from_numpy(
                np.vstack([e.rewards for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.dones for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )

        return states_list, actions_list, rewards, next_states_list, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
