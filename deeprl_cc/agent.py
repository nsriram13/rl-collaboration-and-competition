# -*- coding: utf-8 -*-
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import FullyConnectedActor, FullyConnectedCritic, tensor
from .utils import OUNoise


class DDPGAgent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        sizes: List[int] = None,
        activations: List[str] = None,
        gamma: float = 0.99,
        lrate_actor: float = 3e-4,
        lrate_critic: float = 3e-4,
        tau: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        update_freq: int = 1,
        update_passes: int = 5,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        seed: int = 0,
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.gamma = gamma
        self.tau = tau

        self.lrate_actor = lrate_actor
        self.lrate_critic = lrate_critic
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.update_epochs = update_passes

        if sizes is None:
            sizes = [64, 64]

        if activations is None:
            activations = ["relu", "relu"]

        # actor network w/ target network
        self.actor_local = FullyConnectedActor(
            state_dim, action_dim, sizes, activations, seed=seed
        ).to(self.device)

        self.actor_target = FullyConnectedActor(
            state_dim, action_dim, sizes, activations, seed=seed
        ).to(self.device)

        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(),
            lr=self.lrate_actor,
            weight_decay=self.weight_decay,
        )

        # critic network w/ target network
        self.critic_local = FullyConnectedCritic(
            state_dim, action_dim, sizes, activations, num_agents, seed=seed
        ).to(self.device)

        self.critic_target = FullyConnectedCritic(
            state_dim, action_dim, sizes, activations, num_agents, seed=seed
        ).to(self.device)

        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=self.lrate_critic,
            weight_decay=self.weight_decay,
        )

        # set target network weights to an exact copy of local network
        self.soft_update(self.actor_local, self.actor_target, tau=1)
        self.soft_update(self.critic_local, self.critic_target, tau=1)

        self.noise = OUNoise(action_dim)

    def step(self, shared_memory, timestep):
        if len(shared_memory) > self.batch_size and timestep % self.update_freq == 0:
            for _ in range(self.update_epochs):
                experiences = shared_memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = tensor(state)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action = self.noise.get_action(action)

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states_list, actions_list, rewards, next_states_list, dones = experiences

        next_states_tensor = torch.cat(next_states_list, dim=1).to(self.device)
        states_tensor = torch.cat(states_list, dim=1).to(self.device)
        actions_tensor = torch.cat(actions_list, dim=1).to(self.device)

        # First, optimize Q networks; minimizing MSE between
        # Q(s, a) & r + discount * Q_target(next_s, next_a)
        # Utilize data from both agents for training the critic networks
        # Q(s, a)
        Q_expected = self.critic_local(states_tensor, actions_tensor)

        # r + discount * Q_target(next_s, next_a)
        next_actions = [self.actor_target(states) for states in states_list]
        next_actions_tensor = torch.cat(next_actions, dim=1).to(self.device)
        Q_targets_next = self.critic_target(next_states_tensor, next_actions_tensor)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # MSE loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Second, optimize the actor; maximizing Q(s, actor_action)
        # or minimizing -Q(s, actor_action)
        actions_pred = [self.actor_local(states) for states in states_list]
        actions_pred_tensor = torch.cat(actions_pred, dim=1).to(self.device)

        # -1 * (maximize) Q value for the current prediction
        actor_loss = -self.critic_local(states_tensor, actions_pred_tensor).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft-update target networks
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

        # reset exploration noise
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau=None):

        if tau is None:
            tau = self.tau

        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
