# -*- coding: utf-8 -*-
import numpy as np
from typing import List

import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(DEVICE)
    return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


# Source: https://github.com/ShangtongZhang/DeepRL/blob/932ea88082e0194126b87742bd4a28c4599aa1b8/deep_rl/network/network_utils.py#L23  # noqa: E501
# Fills the input Tensor with a (semi) orthogonal matrix, as described in Exact
# solutions to the nonlinear dynamics of learning in deep linear neural networks
# - Saxe, A. et al. (2013).
def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


# Utility module to initialize a fully connected neural network
# Source: https://github.com/facebookresearch/ReAgent/blob/86b92279b635b38b775032d68979d2e4b52f415c/reagent/models/fully_connected_network.py#L31  # noqa: E501
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

ACTIVATION_MAP = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "linear": Identity,
}


class FullyConnectedNetwork(nn.Module):
    def __init__(
        self,
        layers,
        activations,
        *,
        use_batch_norm=False,
        dropout_ratio=0.0,
        use_layer_norm=False,
        normalize_output=False,
    ) -> None:
        super(FullyConnectedNetwork, self).__init__()

        self.input_dim = layers[0]

        modules: List[nn.Module] = []

        assert len(layers) == len(activations) + 1

        for i, ((in_dim, out_dim), activation) in enumerate(
            zip(zip(layers, layers[1:]), activations)
        ):
            # Add BatchNorm1d
            if use_batch_norm:
                modules.append(nn.BatchNorm1d(in_dim))

            # Add Linear
            if activation == "linear":
                w_scale = 1e-3
            else:
                w_scale = 1
            linear = layer_init(nn.Linear(in_dim, out_dim), w_scale)
            modules.append(linear)

            # Add LayerNorm
            if use_layer_norm and (normalize_output or i < len(activations) - 1):
                modules.append(nn.LayerNorm(out_dim))  # type: ignore

            # Add activation
            if activation in ACTIVATION_MAP:
                modules.append(ACTIVATION_MAP[activation]())
            else:
                # See if it matches any of the nn modules
                modules.append(getattr(nn, activation)())

            # Add Dropout
            if dropout_ratio > 0.0 and (normalize_output or i < len(activations) - 1):
                modules.append(nn.Dropout(p=dropout_ratio))

        self.dnn = nn.Sequential(*modules)  # type: ignore

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for generic feed-forward DNNs. Assumes activation names
        are valid pytorch activation names.
        :param input tensor
        """
        return self.dnn(input)


class FullyConnectedActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sizes: List[int] = None,
        activations: List[str] = None,
        output_activation: str = "tanh",
        seed: int = 0,
    ):
        super().__init__()

        if sizes is None:
            sizes = [64, 64]

        if activations is None:
            activations = ["relu", "relu"]

        assert len(sizes) == len(
            activations
        ), "The numbers of sizes and activations must match; got {} vs {}".format(
            len(sizes), len(activations)
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.random_seed = torch.manual_seed(seed)
        self.fc = FullyConnectedNetwork(
            [state_dim] + sizes + [action_dim],
            activations + [output_activation],
            use_batch_norm=False,
            use_layer_norm=False,
        )

    def forward(self, state):
        state = tensor(state)
        return self.fc(state)


class FullyConnectedCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sizes: List[int] = None,
        activations: List[str] = None,
        num_agents: int = 1,
        seed: int = 0,
    ):
        super().__init__()

        if sizes is None:
            sizes = [64, 64]

        if activations is None:
            activations = ["relu", "relu"]

        assert len(sizes) == len(
            activations
        ), "The numbers of sizes and activations must match; got {} vs {}".format(
            len(sizes), len(activations)
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.random_seed = torch.manual_seed(seed)
        self.fc = FullyConnectedNetwork(
            [(state_dim + action_dim) * num_agents] + sizes + [1],
            activations + ["linear"],
            use_batch_norm=False,
            use_layer_norm=False,
        )

    def forward(self, state, action):
        cat_input = torch.cat((state, action), dim=1)
        return self.fc(cat_input)
