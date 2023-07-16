from typing import Any

import torch
from torch import nn


class SingleInputModel(nn.Module):
    """Model taking a single input."""

    def forward(self, x: Any):
        return x


class DoubleInputModel(nn.Module):
    """Model taking two inputs."""

    def forward(self, x: Any, y: Any):
        return x, y


class UnsupportedInputModel(nn.Module):
    """Model taking an unsupported input."""

    def forward(self, x: torch.Tensor, y: str):
        y = f"unsupported input: {y}"

        return x
